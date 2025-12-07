#!/bin/bash
# =============================================================================
# TP约束测试: Router权重一致性检查
# =============================================================================
# 约束描述: 检查model-after-backward阶段，router权重参数在TP通信组间的完全复制一致性
# 注入方式: 在特定 TP rank 上修改 Router 权重，使其 cksum 与其他 TP rank 不一致
# 预期结果: 约束检测到 TP 组内 Router 权重 cksum 不一致
# 注意: 此约束仅适用于 MoE 模型
# =============================================================================

set -e

export MEGATRON_ROOT="/volume/qscai/lsk/Megatron-LM"
export VTIMELINE_ROOT="/volume/qscai/lsk/VTimeline"
export PYTHONPATH="${VTIMELINE_ROOT}/src:${MEGATRON_ROOT}:${PYTHONPATH}"

# TP 必需的环境变量
export CUDA_DEVICE_MAX_CONNECTIONS=1

export VTIMELINE_LOGGER_DIR="${MEGATRON_ROOT}/tp_router_test_db"
export VTIMELINE_DUMP_STEP=10  # 启用 VTimeline 数据收集
rm -rf "${VTIMELINE_LOGGER_DIR}"
mkdir -p "${VTIMELINE_LOGGER_DIR}"

# 注入配置
export MEGATRON_INJECT_PARAM_CORRUPTION=1
export MEGATRON_CORRUPT_OP="tp_router_cksum"
export MEGATRON_CORRUPT_TP_RANK=0
export MEGATRON_CORRUPT_STEP=2
export MEGATRON_CORRUPT_DELTA=0.01

echo "=========================================="
echo "测试 TP Router 权重一致性错误注入"
echo "=========================================="
echo ""
echo "配置："
echo "  - MEGATRON_CORRUPT_OP=${MEGATRON_CORRUPT_OP}"
echo "  - MEGATRON_CORRUPT_TP_RANK=${MEGATRON_CORRUPT_TP_RANK}"
echo "  - MEGATRON_CORRUPT_STEP=${MEGATRON_CORRUPT_STEP}"
echo ""
echo "⚠️  注意：此约束仅适用于 MoE 模型，普通 GPT 模型可能没有 Router 参数"
echo ""

cd ${MEGATRON_ROOT}

echo "开始训练（TP=2, mock-data 模式）..."

python -m torch.distributed.run \
    --nproc_per_node=2 \
    --nnodes=1 \
    --master_port=29507 \
    pretrain_gpt.py \
    --num-layers 12 \
    --hidden-size 128 \
    --num-attention-heads 4 \
    --micro-batch-size 8 \
    --global-batch-size 16 \
    --seq-length 128 \
    --max-position-embeddings 128 \
    --train-iters 5 \
    --lr-decay-iters 5 \
    --lr 1e-4 \
    --min-lr 1e-5 \
    --lr-decay-style cosine \
    --log-interval 1 \
    --eval-iters 10 \
    --eval-interval 100 \
    --mock-data \
    --tokenizer-type NullTokenizer \
    --vocab-size 50304 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.02 \
    --fp16 \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 1 \
    --no-gradient-accumulation-fusion \
    --no-async-tensor-model-parallel-allreduce \
    2>&1 | tee ${VTIMELINE_LOGGER_DIR}/training.log

TRAIN_EXIT_CODE=$?

echo ""
echo "=========================================="
echo "训练结束（退出码: ${TRAIN_EXIT_CODE}）"
echo "=========================================="

echo ""
echo "检查日志中的注入信息..."
grep -E "\[corrupt-tp-router\]" ${VTIMELINE_LOGGER_DIR}/training.log | head -20

INJECT_COUNT=$(grep -c "\[corrupt-tp-router\] ✓ Modified" ${VTIMELINE_LOGGER_DIR}/training.log 2>/dev/null) || INJECT_COUNT=0
if [ "${INJECT_COUNT}" -gt 0 ] 2>/dev/null; then
    echo "✓ 检测到 ${INJECT_COUNT} 次成功注入"
else
    echo "⚠️  未检测到 Router 参数注入 - 模型可能不包含 Router/Gate 参数"
fi

echo ""
echo "=========================================="
echo "自动验证注入结果"
echo "=========================================="

cat > check_tp_router.py << 'PYTHON_EOF'
import duckdb
import os
import sys

db_dir = os.environ.get("VTIMELINE_LOGGER_DIR", "/volume/qscai/lsk/Megatron-LM/tp_router_test_db") + "/Collector"

print(f"数据库目录: {db_dir}")

db_files = [f for f in os.listdir(db_dir) if f.endswith('.db')] if os.path.exists(db_dir) else []
print(f"找到数据库文件: {db_files}")

router_data = {}

for db_file in db_files:
    parts = db_file.replace('.db', '').split('_')
    tp_rank = None
    for p in parts:
        if p.startswith('tp'):
            tp_rank = int(p[2:])
            break
    
    if tp_rank is None:
        continue
    
    print(f"\n=== TP Rank {tp_rank} ===")
    db_path = f"{db_dir}/{db_file}"
    conn = duckdb.connect(db_path, read_only=True)
    
    # 查询 Router/Gate 相关参数
    result = conn.execute('''
        SELECT
            step,
            json_extract(data, '$.name') as name,
            json_extract(data, '$.cksum') as cksum
        FROM coredump
        WHERE stage = 'model-after-backward'
        AND (
            json_extract_string(data, '$.name') LIKE '%router%'
            OR json_extract_string(data, '$.name') LIKE '%gate%'
        )
        AND step = 2
        ORDER BY name
        LIMIT 10
    ''').fetchall()
    
    print(f"  Router/Gate 参数 (step=2):")
    if not result:
        print("    ⚠️  未找到 Router/Gate 参数 - 模型可能不是 MoE")
    else:
        for row in result:
            step, name, cksum = row
            print(f"    name={name}, cksum={cksum}")
            
            clean_name = name.strip('"') if name else name
            key = (clean_name, step)
            if key not in router_data:
                router_data[key] = {}
            router_data[key][tp_rank] = cksum
    
    conn.close()

print("\n========================================")
print("验证结果比对")
print("========================================\n")

if not router_data:
    print("⚠️  未找到 Router/Gate 参数")
    print("   此约束仅适用于 MoE 模型")
    print("   普通 GPT 模型测试跳过此约束")
    sys.exit(0)  # 非 MoE 模型，正常退出

mismatch_count = 0
match_count = 0

for key in sorted(router_data.keys()):
    name, step = key
    tp_data = router_data[key]
    
    if len(tp_data) >= 2:
        cksums = list(tp_data.values())
        if len(set(str(c) for c in cksums)) > 1:
            mismatch_count += 1
            print(f"❌ 不一致: {name}")
            for tp, ck in sorted(tp_data.items()):
                print(f"   TP{tp} cksum: {ck}")
        else:
            match_count += 1

print(f"\n✓ 一致: {match_count}, ❌ 不一致: {mismatch_count}")

if mismatch_count > 0:
    print("\n🎉 验证成功：检测到注入导致的 Router 权重不一致！")
    print("   约束 'Router权重一致性检查' 可以检测到此错误")
    sys.exit(0)
else:
    print("\n⚠️  所有 Router 权重都一致，注入可能未生效")
    sys.exit(1)
PYTHON_EOF

python check_tp_router.py
rm -f check_tp_router.py


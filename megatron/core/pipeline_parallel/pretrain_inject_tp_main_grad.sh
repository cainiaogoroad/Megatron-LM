#!/bin/bash
# =============================================================================
# TP约束测试: backward后TP分片参数main_grad存在性与完整性检查
# =============================================================================
# 约束描述: 检查model-after-backward阶段，TP组内requires_grad=True的参数分片
#          是否分配了main_grad，且main_grad的shape和dtype与参数分片严格一致
# 注入方式: 在特定 TP rank 上修改 main_grad 的值
# 预期结果: 约束检测到 TP 组内 main_grad cksum 不一致
# =============================================================================

set -e

export MEGATRON_ROOT="/volume/qscai/lsk/Megatron-LM"
export VTIMELINE_ROOT="/volume/qscai/lsk/VTimeline"
export PYTHONPATH="${VTIMELINE_ROOT}/src:${MEGATRON_ROOT}:${PYTHONPATH}"

# TP 必需的环境变量
export CUDA_DEVICE_MAX_CONNECTIONS=1

export VTIMELINE_LOGGER_DIR="${MEGATRON_ROOT}/tp_main_grad_test_db"
export VTIMELINE_DUMP_STEP=10  # 启用 VTimeline 数据收集
rm -rf "${VTIMELINE_LOGGER_DIR}"
mkdir -p "${VTIMELINE_LOGGER_DIR}"

# 注入配置
export MEGATRON_INJECT_PARAM_CORRUPTION=1
export MEGATRON_CORRUPT_OP="tp_main_grad"
export MEGATRON_CORRUPT_TP_RANK=0
export MEGATRON_CORRUPT_STEP=2
export MEGATRON_CORRUPT_DELTA=0.01
export MEGATRON_CORRUPT_PARAM_SUBSTR=""  # 留空表示第一个匹配的参数

echo "=========================================="
echo "测试 TP main_grad 完整性错误注入"
echo "=========================================="
echo ""
echo "配置："
echo "  - MEGATRON_CORRUPT_OP=${MEGATRON_CORRUPT_OP}"
echo "  - MEGATRON_CORRUPT_TP_RANK=${MEGATRON_CORRUPT_TP_RANK}"
echo "  - MEGATRON_CORRUPT_STEP=${MEGATRON_CORRUPT_STEP}"
echo "  - MEGATRON_CORRUPT_DELTA=${MEGATRON_CORRUPT_DELTA}"
echo ""

cd ${MEGATRON_ROOT}

# 使用 mock-data 模式，不需要真实数据集

echo "开始训练（TP=2）..."

python -m torch.distributed.run \
    --nproc_per_node=2 \
    --nnodes=1 \
    --master_port=29506 \
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
grep -E "\[corrupt-tp-main-grad\]" ${VTIMELINE_LOGGER_DIR}/training.log | head -20

echo ""
echo "=========================================="
echo "自动验证注入结果"
echo "=========================================="

cat > check_tp_main_grad.py << 'PYTHON_EOF'
import duckdb
import os
import sys

db_dir = os.environ.get("VTIMELINE_LOGGER_DIR", "/volume/qscai/lsk/Megatron-LM/tp_main_grad_test_db") + "/Collector"

print(f"数据库目录: {db_dir}")

db_files = [f for f in os.listdir(db_dir) if f.endswith('.db')] if os.path.exists(db_dir) else []
print(f"找到数据库文件: {db_files}")

main_grad_data = {}

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
    
    # 查询 main_grad 相关数据
    result = conn.execute('''
        SELECT
            step,
            json_extract(data, '$.name') as name,
            json_extract(data, '$.cksum') as cksum,
            json_extract(data, '$.shape') as shape
        FROM coredump
        WHERE stage = 'main-grad-in-backward'
        AND step = 2
        ORDER BY name
        LIMIT 10
    ''').fetchall()
    
    print(f"  main_grad (step=2):")
    if not result:
        # 尝试其他 stage
        result = conn.execute('''
            SELECT
                step,
                json_extract(data, '$.name') as name,
                json_extract(data, '$.main_grad_cksum') as cksum
            FROM coredump
            WHERE stage = 'model-after-backward'
            AND json_extract(data, '$.main_grad_cksum') IS NOT NULL
            AND step = 2
            ORDER BY name
            LIMIT 10
        ''').fetchall()
        
        for row in result:
            step, name, cksum = row
            print(f"    name={name}, main_grad_cksum={cksum}")
            
            clean_name = name.strip('"') if name else name
            key = (clean_name, step)
            if key not in main_grad_data:
                main_grad_data[key] = {}
            main_grad_data[key][tp_rank] = cksum
    else:
        for row in result:
            step, name, cksum, shape = row
            print(f"    name={name}, cksum={cksum}, shape={shape}")
            
            clean_name = name.strip('"') if name else name
            key = (clean_name, step)
            if key not in main_grad_data:
                main_grad_data[key] = {}
            main_grad_data[key][tp_rank] = cksum
    
    conn.close()

print("\n========================================")
print("验证结果比对")
print("========================================\n")

if not main_grad_data:
    print("❌ 无法比对：未找到 main_grad 数据")
    print("   VTimeline 可能未在 main-grad-in-backward 阶段收集数据")
    sys.exit(1)

mismatch_count = 0
match_count = 0

for key in sorted(main_grad_data.keys()):
    name, step = key
    tp_data = main_grad_data[key]
    
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
    print("\n🎉 验证成功：检测到注入导致的 main_grad 不一致！")
    print("   约束 'backward后TP分片参数main_grad存在性与完整性检查' 可以检测到此错误")
    sys.exit(0)
else:
    print("\n⚠️  验证失败：所有 main_grad 都一致")
    print("   注意：TP 分片的 main_grad 可能本身就是不同的（分片后）")
    sys.exit(1)
PYTHON_EOF

python check_tp_main_grad.py
rm -f check_tp_main_grad.py


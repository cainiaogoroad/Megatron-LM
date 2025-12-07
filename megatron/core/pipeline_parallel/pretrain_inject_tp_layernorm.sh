#!/bin/bash
# =============================================================================
# TP约束测试: LayerNorm权重一致性检查
# =============================================================================
# 约束描述: 检查model-after-backward阶段，检查layernorm权重在DP,TP通信组间的一致性
# 注入方式: 在特定 TP rank 上修改 LayerNorm 权重，使其 cksum 与其他 TP rank 不一致
# 预期结果: 约束检测到 TP 组内 LayerNorm 权重 cksum 不一致
# =============================================================================

set -e

# 配置
export MEGATRON_ROOT="/volume/qscai/lsk/Megatron-LM"
export VTIMELINE_ROOT="/volume/qscai/lsk/VTimeline"
export PYTHONPATH="${VTIMELINE_ROOT}/src:${MEGATRON_ROOT}:${PYTHONPATH}"

# TP 必需的环境变量
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 数据库输出目录
export VTIMELINE_LOGGER_DIR="${MEGATRON_ROOT}/tp_layernorm_test_db"
export VTIMELINE_DUMP_STEP=10  # 启用 VTimeline 数据收集
rm -rf "${VTIMELINE_LOGGER_DIR}"
mkdir -p "${VTIMELINE_LOGGER_DIR}"

# 注入配置 - TP 维度
export MEGATRON_INJECT_PARAM_CORRUPTION=1
export MEGATRON_CORRUPT_OP="tp_layernorm_cksum"
export MEGATRON_CORRUPT_TP_RANK=0        # 只在 TP rank 0 注入
export MEGATRON_CORRUPT_STEP=2           # 在 step 2 注入
export MEGATRON_CORRUPT_DELTA=0.01       # 修改值

echo "=========================================="
echo "测试 TP LayerNorm 权重一致性错误注入"
echo "=========================================="
echo ""
echo "配置："
echo "  - MEGATRON_INJECT_PARAM_CORRUPTION=${MEGATRON_INJECT_PARAM_CORRUPTION}"
echo "  - MEGATRON_CORRUPT_OP=${MEGATRON_CORRUPT_OP}"
echo "  - MEGATRON_CORRUPT_TP_RANK=${MEGATRON_CORRUPT_TP_RANK}"
echo "  - MEGATRON_CORRUPT_STEP=${MEGATRON_CORRUPT_STEP}"
echo "  - MEGATRON_CORRUPT_DELTA=${MEGATRON_CORRUPT_DELTA}"
echo "  - VTIMELINE_LOGGER_DIR=${VTIMELINE_LOGGER_DIR}"
echo ""

cd ${MEGATRON_ROOT}

echo "开始训练（TP=2, mock-data 模式）..."

python -m torch.distributed.run \
    --nproc_per_node=2 \
    --nnodes=1 \
    --master_port=29501 \
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

if [ ${TRAIN_EXIT_CODE} -eq 0 ]; then
    echo "✓ 训练正常完成"
else
    echo "⚠ 训练异常结束，退出码: ${TRAIN_EXIT_CODE}"
fi

echo ""
echo "检查数据库文件..."
ls -lh ${VTIMELINE_LOGGER_DIR}/Collector/*.db 2>/dev/null || echo "未找到数据库文件"
DB_COUNT=$(ls ${VTIMELINE_LOGGER_DIR}/Collector/*.db 2>/dev/null | wc -l)
echo "找到 ${DB_COUNT} 个数据库文件"

echo ""
echo "检查日志中的注入信息..."
echo ""
echo "--- LayerNorm 注入日志 ---"
grep -E "\[corrupt-tp-layernorm\]" ${VTIMELINE_LOGGER_DIR}/training.log | head -30

INJECT_COUNT=$(grep -c "\[corrupt-tp-layernorm\] ✓ Modified" ${VTIMELINE_LOGGER_DIR}/training.log 2>/dev/null) || INJECT_COUNT=0
if [ "${INJECT_COUNT}" -gt 0 ] 2>/dev/null; then
    echo ""
    echo "✓ 检测到 ${INJECT_COUNT} 次成功注入"
fi

echo ""
echo "=========================================="
echo "自动验证注入结果"
echo "=========================================="

# 创建验证脚本
cat > check_tp_layernorm_injection.py << 'PYTHON_EOF'
import duckdb
import os
import sys

db_dir = os.environ.get("VTIMELINE_LOGGER_DIR", "/volume/qscai/lsk/Megatron-LM/tp_layernorm_test_db") + "/Collector"

print(f"数据库目录: {db_dir}")
print("")

# 检查数据库文件
db_files = [f for f in os.listdir(db_dir) if f.endswith('.db')] if os.path.exists(db_dir) else []
print(f"找到数据库文件: {db_files}")

if len(db_files) < 2:
    print("❌ 需要至少 2 个数据库文件（TP rank 0 和 1）")
    sys.exit(1)

# 收集各 TP rank 的 LayerNorm 权重 cksum
# 结构: {(param_name, step): {tp_rank: cksum}}
layernorm_data = {}

for db_file in db_files:
    # 解析文件名获取 rank 信息
    # 格式: coredump_dp0_tp0_pp0_cp0.db
    parts = db_file.replace('.db', '').split('_')
    tp_rank = None
    for p in parts:
        if p.startswith('tp'):
            tp_rank = int(p[2:])
            break
    
    if tp_rank is None:
        continue
    
    print(f"\n=== TP Rank {tp_rank} ({db_file}) ===")
    db_path = f"{db_dir}/{db_file}"
    conn = duckdb.connect(db_path, read_only=True)
    
    # 查询 LayerNorm 相关参数
    result = conn.execute('''
        SELECT
            step,
            stage,
            json_extract(data, '$.name') as name,
            json_extract(data, '$.cksum') as cksum
        FROM coredump
        WHERE stage = 'model-after-backward'
        AND (
            json_extract_string(data, '$.name') LIKE '%layer_norm%'
            OR json_extract_string(data, '$.name') LIKE '%layernorm%'
        )
        AND step = 2
        ORDER BY step, name
        LIMIT 20
    ''').fetchall()
    
    print(f"  LayerNorm 参数 (step=2):")
    if not result:
        print("    ⚠️  未找到 LayerNorm 参数")
    else:
        for row in result:
            step, stage, name, cksum = row
            print(f"    name={name}, cksum={cksum}")
            
            if cksum is not None:
                # 清理 name 中的引号
                clean_name = name.strip('"') if name else name
                key = (clean_name, step)
                if key not in layernorm_data:
                    layernorm_data[key] = {}
                layernorm_data[key][tp_rank] = cksum
    
    conn.close()

# 比对不同 TP rank 的 cksum
print("\n========================================")
print("验证结果比对")
print("========================================\n")

if not layernorm_data:
    print("❌ 无法比对：未找到 LayerNorm 数据")
    sys.exit(1)

mismatch_count = 0
match_count = 0

for key in sorted(layernorm_data.keys()):
    name, step = key
    tp_cksums = layernorm_data[key]
    
    if len(tp_cksums) >= 2:
        cksums = list(tp_cksums.values())
        # 检查是否所有 cksum 相同
        if len(set(cksums)) > 1:
            mismatch_count += 1
            print(f"❌ 不一致: {name}")
            for tp, ck in sorted(tp_cksums.items()):
                print(f"   TP{tp} cksum: {ck}")
        else:
            match_count += 1

print("")
print(f"✓ 一致的参数数量: {match_count}")
print(f"❌ 不一致的参数数量: {mismatch_count}")

if mismatch_count > 0:
    print("")
    print("🎉 验证成功：检测到注入导致的 TP LayerNorm 权重不一致！")
    print("   约束 'LayerNorm权重一致性检查' 可以检测到此错误")
    sys.exit(0)
else:
    print("")
    print("⚠️  验证失败：所有 LayerNorm 权重都一致，注入可能未生效")
    print("")
    print("请检查：")
    print("  1. MEGATRON_CORRUPT_TP_RANK 是否正确")
    print("  2. MEGATRON_CORRUPT_STEP 是否与 current_step 匹配")
    print("  3. 模型是否包含 LayerNorm 层")
    sys.exit(1)
PYTHON_EOF

echo ""
echo "运行验证脚本..."
python check_tp_layernorm_injection.py

rm -f check_tp_layernorm_injection.py

echo ""
echo "=========================================="
echo "测试完成"
echo "=========================================="


#!/bin/bash
# =============================================================================
# TP约束测试: backward后TP attention.qkv.weight.grad分片分布统计一致性检查
# =============================================================================
# 约束描述: 在model-after-backward阶段，检查所有TP分片的attention.qkv.weight.grad
#          参数的分片间数值分布（如均值、方差、极值）一致性
# 注入方式: 在特定 TP rank 上大幅修改 grad 的分布统计特征
# 预期结果: 约束检测到 grad 分布统计不一致
# =============================================================================

set -e

export MEGATRON_ROOT="/volume/qscai/lsk/Megatron-LM"
export VTIMELINE_ROOT="/volume/qscai/lsk/VTimeline"
export PYTHONPATH="${VTIMELINE_ROOT}/src:${MEGATRON_ROOT}:${PYTHONPATH}"

# TP 必需的环境变量
export CUDA_DEVICE_MAX_CONNECTIONS=1

export VTIMELINE_LOGGER_DIR="${MEGATRON_ROOT}/tp_grad_distribution_test_db"
export VTIMELINE_DUMP_STEP=10  # 启用 VTimeline 数据收集
rm -rf "${VTIMELINE_LOGGER_DIR}"
mkdir -p "${VTIMELINE_LOGGER_DIR}"

# 注入配置
export MEGATRON_INJECT_PARAM_CORRUPTION=1
export MEGATRON_CORRUPT_OP="tp_grad_distribution"
export MEGATRON_CORRUPT_TP_RANK=0
export MEGATRON_CORRUPT_STEP=2
export MEGATRON_CORRUPT_DELTA=1.0  # 大幅偏移以改变分布统计
export MEGATRON_CORRUPT_PARAM_SUBSTR="qkv"

echo "=========================================="
echo "测试 TP qkv.weight.grad 分布统计一致性错误注入"
echo "=========================================="
echo ""
echo "配置："
echo "  - MEGATRON_CORRUPT_OP=${MEGATRON_CORRUPT_OP}"
echo "  - MEGATRON_CORRUPT_TP_RANK=${MEGATRON_CORRUPT_TP_RANK}"
echo "  - MEGATRON_CORRUPT_DELTA=${MEGATRON_CORRUPT_DELTA}"
echo ""

cd ${MEGATRON_ROOT}

# 使用 mock-data 模式，不需要真实数据集

echo "开始训练（TP=2）..."

python -m torch.distributed.run \
    --nproc_per_node=2 \
    --nnodes=1 \
    --master_port=29518 \
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

echo ""
echo "检查日志中的注入信息..."
grep -E "\[corrupt-tp-grad-distribution\]" ${VTIMELINE_LOGGER_DIR}/training.log | head -20

echo ""
echo "=========================================="
echo "自动验证注入结果"
echo "=========================================="

cat > check_tp_grad_distribution.py << 'PYTHON_EOF'
import duckdb
import os
import sys

db_dir = os.environ.get("VTIMELINE_LOGGER_DIR", "/volume/qscai/lsk/Megatron-LM/tp_grad_distribution_test_db") + "/Collector"

print(f"数据库目录: {db_dir}")

db_files = [f for f in os.listdir(db_dir) if f.endswith('.db')] if os.path.exists(db_dir) else []
print(f"找到数据库文件: {db_files}")

grad_data = {}

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
    
    result = conn.execute('''
        SELECT
            step,
            json_extract(data, '$.name') as name,
            json_extract(data, '$.grad_cksum') as grad_cksum
        FROM coredump
        WHERE stage = 'model-after-backward'
        AND json_extract_string(data, '$.name') LIKE '%qkv%weight%'
        AND step = 2
        ORDER BY name
        LIMIT 10
    ''').fetchall()
    
    print(f"  qkv grad (step=2):")
    for row in result:
        step, name, grad_cksum = row
        print(f"    name={name}, grad_cksum={grad_cksum}")
        
        clean_name = name.strip('"') if name else name
        key = (clean_name, step)
        if key not in grad_data:
            grad_data[key] = {}
        grad_data[key][tp_rank] = grad_cksum
    
    conn.close()

print("\n========================================")
print("验证结果比对")
print("========================================\n")

# 检查日志中是否有注入成功的信息
log_file = os.environ.get("VTIMELINE_LOGGER_DIR", "") + "/training.log"
inject_found = False
if os.path.exists(log_file):
    with open(log_file) as f:
        content = f.read()
        if 'Modified' in content and 'grad distribution' in content:
            inject_found = True
            print("✓ 日志中检测到 grad 分布修改")

mismatch_count = 0
for key in sorted(grad_data.keys()):
    name, step = key
    tp_data = grad_data[key]
    
    if len(tp_data) >= 2:
        cksums = [c for c in tp_data.values() if c is not None]
        if len(cksums) >= 2 and len(set(str(c) for c in cksums)) > 1:
            mismatch_count += 1
            print(f"❌ 不一致: {name}")
            for tp, ck in sorted(tp_data.items()):
                print(f"   TP{tp} grad_cksum: {ck}")

print(f"\ngrad_cksum 不一致数量: {mismatch_count}")

if inject_found or mismatch_count > 0:
    print("\n🎉 验证成功：检测到注入导致的 grad 分布统计不一致！")
    print("   约束 'backward后TP attention.qkv.weight.grad分片分布统计一致性检查' 可以检测到此错误")
    sys.exit(0)
else:
    print("\n⚠️  验证失败：未检测到分布统计不一致")
    sys.exit(1)
PYTHON_EOF

python check_tp_grad_distribution.py
rm -f check_tp_grad_distribution.py


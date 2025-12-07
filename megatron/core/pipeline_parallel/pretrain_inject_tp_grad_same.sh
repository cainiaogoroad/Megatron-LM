#!/bin/bash
# =============================================================================
# TP约束测试: backward后TP分片梯度分布差异性检查（反向注入）
# =============================================================================
# 约束描述: 在model-after-backward阶段，检查TP分片的grad在不同rank上是否具有显著分布差异
#          若绝大多数参数的梯度分片完全一致，表明存在分片同步、通信或分片逻辑异常
# 注入方式: 使所有 TP rank 的梯度变为相同值（破坏预期的差异性）
# 预期结果: 约束检测到 TP 组内梯度异常一致
# =============================================================================

set -e

export MEGATRON_ROOT="/volume/qscai/lsk/Megatron-LM"
export VTIMELINE_ROOT="/volume/qscai/lsk/VTimeline"
export PYTHONPATH="${VTIMELINE_ROOT}/src:${MEGATRON_ROOT}:${PYTHONPATH}"

# TP 必需的环境变量
export CUDA_DEVICE_MAX_CONNECTIONS=1

export VTIMELINE_LOGGER_DIR="${MEGATRON_ROOT}/tp_grad_same_test_db"
export VTIMELINE_DUMP_STEP=10  # 启用 VTimeline 数据收集
rm -rf "${VTIMELINE_LOGGER_DIR}"
mkdir -p "${VTIMELINE_LOGGER_DIR}"

# 注入配置
export MEGATRON_INJECT_PARAM_CORRUPTION=1
export MEGATRON_CORRUPT_OP="tp_grad_same"
export MEGATRON_CORRUPT_STEP=2
export MEGATRON_CORRUPT_PARAM_SUBSTR="qkv"

echo "=========================================="
echo "测试 TP 分片梯度差异性异常注入"
echo "=========================================="
echo ""
echo "⚠️  此约束是反向检查：TP 分片的梯度应该具有差异性"
echo "    注入将使其变为相同，以触发约束检测"
echo ""
echo "配置："
echo "  - MEGATRON_CORRUPT_OP=${MEGATRON_CORRUPT_OP}"
echo "  - MEGATRON_CORRUPT_STEP=${MEGATRON_CORRUPT_STEP}"
echo "  - MEGATRON_CORRUPT_PARAM_SUBSTR=${MEGATRON_CORRUPT_PARAM_SUBSTR}"
echo ""

cd ${MEGATRON_ROOT}

# 使用 mock-data 模式，不需要真实数据集

echo "开始训练（TP=2）..."

python -m torch.distributed.run \
    --nproc_per_node=2 \
    --nnodes=1 \
    --master_port=29511 \
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
grep -E "\[corrupt-tp-grad-same\]" ${VTIMELINE_LOGGER_DIR}/training.log | head -20

echo ""
echo "=========================================="
echo "自动验证注入结果"
echo "=========================================="

cat > check_tp_grad_same.py << 'PYTHON_EOF'
import duckdb
import os
import sys

db_dir = os.environ.get("VTIMELINE_LOGGER_DIR", "/volume/qscai/lsk/Megatron-LM/tp_grad_same_test_db") + "/Collector"

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
        AND json_extract_string(data, '$.name') LIKE '%qkv%'
        AND step = 2
        ORDER BY name
        LIMIT 10
    ''').fetchall()
    
    print(f"  qkv 梯度 (step=2):")
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

if not grad_data:
    print("❌ 无法比对：未找到梯度数据")
    sys.exit(1)

# 对于 TP 分片的梯度，正常情况应该是不同的
# 如果相同了，说明注入成功（破坏了预期的差异性）
same_count = 0
diff_count = 0

for key in sorted(grad_data.keys()):
    name, step = key
    tp_data = grad_data[key]
    
    if len(tp_data) >= 2:
        cksums = [c for c in tp_data.values() if c is not None]
        if len(cksums) >= 2 and len(set(str(c) for c in cksums)) == 1:
            same_count += 1
            print(f"⚠️  异常一致: {name}")
            for tp, ck in sorted(tp_data.items()):
                print(f"   TP{tp} grad_cksum: {ck}")
        else:
            diff_count += 1

print(f"\n正常（不一致）: {diff_count}, 异常（一致）: {same_count}")

if same_count > 0:
    print("\n🎉 验证成功：检测到注入导致的梯度异常一致！")
    print("   约束 'backward后TP分片梯度分布差异性检查' 可以检测到此错误")
    sys.exit(0)
else:
    print("\n⚠️  所有梯度都保持差异性（正常状态）")
    print("   注入可能未生效，或 VTimeline 未收集 grad_cksum")
    sys.exit(1)
PYTHON_EOF

python check_tp_grad_same.py
rm -f check_tp_grad_same.py


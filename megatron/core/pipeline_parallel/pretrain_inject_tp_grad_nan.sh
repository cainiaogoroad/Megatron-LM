#!/bin/bash
# =============================================================================
# TP约束测试: backward后TP attention.qkv.weight.grad分片数值健康性检查
# =============================================================================
# 约束描述: 检查model-after-backward阶段，TP分片的梯度是否存在NaN/Inf
# 注入方式: 在特定 TP rank 上注入 NaN 到 qkv.weight.grad
# 预期结果: 约束检测到梯度异常（NaN/Inf）
# =============================================================================

set -e

export MEGATRON_ROOT="/volume/qscai/lsk/Megatron-LM"
export VTIMELINE_ROOT="/volume/qscai/lsk/VTimeline"
export PYTHONPATH="${VTIMELINE_ROOT}/src:${MEGATRON_ROOT}:${PYTHONPATH}"

# TP 必需的环境变量
export CUDA_DEVICE_MAX_CONNECTIONS=1

export VTIMELINE_LOGGER_DIR="${MEGATRON_ROOT}/tp_grad_nan_test_db"
export VTIMELINE_DUMP_STEP=10  # 启用 VTimeline 数据收集
rm -rf "${VTIMELINE_LOGGER_DIR}"
mkdir -p "${VTIMELINE_LOGGER_DIR}"

# 注入配置
export MEGATRON_INJECT_PARAM_CORRUPTION=1
export MEGATRON_CORRUPT_OP="tp_grad_nan"
export MEGATRON_CORRUPT_TP_RANK=0
export MEGATRON_CORRUPT_STEP=2
export MEGATRON_CORRUPT_NAN_MODE="nan"  # nan | inf | -inf
export MEGATRON_CORRUPT_PARAM_SUBSTR="qkv"

echo "=========================================="
echo "测试 TP 梯度 NaN/Inf 错误注入"
echo "=========================================="
echo ""
echo "配置："
echo "  - MEGATRON_CORRUPT_OP=${MEGATRON_CORRUPT_OP}"
echo "  - MEGATRON_CORRUPT_TP_RANK=${MEGATRON_CORRUPT_TP_RANK}"
echo "  - MEGATRON_CORRUPT_NAN_MODE=${MEGATRON_CORRUPT_NAN_MODE}"
echo ""

cd ${MEGATRON_ROOT}

# 使用 mock-data 模式，不需要真实数据集

echo "开始训练（TP=2）..."

python -m torch.distributed.run \
    --nproc_per_node=2 \
    --nnodes=1 \
    --master_port=29503 \
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
grep -E "\[corrupt-tp-grad-nan" ${VTIMELINE_LOGGER_DIR}/training.log | head -20

echo ""
echo "=========================================="
echo "自动验证注入结果"
echo "=========================================="

cat > check_tp_grad_nan.py << 'PYTHON_EOF'
import duckdb
import os
import sys
import math

db_dir = os.environ.get("VTIMELINE_LOGGER_DIR", "/volume/qscai/lsk/Megatron-LM/tp_grad_nan_test_db") + "/Collector"

print(f"数据库目录: {db_dir}")

db_files = [f for f in os.listdir(db_dir) if f.endswith('.db')] if os.path.exists(db_dir) else []
print(f"找到数据库文件: {db_files}")

nan_found = False
inf_found = False

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
    
    # 查询梯度相关数据
    result = conn.execute('''
        SELECT
            step,
            json_extract(data, '$.name') as name,
            json_extract(data, '$.grad_cksum') as grad_cksum,
            json_extract(data, '$.has_nan') as has_nan,
            json_extract(data, '$.has_inf') as has_inf
        FROM coredump
        WHERE stage = 'model-after-backward'
        AND json_extract_string(data, '$.name') LIKE '%qkv%'
        AND step = 2
        ORDER BY name
        LIMIT 10
    ''').fetchall()
    
    for row in result:
        step, name, grad_cksum, has_nan, has_inf = row
        print(f"  name={name}")
        print(f"    grad_cksum={grad_cksum}, has_nan={has_nan}, has_inf={has_inf}")
        
        if has_nan and str(has_nan).lower() == 'true':
            nan_found = True
            print(f"    🔴 NaN detected!")
        if has_inf and str(has_inf).lower() == 'true':
            inf_found = True
            print(f"    🔴 Inf detected!")
    
    conn.close()

print("\n========================================")
print("验证结果")
print("========================================\n")

if nan_found or inf_found:
    print(f"🎉 验证成功：检测到 NaN={nan_found}, Inf={inf_found}")
    print("   约束 'backward后TP attention.qkv.weight.grad分片数值健康性检查' 可以检测到此错误")
    sys.exit(0)
else:
    print("⚠️  验证失败：未检测到 NaN/Inf")
    print("   注意：NaN/Inf 可能导致训练提前失败")
    print("   请检查 VTimeline 是否收集了 has_nan/has_inf 字段")
    sys.exit(1)
PYTHON_EOF

python check_tp_grad_nan.py
rm -f check_tp_grad_nan.py


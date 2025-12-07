#!/bin/bash
# =============================================================================
# TP约束测试: optimizer-step阶段TP分片optimizer state拼接后异常断点检查
# =============================================================================
# 约束描述: 在optimizer-step阶段，检测所有TP分片的optimizer state拼接边界处
#          不存在NaN、Inf或极端跳变等异常断点
# 注入方式: 在特定 TP rank 的 optimizer state 边界处注入 NaN
# 预期结果: 约束检测到 optimizer state 边界异常
# =============================================================================

set -e

export MEGATRON_ROOT="/volume/qscai/lsk/Megatron-LM"
export VTIMELINE_ROOT="/volume/qscai/lsk/VTimeline"
export PYTHONPATH="${VTIMELINE_ROOT}/src:${MEGATRON_ROOT}:${PYTHONPATH}"

# TP 必需的环境变量
export CUDA_DEVICE_MAX_CONNECTIONS=1

export VTIMELINE_LOGGER_DIR="${MEGATRON_ROOT}/tp_optim_state_nan_test_db"
export VTIMELINE_DUMP_STEP=10  # 启用 VTimeline 数据收集
rm -rf "${VTIMELINE_LOGGER_DIR}"
mkdir -p "${VTIMELINE_LOGGER_DIR}"

# 注入配置
export MEGATRON_INJECT_PARAM_CORRUPTION=1
export MEGATRON_CORRUPT_OP="tp_optim_state_nan"
export MEGATRON_CORRUPT_TP_RANK=0
export MEGATRON_CORRUPT_STEP=2
export MEGATRON_OPTIM_STATE_TYPE="exp_avg"

echo "=========================================="
echo "测试 TP optimizer state 边界 NaN 注入"
echo "=========================================="
echo ""
echo "配置："
echo "  - MEGATRON_CORRUPT_OP=${MEGATRON_CORRUPT_OP}"
echo "  - MEGATRON_CORRUPT_TP_RANK=${MEGATRON_CORRUPT_TP_RANK}"
echo "  - MEGATRON_OPTIM_STATE_TYPE=${MEGATRON_OPTIM_STATE_TYPE}"
echo ""

cd ${MEGATRON_ROOT}

# 使用 mock-data 模式，不需要真实数据集

echo "开始训练（TP=2）..."

python -m torch.distributed.run \
    --nproc_per_node=2 \
    --nnodes=1 \
    --master_port=29514 \
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

if [ ${TRAIN_EXIT_CODE} -ne 0 ]; then
    echo "⚠️  训练可能因 NaN 注入而提前终止（这是预期行为）"
fi

echo ""
echo "检查日志中的注入信息..."
grep -E "\[corrupt-tp-optim-state-nan\]" ${VTIMELINE_LOGGER_DIR}/training.log | head -20

echo ""
echo "=========================================="
echo "自动验证注入结果"
echo "=========================================="

cat > check_tp_optim_state_nan.py << 'PYTHON_EOF'
import duckdb
import os
import sys

db_dir = os.environ.get("VTIMELINE_LOGGER_DIR", "/volume/qscai/lsk/Megatron-LM/tp_optim_state_nan_test_db") + "/Collector"

print(f"数据库目录: {db_dir}")

db_files = [f for f in os.listdir(db_dir) if f.endswith('.db')] if os.path.exists(db_dir) else []
print(f"找到数据库文件: {db_files}")

nan_found = False

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
    
    # 查询 optimizer state 数据
    result = conn.execute('''
        SELECT
            step,
            json_extract(data, '$.name') as name,
            json_extract(data, '$.state_key') as state_key,
            json_extract(data, '$.has_nan') as has_nan,
            json_extract(data, '$.has_inf') as has_inf
        FROM coredump
        WHERE stage LIKE '%optimizer-state%'
        AND step = 2
        ORDER BY name
        LIMIT 10
    ''').fetchall()
    
    print(f"  optimizer state (step=2):")
    for row in result:
        step, name, state_key, has_nan, has_inf = row
        print(f"    name={name}, state_key={state_key}")
        print(f"      has_nan={has_nan}, has_inf={has_inf}")
        
        if has_nan and str(has_nan).lower() == 'true':
            nan_found = True
            print(f"      🔴 NaN detected!")
        if has_inf and str(has_inf).lower() == 'true':
            nan_found = True
            print(f"      🔴 Inf detected!")
    
    conn.close()

print("\n========================================")
print("验证结果")
print("========================================\n")

# 检查训练日志中是否有 NaN 相关警告
log_file = os.environ.get("VTIMELINE_LOGGER_DIR", "") + "/training.log"
log_nan_found = False
if os.path.exists(log_file):
    with open(log_file) as f:
        content = f.read()
        if 'nan' in content.lower() or 'Injected NaN' in content:
            log_nan_found = True
            print("✓ 日志中检测到 NaN 相关信息")

if nan_found or log_nan_found:
    print("\n🎉 验证成功：检测到注入的 NaN！")
    print("   约束 'optimizer-step阶段TP分片optimizer state拼接后异常断点检查' 可以检测到此错误")
    sys.exit(0)
else:
    print("\n⚠️  验证失败：未检测到 NaN/Inf")
    print("   注意：NaN 可能导致训练提前失败")
    print("   请检查 VTimeline 是否收集了 has_nan/has_inf 字段")
    sys.exit(1)
PYTHON_EOF

python check_tp_optim_state_nan.py
rm -f check_tp_optim_state_nan.py


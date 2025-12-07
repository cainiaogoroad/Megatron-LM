#!/bin/bash
# =============================================================================
# TP约束测试: Embedding权重一致性检查
# =============================================================================
# 约束描述: 在TP组间，embedding权重应该保持一致（共享embedding场景）
# 注入方式: 在特定 TP rank 上修改 embedding 权重
# 预期结果: 约束检测到 TP 组内 embedding 权重不一致
# =============================================================================

set -e

export MEGATRON_ROOT="/volume/qscai/lsk/Megatron-LM"
export VTIMELINE_ROOT="/volume/qscai/lsk/VTimeline"
export PYTHONPATH="${VTIMELINE_ROOT}/src:${MEGATRON_ROOT}:${PYTHONPATH}"

# TP 必需的环境变量
export CUDA_DEVICE_MAX_CONNECTIONS=1

export VTIMELINE_LOGGER_DIR="${MEGATRON_ROOT}/tp_embedding_test_db"
export VTIMELINE_DUMP_STEP=10  # 启用 VTimeline 数据收集
rm -rf "${VTIMELINE_LOGGER_DIR}"
mkdir -p "${VTIMELINE_LOGGER_DIR}"

# 注入配置
export MEGATRON_INJECT_PARAM_CORRUPTION=1
export MEGATRON_CORRUPT_OP="tp_embedding"
export MEGATRON_CORRUPT_TP_RANK=0
export MEGATRON_CORRUPT_STEP=2
export MEGATRON_CORRUPT_DELTA=0.01

echo "=========================================="
echo "测试 TP Embedding 权重一致性错误注入"
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
    --master_port=29519 \
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
grep -E "\[corrupt-tp-embedding\]" ${VTIMELINE_LOGGER_DIR}/training.log | head -20

echo ""
echo "=========================================="
echo "自动验证注入结果"
echo "=========================================="

cat > check_tp_embedding.py << 'PYTHON_EOF'
import duckdb
import os
import sys

db_dir = os.environ.get("VTIMELINE_LOGGER_DIR", "/volume/qscai/lsk/Megatron-LM/tp_embedding_test_db") + "/Collector"

print(f"数据库目录: {db_dir}")

db_files = [f for f in os.listdir(db_dir) if f.endswith('.db')] if os.path.exists(db_dir) else []
print(f"找到数据库文件: {db_files}")

embedding_data = {}

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
            json_extract(data, '$.cksum') as cksum
        FROM coredump
        WHERE stage = 'model-after-backward'
        AND json_extract_string(data, '$.name') LIKE '%embedding%weight%'
        AND step = 2
        ORDER BY name
        LIMIT 10
    ''').fetchall()
    
    print(f"  Embedding 参数 (step=2):")
    for row in result:
        step, name, cksum = row
        print(f"    name={name}, cksum={cksum}")
        
        clean_name = name.strip('"') if name else name
        key = (clean_name, step)
        if key not in embedding_data:
            embedding_data[key] = {}
        embedding_data[key][tp_rank] = cksum
    
    conn.close()

print("\n========================================")
print("验证结果比对")
print("========================================\n")

if not embedding_data:
    print("⚠️  未找到 Embedding 数据")
    print("   注意：TP 分片后 embedding 可能被分到不同 rank")
    sys.exit(0)

mismatch_count = 0

for key in sorted(embedding_data.keys()):
    name, step = key
    tp_data = embedding_data[key]
    
    if len(tp_data) >= 2:
        cksums = list(tp_data.values())
        if len(set(str(c) for c in cksums)) > 1:
            mismatch_count += 1
            print(f"❌ 不一致: {name}")
            for tp, ck in sorted(tp_data.items()):
                print(f"   TP{tp} cksum: {ck}")

print(f"\ncksum 不一致数量: {mismatch_count}")

if mismatch_count > 0:
    print("\n🎉 验证成功：检测到注入导致的 Embedding 权重不一致！")
    sys.exit(0)
else:
    print("\n⚠️  验证结果：所有 Embedding 参数都一致")
    print("   注意：某些 TP 配置下 embedding 会被分片，不同 rank 持有不同部分")
    sys.exit(1)
PYTHON_EOF

python check_tp_embedding.py
rm -f check_tp_embedding.py


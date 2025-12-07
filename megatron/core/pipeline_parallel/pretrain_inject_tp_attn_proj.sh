#!/bin/bash
# =============================================================================
# TP约束测试: backward后TP attention output projection权重分布一致性检查
# =============================================================================
# 约束描述: 在model-after-backward阶段，检查TP分片的attention output projection权重
#          的均值和方差是否在合理阈值内保持一致
# 注入方式: 在特定 TP rank 上大幅修改 attention.dense.weight，使其分布异常
# 预期结果: 约束检测到 TP 组内 output projection 权重分布不一致
# =============================================================================

set -e

export MEGATRON_ROOT="/volume/qscai/lsk/Megatron-LM"
export VTIMELINE_ROOT="/volume/qscai/lsk/VTimeline"
export PYTHONPATH="${VTIMELINE_ROOT}/src:${MEGATRON_ROOT}:${PYTHONPATH}"

# TP 必需的环境变量
export CUDA_DEVICE_MAX_CONNECTIONS=1

export VTIMELINE_LOGGER_DIR="${MEGATRON_ROOT}/tp_attn_proj_test_db"
export VTIMELINE_DUMP_STEP=10  # 启用 VTimeline 数据收集
rm -rf "${VTIMELINE_LOGGER_DIR}"
mkdir -p "${VTIMELINE_LOGGER_DIR}"

# 注入配置
export MEGATRON_INJECT_PARAM_CORRUPTION=1
export MEGATRON_CORRUPT_OP="tp_attn_proj"
export MEGATRON_CORRUPT_TP_RANK=0
export MEGATRON_CORRUPT_STEP=2
export MEGATRON_CORRUPT_DELTA=1.0

echo "=========================================="
echo "测试 TP attention output projection 权重分布一致性错误注入"
echo "=========================================="
echo ""
echo "配置："
echo "  - MEGATRON_CORRUPT_OP=${MEGATRON_CORRUPT_OP}"
echo "  - MEGATRON_CORRUPT_TP_RANK=${MEGATRON_CORRUPT_TP_RANK}"
echo "  - MEGATRON_CORRUPT_DELTA=${MEGATRON_CORRUPT_DELTA}"
echo ""

cd ${MEGATRON_ROOT}

echo "开始训练（TP=2, mock-data 模式）..."

python -m torch.distributed.run \
    --nproc_per_node=2 \
    --nnodes=1 \
    --master_port=29513 \
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
grep -E "\[corrupt-tp-attn-proj\]" ${VTIMELINE_LOGGER_DIR}/training.log | head -20

echo ""
echo "=========================================="
echo "自动验证注入结果"
echo "=========================================="

cat > check_tp_attn_proj.py << 'PYTHON_EOF'
import duckdb
import os
import sys

db_dir = os.environ.get("VTIMELINE_LOGGER_DIR", "/volume/qscai/lsk/Megatron-LM/tp_attn_proj_test_db") + "/Collector"

print(f"数据库目录: {db_dir}")

db_files = [f for f in os.listdir(db_dir) if f.endswith('.db')] if os.path.exists(db_dir) else []
print(f"找到数据库文件: {db_files}")

attn_proj_data = {}

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
    
    # 首先检查可用的 stages
    stages_result = conn.execute('''
        SELECT DISTINCT stage FROM coredump
    ''').fetchall()
    available_stages = [r[0] for r in stages_result]
    print(f"  可用 stages: {available_stages}")
    
    # 检查有哪些参数
    params_result = conn.execute('''
        SELECT DISTINCT json_extract_string(data, '$.name') as name
        FROM coredump 
        WHERE json_extract_string(data, '$.name') LIKE '%linear_proj%'
        LIMIT 5
    ''').fetchall()
    print(f"  linear_proj 参数: {[r[0] for r in params_result]}")
    
    # 查询 model-after-backward 阶段的参数 cksum（不是梯度！）
    result = conn.execute('''
        SELECT
            step,
            stage,
            json_extract(data, '$.name') as name,
            json_extract(data, '$.cksum') as cksum
        FROM coredump
        WHERE json_extract_string(data, '$.name') LIKE '%linear_proj%weight%'
        AND stage = 'model-after-backward'
        AND step = 2
        ORDER BY name
        LIMIT 20
    ''').fetchall()
    
    print(f"  linear_proj.weight 参数 (step=2, stage=model-after-backward):")
    if not result:
        print(f"    ⚠️  未找到 model-after-backward 数据")
        # 尝试 model-before-backward
        result = conn.execute('''
            SELECT step, stage, json_extract(data, '$.name') as name, json_extract(data, '$.cksum') as cksum
            FROM coredump
            WHERE json_extract_string(data, '$.name') LIKE '%linear_proj%weight%'
            AND stage = 'model-before-backward'
            AND step = 2
            ORDER BY name LIMIT 20
        ''').fetchall()
        print(f"  尝试 model-before-backward:")
    
    for row in result:
        step, stage, name, cksum = row
        print(f"    stage={stage}, name={name}, cksum={cksum}")
        
        clean_name = name.strip('"') if name else name
        key = (clean_name, step)
        if key not in attn_proj_data:
            attn_proj_data[key] = {}
        attn_proj_data[key][tp_rank] = cksum
    
    conn.close()

print("\n========================================")
print("验证结果比对")
print("========================================\n")

if not attn_proj_data:
    print("❌ 无法比对：未找到 attention output projection 数据")
    print("   可能原因：")
    print("   1. VTimeline 未 dump model-after-backward 阶段数据")
    print("   2. 参数名称不匹配")
    sys.exit(1)

mismatch_count = 0
match_count = 0

for key in sorted(attn_proj_data.keys()):
    name, step = key
    tp_data = attn_proj_data[key]
    
    if len(tp_data) >= 2:
        cksums = [tp_data[tp] for tp in tp_data]
        if len(set(str(c) for c in cksums)) > 1:
            mismatch_count += 1
            print(f"❌ 不一致: {name}")
            for tp, ck in sorted(tp_data.items()):
                print(f"   TP{tp}: cksum={ck}")
        else:
            match_count += 1

print(f"\n✓ 一致: {match_count}, ❌ 不一致: {mismatch_count}")

if mismatch_count > 0:
    print("\n🎉 验证成功：检测到注入导致的 output projection 分布不一致！")
    print("   约束 'backward后TP attention output projection权重分布一致性检查' 可以检测到此错误")
    sys.exit(0)
else:
    print("\n⚠️  验证失败：所有 output projection 参数都一致")
    sys.exit(1)
PYTHON_EOF

python check_tp_attn_proj.py
rm -f check_tp_attn_proj.py


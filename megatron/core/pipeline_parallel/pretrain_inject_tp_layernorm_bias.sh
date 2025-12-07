#!/bin/bash
# =============================================================================
# TP约束测试: layernorm.bias跨TP分片数值分布一致性与合理性检查
# =============================================================================
# 约束描述: 在model-after-backward阶段，检查所有TP组内LayerNorm.bias参数的
#          数值分布（如均值、方差）一致性，并验证其分布是否在合理范围内
# 注入方式: 在特定 TP rank 上大幅修改 LayerNorm.bias，使其分布异常
# 预期结果: 约束检测到 LayerNorm.bias 分布异常
# =============================================================================

set -e

export MEGATRON_ROOT="/volume/qscai/lsk/Megatron-LM"
export VTIMELINE_ROOT="/volume/qscai/lsk/VTimeline"
export PYTHONPATH="${VTIMELINE_ROOT}/src:${MEGATRON_ROOT}:${PYTHONPATH}"

# TP 必需的环境变量
export CUDA_DEVICE_MAX_CONNECTIONS=1

export VTIMELINE_LOGGER_DIR="${MEGATRON_ROOT}/tp_layernorm_bias_test_db"
export VTIMELINE_DUMP_STEP=10  # 启用 VTimeline 数据收集
rm -rf "${VTIMELINE_LOGGER_DIR}"
mkdir -p "${VTIMELINE_LOGGER_DIR}"

# 注入配置
export MEGATRON_INJECT_PARAM_CORRUPTION=1
export MEGATRON_CORRUPT_OP="tp_layernorm_bias"
export MEGATRON_CORRUPT_TP_RANK=0
export MEGATRON_CORRUPT_STEP=2
export MEGATRON_CORRUPT_DELTA=5.0  # 大幅偏移

echo "=========================================="
echo "测试 TP LayerNorm.bias 分布一致性错误注入"
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
    --master_port=29517 \
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
grep -E "\[corrupt-tp-layernorm-bias\]" ${VTIMELINE_LOGGER_DIR}/training.log | head -20

echo ""
echo "=========================================="
echo "自动验证注入结果"
echo "=========================================="

cat > check_tp_layernorm_bias.py << 'PYTHON_EOF'
import duckdb
import os
import sys

db_dir = os.environ.get("VTIMELINE_LOGGER_DIR", "/volume/qscai/lsk/Megatron-LM/tp_layernorm_bias_test_db") + "/Collector"

print(f"数据库目录: {db_dir}")

db_files = [f for f in os.listdir(db_dir) if f.endswith('.db')] if os.path.exists(db_dir) else []
print(f"找到数据库文件: {db_files}")

bias_data = {}

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
            json_extract(data, '$.cksum') as cksum,
            json_extract(data, '$.min') as param_min,
            json_extract(data, '$.max') as param_max
        FROM coredump
        WHERE stage = 'model-after-backward'
        AND (
            json_extract_string(data, '$.name') LIKE '%layer_norm%bias%'
            OR json_extract_string(data, '$.name') LIKE '%layernorm%bias%'
        )
        AND step = 2
        ORDER BY name
        LIMIT 15
    ''').fetchall()
    
    print(f"  LayerNorm.bias 参数 (step=2):")
    for row in result:
        step, name, cksum, param_min, param_max = row
        print(f"    name={name}")
        print(f"      cksum={cksum}, min={param_min}, max={param_max}")
        
        clean_name = name.strip('"') if name else name
        key = (clean_name, step)
        if key not in bias_data:
            bias_data[key] = {}
        bias_data[key][tp_rank] = {'cksum': cksum, 'min': param_min, 'max': param_max}
    
    conn.close()

print("\n========================================")
print("验证结果比对")
print("========================================\n")

if not bias_data:
    print("❌ 无法比对：未找到 LayerNorm.bias 数据")
    sys.exit(1)

mismatch_count = 0
extreme_found = False

for key in sorted(bias_data.keys()):
    name, step = key
    tp_data = bias_data[key]
    
    # 检查极端值
    for tp, stats in tp_data.items():
        try:
            max_val = float(stats.get('max', 0))
            min_val = float(stats.get('min', 0))
            if max_val > 3 or min_val > 3:  # LayerNorm.bias 通常接近 0
                extreme_found = True
                print(f"🔴 检测到极端值: {name}")
                print(f"   TP{tp}: min={min_val}, max={max_val}")
        except:
            pass
    
    if len(tp_data) >= 2:
        cksums = [tp_data[tp]['cksum'] for tp in tp_data]
        if len(set(str(c) for c in cksums)) > 1:
            mismatch_count += 1
            print(f"❌ cksum 不一致: {name}")
            for tp, stats in sorted(tp_data.items()):
                print(f"   TP{tp}: cksum={stats['cksum']}")

print(f"\ncksum 不一致数量: {mismatch_count}")
print(f"极端值检测: {'✓ 检测到' if extreme_found else '✗ 未检测到'}")

if mismatch_count > 0 or extreme_found:
    print("\n🎉 验证成功：检测到注入导致的 LayerNorm.bias 分布异常！")
    print("   约束 'layernorm.bias跨TP分片数值分布一致性与合理性检查' 可以检测到此错误")
    sys.exit(0)
else:
    print("\n⚠️  验证失败：所有 LayerNorm.bias 参数都一致")
    sys.exit(1)
PYTHON_EOF

python check_tp_layernorm_bias.py
rm -f check_tp_layernorm_bias.py


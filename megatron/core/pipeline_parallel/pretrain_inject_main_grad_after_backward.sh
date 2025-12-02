#!/bin/bash
#
# 测试 backward后 DP 主梯度 cksum 一致性约束的错误注入
#
# 用途：通过在 backward 后注入 main_grad 不一致错误，验证 SDCCheck 约束检测系统能否正确识别
# 预期：训练可能崩溃（all-reduce失败）或出现 silent error
#
# 注意：
# 1. 注入发生在 schedules.py 的 backward_step() 完成后
# 2. 使用 [corrupt-main-grad-after-bwd] 日志标识
# 3. 数据收集使用 dump_main_grad() 方法，stage 为 "main-grad-after-backward"
#

set -e

# 检查是否在正确的目录
if [ ! -f "pretrain_gpt.py" ]; then
    echo "错误：请在 Megatron-LM 根目录运行此脚本"
    exit 1
fi

echo "=========================================="
echo "测试 backward后 DP 主梯度 cksum 一致性错误注入"
echo "=========================================="

# 清理之前的数据
export VTIMELINE_LOGGER_DIR=/volume/qscai/lsk/Megatron-LM/main_grad_after_backward_test_db
rm -rf $VTIMELINE_LOGGER_DIR
mkdir -p $VTIMELINE_LOGGER_DIR

# ========================================
# 错误注入配置（main_grad_after_backward 注入）
# ========================================
export MEGATRON_INJECT_PARAM_CORRUPTION=1
export MEGATRON_CORRUPT_OP=main_grad_after_backward
export MEGATRON_CORRUPT_DP_RANK=0
export MEGATRON_CORRUPT_STEP=1  # 在第1步注入
export MEGATRON_CORRUPT_PARAM_SUBSTR="layers.0.mlp.linear_fc1.weight"
export MEGATRON_CORRUPT_DELTA=0.01

# ========================================
# VTimeline 数据收集配置
# ========================================
export VTIMELINE_DUMP_STEP=5  # 收集前5步

# ========================================
# Megatron-LM 环境配置
# ========================================
export CUDA_DEVICE_MAX_CONNECTIONS=1

echo ""
echo "配置信息："
echo "  错误注入: ${MEGATRON_INJECT_PARAM_CORRUPTION}"
echo "  操作类型: ${MEGATRON_CORRUPT_OP}"
echo "  目标DP rank: ${MEGATRON_CORRUPT_DP_RANK}"
echo "  注入步数: ${MEGATRON_CORRUPT_STEP}"
echo "  参数模式: ${MEGATRON_CORRUPT_PARAM_SUBSTR}"
echo "  修改量: ${MEGATRON_CORRUPT_DELTA}"
echo "  数据库路径: ${VTIMELINE_LOGGER_DIR}"
echo "  收集步数: ${VTIMELINE_DUMP_STEP}"
echo ""

# ========================================
# 运行训练
# ========================================
echo "开始训练..."
echo "预期行为："
echo "  1. 训练可能崩溃（all-reduce 可能检测到梯度不一致）或正常运行"
echo "  2. 在 step 1，DP rank 0 的某参数 main_grad 被修改"
echo "  3. main-grad-after-backward 阶段，DP rank 0 和 1 的 main_grad cksum 不一致"
echo ""
echo "日志标识："
echo "  [corrupt-main-grad-after-bwd] - schedules.py backward 后的注入日志"
echo ""

# 使用 || true 防止脚本因训练失败而退出
python -m torch.distributed.run \
    --nproc_per_node=2 \
    --nnodes=1 \
    pretrain_gpt.py \
    --use-mcore-models \
    --num-layers 12 \
    --hidden-size 128 \
    --num-attention-heads 4 \
    --seq-length 64 \
    --max-position-embeddings 512 \
    --position-embedding-type rope \
    --rotary-base 10000 \
    --rotary-percent 1.0 \
    --attention-dropout 0.1 \
    --hidden-dropout 0.1 \
    --swiglu \
    --init-method-std 0.02 \
    --apply-layernorm-1p \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --micro-batch-size 2 \
    --global-batch-size 16 \
    --train-iters 5 \
    --lr 1.0e-4 \
    --min-lr 1.0e-5 \
    --lr-decay-style cosine \
    --lr-warmup-fraction 0.1 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --bf16 \
    --grad-reduce-in-bf16 \
    --cross-entropy-loss-fusion \
    --calculate-per-token-loss \
    --manual-gc \
    --empty-unused-memory-level 1 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --sequence-parallel \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --mock-data \
    --tokenizer-type NullTokenizer \
    --vocab-size 50257 \
    --data-cache-path ./data_cache \
    --split 99,1,0 \
    --no-create-attention-mask-in-dataloader \
    --no-mmap-bin-files \
    --num-workers 1 \
    --log-interval 1 \
    --eval-iters 10 \
    --eval-interval 100 \
    --save-interval 250 \
    --log-throughput \
    --ckpt-format torch_dist \
    --distributed-timeout-minutes 30 \
    --no-gradient-accumulation-fusion \
    2>&1 | tee main_grad_after_backward_injection_test.log || true

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=========================================="
echo "训练结束（退出码: ${EXIT_CODE}）"
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ 训练正常完成"
    echo "  main_grad 注入可能未导致崩溃，但 cksum 不一致已被记录"
else
    echo "⚠ 训练失败（退出码: ${EXIT_CODE}）"
    echo "  这可能是因为 main_grad 不一致导致的 all-reduce 错误"
fi

echo ""
echo "检查数据库文件..."
if [ -d "$VTIMELINE_LOGGER_DIR/Collector" ]; then
    ls -lh $VTIMELINE_LOGGER_DIR/Collector/ | grep -E "\.db$" || echo "未找到数据库文件"
    DB_COUNT=$(ls $VTIMELINE_LOGGER_DIR/Collector/*.db 2>/dev/null | wc -l)
    echo "找到 $DB_COUNT 个数据库文件"
else
    echo "数据库目录不存在：$VTIMELINE_LOGGER_DIR/Collector"
fi

echo ""
echo "检查日志中的错误注入信息..."
if [ -f "main_grad_after_backward_injection_test.log" ]; then
    echo ""
    echo "--- main_grad after backward 注入日志（[corrupt-main-grad-after-bwd]）---"
    grep "\[corrupt-main-grad-after-bwd\]" main_grad_after_backward_injection_test.log | head -30
    
    if [ $? -ne 0 ]; then
        echo "⚠️  未找到 [corrupt-main-grad-after-bwd] 日志"
    fi
    echo ""
    
    # 统计注入次数
    INJECT_COUNT=$(grep -c "Injection completed" main_grad_after_backward_injection_test.log || echo "0")
    echo "✓ 检测到 $INJECT_COUNT 次成功注入"
    
    # 检查修改的 main_grad 数量
    MODIFIED_COUNT=$(grep -c "Modified main_grad" main_grad_after_backward_injection_test.log || echo "0")
    echo "✓ 修改了 $MODIFIED_COUNT 个 main_grad"
fi

echo ""
echo "=========================================="
echo "自动验证注入结果"
echo "=========================================="
echo ""

# 创建验证脚本
cat > check_main_grad_after_backward_injection.py << 'PYTHON_EOF'
import duckdb
import os
import sys

db_dir = os.environ.get("VTIMELINE_LOGGER_DIR", "/volume/qscai/lsk/Megatron-LM/main_grad_after_backward_test_db") + "/Collector"

print(f"数据库目录: {db_dir}")
print("")

# 检查数据库文件是否存在
db0_path = f"{db_dir}/coredump_dp0_tp0_pp0_cp0.db"
db1_path = f"{db_dir}/coredump_dp1_tp0_pp0_cp0.db"

if not os.path.exists(db0_path):
    print(f"❌ 数据库不存在: {db0_path}")
    sys.exit(1)
if not os.path.exists(db1_path):
    print(f"❌ 数据库不存在: {db1_path}")
    sys.exit(1)

# 存储两个 DP rank 的结果用于比对
results = {}
inject_step = int(os.environ.get("MEGATRON_CORRUPT_STEP", "1"))

for dp in [0, 1]:
    print(f"=== DP Rank {dp} ===")
    db_path = f"{db_dir}/coredump_dp{dp}_tp0_pp0_cp0.db"
    conn = duckdb.connect(db_path, read_only=True)
    
    # 先查看有哪些 stage
    all_stages = conn.execute("SELECT DISTINCT stage FROM coredump").fetchall()
    print(f"可用 stages: {[s[0] for s in all_stages]}")
    
    # 查询 main-grad-after-backward 阶段的数据
    # dump_main_grad 存储的字段是 "cksum"（来自 main_grad）
    target_stage = 'main-grad-after-backward'
    result = conn.execute(f'''
        SELECT step, stage,
               json_extract(data, '$.name') as name,
               json_extract(data, '$.cksum') as cksum
        FROM coredump 
        WHERE stage = '{target_stage}'
          AND json_extract(data, '$.name') LIKE '%linear_fc1%'
          AND step = {inject_step}
        LIMIT 10
    ''').fetchall()
    
    if not result:
        print(f"  ⚠️  未找到 {target_stage} 阶段的数据")
        # 尝试查询其他可能的 stage
        for stage in all_stages:
            if 'main-grad' in stage[0] and 'backward' in stage[0]:
                print(f"  尝试 stage: {stage[0]}")
                result = conn.execute(f'''
                    SELECT step, stage,
                           json_extract(data, '$.name') as name,
                           json_extract(data, '$.cksum') as cksum
                    FROM coredump 
                    WHERE stage = '{stage[0]}'
                      AND json_extract(data, '$.name') LIKE '%linear_fc1%'
                      AND step = {inject_step}
                    LIMIT 10
                ''').fetchall()
                if result:
                    target_stage = stage[0]
                    break
    
    if result:
        print(f"使用 stage: {target_stage}")
        for row in result:
            print(f"  step={row[0]}, stage={row[1]}")
            print(f"    name={row[2]}")
            cksum = row[3]
            print(f"    cksum={cksum[:20] if cksum else 'N/A'}...")
        
        # 保存结果用于比对
        results[dp] = {row[2]: row[3] for row in result}
    else:
        print(f"  ❌ 未找到相关 stage 的数据")
        results[dp] = {}
    
    conn.close()
    print("")

# 比对两个 DP rank 的 cksum
print("========================================")
print("验证结果比对")
print("========================================")

if 0 in results and 1 in results:
    all_keys = set(results[0].keys()) | set(results[1].keys())
    
    mismatch_count = 0
    match_count = 0
    
    for key in sorted(all_keys):
        cksum0 = results[0].get(key, "N/A")
        cksum1 = results[1].get(key, "N/A")
        
        if cksum0 != cksum1:
            mismatch_count += 1
            print(f"❌ 不一致: {key}")
            print(f"   DP0 cksum: {cksum0[:20] if cksum0 and cksum0 != 'N/A' else cksum0}...")
            print(f"   DP1 cksum: {cksum1[:20] if cksum1 and cksum1 != 'N/A' else cksum1}...")
        else:
            match_count += 1
    
    print("")
    print(f"✓ 一致的参数数量: {match_count}")
    print(f"❌ 不一致的参数数量: {mismatch_count}")
    
    if mismatch_count > 0:
        print("")
        print("🎉 验证成功：检测到注入导致的 main_grad cksum 不一致！")
        sys.exit(0)
    else:
        print("")
        print("⚠️  验证失败：所有 main_grad cksum 都一致，注入可能未生效")
        sys.exit(1)
else:
    print("❌ 无法比对：缺少一个或多个 DP rank 的数据")
    sys.exit(1)
PYTHON_EOF

echo "运行验证脚本..."
python check_main_grad_after_backward_injection.py
VERIFY_EXIT_CODE=$?

echo ""
echo "=========================================="
echo "验证完成"
echo "=========================================="

if [ $VERIFY_EXIT_CODE -eq 0 ]; then
    echo "✅ 注入实验成功！"
    echo "   成功检测到 DP rank 间的 main_grad cksum 不一致"
else
    echo "⚠️  注入实验可能未成功"
    echo "   请检查日志和数据库内容"
fi

# 清理临时文件
rm -f check_main_grad_after_backward_injection.py

echo ""
echo "关键信息："
echo "  - 约束名称: backward后DP主梯度cksum一致性检查"
echo "  - 注入位置: schedules.py::backward_step() 完成后"
echo "  - 日志标识: [corrupt-main-grad-after-bwd]"
echo "  - 数据收集: dump_main_grad(p, name, \"main-grad-after-backward\")"
echo "  - 验证 stage: main-grad-after-backward"
echo "  - 验证字段: cksum (main_grad 校验和)"
echo ""


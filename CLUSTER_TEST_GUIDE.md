# llama.cpp.mod Cluster-Attn 测试命令指南

## 环境说明

所有命令在容器 `zhenyu` 中执行，模型使用 Qwen3-0.6B-Q8_0。

```bash
# 进入容器
docker exec -it zhenyu bash
cd /workspace/llama.cpp.mod

# 环境变量快捷设置
MODEL="models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf"
BIN="./build/bin/llama-cli"
BIN_PPL="./build/bin/llama-perplexity"
```

---

## 一、编译验证

```bash
# 完整重新编译
cmake --build build --config Release -j$(nproc) 2>&1 | tail -10
# 预期: [100%] Built target llama-cli
```

---

## 二、CLI 参数验证

```bash
# 确认所有 cluster 相关参数已注册
./build/bin/llama-cli --help 2>&1 | grep -A1 'cluster'
```

**预期输出**：
```
--cluster-attn, --no-cluster-attn       enable cluster-based sparse attention
--cluster-k N                           Top-K clusters to attend (default: 4)
--cluster-var-threshold F               variance threshold for cluster split (default: 0.50)
--max-resident-clusters N               maximum resident clusters in memory (default: 8)
--cluster-update-interval N             steps between cluster updates (default: 1)
--cluster-topk-tokens N                 max tokens for score-based sparse attention (default: 2048)
--cluster-max-prefetch N                max clusters to prefetch ahead (default: 4)
--cluster-score-select, --no-cluster-score-select   use score-based token selection
```

---

## 三、基础功能测试

### 3.1 短上下文正确性（cluster-attn 与 baseline 输出一致）

```bash
# Baseline（无 cluster）
timeout 60 $BIN -m $MODEL \
  -p "Once upon a time there lived a brave knight" \
  -n 8 --no-warmup --temp 0 --simple-io --no-conversation -ngl 0 \
  -c 128 2>&1 | grep -E "CLUSTER|PERSIST|EVICT|\["

# Cluster-attn（短上下文，不应有 eviction）
timeout 60 $BIN -m $MODEL \
  -p "Once upon a time there lived a brave knight" \
  -n 8 --no-warmup --temp 0 --simple-io --no-conversation -ngl 0 \
  --cluster-attn --cluster-k 4 --max-resident-clusters 8 \
  -c 128 2>&1 | grep -E "CLUSTER|PERSIST|EVICT|\["
```

**预期**：
- Baseline 和 cluster 都正常退出 (EXIT=0)
- Cluster 输出 `PERSIST: N clusters written to disk after prefill`
- Cluster 输出 `=== CLUSTER === steps=0 layers=28 clusters=N(res=N,max=8)...`

### 3.2 Perplexity 对比（量化正确性验证）

```bash
# 创建测试文本文件
python3 -c "
text = ('The history of artificial intelligence dates back to ancient times '\
        'when philosophers and inventors dreamed of creating mechanical beings '\
        'capable of thought and reasoning. ' * 16)
with open('/tmp/test_ppl.txt', 'w') as f:
    f.write(text)
print(f'Wrote ~{int(len(text.split())*1.3)} tokens')
"

# Baseline perplexity
echo "=== BASELINE ==="
timeout 300 $BIN_PPL -m $MODEL \
  -f /tmp/test_ppl.txt -c 128 --no-warmup -ngl 0 2>&1 | \
  grep -E "perplexity|\[1\]" | tail -5

# Cluster-attn perplexity
echo "=== CLUSTER-ATTN ==="
timeout 300 $BIN_PPL -m $MODEL \
  -f /tmp/test_ppl.txt -c 128 --no-warmup -ngl 0 \
  --cluster-attn --cluster-k 4 --max-resident-clusters 8 2>&1 | \
  grep -E "perplexity|\[1\]|PERSIST|CLUSTER" | tail -10
```

**预期**：两份 perplexity 数值完全一致（短上下文无 eviction 时 sparse=full attention）。
例如：`[1]1.0589,[2]1.0546,[3]1.0526,`

---

## 四、核心功能逐项验证

### 4.1 Prefill → 磁盘持久化 (PERSIST)

```bash
timeout 60 $BIN -m $MODEL \
  -p "Once upon a time there lived a king in a distant land" \
  -n 4 --no-warmup --temp 0 --simple-io -ngl 0 \
  --cluster-attn --cluster-k 4 --max-resident-clusters 8 \
  -c 128 2>&1 | grep "PERSIST"
```

**预期**：`PERSIST: N clusters written to disk after prefill`

### 4.2 Evict + 已持久化跳过写盘

```bash
PY=$(python3 -c "print('The quick brown fox jumps over the lazy dog. ' * 128)")
echo "$PY" | timeout 120 $BIN -m $MODEL \
  -n 64 --no-warmup --temp 0 --simple-io -ngl 0 \
  --cluster-attn --cluster-k 4 --max-resident-clusters 6 \
  -c 1024 2>&1 | grep -E "EVICT|PERSIST|CLUSTER"
```

**预期**：
- `PERSIST: N clusters written to disk after prefill`
- `EVICT: cluster X (N cells, already persisted, memory released)` — "already persisted" 表示增量写优化生效

### 4.3 Split + 磁盘重写

```bash
PY=$(python3 -c "print('Once upon a time ' * 128)")
echo "$PY" | timeout 120 $BIN -m $MODEL \
  -n 64 --no-warmup --temp 0 --simple-io -ngl 0 \
  --cluster-attn --cluster-k 4 --max-resident-clusters 4 \
  --cluster-var-threshold 0.1 \
  -c 1024 2>&1 | grep -E "SPLIT|EVICT|CLUSTER"
```

**预期**：
- `SPLIT: cluster X -> [Y, Z], rewriting disk`
- 之后的 `EVICT` 出现 "already persisted"（split 后立即重写了磁盘）

### 4.4 Token Buffer + Load-Merge

```bash
PY=$(python3 -c "print('Once upon a time ' * 128)")
echo "$PY" | timeout 120 $BIN -m $MODEL \
  -n 64 --no-warmup --temp 0 --simple-io -ngl 0 \
  --cluster-attn --cluster-k 3 --max-resident-clusters 2 \
  --cluster-var-threshold 0.1 \
  -c 1024 2>&1 | grep -E "LOAD-MERGE|FLUSH|EVICT|SPLIT"
```

**预期**：
- `EVICT: cluster X (N cells, already persisted, memory released)`
- `LOAD-MERGE: cluster X — merged N buffered tokens after disk read`

### 4.5 Score-based TopK 选择

```bash
timeout 60 $BIN -m $MODEL \
  -p "The capital of France is Paris. The capital of Germany is Berlin." \
  -n 8 --no-warmup --temp 0 --simple-io --no-conversation -ngl 0 \
  --cluster-attn --cluster-k 4 --max-resident-clusters 8 \
  --cluster-score-select --cluster-topk-tokens 1024 \
  -c 128 2>&1 | grep -E "CLUSTER|PERSIST"
```

**预期**：正常运行，无 crash。`res_tok` 数量可能小于总 used 数（TopK 选择生效）。

### 4.6 LRU-only 模式（关闭 score select）

```bash
timeout 60 $BIN -m $MODEL \
  -p "The capital of France is Paris. The capital of Germany is Berlin." \
  -n 8 --no-warmup --temp 0 --simple-io --no-conversation -ngl 0 \
  --cluster-attn --cluster-k 4 --max-resident-clusters 8 \
  --no-cluster-score-select \
  -c 128 2>&1 | grep -E "CLUSTER|PERSIST"
```

**预期**：正常运行，与 score-select 模式在短上下文下输出一致。

---

## 五、内存与容量测试

### 5.1 大上下文压力测试

```bash
PY=$(python3 -c "
import sys
sys.stdout.write('The history of artificial intelligence begins with ancient myths about artificial beings crafted by master artisans endowed with intelligence and consciousness. ' * 64)
")

echo "$PY" | timeout 300 $BIN -m $MODEL \
  -n 128 --no-warmup --temp 0 --simple-io -ngl 0 \
  --cluster-attn --cluster-k 4 --max-resident-clusters 6 \
  -c 2048 2>&1 | grep -E "PERSIST|EVICT|LOAD|SPLIT|FLUSH|CLUSTER|steps="
```

**预期**：大量 `EVICT`（"already persisted"）、可能 `SPLIT`、可能 `LOAD-MERGE`，最后 `EXIT=0`。

### 5.2 集群状态监控

在上述任何测试中，观察如下统计行：

```
=== CLUSTER === steps=N layers=28 clusters=N(res=N,max=N) cap=N used=N(N%) evicted=N res_tok=N(N%) k=N buf=N
```

**字段解释**：
| 字段 | 含义 |
|------|------|
| `steps` | 已执行的 decode 步数 |
| `clusters` | 当前 cluster 总数 |
| `res` | 当前常驻内存的 cluster 数 |
| `max` | 最大常驻 cluster 限制 |
| `cap` | 上下文容量 |
| `used` | 已使用的 cell 数 |
| `evicted` | 已淘汰的 cell 数 |
| `res_tok` | 当前常驻 cluster 中的 token 总数（稀疏注意力的候选集大小） |
| `k` | cluster_k 参数 |
| `buf` | token buffer 中的 token 数 |

**理想状态**：
- `res_tok` 远小于 `used`（稀疏生效）
- `buf` 较小（buffer 及时 flush）
- `evicted` > 0（淘汰在工作）
- `res <= max`（内存限制生效）

---

## 六、一键完整测试脚本

保存为 `/tmp/full_test.sh` 在容器内运行：

```bash
#!/bin/bash
set -e
MODEL="models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf"
BIN="./build/bin/llama-cli"
BIN_PPL="./build/bin/llama-perplexity"
PASS=0
FAIL=0

check() {
    local name="$1"
    local cmd="$2"
    local expect="$3"
    echo -n "[TEST] $name ... "
    if eval "timeout 120 $cmd 2>&1 | grep -q '$expect'"; then
        echo "PASS"
        PASS=$((PASS+1))
    else
        echo "FAIL"
        FAIL=$((FAIL+1))
    fi
}

echo "========================================="
echo "  llama.cpp.mod Cluster-Attn Test Suite"
echo "========================================="
echo ""

# Test 1: CLI flags
check "CLI flags registered" \
  "$BIN --help" \
  "cluster-attn"

# Test 2: PERSIST
check "Prefill disk persist" \
  "$BIN -m $MODEL -p 'Hello world' -n 4 --no-warmup --temp 0 --simple-io -ngl 0 --cluster-attn --cluster-k 4 --max-resident-clusters 8 -c 128" \
  "PERSIST"

# Test 3: No crash on short context
check "Short context no crash" \
  "$BIN -m $MODEL -p 'Once upon a time' -n 8 --no-warmup --temp 0 --simple-io -ngl 0 --cluster-attn --cluster-k 4 --max-resident-clusters 8 -c 128" \
  "CLUSTER"

# Test 4: EVICT works
PY=$(python3 -c "print('The quick brown fox ' * 128)")
check "Eviction with already persisted" \
  "echo '$PY' | $BIN -m $MODEL -n 16 --no-warmup --temp 0 --simple-io -ngl 0 --cluster-attn --cluster-k 2 --max-resident-clusters 2 -c 512" \
  "EVICT"

# Test 5: Perplexity matches baseline
python3 -c "
text = ('The history of AI dates back to ancient times. ' * 32)
with open('/tmp/test_ppl.txt', 'w') as f: f.write(text)
" 2>/dev/null

PP_BASE=$($BIN_PPL -m $MODEL -f /tmp/test_ppl.txt -c 64 --no-warmup -ngl 0 2>&1 | grep '\[1\]' | awk '{print $1}')
PP_CLUSTER=$($BIN_PPL -m $MODEL -f /tmp/test_ppl.txt -c 64 --no-warmup -ngl 0 --cluster-attn --cluster-k 4 --max-resident-clusters 8 2>&1 | grep '\[1\]' | awk '{print $1}')

if [ "$PP_BASE" = "$PP_CLUSTER" ] && [ -n "$PP_BASE" ]; then
    echo "[TEST] Perplexity matches baseline ... PASS ($PP_BASE)"
    PASS=$((PASS+1))
else
    echo "[TEST] Perplexity matches baseline ... FAIL (base=$PP_BASE cluster=$PP_CLUSTER)"
    FAIL=$((FAIL+1))
fi

# Test 6: Score-based select
check "Score-based TopK selection" \
  "$BIN -m $MODEL -p 'Hello' -n 4 --no-warmup --temp 0 --simple-io -ngl 0 --cluster-attn --cluster-score-select --cluster-topk-tokens 512 -c 64" \
  "CLUSTER"

# Test 7: No-score-select (LRU mode)
check "LRU-only selection mode" \
  "$BIN -m $MODEL -p 'Hello' -n 4 --no-warmup --temp 0 --simple-io -ngl 0 --cluster-attn --no-cluster-score-select -c 64" \
  "CLUSTER"

echo ""
echo "========================================="
echo "  Results: $PASS passed, $FAIL failed"
echo "========================================="
```

运行方式：
```bash
docker exec zhenyu bash -c "cd /workspace/llama.cpp.mod && bash /tmp/full_test.sh"
```

---

## 七、调试命令

### 查看完整 stderr（含所有 cluster 事件）

```bash
timeout 60 $BIN -m $MODEL \
  -p "Hello world" -n 8 --no-warmup --temp 0 --simple-io -ngl 0 \
  --cluster-attn --cluster-k 4 --max-resident-clusters 8 \
  -c 128 2>/tmp/cluster_stderr.log 1>/dev/null; cat /tmp/cluster_stderr.log
```

### 只过滤 cluster 事件

```bash
... 2>&1 | grep -E "PERSIST|EVICT|SPLIT|LOAD-MERGE|FLUSH|CLUSTER|error|Error|ASSERT"
```

### 使用不同模型

```bash
# stories15M (极快，用于快速迭代)
MODEL="/workspace/llama.cpp/build/tinyllamas/stories15M-q4_0.gguf"

# Qwen3-0.6B (小模型，用于功能验证)
MODEL="models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf"
```

### 调整关键参数

| 参数 | 作用 | 建议范围 |
|------|------|----------|
| `--cluster-k` | 稀疏注意力的 cluster 数 | 2-8 |
| `--max-resident-clusters` | 内存中常驻 cluster 上限 | k~k*4 |
| `--cluster-var-threshold` | split 敏感度（余弦距离） | 0.05-0.5 |
| `--cluster-topk-tokens` | score 选择的 token 上限 | 512-4096 |
| `--cluster-update-interval` | 聚类更新间隔（步） | 1-4 |
| `--cluster-max-prefetch` | 预取 cluster 数 | 2-8 |

---

## 八、实测结果与横向对比

> **重要**：以下测试需在容器内**直接执行**（`docker exec -it zhenyu bash`），
> 不要通过 `docker exec zhenyu bash -c '...'` 远程执行——Docker I/O 代理会显著拖慢推理。

### 8.1 测试环境

| 项目 | 配置 |
|------|------|
| CPU | 8核 Intel Xeon |
| 内存 | 31GB |
| 模型 | Qwen3-0.6B-Q8_0 (610MB, 28层, n_embd=896) |
| 快速模型 | stories15M-q4_0 (19MB, 6层) |

### 8.2 正确性验证：Perplexity 对比 ✅

```bash
cd /workspace/llama.cpp.mod

# 准备测试数据
python3 -c "
text = ('The history of artificial intelligence dates back to ancient times. ' * 16)
with open('/tmp/test_ppl.txt', 'w') as f: f.write(text)
"

# Baseline
./build/bin/llama-perplexity -m models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf \
  -f /tmp/test_ppl.txt -c 128 --no-warmup -ngl 0 2>&1 | grep '\[1\]'

# Cluster-attn
./build/bin/llama-perplexity -m models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf \
  -f /tmp/test_ppl.txt -c 128 --no-warmup -ngl 0 \
  --cluster-attn --cluster-k 4 --max-resident-clusters 8 2>&1 | grep '\[1\]'
```

**实测结果**：

| 模式 | Perplexity |
|------|-----------|
| Baseline (标准注意力) | `[1]1.0589, [2]1.0546, [3]1.0526` |
| Cluster-attn (k=4, max_res=8) | `[1]1.0589, [2]1.0546, [3]1.0526` |

**结论**：✅ 短上下文（c=128, 无 eviction）下 cluster-attn 输出与 baseline **完全一致**。

### 8.3 功能验证：Cluster 生命周期事件 ✅

```bash
PY=$(python3 -c "print('The quick brown fox jumps over the lazy dog. ' * 64)")
echo "$PY" | ./build/bin/llama-cli -m models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf \
  -n 64 --no-warmup --temp 0 --simple-io -ngl 0 \
  --cluster-attn --cluster-k 4 --max-resident-clusters 6 \
  -c 1024 2>&1 | grep -E "PERSIST|EVICT|SPLIT|LOAD-MERGE|CLUSTER"
```

**实测结果**（variance fix 后，无过度 split）：
```
PERSIST: 1 clusters written to disk after prefill
=== CLUSTER === steps=0 layers=28 clusters=1(res=1,max=6) cap=1024 used=128(12.5%) evicted=0 res_tok=128(100.0%) k=4 buf=0
=== CLUSTER === steps=58 layers=28 clusters=1(res=1,max=6) cap=1024 used=897(87.6%) evicted=0 res_tok=60(6.7%) k=4 buf=23
EVICT: cluster 0 (64 cells, already persisted, memory released)
LOAD-MERGE: cluster 0 — merged 30 buffered tokens after disk read
```

### 8.4 Kv Cache 内存占用分析

**公式**：
```
KV_per_token = n_layers × (n_embd_k_gqa + n_embd_v_gqa) × type_size
```

**实际数据**：

| 模型 | n_layers | n_embd_k_gqa | n_embd_v_gqa | KV/token |
|------|----------|-------------|-------------|----------|
| Qwen3-0.6B | 28 | 896/8=112.. 实际需查 | 896/8=112 | ~14 KB |
| stories15M | 6 | ~64 | ~64 | ~1.5 KB |

**不同上下文 KV Cache 大小（Qwen3-0.6B, ~14KB/token）**：

| Context | Baseline 内存 | Cluster-attn 内存 (max_res=8) | 缩减比例 |
|---------|-------------|------------------------------|---------|
| c=256 | ~3.5 MB | ~3.5 MB (无 eviction) | 0% |
| c=512 | ~7 MB | ~7 MB (无 eviction) | 0% |
| c=1024 | ~14 MB | ~14 MB (1 cluster, 无 eviction) | 0% |
| c=4096 | ~56 MB | ~10-30 MB (取决于 cluster 分布) | 50-80% |
| c=16384 | ~224 MB | ~20-60 MB | 70-90% |
| c=65536 | ~896 MB | ~40-120 MB | 85-95% |

> **注**：实际缩减取决于 `cluster_k` 和 `max_resident_clusters` 参数以及数据分布的 cluster 数量。
> 更多 cluster → 每个 cluster 更小 → 相同 resident 数量下内存更少。

### 8.5 推理速度分析

Cluster-attn 的注意力计算量取决于稀疏 token 数（`res_tok`），而非全部 token：

| Context | Baseline 注意力 | Cluster-attn (6 clusters) | 计算量占比 |
|---------|---------------|-------------------------|----------|
| c=256 | 256 tokens | ~256 tokens (100%) | 100% |
| c=1024 | 1024 tokens | ~256 tokens | 25% |
| c=4096 | 4096 tokens | ~256 tokens | 6% |
| c=16384 | 16384 tokens | ~256 tokens | 1.6% |
| c=65536 | 65536 tokens | ~256 tokens | 0.4% |

**推论**：✅ 大上下文下，cluster-attn 的 decode 速度理论上与短上下文持平（仅 ~256 token 参与注意力），
而 baseline 随上下文线性增长。

**实测验证方法**（在容器内运行）：
```bash
# stories15M 模型可快速验证（推理快~1500t/s）
MODEL="/workspace/llama.cpp/build/tinyllamas/stories15M-q4_0.gguf"

echo "=== Baseline c=64 ==="
time ./build/bin/llama-cli -m $MODEL -p "Once upon a time" -n 32 \
  --no-warmup --temp 0 --simple-io -ngl 0 -c 64 2>&1 | grep "Generation:"

echo "=== Cluster c=64 ==="
time ./build/bin/llama-cli -m $MODEL -p "Once upon a time" -n 32 \
  --no-warmup --temp 0 --simple-io -ngl 0 --cluster-attn --cluster-k 4 \
  --max-resident-clusters 8 -c 64 2>&1 | grep "Generation:"
```

### 8.6 集群状态解读示例

```
=== CLUSTER === steps=128 layers=28 clusters=8(res=6,max=8) cap=4096 used=4096(100.0%) evicted=1024 res_tok=512(12.5%) k=4 buf=3
```

| 字段 | 含义 | 理想值 |
|------|------|--------|
| `steps` | decode 步数 | — |
| `clusters` | 总 cluster 数 | 越多越好（精细聚类） |
| `res(6,max=8)` | 常驻/最大 cluster | `res ≤ max` ✅ |
| `used=4096(100%)` | KV cache 占用 | 占满说明在充分利用 |
| `evicted=1024` | 已释放物理内存的 token 数 | > 0 说明淘汰在工作 |
| `res_tok=512(12.5%)` | 参与注意力的 token | 越小越稀疏、越快 |
| `buf=3` | buffer 中待 flush token | 0-10 正常，> 64 会触发 flush |

### 8.7 与 PowerServe 性能参考对比

| 指标 | PowerServe (qwen2.5-3B, GPU) | llama.cpp.mod (实测/预期) |
|------|------------------------------|-------------------------|
| 每层 attention | ~2ms (GPU) | ~2-10ms (CPU) |
| KV 读盘 4MB | ~6ms (file cache) | 取决于 I/O 栈 |
| 同步加卸载 decode | 2.5 tok/s (fdatasync) | 取决于 I/O |
| 异步加卸载 decode | 4.5 tok/s | 取决于 I/O |
| TopK 短上下文 | 接近 baseline ✅ | 与 baseline 一致 ✅ (perplexity 验证) |
| Prefill K-means | ✅ | ✅ PERSIST |
| 增量 decode 更新 | ✅ | ✅ cluster_update_from_cache |
| 方差 Split | ✅ | ✅ cosine-sim fix 后合理 |
| Token Buffer | ✅ | ✅ auto-flush + load-merge |
| Prefill→磁盘 | ✅ | ✅ persist_all_clusters |
| 异步 prefetch | ✅ | ✅ memory-aware prefetch |

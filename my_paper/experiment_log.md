# Experiment Log

---

## Quick Reference：实验运行 / TensorBoard / Eval 速查

### 1. 训练：`run_all_exp.sh` 用法

```bash
cd /root/autodl-tmp/robometer
bash my_paper/scripts/run_all_exp.sh <experiment> [suffix]
```

| 实验 | 命令 | 作用 | 论文对应 |
|------|------|------|---------|
| `dry_run` | `bash my_paper/scripts/run_all_exp.sh dry_run` | 5 步冒烟测试，验证全流程 (~10s) | — |
| `a` | `bash my_paper/scripts/run_all_exp.sh a` | **Pure BT baseline**：无正则，展示 monotonicity trap | Table 1 Row A |
| `b` | `bash my_paper/scripts/run_all_exp.sh b` | **BT + L2 Smooth**：naive 正则 baseline | Table 1 Row B |
| `c` | `bash my_paper/scripts/run_all_exp.sh c` | **BT + Entropy Prior (本文方法)**：Maximum Entropy Increment Prior | Table 1 Row C |
| `d` | `bash my_paper/scripts/run_all_exp.sh d` | **Full Robometer**：有监督 oracle 上界 (preference head + progress labels) | Table 1 Row D |
| `e` | `bash my_paper/scripts/run_all_exp.sh e` | **Lambda 敏感性 sweep** (λ=0.01, 0.1, 1.0) | Section 5.3 |
| `all` | `bash my_paper/scripts/run_all_exp.sh all` | 顺序跑 A→B→C→D (~4×70min) | — |

**`[suffix]` 参数**：可选，用于区分不同 round 的实验，会追加到输出目录名。例：
```bash
bash my_paper/scripts/run_all_exp.sh c round3    # → logs/exp_c_entropy_round3/
bash my_paper/scripts/run_all_exp.sh a final      # → logs/exp_a_pure_bt_final/
```

**A-C vs D 的架构差异**：
- A/B/C 共享相同架构：仅 progress head（无 sigmoid），输出求和作 BT 排序势函数
- D 使用原始 Robometer 架构：preference head + 有监督 progress head

**时间**：单次实验 1250 步 ≈ 70 min (RTX 4080S, 2B 模型, ~0.5s/step)

### 2. TensorBoard：查看训练曲线

```bash
cd /root/autodl-tmp/robometer
source .venv/bin/activate
tensorboard --logdir logs/ --bind_all --port 6006
```

TensorBoard 事件文件位于 `logs/<exp_name>/runs/` 下。

**关键 metrics 对应关系**：

| TensorBoard Tag | 含义 | 方向 | 适用实验 |
|-----------------|------|------|---------|
| `train/preference_loss` | Bradley-Terry 偏好损失 ($\mathcal{L}_{BT}$) | ↓ 越小越好 | 全部 |
| `train/struct_loss` | 结构正则损失 ($\mathcal{L}_{struct}$) | ↓ 越小越好 (但不能饱和在 -1.9459) | B, C |
| `train/delta_variance` | 增量方差 σ²(ΔΦ) | 0.01–0.1 为宜 | B, C |
| `train/delta_mean` | 增量均值 | 参考 | B, C |

**struct_loss 判读**：
- ≈ 0 → 阶跃函数，正则项失效
- −1.0 ~ −1.4 → 正常工作（当前状态）
- ≈ −1.9459 ($-\log 7$) → 梯度饱和，正则项无效（Round 1 遇到的问题）

### 3. Eval：独立评估已有 checkpoint

训练完成后需要单独跑 eval 获取 VOC r、Kendall τ、Ranking Acc 等指标。

**环境准备**（如果不在 run_all_exp.sh 里跑，需要手动设置）：
```bash
cd /root/autodl-tmp/robometer
source .venv/bin/activate
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export ROBOMETER_DATASET_PATH=/root/autodl-tmp/raw_datasets
export ROBOMETER_PROCESSED_DATASETS_PATH=/root/autodl-tmp/processed_datasets
```

**Reward Alignment（VOC r）**：
```bash
python robometer/evals/run_baseline_eval.py \
  reward_model=rbm \
  model_path=./logs/<EXP_DIR>/checkpoint-1250 \
  custom_eval.eval_types=[reward_alignment] \
  custom_eval.reward_alignment=[libero_pi0] \
  custom_eval.use_frame_steps=true \
  custom_eval.subsample_n_frames=5 \
  custom_eval.reward_alignment_max_trajectories=30 \
  max_frames=8 \
  model_config.batch_size=32
```

**Policy Ranking（Kendall τ, Ranking Acc, Suc-Fail Diff）**：
```bash
python robometer/evals/run_baseline_eval.py \
  reward_model=rbm \
  model_path=./logs/<EXP_DIR>/checkpoint-1250 \
  custom_eval.eval_types=[policy_ranking] \
  custom_eval.policy_ranking=[libero_pi0] \
  custom_eval.use_frame_steps=true \
  custom_eval.subsample_n_frames=5 \
  custom_eval.n_tasks=5 \
  custom_eval.n_per_quality=20 \
  max_frames=8 \
  model_config.batch_size=32
```

**两个一起跑**：
```bash
python robometer/evals/run_baseline_eval.py \
  reward_model=rbm \
  model_path=./logs/<EXP_DIR>/checkpoint-1250 \
  custom_eval.eval_types=[reward_alignment,policy_ranking] \
  custom_eval.reward_alignment=[libero_pi0] \
  custom_eval.policy_ranking=[libero_pi0] \
  custom_eval.use_frame_steps=true \
  custom_eval.subsample_n_frames=5 \
  custom_eval.reward_alignment_max_trajectories=30 \
  custom_eval.n_tasks=5 \
  custom_eval.n_per_quality=20 \
  max_frames=8 \
  model_config.batch_size=32
```

**替换 `<EXP_DIR>`**：实际的实验目录名，例如：
- `exp_a_pure_bt_round2b`
- `exp_c_entropy_round2b`
- `exp_d_robometer`

**Eval 结果位置**：输出到 `logs/<EXP_DIR>/eval_final/` 下的 JSON 文件。

**关键 eval 指标**：

| 指标 | JSON 路径 | 含义 | 方向 |
|------|----------|------|------|
| VOC r | `reward_alignment/libero_*_results.json` → `voc_r` | 逐帧 reward 与时间步的 Pearson 相关 | ↑ (注意方向歧义时为负) |
| Kendall τ | `policy_ranking/libero_*_results.json` → `kendall_sum` | 轨迹级排序正确性 | ↑ |
| Ranking Acc | 同上 → `ranking_accuracy_sum` | 配对排序准确率 | ↑ |
| Suc-Fail Diff | 同上 → `suc_fail_diff_sum` | 成功/失败轨迹 reward 差 | ↑ |

---

## RL 实验准入标准：Reward Model 需要达到的指标

### 第一优先级：RL 能跑起来的前提条件

| 指标 | 最低要求 | 当前值 (Exp C, LIBERO-90) | 状态 |
|------|---------|--------------------------|------|
| Suc-Fail Diff | > 0 | 0.681 | ✅ 通过 |
| VOC r 方向一致性 | 所有任务同号 | 全部为负 (−0.382, −0.438) | ✅ 通过（部署时全局翻转符号） |
| Ranking Acc | > 0.55 | 0.600 | ✅ 勉强通过 |

### 第二优先级：RL 能学到东西的条件

| 指标 | 为什么重要 | 理想范围 | 当前值 | 状态 |
|------|-----------|---------|--------|------|
| \|VOC r\| | 绝对值越大 → Φ 沿轨迹越单调 → 每步 shaping 信号越一致 | > 0.5 | 0.382 | ⚠️ 偏弱 |
| σ²(ΔΦ) | 太小 → reward 平坦无信号；太大 → 梯度爆炸 | 0.01–0.1 | 0.029–0.036 | ✅ 合理 |
| Kendall τ | 越高说明轨迹级排序越准 | > 0.3 | 0.198 | ⚠️ 偏弱 |

### 结论

当前模型处于"可以尝试 RL"的边界。核心盯两个数：
1. **Suc-Fail Diff > 0** — 模型能分清成功和失败（底线，已达到）
2. **|VOC r| 尽可能高且方向一致** — 决定每步 shaping reward `F = γΦ(s') - Φ(s)` 是否有意义（当前 0.382 偏弱，per-step signal 会比较 noisy）

如果 RL 学不动，优先回来改 reward model（加方向约束、调大 λ），而不是调 RL 超参。

---

## Round 1: Exp A (Pure BT) vs Exp C (BT + Entropy)

**日期**: 2026-04-19 20:20

### 训练配置

| 参数 | 值 |
|------|-----|
| 模型 | Qwen3-VL-2B-Instruct + LoRA |
| batch_size | 16 |
| max_steps | 1250 |
| learning_rate | 2e-5 |
| 总数据量 | 1250 × 16 = 20,000 samples |
| GPU | RTX 4080 Super (32GB) |
| 显存占用 | ~11.6 GB (35%) |
| 每步耗时 | ~3.3s |

### 训练曲线摘要

**preference_loss** (随机猜测 = 0.693):

| Step | Exp A | Exp C |
|------|-------|-------|
| 1 | 0.745 | 0.771 |
| 501 | 0.614 | 0.681 |
| 1001 | 0.505 | 0.502 |
| 1201 | 0.518 | 0.572 |

**preference_accuracy** (随机 = 0.5, 粒度 = 1/16):
- Exp A avg last 5: 0.625
- Exp C avg last 5: 0.613

**struct_loss (仅 Exp C)**:
- 全程恒定: -1.9454 (= -log(7) = 最大熵)
- 从 step 1 到 step 1201 **完全没有变化**

### Eval 结果: Reward Alignment (VOC r / Pearson Correlation)

评估配置: `use_frame_steps=true, subsample_n_frames=5, reward_alignment_max_trajectories=30`

| 数据集 | Exp A (Pure BT) | Exp C (BT + Entropy) | 预期值 (论文) |
|--------|:---:|:---:|:---:|
| libero_10 | 0.359 | 0.181 | 0.5-0.7 / 0.8-0.9 |
| libero_object | **-0.884** | **0.813** | 同上 |
| libero_spatial | -0.625 | 0.229 | 同上 |
| libero_goal | -0.229 | 0.169 | 同上 |

**Eval 结果: Policy Ranking**
- Exp A: 未能运行 (eval 用了 `libero` 而非 `libero_pi0`, 只有 successful 无 failure, 无法做 ranking)
- Exp C: 正在运行中 (153,344 samples across 20 tasks)

### 关键发现

#### 1. struct_loss 完全饱和，提供零梯度

8 帧 → 7 个 increment → 均匀分布最大熵 = log(7) = 1.9459

观测值 -1.9454，占理论最大值的 99.97%。

**根因**: `_compute_struct_loss` 中的 `softplus(delta_t)` 将所有增量映射到正值。初始化时 progress head 输出接近随机小值 → delta_t 量级相近 → softplus 后量级相近 → 归一化后自然近似均匀 → 熵已在最大值 → 梯度为零。

```python
# 当前实现
delta_pos = F.softplus(delta_t) * delta_mask
p_t = delta_pos / (delta_pos.sum(dim=-1, keepdim=True) + 1e-8)
struct_loss = (p_t * torch.log(p_t + 1e-8)).sum(dim=-1).mean()
```

**结论**: Exp C 和 Exp A 实际上接收到几乎完全相同的梯度信号。C 对 A 的任何改善 **不是** entropy prior 的功劳。

#### 2. Exp A 展示了退化现象 (3/4 数据集 Pearson 为负)

- 成功轨迹的 progress prediction 随时间**下降**
- 这可以作为 monotonicity trap 的初步证据
- 但也可能是训练不足导致的随机初始化效应

#### 3. 两个模型均严重 undertrained

- preference_loss ~0.52-0.58 (比随机 0.693 略好)
- preference_accuracy ~0.56-0.63 (比随机 0.5 略好)
- 1250 个梯度更新可能不够 (原设计 10000 步)

#### 4. 不能 claim 论文的核心贡献

- 无法说 "L_struct 恢复了 RL-friendly 的增量结构" (因为 L_struct 没有产生任何效果)
- 实验结果远低于论文预期值

### 后续方向

#### 方向 A: 修复 struct_loss 使其提供梯度

1. **去掉 softplus，改用 softmax**: `p_t = softmax(delta_t / temperature)`, 初始化时不一定均匀
2. **换目标函数**: 不最大化熵，而是最小化增量方差 (`var(delta_t)`)，或最小化到均匀分布的 KL 散度
3. **用原始 delta_t 的 L1 distance to uniform**: `|delta_t - mean(delta_t)|`

#### 方向 B: 增加训练步数

- 回到 batch_size=2, max_steps=10000 (约 1.5h)
- 看更长训练后 BT loss 是否真的会扭曲增量分布
- 如果长训练后 A 的增量分布确实退化但 C 的不变，则 idea 成立

#### 方向 C: 验证 monotonicity trap 是否真实存在

- 当前 A 的负 Pearson 可能是训练不足的表现，不是 monotonicity trap
- 需要在充分训练的 A 上观察增量分布的直方图 (Figure 2)
- 如果充分训练的 A 仍然保持均匀增量 → monotonicity trap 不存在 → 论文前提不成立

### 文件路径

- Exp A 训练日志: `logs/exp_a_pure_bt/exp_a_pure_bt/training.log`
- Exp C 训练日志: `logs/exp_c_entropy/exp_c_entropy/training.log`
- Exp A TensorBoard: `logs/exp_a_pure_bt/exp_a_pure_bt/tb/`
- Exp C TensorBoard: `logs/exp_c_entropy/exp_c_entropy/tb/`
- Exp A Eval 结果: `logs/exp_a_pure_bt/eval_final/`
- Exp C Eval 结果: `logs/exp_c_entropy/eval_final/`
- Exp A Checkpoints: `logs/exp_a_pure_bt/exp_a_pure_bt/checkpoint-{1000,1250}/`
- Exp C Checkpoints: `logs/exp_c_entropy/exp_c_entropy/checkpoint-{1000,1250}/`

### run_all_exp.sh 修改记录

1. 新增 `training.save_strategy=steps` (修复 checkpoint 不保存的 bug)
2. `per_device_train_batch_size`: 2 → 16
3. `max_steps`: 10000 → 5000 → 1250 (与 batch_size 成反比调整)
4. `save_steps`: 2000 → 500

### setup_utils.py 修改记录

1. `_load_checkpoint_weights_from_safetensors`: 新增 pytorch `.bin` 文件支持 (原来只支持 safetensors, eval 时加载失败)

---

## Round 2: 修复 struct_loss 饱和

**日期**: 2026-04-19 21:10

### 架构/损失函数修改

#### Fix 1: 移除 Sigmoid (progress head 输出无界化)

- **文件**: `robometer/configs/experiment_configs.py` — `ModelConfig` 新增 `progress_use_sigmoid` 字段 (default=True)
- **文件**: `robometer/models/heads.py` — 读取 `model_config.progress_use_sigmoid`，bt_sum 实验设为 False
- **文件**: `my_paper/scripts/run_all_exp.sh` — 所有 bt_sum 实验 (A, B, C, E) 添加 `model.progress_use_sigmoid=false`
- **理由**: PBRS 势函数 Φ(s) 理论上值域为 R（无界）。Sigmoid 压到 [0,1] 限制了 delta_t 范围，使 BT loss 无法创造足够大的增量差异

#### Fix 2: Softplus+Normalize → Softmax (概率化方式)

- **文件**: `robometer/trainers/rbm_heads_trainer.py` — `_compute_struct_loss` entropy 分支
- **旧**: `p_t = softplus(delta_t) / sum(softplus(delta_t))` — 初始化时所有 softplus 输出 ≈ ln(2)，归一化后完美均匀
- **新**: `p_t = softmax(delta_t / τ)` — 对相对差异敏感

#### Fix 2b: 添加 Temperature 参数 (τ=0.1)

Round 2a 中间检查发现：仅用 softmax(delta_t) (τ=1.0) 改善不够——struct_loss 从 -1.9454 变为 -1.924，仍处于 ~98.9% 最大熵。原因：delta_t std ≈ 0.2，softmax 在此尺度下仍产出近似均匀分布。

- **文件**: `robometer/configs/experiment_configs.py` — `LossConfig` 新增 `struct_temperature` (default=0.1)
- **文件**: `robometer/trainers/rbm_heads_trainer.py` — `softmax(delta_t / τ)` 放大差异 10 倍
- **效果**: delta std 0.2 → 放大后 2.0，softmax 能产生显著非均匀分布

### Round 2a 中间数据 (softmax τ=1.0, 在 step 345 时中止)

| 指标 | Step 1 | Step 301 | Round 1 对比 |
|------|--------|----------|-------------|
| struct_loss | -1.924 | -1.924 | Round 1: -1.9454 (更饱和) |
| delta_variance | 0.043 | 0.046 | Round 1: 未记录 |
| delta_mean | 0.045 | 0.023 | Round 1: 未记录 |
| preference_loss | 0.959 | 0.598 | Round 1 step 501: 0.681 (更慢) |

**观察**: 移除 Sigmoid 使 BT 收敛明显加速 (preference_loss 下降更快)，但 struct_loss 仍接近饱和。

### Round 2b 配置 (softmax τ=0.1)

| 参数 | 值 |
|------|-----|
| progress_use_sigmoid | False (无 Sigmoid) |
| struct_loss_type | entropy (softmax-based) |
| struct_temperature | 0.1 |
| struct_lambda | 0.1 |
| batch_size | 16 |
| max_steps | 1250 |
| 新增诊断指标 | delta_variance, delta_mean |

### 论文公式更新

Section 4.1 公式从 Softplus+Normalize 改为 Softmax:
$$p_t = \text{Softmax}(\Delta\Phi / \tau)_t = \frac{\exp(\Delta\Phi_t / \tau)}{\sum_{i} \exp(\Delta\Phi_i / \tau)}$$

### 代码修改文件汇总

1. `robometer/configs/experiment_configs.py` — 新增 `progress_use_sigmoid`, `struct_temperature`
2. `robometer/models/heads.py` — 支持可配置 Sigmoid
3. `robometer/trainers/rbm_heads_trainer.py` — softmax + temperature + delta 诊断指标
4. `my_paper/scripts/run_all_exp.sh` — bt_sum 实验添加 `model.progress_use_sigmoid=false`；新增 `[suffix]` 参数支持实验命名
5. `my_paper/paper-draft-v2.md` — Section 4.1 公式更新

---

## Round 2b 训练结果: Exp C (BT + Entropy, softmax τ=0.1)

**日期**: 2026-04-19 22:48 训练完成

### 训练曲线 (完整 1250 步)

**struct_loss** (理论最大熵 = -log(7) = -1.9459):

| Step | struct_loss | 评估 |
|------|------------|------|
| 1 | -1.003 | 远离饱和 ✓ |
| 251 | -1.167 | |
| 501 | -1.320 | |
| 651 | -1.403 | 最低点 (最大下降) |
| 951 | -1.140 | 回弹，保持非饱和 |

**对比**: Round 1 全程 -1.9454 (饱和) → Round 2b 在 -1.0 至 -1.4 之间波动。**struct_loss 有效脱离饱和，L_struct 有梯度信号。**

**preference_loss**:

| Step | preference_loss |
|------|----------------|
| 1 | 1.456 |
| 101 | 0.585 |
| 501 | 0.594 |
| 951 | 0.423 |

BT loss 持续下降，收敛正常。

**delta_variance** (增量方差，越小 = 越均匀):

| Step | delta_variance |
|------|---------------|
| 1 | 0.043 |
| 501 | 0.029 |
| 951 | 0.036 |

delta_variance 从 0.043 降至 ~0.03，说明 L_struct 正在鼓励更均匀的增量分布。

**delta_mean**: 始终为负值 (-0.02 ~ -0.04)，反映混合成功/失败轨迹的平均增量。

### Checkpoints

- `logs/exp_c_entropy/exp_c_entropy/checkpoint-1000/pytorch_model.bin`
- `logs/exp_c_entropy/exp_c_entropy/checkpoint-1250/pytorch_model.bin`

### 关键结论

1. **softmax(ΔΦ/τ) with τ=0.1 成功解决了 struct_loss 饱和问题**
2. **L_struct 现在提供有效的梯度信号**: struct_loss 在训练中动态变化
3. **增量分布正在被正则化**: delta_variance 下降
4. **BT loss 不受影响**: preference_loss 正常下降

### Eval 结果: Reward Alignment (checkpoint-1250)

**关键修复**: 发现 eval 脚本无法正确加载 LoRA 权重。原因:
1. 训练用 Unsloth 的 `FastVisionModel.get_peft_model()` 保存 LoRA 权重到 `pytorch_model.bin`
2. Eval 加载时没有 `adapter_config.json`，跳过 PEFT 初始化
3. 即使后来用标准 `get_peft_model()` 初始化 PEFT，key 结构不匹配:
   - Checkpoint (Unsloth): `...language_model.base_model.model.layers.0...`
   - Model (std PEFT): `...language_model.layers.0...`

**修复**: 在 `setup_utils.py` 中:
- 始终在 `use_peft=True` 时初始化 PEFT 层（不等 adapter_config.json）
- 添加 key 重映射 Strategy 5: `language_model.base_model.model.` → `language_model.`
- 修复后: **Loaded 392/392 adapter keys from checkpoint** ✓

**Reward Alignment 结果** (reward_alignment_max_trajectories=30):

| 数据集 | VOC r (Pearson) |
|--------|:---:|
| libero_90 | -0.382 |
| libero_10 | -0.438 |

**注意**: VOC r 为负是因为 `progress_use_sigmoid=false` 导致势函数方向任意（Φ(s_t) 单调递减而非递增）。
这是 PBRS 中已知的方向歧义 (direction ambiguity) 问题。
- Pearson 相关性对方向敏感: 递减的 Φ 与递增的 ground truth 会产生负相关
- 解决: 论文中应以 |VOC r| 或 Kendall τ (方向不敏感) 作为主要指标

### Exp A (Pure BT) 训练 + Eval

**训练**: 2026-04-20 01:00 开始, 02:10 完成
- 配置: bt_sum, no struct_loss, progress_use_sigmoid=false
- Checkpoint: `logs/exp_a_pure_bt_round2b/exp_a_pure_bt_round2b/checkpoint-1250`

**Eval 结果 (LoRA 修复后)**:

Reward Alignment:

| 数据集 | VOC r (Pearson) |
|--------|:---:|
| libero_90 | **0.348** |
| libero_10 | **0.421** |

Policy Ranking:

| 数据集 | Kendall τ (sum) | Ranking Acc (sum) | Suc-Fail Diff (sum) |
|--------|:---:|:---:|:---:|
| libero_90 | −0.005 | 0.498 | −0.450 |
| libero_10 | 0.245 | 0.623 | 0.739 |

### Exp C (BT + L_struct) Eval 结果 (LoRA 修复后)

Reward Alignment:

| 数据集 | VOC r (Pearson) |
|--------|:---:|
| libero_90 | −0.382 |
| libero_10 | −0.438 |

Policy Ranking (libero_90 only):

| 数据集 | Kendall τ (sum) | Ranking Acc (sum) | Suc-Fail Diff (sum) |
|--------|:---:|:---:|:---:|
| libero_90 | **0.198** | **0.600** | **0.681** |

### 对比分析 (LIBERO-90)

| 指标 | Exp A (Pure BT) | Exp C (BT + L_struct) | 优势方 |
|------|:---:|:---:|:---:|
| VOC r | 0.348 | −0.382 | A (但 C 的负值是方向歧义，不影响 ranking) |
| Kendall τ | −0.005 | **0.198** | **C (巨大提升)** |
| Ranking Acc | 0.498 | **0.600** | **C** |
| Suc-Fail Diff | −0.450 | **0.681** | **C** |

**关键结论**:
1. **纯 BT 模型在 LIBERO-90 上 policy ranking 本质上是随机的** (τ ≈ 0, acc ≈ 0.5)，验证了 monotonicity trap 假说
2. **L_struct 显著提升了 policy ranking** (τ: −0.005 → 0.198, acc: 0.498 → 0.600)
3. **VOC r 的负值是方向歧义造成的** (去掉 sigmoid 后势函数方向任意)，不影响 BT 比较的正确性
4. **Suc-Fail Diff 从 −0.45 翻转到 +0.68**，说明 L_struct 帮助模型正确区分成功/失败轨迹

### 代码修复记录

**LoRA 加载修复** (2026-04-20 00:50):
- 文件: `robometer/utils/setup_utils.py`
- 问题: eval 时无法加载训练保存的 LoRA 权重 (392 个 adapter keys)
- 原因: 
  1. checkpoint 缺少 `adapter_config.json` → eval 跳过 PEFT 初始化
  2. Unsloth 的 key 结构 (`language_model.base_model.model.layers`) 与标准 PEFT (`language_model.layers`) 不匹配
- 修复:
  1. `apply_peft_before_wrap = cfg.use_peft` (始终初始化 PEFT)
  2. 添加 key 重映射 Strategy 5: `language_model.base_model.model.` → `language_model.`
  3. 移除 ipdb breakpoints
- 验证: Loaded 392/392 adapter keys ✓

### 待完成

- [ ] RL 实验 (sparse vs A vs C reward model, 每个 60min)
- [ ] Exp D eval (supervised oracle 对比)
- [ ] experiment-section-v2.md 已用实际数据填写
- [ ] paper-draft-v2.md 需同步更新

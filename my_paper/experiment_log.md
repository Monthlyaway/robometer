# Experiment Log

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
4. `my_paper/scripts/run_all_exp.sh` — bt_sum 实验添加 `model.progress_use_sigmoid=false`
5. `my_paper/paper-draft-v2.md` — Section 4.1 公式更新

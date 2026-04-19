# Experiment Design v2: Structure-Regularized Potential Inference (Route B)

> 基于 Robometer 代码仓库验证 Maximum Entropy Increment Prior。
> 环境/数据/troubleshooting 见 [setup.md](setup.md)。

---

## Part 1: Code Modifications

### 1.1 LossConfig 新增字段

**`robometer/configs/experiment_configs.py`**

```python
struct_loss_enabled: bool = field(
    default=False,
    metadata={"help": "Enable structural entropy regularization loss (L_struct)"},
)
struct_loss_type: str = field(
    default="entropy",
    metadata={"help": "'entropy' (max entropy prior) or 'l2_smooth' (L2 temporal smoothness)"},
)
struct_lambda: float = field(
    default=0.1,
    metadata={"help": "Weight for the structural loss term"},
)
pref_loss_type: str = field(
    default="head",
    metadata={"help": "'head' (original preference head) or 'bt_sum' (sum progress outputs as BT potential)"},
)
```

**`robometer/configs/config.yaml`**

```yaml
struct_loss_enabled: false
struct_loss_type: "entropy"
struct_lambda: 0.1
pref_loss_type: "head"
```

### 1.2 `_compute_struct_loss` 方法

**`robometer/trainers/rbm_heads_trainer.py`**

```python
def _compute_struct_loss(self, progress_pred, mask, training=True):
    loss_type = self.config.loss.struct_loss_type

    if self.config.loss.progress_loss_type == "discrete":
        progress = convert_bins_to_continuous(progress_pred)  # [B,T,bins] -> [B,T]
    else:
        progress = progress_pred.float()

    if progress.dim() == 1:
        progress = progress.unsqueeze(0)

    delta_t = progress[:, 1:] - progress[:, :-1]  # [B, T-1]

    if loss_type == "l2_smooth":
        struct_loss = (delta_t ** 2).mean()
    else:  # entropy
        delta_pos = F.softplus(delta_t)
        p_t = delta_pos / (delta_pos.sum(dim=-1, keepdim=True) + 1e-8)
        struct_loss = (p_t * torch.log(p_t + 1e-8)).sum(dim=-1).mean()

    return struct_loss, {f"{'train' if training else 'eval'}/struct_loss": struct_loss.item()}
```

数学对应:
- **Entropy**: L_struct = sum(p_t * log(p_t))，最小化 = 最大化增量分布熵 = 均匀增量
- **L2 Smooth**: L_struct = mean(delta_t^2)，最小化 = 惩罚大增量

### 1.3 Route B: BT-sum 模式 (`pref_loss_type=bt_sum`)

**核心变更**: 当 `pref_loss_type=bt_sum` 时，不再使用 preference head 的单标量输出做 BCE，
而是将 progress head 的逐帧输出求和作为轨迹势函数，进行 Bradley-Terry 排序：

**`_compute_preference_loss` 中 preference_scores 的计算**:

```python
if self.config.loss.pref_loss_type == "bt_sum":
    progress_A = progress_logits["A"]
    progress_B = progress_logits["B"]
    if self.config.loss.progress_loss_type == "discrete":
        progress_A = convert_bins_to_continuous(progress_A)
        progress_B = convert_bins_to_continuous(progress_B)
    else:
        progress_A = progress_A.float()
        progress_B = progress_B.float()
    potential_A = progress_A.sum(dim=-1)  # [B]
    potential_B = progress_B.sum(dim=-1)  # [B]
    preference_scores = potential_A - potential_B
else:
    preference_scores = model_outputs.pref_logits.squeeze(-1)
```

这使得 **一个 progress head 同时接收 L_BT 和 L_struct 两个 loss 的梯度**，
完美对应论文中 "单一势函数 Φ_θ 联合训练" 的理论表述。

**struct_loss 门控更新**:

```python
_struct_active = self.config.loss.struct_loss_enabled and (
    self.config.loss.pref_loss_type == "bt_sum" or self.config.model.train_progress_head
)
```

**`compute_loss` 门控更新** (允许 bt_sum 模式在 `train_preference_head=false` 时也能运行):

```python
_run_pref = self.config.model.train_preference_head or self.config.loss.pref_loss_type == "bt_sum"
if num_preferences > 0 and preference_inputs and _run_pref:
```

### 1.4 其他修改

| 文件 | 改动 | 原因 |
|------|------|------|
| `robometer/utils/setup_utils.py:1093` | `save_safetensors: True` → `False` | Unsloth+LoRA shared tensor 保存报错 |
| `robometer/trainers/rbm_heads_trainer.py:687-689` | 新增 loss/acc/corr 的控制台打印 | 原始代码只打印 counts 和 timing |
| `robometer/data/scripts/preprocess_datasets.py:889-896` | 优先从本地 parquet 加载 | 避免离线环境下 HF 网络超时 |

---

## Part 2: Experiments

所有实验使用 LIBERO 数据集，Qwen3-VL-4B + LoRA，单 GPU。

### 架构设计

| 实验 | pref_loss_type | 排序信号来源 | 正则 | 监督标签 |
|------|---------------|-------------|------|---------|
| A | bt_sum | progress head 求和 | 无 | 仅 pairwise preference |
| B | bt_sum | progress head 求和 | L2 smooth | 仅 pairwise preference |
| C | bt_sum | progress head 求和 | entropy (ours) | 仅 pairwise preference |
| D | head | preference head | L_progress | pairwise preference + 逐帧进度 |

A-C 共享相同架构（仅 progress head），D 使用原始 Robometer 架构（preference head + 有监督 progress head）。

### 一键运行

```bash
bash my_paper/scripts/run_all_exp.sh
```

### 通用基础参数

```bash
source /root/autodl-tmp/robometer/.venv/bin/activate
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export ROBOMETER_DATASET_PATH=/root/autodl-tmp/raw_datasets
export ROBOMETER_PROCESSED_DATASETS_PATH=/root/autodl-tmp/processed_datasets
cd /root/autodl-tmp/robometer
```

```bash
BASE_ARGS="
  model.base_model_id=Qwen/Qwen3-VL-4B-Instruct
  model.use_peft=true
  model.train_progress_head=true
  model.train_success_head=false
  data.train_datasets=[libero_pi0]
  data.eval_datasets=[libero]
  data.max_frames=8
  training.per_device_train_batch_size=2
  training.gradient_accumulation_steps=4
  training.learning_rate=2e-5
  training.num_train_epochs=2
  training.eval_steps=200
  training.custom_eval_steps=200
  training.save_steps=200
  training.logging_steps=10
"
```

注意:
- `data.eval_datasets=[libero]` 而非 `[libero_pi0]`（后者含嵌套列表会报错）
- A-C: `train_preference_head=false`，D: `train_preference_head=true`

---

### Exp A: Pure BT（无正则 baseline）

论文对应: Table 1 Row A — 展示 monotonicity trap

```bash
accelerate launch --config_file robometer/configs/distributed/fsdp.yaml --num_processes=1 \
  train.py $BASE_ARGS \
  model.train_preference_head=false \
  "data.sample_type_ratio=[1,0,0]" \
  training.predict_pref_progress=false \
  loss.pref_loss_type=bt_sum \
  loss.struct_loss_enabled=false \
  training.output_dir=./logs/exp_a_pure_bt \
  training.exp_name=exp_a_pure_bt \
  "logging.log_to=[tensorboard]"
```

### Exp B: BT + L2 Smooth（naive 正则 baseline）

论文对应: Table 1 Row B — L2 temporal smoothing 不够

```bash
accelerate launch --config_file robometer/configs/distributed/fsdp.yaml --num_processes=1 \
  train.py $BASE_ARGS \
  model.train_preference_head=false \
  "data.sample_type_ratio=[1,0,0]" \
  training.predict_pref_progress=false \
  loss.pref_loss_type=bt_sum \
  loss.struct_loss_enabled=true \
  loss.struct_loss_type=l2_smooth \
  loss.struct_lambda=0.1 \
  training.output_dir=./logs/exp_b_l2_smooth \
  training.exp_name=exp_b_l2_smooth \
  "logging.log_to=[tensorboard]"
```

### Exp C: BT + Entropy Prior（本文方法）

论文对应: Table 1 Row C — Maximum Entropy Increment Prior，**本文核心贡献**

```bash
accelerate launch --config_file robometer/configs/distributed/fsdp.yaml --num_processes=1 \
  train.py $BASE_ARGS \
  model.train_preference_head=false \
  "data.sample_type_ratio=[1,0,0]" \
  training.predict_pref_progress=false \
  loss.pref_loss_type=bt_sum \
  loss.struct_loss_enabled=true \
  loss.struct_loss_type=entropy \
  loss.struct_lambda=0.1 \
  training.output_dir=./logs/exp_c_entropy \
  training.exp_name=exp_c_entropy \
  "logging.log_to=[tensorboard]"
```

### Exp D: Full Robometer（有监督 oracle 上界）

论文对应: Table 1 Row D — 使用 preference head + 有监督 progress labels

```bash
accelerate launch --config_file robometer/configs/distributed/fsdp.yaml --num_processes=1 \
  train.py $BASE_ARGS \
  model.train_preference_head=true \
  "data.sample_type_ratio=[1,0,0]" \
  training.predict_pref_progress=true \
  loss.pref_loss_type=head \
  loss.struct_loss_enabled=false \
  training.output_dir=./logs/exp_d_robometer \
  training.exp_name=exp_d_robometer \
  "logging.log_to=[tensorboard]"
```

### Exp E: Lambda 敏感性

论文对应: Section 5.3 超参数分析

```bash
for LAMBDA in 0.01 0.1 1.0; do
  accelerate launch --config_file robometer/configs/distributed/fsdp.yaml --num_processes=1 \
    train.py $BASE_ARGS \
    model.train_preference_head=false \
    "data.sample_type_ratio=[1,0,0]" \
    training.predict_pref_progress=false \
    loss.pref_loss_type=bt_sum \
    loss.struct_loss_enabled=true \
    loss.struct_loss_type=entropy \
    loss.struct_lambda=$LAMBDA \
    training.output_dir=./logs/exp_e_lambda_${LAMBDA} \
    training.exp_name=exp_e_lambda_${LAMBDA} \
    "logging.log_to=[tensorboard]"
done
```

预期: lambda=0.1 附近效果最好，0.01 退化为 Pure BT，1.0 过度约束。

---

## Part 3: 查看结果

### TensorBoard

```bash
tensorboard --logdir ./logs --bind_all --port 6006
```

AutoDL 端口映射或 SSH 隧道: `ssh -L 6006:localhost:6006 root@<地址>`，浏览器打开 `http://localhost:6006`。

关键 metrics:

| TensorBoard Tag | 含义 | 越大越好 |
|-----------------|------|---------|
| `train/preference_loss` | Bradley-Terry 偏好损失 | 否 (loss) |
| `train/struct_loss` | 结构正则损失 (仅 Exp B/C) | 否 (loss) |
| `eval_p_rank/voc_r_*` | VOC r (Pearson correlation) | 是 |
| `eval_p_rank/kendall_last_*` | Kendall τ | 是 |
| `eval_p_rank/suc_fail_diff_*` | 成功-失败差异 | 是 |

### 本地日志文件

```
./logs/exp_c_entropy/
├── trainer_state.json        # loss 历史、训练进度
├── checkpoint-200/           # 模型权重 (每 save_steps 保存)
├── checkpoint-400/
├── eval_results_*.json       # 评估指标 (每 eval_steps 保存)
└── runs/                     # TensorBoard 事件文件
```

### 独立评估（对已有 checkpoint 重新评估）

```bash
python robometer/evals/run_baseline_eval.py \
  reward_model=rbm \
  model_path=./logs/exp_c_entropy/checkpoint-400 \
  custom_eval.eval_types=[reward_alignment,policy_ranking] \
  custom_eval.reward_alignment=[libero] \
  custom_eval.policy_ranking=[libero] \
  custom_eval.use_frame_steps=true \
  custom_eval.subsample_n_frames=5 \
  custom_eval.reward_alignment_max_trajectories=30 \
  max_frames=8 \
  model_config.batch_size=32
```

---

## Part 4: 论文占位符映射

| 论文占位符 | 数据来源 | 实验 |
|-----------|---------|------|
| **Table 1**: VOC r / Kendall τ / Suc-Fail Diff | TensorBoard 或 eval_results JSON | Exp A-D |
| **Figure 1**: 势函数曲线 Φ(s_t) vs t | 推理脚本: 加载 checkpoint → 逐帧 predict progress | Exp A vs Exp C |
| **Figure 2**: 增量分布直方图 | 同上，计算 Δ_t = Φ(s_{t+1}) - Φ(s_t) 的分布 | Exp A vs Exp C |
| **Table/Figure 3**: λ 敏感性 | 各 lambda 实验的 VOC r / Kendall τ 汇总 | Exp E |

推理脚本参考: `scripts/example_inference_local.py`。
模型加载: `robometer/utils/save.py` 的 `load_model_from_hf()`。

---

## Part 5: 预期结果与时间估算

### 预期结果

| Model | VOC r | Kendall τ | Suc-Fail Diff |
|-------|-------|-----------|---------------|
| A. Pure BT (bt_sum, no reg) | ~0.5-0.7 | ~0.3-0.5 | ~0.05-0.10 |
| B. BT + L2 Smooth (bt_sum) | ~0.6-0.75 | ~0.4-0.55 | ~0.08-0.12 |
| C. BT + Entropy (bt_sum, ours) | ~0.80-0.90 | ~0.6-0.8 | ~0.15-0.30 |
| D. Full Robometer (head + supervised) | ~0.85-0.95 | ~0.7-0.85 | ~0.20-0.35 |

参考值来自 Robometer 原文 Table 3 的 LIBERO-90 ablation。C 应接近 D，远好于 A/B。

### 时间估算

| 实验 | 模型数 | 预估时间 (RTX 4080S) |
|------|--------|---------------------|
| Exp A-D: Core Ablation | 4 | ~8-12h 总计 |
| Exp E: Lambda Sweep | 3 | ~6-9h |
| Eval (独立) | - | ~15min/checkpoint |

建议优先级: **Exp A-D** → Exp E → 可视化脚本

---

## Checklist

- [x] 环境安装成功
- [x] LIBERO 数据集下载并预处理完成 (10/10)
- [x] 模型缓存就绪 (Qwen3-VL-4B)
- [x] 代码修改完成 (configs + trainer + utils + Route B bt_sum)
- [x] Dry run 成功 (bt_sum 模式: preference_loss + struct_loss 均正常)
- [ ] Exp A-D 训练完成
- [ ] Exp E lambda sweep 完成
- [ ] Table 1 填写
- [ ] Figure 1/2 可视化脚本编写并生成
- [ ] Figure/Table 3 lambda 敏感性汇总

# Experiment Design v2: Structure-Regularized Potential Inference

> 基于 Robometer 代码仓库验证 Maximum Entropy Increment Prior。
> 环境/数据/troubleshooting 见 [setup.md](setup.md)。

---

## Part 1: Code Modifications

### 1.1 LossConfig 新增字段

**`robometer/configs/experiment_configs.py`** (line 440-451)

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
```

**`robometer/configs/config.yaml`** (line 220-222)

```yaml
struct_loss_enabled: false
struct_loss_type: "entropy"
struct_lambda: 0.1
```

### 1.2 `_compute_struct_loss` 方法

**`robometer/trainers/rbm_heads_trainer.py`** (line 2438-2471)

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

### 1.3 集成到 `_compute_preference_loss`

同文件，在 success_loss 之后 (line 2540):

```python
if self.config.loss.struct_loss_enabled and self.config.model.train_progress_head:
    progress_pred_A = progress_logits["A"]
    struct_loss, struct_log = self._compute_struct_loss(
        progress_pred_A, target_progress_A_mask, training=training
    )
    if not torch.isnan(struct_loss).any():
        final_loss += self.config.loss.struct_lambda * struct_loss
```

设计理由: struct_loss 放在 preference loss 流程而非 progress loss 流程，因为 `progress_logits["A"]` 在 preference forward pass 中始终可用，不需要 progress target labels。

### 1.4 其他修改

| 文件 | 改动 | 原因 |
|------|------|------|
| `robometer/utils/setup_utils.py:1093` | `save_safetensors: True` → `False` | Unsloth+LoRA shared tensor 保存报错 |
| `robometer/trainers/rbm_heads_trainer.py:687-689` | 新增 loss/acc/corr 的控制台打印 | 原始代码只打印 counts 和 timing |
| `robometer/data/scripts/preprocess_datasets.py:889-896` | 优先从本地 parquet 加载 | 避免离线环境下 HF 网络超时 |

---

## Part 2: Experiments

所有实验使用 LIBERO 数据集，Qwen3-VL-4B + LoRA，单 GPU。

### 通用基础参数

```bash
# 每次开终端先执行（或写进 .bashrc）
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
  model.train_preference_head=true
  model.train_success_head=false
  data.train_datasets=[libero_pi0]
  data.eval_datasets=[libero]
  data.max_frames=8
  data.sample_type_ratio=[1,0,0]
  training.per_device_train_batch_size=2
  training.gradient_accumulation_steps=4
  training.learning_rate=2e-5
  training.num_train_epochs=2
  training.eval_steps=200
  training.custom_eval_steps=200
  training.save_steps=200
  training.predict_pref_progress=false
  training.logging_steps=10
  logging.log_to=[tensorboard]
"
```

注意:
- `data.eval_datasets=[libero]` 而非 `[libero_pi0]`（后者含嵌套列表会报错）
- `logging.log_to=[tensorboard]` 日志写到 output_dir 下

---

### Exp 1: BT-only baseline（无进度头，无正则）

论文对应: Table 1 第一行（Pure Bradley-Terry），Figure 1 中的病态曲线

```bash
accelerate launch --config_file robometer/configs/distributed/fsdp.yaml --num_processes=1 \
  train.py $BASE_ARGS \
  model.train_progress_head=false \
  loss.struct_loss_enabled=false \
  training.output_dir=./logs/exp1_bt_only \
  training.exp_name=exp1_bt_only
```

### Exp 2: BT + Progress（无正则 baseline）

论文对应: Table 1 第二行（Robometer 的 preference + progress 配置，无结构正则）

```bash
accelerate launch --config_file robometer/configs/distributed/fsdp.yaml --num_processes=1 \
  train.py $BASE_ARGS \
  loss.struct_loss_enabled=false \
  training.output_dir=./logs/exp2_bt_progress \
  training.exp_name=exp2_bt_progress
```

### Exp 3: BT + Progress + L2 Smooth（baseline 正则）

论文对应: Table 1 第三行（L2 temporal smoothness 作为对照正则方法）

```bash
accelerate launch --config_file robometer/configs/distributed/fsdp.yaml --num_processes=1 \
  train.py $BASE_ARGS \
  loss.struct_loss_enabled=true \
  loss.struct_loss_type=l2_smooth \
  loss.struct_lambda=0.1 \
  training.output_dir=./logs/exp3_l2_smooth \
  training.exp_name=exp3_l2_smooth
```

### Exp 4: BT + Progress + Entropy Prior（本文方法）

论文对应: Table 1 第四行（Maximum Entropy Increment Prior，本文贡献）

```bash
accelerate launch --config_file robometer/configs/distributed/fsdp.yaml --num_processes=1 \
  train.py $BASE_ARGS \
  loss.struct_loss_enabled=true \
  loss.struct_loss_type=entropy \
  loss.struct_lambda=0.1 \
  training.output_dir=./logs/exp4_entropy \
  training.exp_name=exp4_entropy
```

### Exp 5: Lambda 敏感性

论文对应: Section 5.3 超参数分析

```bash
for LAMBDA in 0.01 0.1 1.0; do
  accelerate launch --config_file robometer/configs/distributed/fsdp.yaml --num_processes=1 \
    train.py $BASE_ARGS \
    loss.struct_loss_enabled=true \
    loss.struct_loss_type=entropy \
    loss.struct_lambda=$LAMBDA \
    training.output_dir=./logs/exp5_lambda_${LAMBDA} \
    training.exp_name=exp5_lambda_${LAMBDA}
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
| `train/pref_bt_loss` | Bradley-Terry 偏好损失 | 否 (loss) |
| `train/struct_loss` | 结构正则损失 (仅 Exp 3/4) | 否 (loss) |
| `eval_p_rank/voc_r_*` | VOC r (Pearson correlation) | 是 |
| `eval_p_rank/kendall_last_*` | Kendall τ | 是 |
| `eval_p_rank/suc_fail_diff_*` | 成功-失败差异 | 是 |

### 本地日志文件

```
./logs/exp4_entropy/
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
  model_path=./logs/exp4_entropy/checkpoint-400 \
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
| **Table 1**: VOC r / Kendall τ / Suc-Fail Diff | TensorBoard 或 eval_results JSON | Exp 1-4 |
| **Figure 1**: 势函数曲线 Φ(s_t) vs t | 推理脚本: 加载 checkpoint → 逐帧 predict progress | Exp 1 vs Exp 4 |
| **Figure 2**: 增量分布直方图 | 同上，计算 Δ_t = Φ(s_{t+1}) - Φ(s_t) 的分布 | Exp 1 vs Exp 4 |
| **Table/Figure 3**: λ 敏感性 | 各 lambda 实验的 VOC r / Kendall τ 汇总 | Exp 5 |

推理脚本参考: `scripts/example_inference_local.py`。
模型加载: `robometer/utils/save.py` 的 `load_model_from_hf()`。

---

## Part 5: 预期结果与时间估算

### 预期结果

| Model | VOC r | Kendall τ | Suc-Fail Diff |
|-------|-------|-----------|---------------|
| A. Pure BT (Exp 1) | ~0.5-0.7 | ~0.3-0.5 | ~0.05-0.10 |
| B. BT + Progress (Exp 2) | ~0.7-0.8 | ~0.5-0.6 | ~0.10-0.15 |
| C. BT + L2 Smooth (Exp 3) | ~0.7-0.8 | ~0.5-0.6 | ~0.10-0.15 |
| D. BT + Entropy (Exp 4) | ~0.85-0.95 | ~0.7-0.85 | ~0.20-0.35 |

参考值来自 Robometer 原文 Table 3 的 LIBERO-90 ablation。D 应远好于 A/B/C。

### 时间估算

| 实验 | 模型数 | 预估时间 (RTX 4080S) |
|------|--------|---------------------|
| Exp 1-4: Core Ablation | 4 | ~8-12h 总计 |
| Exp 5: Lambda Sweep | 3 | ~6-9h |
| Eval (独立) | - | ~15min/checkpoint |

建议优先级: **Exp 1-4** → Exp 5 → 可视化脚本

---

## Checklist

- [x] 环境安装成功
- [x] LIBERO 数据集下载并预处理完成 (10/10)
- [x] 模型缓存就绪 (Qwen3-VL-4B)
- [x] 代码修改完成 (configs + trainer + utils)
- [x] Dry run 成功 (struct_loss 出现在日志中)
- [ ] Exp 1-4 训练完成
- [ ] Exp 5 lambda sweep 完成
- [ ] Table 1 填写
- [ ] Figure 1/2 可视化脚本编写并生成
- [ ] Figure/Table 3 lambda 敏感性汇总

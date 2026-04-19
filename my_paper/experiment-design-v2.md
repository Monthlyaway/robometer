# Experiment Design v2: Code Modifications

本文档记录为实现 Maximum Entropy Increment Prior 所做的全部代码修改。

---

## 修改总览

| 文件 | 修改类型 | 目的 |
|------|---------|------|
| `robometer/configs/experiment_configs.py` | 新增 3 个字段 | 配置 struct loss 参数 |
| `robometer/configs/config.yaml` | 新增 3 行默认值 | YAML 端配置对齐 |
| `robometer/trainers/rbm_heads_trainer.py` | 新增方法 + 集成 + 日志 | 核心 loss 实现 |
| `robometer/utils/setup_utils.py` | 改 1 行 | 修复 Unsloth+LoRA 保存问题 |
| `robometer/data/scripts/preprocess_datasets.py` | 改 5 行 | 本地 parquet 加载 hack |

---

## 1. LossConfig 新增字段

**文件**: `robometer/configs/experiment_configs.py`，`LossConfig` dataclass 末尾 (line 440-451)

```python
struct_loss_enabled: bool = field(
    default=False,
    metadata={"help": "Enable structural entropy regularization loss (L_struct)"},
)
struct_loss_type: str = field(
    default="entropy",
    metadata={"help": "Type of structural loss: 'entropy' (max entropy prior) or 'l2_smooth' (L2 temporal smoothness)"},
)
struct_lambda: float = field(
    default=0.1,
    metadata={"help": "Weight for the structural loss term"},
)
```

**文件**: `robometer/configs/config.yaml`，`loss:` section (line 220-222)

```yaml
struct_loss_enabled: false
struct_loss_type: "entropy"  # Options: "entropy" or "l2_smooth"
struct_lambda: 0.1
```

---

## 2. `_compute_struct_loss` 方法

**文件**: `robometer/trainers/rbm_heads_trainer.py`，line 2438-2471

新增方法，位于 `_compute_preference_loss` 之前。

```python
def _compute_struct_loss(
    self,
    progress_pred: torch.Tensor,
    mask: torch.Tensor,
    training: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    loss_type = self.config.loss.struct_loss_type
    prefix = "train" if training else "eval"

    # 离散模式: [B, T, num_bins] -> [B, T] 连续值
    if self.config.loss.progress_loss_type == "discrete":
        progress = convert_bins_to_continuous(progress_pred)
    else:
        progress = progress_pred.float()

    if progress.dim() == 1:
        progress = progress.unsqueeze(0)

    # 相邻帧增量 [B, T-1]
    delta_t = progress[:, 1:] - progress[:, :-1]

    if loss_type == "l2_smooth":
        # L2 Temporal Smoothness baseline: min sum(delta^2)
        struct_loss = (delta_t ** 2).mean()
    else:
        # Maximum Entropy Increment Prior: min sum(p * log(p))
        # softplus 保证 delta_pos > 0，然后归一化为概率分布
        delta_pos = F.softplus(delta_t)
        p_t = delta_pos / (delta_pos.sum(dim=-1, keepdim=True) + 1e-8)
        struct_loss = (p_t * torch.log(p_t + 1e-8)).sum(dim=-1).mean()

    log_dict = {f"{prefix}/struct_loss": struct_loss.item()}
    return struct_loss, log_dict
```

**数学对应**:
- Entropy 模式: \(\mathcal{L}_{\text{struct}} = \sum_t p_t \log p_t\)，其中 \(p_t = \text{softplus}(\Delta_t) / \sum \text{softplus}(\Delta_t)\)。最小化此值 = 最大化增量分布的熵 = 鼓励均匀增量。
- L2 Smooth 模式: \(\mathcal{L}_{\text{struct}} = \frac{1}{T-1}\sum_t \Delta_t^2\)。最小化 = 惩罚大增量。

---

## 3. 集成到 `_compute_preference_loss`

**文件**: `robometer/trainers/rbm_heads_trainer.py`

在 success_loss 之后、`return` 之前 (line 2540-2548):

```python
if self.config.loss.struct_loss_enabled and self.config.model.train_progress_head:
    progress_pred_A = progress_logits["A"]
    struct_loss, struct_log = self._compute_struct_loss(
        progress_pred_A, target_progress_A_mask, training=training
    )
    if not torch.isnan(struct_loss).any():
        final_loss += self.config.loss.struct_lambda * struct_loss
    else:
        logger.warning("NaN detected in struct loss")
```

在 `if return_outputs:` block 内 (line 2556-2557):

```python
if self.config.loss.struct_loss_enabled and self.config.model.train_progress_head:
    outputs_dict.update(struct_log)
```

**设计说明**: struct_loss 放在 preference loss 流程而非 progress loss 流程，因为:
- `progress_logits["A"]` 在 preference forward pass 中始终可用（无论 `predict_pref_progress` 设置）
- struct_loss 不需要 progress target labels，只需模型自身的 progress predictions
- 这样当 `sample_type_ratio=[1,0,0]`（全 preference 样本）时也能生效

---

## 4. 控制台日志增强

**文件**: `robometer/trainers/rbm_heads_trainer.py`，`_log_metadata` 方法 (line 687-689)

原始代码只在控制台打印 counts 和 timing，loss 指标只发 wandb。新增:

```python
for key in sorted(log_metadata):
    if "loss" in key or "acc" in key or "corr" in key:
        logger.info(f"  {key}: {log_metadata[key]:.6f}")
```

效果: 每步在控制台可见 `train/struct_loss: -1.945312` 等 loss 指标。

---

## 5. 模型保存修复

**文件**: `robometer/utils/setup_utils.py`，`create_training_arguments` 函数 (line 1093)

```python
# 原始: "save_safetensors": True,
# 修改: "save_safetensors": False,
```

原因: Unsloth 对 Qwen3-VL 做了 weight tying，导致 `save_pretrained` 检测到 shared tensors 报错 `RuntimeError: The weights trying to be saved contained shared tensors`。改为 pickle 格式避免。

---

## 6. 数据预处理本地加载 hack

**文件**: `robometer/data/scripts/preprocess_datasets.py`，`_load_dataset_from_path` 方法 (line 889-896)

```python
# 优先从本地 parquet 文件加载，避免网络请求
local_parquet_dir = os.path.join(dataset_root, dataset_name, subset)
local_parquet = os.path.join(local_parquet_dir, "train-00000-of-00001.parquet")
if os.path.exists(local_parquet):
    rank_0_print(f"Loading from local parquet: {local_parquet_dir}")
    dataset = load_dataset("parquet", data_dir=local_parquet_dir, split="train")
else:
    dataset = load_dataset(dataset_path, name=subset, split="train")
```

原因: `huggingface datasets.load_dataset()` 即使有本地缓存也会尝试 metadata 网络请求，在离线环境下超时。此 hack 让预处理脚本直接从 `$ROBOMETER_DATASET_PATH` 下的 parquet 文件加载。

---

## 配置参数速查

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `loss.struct_loss_enabled` | bool | `false` | 开启结构正则 |
| `loss.struct_loss_type` | str | `"entropy"` | `"entropy"` 或 `"l2_smooth"` |
| `loss.struct_lambda` | float | `0.1` | 正则项权重 λ |

命令行覆盖示例:

```bash
loss.struct_loss_enabled=true loss.struct_loss_type=entropy loss.struct_lambda=0.1
```

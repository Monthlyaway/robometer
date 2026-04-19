# Experiment Design v1: Structure-Regularized Potential Inference

> 毕业论文实验手册 -- 基于 Robometer 代码仓库验证 Maximum Entropy Increment Prior

---

## 数据与模型总览

```
/root/autodl-tmp/
├── raw_datasets/                              # ROBOMETER_DATASET_PATH
│   ├── libero_rfm/                            # abraranwar/libero_rfm (HF 下载)
│   │   ├── libero256_10/                      # 训练: 388 条成功轨迹
│   │   ├── libero256_object/                  # 训练: 456 条成功轨迹
│   │   ├── libero256_spatial/                 # 训练: 433 条成功轨迹
│   │   ├── libero256_goal/                    # 训练: 456 条成功轨迹
│   │   └── libero256_90/                      # 评估: ~8000 条成功轨迹
│   └── libero_failure_rfm/                    # ykorkmaz/libero_failure_rfm (HF 下载)
│       ├── libero_10_failure/                 # 训练: 498 条失败轨迹
│       ├── libero_object_failure/             # 训练: 490 条失败轨迹
│       ├── libero_spatial_failure/            # 训练: 486 条失败轨迹
│       ├── libero_goal_failure/               # 训练: 456 条失败轨迹
│       └── libero_90_failure/                 # 评估: 失败轨迹
├── processed_datasets/                        # ROBOMETER_PROCESSED_DATASETS_PATH
│   ├── abraranwar_libero_rfm_libero256_10/    # 预处理后 (frames/ + index)
│   ├── abraranwar_libero_rfm_libero256_object/
│   ├── abraranwar_libero_rfm_libero256_spatial/
│   ├── abraranwar_libero_rfm_libero256_goal/
│   ├── abraranwar_libero_rfm_libero256_90/
│   ├── ykorkmaz_libero_failure_rfm_libero_10_failure/
│   ├── ykorkmaz_libero_failure_rfm_libero_object_failure/
│   ├── ykorkmaz_libero_failure_rfm_libero_spatial_failure/
│   ├── ykorkmaz_libero_failure_rfm_libero_goal_failure/
│   └── ykorkmaz_libero_failure_rfm_libero_90_failure/
└── .cache/huggingface/hub/                    # HF_HOME (模型缓存)
    ├── models--Qwen--Qwen3-VL-4B-Instruct/
    └── models--robometer--Robometer-4B/
```

---

## Part 0: Environment and Infrastructure

### 0.1 Python + uv 安装

项目要求 **Python 3.10**（硬性，`pyproject.toml` 中 `requires-python = "==3.10.*"`）。

```bash
# 安装 uv 包管理器
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

### 0.2 配置国内镜像

```bash
# --- PyPI 镜像（清华源） ---
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
echo 'export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple' >> ~/.bashrc

# --- HuggingFace 镜像 ---
export HF_ENDPOINT=https://hf-mirror.com
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc

source ~/.bashrc
```

> 注意: PyTorch CUDA wheel 从 `https://download.pytorch.org/whl/cu128` 下载,
> `pyproject.toml` 第 163-165 行已配置此 index, uv 会自动处理, 无需额外设置。

### 0.3 安装项目依赖

```bash
cd /root/autodl-tmp/robometer
uv sync --extra robometer
```

这会安装 `torch==2.8.0` (CUDA 12.8)、`transformers>=4.57`、`trl==0.20.0` 等全部依赖。

### 0.4 修复 torchcodec 版本兼容性

`uv sync` 会安装最新的 `torchcodec`（如 0.11.1），但它要求 `torch>=2.11`，
与本项目的 `torch==2.8.0` 不兼容。需要手动降级到对应版本。

版本对应关系（来源: [torchcodec README](https://github.com/pytorch/torchcodec)）:

| torchcodec | torch |
|------------|-------|
| 0.7 / 0.6  | 2.8   |
| 0.5 / 0.4  | 2.7   |
| 0.2        | 2.6   |

```bash
# 激活 venv 后降级 torchcodec
source .venv/bin/activate
pip install torchcodec==0.7

# 验证
python -c "import torchcodec; print('torchcodec OK:', torchcodec.__version__)"
```

> **重要**: AutoDL 服务器重启后 pip 安装的包不会丢失（.venv 在数据盘），
> 但如果重新执行 `uv sync` 会覆盖回不兼容的版本，需要再次降级。

### 0.5 验证安装

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
uv run python -c "import transformers; print(transformers.__version__)"
uv run python -c "from robometer.models.rbm import RBM; print('OK')"
```

### 0.6 WandB 配置

训练默认用 WandB 记日志。两种选择:

```bash
# 方案 A: 使用 WandB（推荐）
uv run wandb login   # 输入你的 API key

# 方案 B: 禁用 WandB
# 在训练命令中加: logging.log_to=[]
```

### 0.7 AutoDL 特别说明

- 确认 CUDA driver >= 12.8: `nvidia-smi`
- 确认磁盘空间 >= 50GB（数据 + 模型 + checkpoints）
- AutoDL 学术加速: `source /etc/network_turbo`（如果平台支持）

### 0.8 服务器重启后恢复清单

AutoDL 服务器重启后, `.venv/` 中的包不会丢失, 但环境变量和某些状态会重置。
每次重启后需要:

```bash
# 1. 恢复环境变量（如果没写入 ~/.bashrc）
source ~/.bashrc

# 2. 验证 torchcodec 版本（uv sync 可能覆盖）
source .venv/bin/activate
python -c "import torchcodec; print(torchcodec.__version__)"
# 应输出 0.7.0; 如果不是, 重新执行: pip install torchcodec==0.7

# 3. 验证两个数据路径都已设置且不同
echo "RAW:       $ROBOMETER_DATASET_PATH"        # 应为 /root/autodl-tmp/raw_datasets
echo "PROCESSED: $ROBOMETER_PROCESSED_DATASETS_PATH"  # 应为 /root/autodl-tmp/processed_datasets
```

---

## Part 1: Data and Model Download

### 1.1 环境变量

项目需要 **两个** 数据路径环境变量, 必须指向 **不同目录**:

| 变量 | 用途 | 路径 |
|------|------|------|
| `ROBOMETER_DATASET_PATH` | 原始 HF 下载（视频 + parquet） | `/root/autodl-tmp/raw_datasets` |
| `ROBOMETER_PROCESSED_DATASETS_PATH` | 预处理后的索引缓存 | `/root/autodl-tmp/processed_datasets` |

```bash
export ROBOMETER_DATASET_PATH=/root/autodl-tmp/raw_datasets
export ROBOMETER_PROCESSED_DATASETS_PATH=/root/autodl-tmp/processed_datasets
echo 'export ROBOMETER_DATASET_PATH=/root/autodl-tmp/raw_datasets' >> ~/.bashrc
echo 'export ROBOMETER_PROCESSED_DATASETS_PATH=/root/autodl-tmp/processed_datasets' >> ~/.bashrc
mkdir -p $ROBOMETER_DATASET_PATH $ROBOMETER_PROCESSED_DATASETS_PATH
```

> **关键**: 这两个目录必须分开。
> - `ROBOMETER_DATASET_PATH` 下存放原始数据: `libero_rfm/`、`libero_failure_rfm/`
> - `ROBOMETER_PROCESSED_DATASETS_PATH` 下存放预处理结果: `abraranwar_libero_rfm_libero256_10/` 等
> - 预处理脚本通过 `ROBOMETER_DATASET_PATH` 找到视频文件, 输出到 `cache_dir`
>   (配置文件中设为 `ROBOMETER_PROCESSED_DATASETS_PATH`)
> - 训练代码只读 `ROBOMETER_PROCESSED_DATASETS_PATH`

### 1.2 HuggingFace 登录

```bash
uv run huggingface-cli login
# 输入你的 HF token (https://huggingface.co/settings/tokens)
# 也可以直接: uv run huggingface-cli login --token hf_xxxxx
```

### 1.3 下载 LIBERO 数据集（仅需要这些）

根据 `robometer/data/dataset_category.py` 第 260-281 行的 `libero_pi0` 定义,
训练需要 LIBERO-10/Object/Spatial/Goal 的成功数据和失败数据,
评估需要 LIBERO-90 的成功数据和失败数据。

```bash
# 成功轨迹（包含 libero256_10, _object, _spatial, _goal, _90 子集）
uv run huggingface-cli download abraranwar/libero_rfm \
  --repo-type dataset \
  --local-dir $ROBOMETER_DATASET_PATH/libero_rfm

# 失败轨迹
uv run huggingface-cli download ykorkmaz/libero_failure_rfm \
  --repo-type dataset \
  --local-dir $ROBOMETER_DATASET_PATH/libero_failure_rfm
```

> 下载后检查: `ls $ROBOMETER_DATASET_PATH/` 应看到 `libero_rfm/` 和
> `libero_failure_rfm/` 两个文件夹，每个下面有子集目录（如 `libero256_10/`）。
> 注意是 `ROBOMETER_DATASET_PATH`（raw_datasets），不是 PROCESSED。

### 1.4 预处理数据集（必需步骤）

下载的原始数据是 HuggingFace 格式（parquet + 视频文件），训练代码需要
预处理后的索引格式。预处理脚本会:
- 从视频中提取帧（最多 64 帧）
- 构建轨迹索引（成功/失败/任务映射）
- 将结果缓存到 `$ROBOMETER_PROCESSED_DATASETS_PATH/` 下的扁平目录

预处理配置文件已准备好（仅包含 LIBERO 数据集）:

```bash
cd /root/autodl-tmp/robometer
source .venv/bin/activate

# ROBOMETER_DATASET_PATH 指向原始数据所在目录
export ROBOMETER_DATASET_PATH=/root/autodl-tmp/raw_datasets

# 运行预处理（约 20-30 分钟，10 个数据子集）
# cache_dir 在 yaml 中已设为 /root/autodl-tmp/processed_datasets
python -m robometer.data.scripts.preprocess_datasets \
  --config robometer/configs/preprocess_libero.yaml
```

> 预处理配置文件: `robometer/configs/preprocess_libero.yaml`
> 其中 `cache_dir: "/root/autodl-tmp/processed_datasets"` 控制输出位置。
>
> 预处理完成后的目录结构:
> ```
> /root/autodl-tmp/
> ├── raw_datasets/                          # ROBOMETER_DATASET_PATH
> │   ├── libero_rfm/                        # HF 下载的原始数据
> │   │   ├── libero256_10/
> │   │   │   ├── batch_0000/
> │   │   │   │   └── trajectory_0000.mp4 ...
> │   │   │   └── train-00000-of-00001.parquet
> │   │   ├── libero256_object/ ...
> │   │   └── libero256_90/ ...
> │   └── libero_failure_rfm/
> │       ├── libero_10_failure/ ...
> │       └── libero_90_failure/ ...
> └── processed_datasets/                    # ROBOMETER_PROCESSED_DATASETS_PATH
>     ├── abraranwar_libero_rfm_libero256_10/
>     │   ├── processed_dataset/             # HF Dataset on disk
>     │   ├── index_mappings.json
>     │   ├── dataset_info.json
>     │   └── frames/                        # 提取的 .npz 帧
>     ├── abraranwar_libero_rfm_libero256_object/ ...
>     ├── ykorkmaz_libero_failure_rfm_libero_10_failure/ ...
>     └── ... (共 10 个子集)
> ```

#### 验证预处理完成

```bash
# 应输出 10 个以 abraranwar_ 或 ykorkmaz_ 开头的目录
ls $ROBOMETER_PROCESSED_DATASETS_PATH/ | grep -E "^(abraranwar|ykorkmaz)" | wc -l
# 应输出: 10

# 检查每个目录结构
ls $ROBOMETER_PROCESSED_DATASETS_PATH/abraranwar_libero_rfm_libero256_10/
# 应输出: dataset_info.json  embeddings_cache  frames  index_mappings.json  processed_dataset
```

### 1.5 下载模型

```bash
# 基础 VLM（训练起点）
uv run huggingface-cli download Qwen/Qwen3-VL-4B-Instruct

# 预训练 Robometer（用于对比实验 + LoRA 微调基础）
uv run huggingface-cli download robometer/Robometer-4B
```

> HF 模型会缓存到 `/root/autodl-tmp/.cache/huggingface/hub/`。
> 如果磁盘空间不足, 可以用 `--local-dir` 指定路径, 然后在训练命令中用本地路径。

### 1.6 验证数据加载

```bash
uv run python -c "
from robometer.data.datasets.base import resolve_dataset_keys
keys = resolve_dataset_keys(['libero_pi0'], 'train')
print(f'Training datasets ({len(keys)}):')
for k in keys:
    print(f'  {k}')
"
```

应输出 8 个数据集名称（4 个成功 + 4 个失败）。

---

## Part 2: Code Modifications

需要修改 **4 个文件** 来实现 L_struct (Maximum Entropy Increment Prior)。

### 2.1 文件: `robometer/configs/experiment_configs.py`

**位置**: `LossConfig` dataclass, 第 420-439 行

**操作**: 在 `progress_discrete_bins` 字段之后, 添加两个新字段:

```python
# --- 在 progress_discrete_bins 字段之后添加 ---
struct_loss_enabled: bool = field(
    default=False,
    metadata={"help": "Whether to enable the structural entropy regularization loss (L_struct)"},
)
struct_lambda: float = field(
    default=0.1,
    metadata={"help": "Weight for the structural entropy loss term"},
)
```

### 2.2 文件: `robometer/configs/config.yaml`

**位置**: `loss:` section, 第 214-219 行

**操作**: 在 `progress_discrete_bins` 之后添加:

```yaml
loss:
  predict_last_frame_progress: false
  progress_loss_type: "discrete"
  progress_discrete_bins: 10
  # --- 新增 ---
  struct_loss_enabled: false   # 默认关闭, 实验中通过命令行覆盖
  struct_lambda: 0.1
```

### 2.3 文件: `robometer/trainers/rbm_heads_trainer.py`

需要做两件事:

#### 2.3a 添加 `_compute_struct_loss` 方法

**位置**: 在 `_compute_preference_loss` 方法 (第 2434 行) 之前, 添加新方法。

**逻辑**（对应论文 Section 4.1 的公式）:

```
输入: progress_pred -- shape [batch_size, seq_len] 或 [batch_size, seq_len, num_bins]
       mask         -- shape [batch_size, 1] 表示哪些样本参与计算

步骤:
1. 如果是 discrete 模式, 先将 logits 转为连续值 (用 convert_bins_to_continuous)
2. 计算相邻帧差分: delta_t = progress_pred[:, t+1] - progress_pred[:, t]
   结果 shape: [batch_size, seq_len - 1]
3. 对 delta 应用 Softplus 保证非负: delta_pos = F.softplus(delta_t)
4. 归一化为概率分布: p_t = delta_pos / delta_pos.sum(dim=-1, keepdim=True)
5. 计算负熵: L_struct = (p_t * torch.log(p_t + 1e-8)).sum(dim=-1)
6. 对 mask 内的样本取均值
7. 返回 struct_loss 和日志 dict
```

> 关键: `progress_pred` 来自 `model_output.progress_logits["A"]`
> (参见 `robometer/models/utils.py` 第 6-12 行的 `ModelOutput` 定义)。
> 对于 discrete 模式, `progress_logits["A"]` 的 shape 是 `[B, T, num_bins]`,
> 需要先调用 `convert_bins_to_continuous()` 转为 `[B, T]`。
> 对于 L1/L2 模式, shape 直接是 `[B, T]`。

#### 2.3b 在 `compute_loss` 中调用 `_compute_struct_loss`

**位置**: `compute_loss` 方法, 第 1823 行之后 (progress loss 计算完成后)

**操作**: 在 progress loss 代码块后、`for key, value in log_metadata.items():` (第 1825 行) 之前, 插入:

```
# 伪代码位置参考 (在第 1823-1824 行之间):
if self.config.loss.struct_loss_enabled:
    # 需要 progress predictions -- 可以从 preference 或 progress forward pass 获取
    # 优先从 progress_inputs 获取（如果有 progress samples）
    # 否则从 preference_inputs 获取（preference forward pass 也会产生 progress_logits["A"]）
    struct_loss, struct_dict = self._compute_struct_loss(...)
    total_loss += self.config.loss.struct_lambda * struct_loss
    log_metadata.update(struct_dict)
```

> 注意: 当 `sample_type_ratio=[1,0,0]`（默认配置, 只有 preference samples）时,
> progress_logits 来自 `_compute_preference_loss` 中的 forward pass。
> 此时需要在 `_compute_preference_loss` 返回 progress_logits,
> 或者在 `compute_loss` 中额外做一次 forward pass。
> 最简单的方案: 在 `_compute_preference_loss` 中计算 struct loss 并返回。

### 2.4 替代方案: L2 Temporal Smoothness Baseline

对照实验 "BT + L2 smooth" 需要实现一个简单的 L2 平滑 loss:

```
L2_smooth = mean( (delta_t)^2 )
其中 delta_t = progress_pred[:, t+1] - progress_pred[:, t]
```

建议在同一个 `_compute_struct_loss` 方法中, 通过 config 字段切换:
- `struct_loss_type: "entropy"` -- 最大熵先验 (你的方法)
- `struct_loss_type: "l2_smooth"` -- L2 平滑 (对照组)

对应需要在 `LossConfig` 中额外加一个字段:

```python
struct_loss_type: str = field(
    default="entropy",
    metadata={"help": "Type of structural loss: 'entropy' (max entropy prior) or 'l2_smooth' (L2 temporal smoothness)"},
)
```

---

## Part 3: Experiments

所有实验使用 **LIBERO** 数据集, LoRA 微调, 单 GPU。

### 通用训练命令模板

```bash
uv run accelerate launch \
  --config_file robometer/configs/distributed/fsdp.yaml \
  --num_processes=1 \
  train.py \
  model.base_model_id=Qwen/Qwen3-VL-4B-Instruct \
  model.use_peft=true \
  data.train_datasets=[libero_pi0] \
  data.eval_datasets=[libero_pi0] \
  data.max_frames=8 \
  training.per_device_train_batch_size=8 \
  training.learning_rate=2e-5 \
  training.warmup_ratio=0.1 \
  training.weight_decay=0.01 \
  training.max_steps=5000 \
  training.eval_steps=250 \
  training.custom_eval_steps=250 \
  training.overwrite_output_dir=True \
  custom_eval.eval_types=[reward_alignment,policy_ranking] \
  custom_eval.reward_alignment=[libero_pi0] \
  custom_eval.policy_ranking=[libero_pi0] \
  logging.log_to=[] \
  # --- 下面的参数各实验不同 ---
  model.train_progress_head=??? \
  model.train_preference_head=??? \
  loss.struct_loss_enabled=??? \
  loss.struct_lambda=??? \
  training.output_dir=??? \
  training.exp_name=???
```

> 注: `num_processes=1` 对应单 GPU; 多 GPU 改为 GPU 数量。
> `logging.log_to=[]` 禁用 WandB; 要用 WandB 则改为 `logging.log_to=[wandb]`。

---

### Experiment 1: Monotonicity Trap 诊断

**目的**: 实证验证论文 Section 3 -- 纯排序模型学到的 potential function 存在病态曲线。

**对应论文**: Figure 1 (Monotonicity Trap 可视化)

#### 1a. 训练 Pure Bradley-Terry 模型

只训练 preference head, 不训练 progress head:

```bash
uv run accelerate launch \
  --config_file robometer/configs/distributed/fsdp.yaml \
  --num_processes=1 \
  train.py \
  model.base_model_id=Qwen/Qwen3-VL-4B-Instruct \
  model.use_peft=true \
  model.train_progress_head=true \
  model.train_preference_head=true \
  model.train_success_head=false \
  data.train_datasets=[libero_pi0] \
  data.eval_datasets=[libero_pi0] \
  data.max_frames=8 \
  data.sample_type_ratio=[1,0,0] \
  training.per_device_train_batch_size=8 \
  training.learning_rate=2e-5 \
  training.max_steps=5000 \
  training.predict_pref_progress=false \
  loss.struct_loss_enabled=false \
  training.output_dir=./logs/exp1_pure_bt \
  training.exp_name=exp1_pure_bt \
  logging.log_to=[] \
  custom_eval.eval_types=[reward_alignment,policy_ranking] \
  custom_eval.reward_alignment=[libero_pi0] \
  custom_eval.policy_ranking=[libero_pi0]
```

> 关键设置:
> - `model.train_progress_head=true` 但 `training.predict_pref_progress=false`:
>   progress head 存在但不参与 loss, 仅用于推理时输出帧级预测
> - 如果上面的组合不能产生 progress 输出, 则改为:
>   `model.train_progress_head=true, training.predict_pref_progress=true`
>   但将 progress loss 的权重设为 0 (需要在代码中添加此功能)
> - 实际上更简单的做法: 保持默认配置
>   `model.train_progress_head=true, training.predict_pref_progress=true`,
>   但在 `_compute_preference_loss` 中不将 progress_loss_A 加入 final_loss。
>   这需要在代码中添加一个 `loss.progress_loss_weight: float = 1.0` 配置项。

#### 1b. 训练 Full Robometer Baseline

完整的 Robometer 训练 (preference + progress + success):

```bash
uv run accelerate launch \
  --config_file robometer/configs/distributed/fsdp.yaml \
  --num_processes=1 \
  train.py \
  model.base_model_id=Qwen/Qwen3-VL-4B-Instruct \
  model.use_peft=true \
  model.train_progress_head=true \
  model.train_preference_head=true \
  model.train_success_head=true \
  data.train_datasets=[libero_pi0] \
  data.eval_datasets=[libero_pi0] \
  data.max_frames=8 \
  training.per_device_train_batch_size=8 \
  training.learning_rate=2e-5 \
  training.max_steps=5000 \
  loss.struct_loss_enabled=false \
  training.output_dir=./logs/exp1_robometer \
  training.exp_name=exp1_robometer \
  logging.log_to=[] \
  custom_eval.eval_types=[reward_alignment,policy_ranking] \
  custom_eval.reward_alignment=[libero_pi0] \
  custom_eval.policy_ranking=[libero_pi0]
```

#### 1c. 可视化分析

训练完成后, 写一个可视化脚本（放在 `my_paper/scripts/` 下）完成:

1. **Potential Curve Visualization**: 
   - 加载两个模型 checkpoint
   - 对 LIBERO-90 中若干成功轨迹做推理, 得到帧级 progress predictions
   - 画 Phi(s_t) vs t 曲线, 对比 Pure BT vs Robometer
   - Pure BT 应该展现阶梯状或指数形的病态曲线

2. **Increment Distribution**:
   - 计算 delta_t = Phi(s_{t+1}) - Phi(s_t)
   - 画 delta_t 的直方图
   - Pure BT 的 delta 分布应该极度集中（大部分接近 0, 偶尔有大 spike）
   - Robometer 的 delta 分布应该更均匀

> 推理代码参考: `scripts/example_inference_local.py` 展示了如何加载模型并对视频做推理。
> 模型加载: `robometer/utils/save.py` 中的 `load_model_from_hf()` 函数。

---

### Experiment 2: Core Ablation (主结果表)

**目的**: 验证论文 Section 4 -- L_struct 能否在不用帧级标签的情况下达到接近 Robometer 的效果。

**对应论文**: 主结果表 (VOC r, Kendall tau, Suc-Fail Diff)

训练 4 个模型变体:

#### 2a. Pure BT (和实验 1a 相同, 可复用)

#### 2b. BT + L2 Temporal Smoothness

```bash
uv run accelerate launch \
  --config_file robometer/configs/distributed/fsdp.yaml \
  --num_processes=1 \
  train.py \
  model.base_model_id=Qwen/Qwen3-VL-4B-Instruct \
  model.use_peft=true \
  model.train_progress_head=true \
  model.train_preference_head=true \
  model.train_success_head=false \
  data.train_datasets=[libero_pi0] \
  data.eval_datasets=[libero_pi0] \
  data.max_frames=8 \
  training.per_device_train_batch_size=8 \
  training.learning_rate=2e-5 \
  training.max_steps=5000 \
  training.predict_pref_progress=false \
  loss.struct_loss_enabled=true \
  loss.struct_loss_type=l2_smooth \
  loss.struct_lambda=0.1 \
  training.output_dir=./logs/exp2_bt_l2smooth \
  training.exp_name=exp2_bt_l2smooth \
  logging.log_to=[] \
  custom_eval.eval_types=[reward_alignment,policy_ranking] \
  custom_eval.reward_alignment=[libero_pi0] \
  custom_eval.policy_ranking=[libero_pi0]
```

#### 2c. BT + L_struct (你的方法)

```bash
uv run accelerate launch \
  --config_file robometer/configs/distributed/fsdp.yaml \
  --num_processes=1 \
  train.py \
  model.base_model_id=Qwen/Qwen3-VL-4B-Instruct \
  model.use_peft=true \
  model.train_progress_head=true \
  model.train_preference_head=true \
  model.train_success_head=false \
  data.train_datasets=[libero_pi0] \
  data.eval_datasets=[libero_pi0] \
  data.max_frames=8 \
  training.per_device_train_batch_size=8 \
  training.learning_rate=2e-5 \
  training.max_steps=5000 \
  training.predict_pref_progress=false \
  loss.struct_loss_enabled=true \
  loss.struct_loss_type=entropy \
  loss.struct_lambda=0.1 \
  training.output_dir=./logs/exp2_bt_struct \
  training.exp_name=exp2_bt_struct \
  logging.log_to=[] \
  custom_eval.eval_types=[reward_alignment,policy_ranking] \
  custom_eval.reward_alignment=[libero_pi0] \
  custom_eval.policy_ranking=[libero_pi0]
```

#### 2d. Full Robometer (和实验 1b 相同, 可复用)

#### 评估与结果收集

训练过程中, 评估结果会自动打印到日志。关键指标:

| 指标 | 日志 key | 说明 |
|------|----------|------|
| VOC r (Reward Alignment) | `eval_rew_align/pearson_*` | Pearson 相关性, 越高越好 |
| Kendall tau (Policy Ranking) | `eval_p_rank/kendall_last_*` | 排序一致性, 越高越好 |
| Suc-Fail Diff | `eval_p_rank/suc_fail_diff_*` | 成功-失败奖励差, 越高越好 |

也可以用 `run_baseline_eval.py` 对保存的 checkpoint 做独立评估:

```bash
uv run python robometer/evals/run_baseline_eval.py \
  reward_model=rbm \
  model_path=./logs/exp2_bt_struct/best_checkpoint \
  custom_eval.eval_types=[reward_alignment,policy_ranking] \
  custom_eval.reward_alignment=[libero_pi0] \
  custom_eval.policy_ranking=[libero_pi0] \
  custom_eval.use_frame_steps=true \
  custom_eval.subsample_n_frames=5 \
  custom_eval.reward_alignment_max_trajectories=30 \
  custom_eval.num_examples_per_quality_pr=1000 \
  max_frames=8 \
  model_config.batch_size=32
```

#### 预期结果表格

| Model | VOC r | Kendall tau | Suc-Fail Diff |
|-------|-------|-------------|---------------|
| A. Pure BT | ~0.5-0.7 | ~0.3-0.5 | ~0.05-0.10 |
| B. BT + L2 smooth | ~0.7-0.8 | ~0.5-0.6 | ~0.10-0.15 |
| C. BT + L_struct (ours) | ~0.85-0.95 | ~0.7-0.85 | ~0.20-0.35 |
| D. Full Robometer | ~0.90-0.98 | ~0.75-0.92 | ~0.30-0.46 |

> 参考值来自 Robometer 原文 Table 3 的 LIBERO-90 ablation。
> C 应该接近 D 的 ~95%（你的论文 claim）, 远好于 A 和 B。

---

### Experiment 3: Lambda Sensitivity

**目的**: 分析超参数 lambda 对 L_struct 效果的影响。

**对应论文**: 超参数分析段落

对 lambda in {0.01, 0.1, 0.5, 1.0, 5.0} 各训练一次:

```bash
for LAMBDA in 0.01 0.1 0.5 1.0 5.0; do
  uv run accelerate launch \
    --config_file robometer/configs/distributed/fsdp.yaml \
    --num_processes=1 \
    train.py \
    model.base_model_id=Qwen/Qwen3-VL-4B-Instruct \
    model.use_peft=true \
    model.train_progress_head=true \
    model.train_preference_head=true \
    model.train_success_head=false \
    data.train_datasets=[libero_pi0] \
    data.eval_datasets=[libero_pi0] \
    data.max_frames=8 \
    training.per_device_train_batch_size=8 \
    training.learning_rate=2e-5 \
    training.max_steps=5000 \
    training.predict_pref_progress=false \
    loss.struct_loss_enabled=true \
    loss.struct_loss_type=entropy \
    loss.struct_lambda=${LAMBDA} \
    training.output_dir=./logs/exp3_lambda_${LAMBDA} \
    training.exp_name=exp3_lambda_${LAMBDA} \
    logging.log_to=[] \
    custom_eval.eval_types=[reward_alignment,policy_ranking] \
    custom_eval.reward_alignment=[libero_pi0] \
    custom_eval.policy_ranking=[libero_pi0]
done
```

**输出**: 画两条曲线
- VOC r vs lambda
- Kendall tau vs lambda

预期: lambda 在 0.1-1.0 范围内效果最好, 过小 (~0.01) 退化为 Pure BT, 过大 (~5.0) 过度约束。

---

### Experiment 4: Downstream RL Performance

**目的**: 验证 reward model 质量改善转化为 RL 训练效果提升。

**对应论文**: Section 4.3 和 "achieve >= 95% sample efficiency" claim

#### 4a. 前置准备

LIBERO RL 实验需要额外安装 LIBERO 环境:

```bash
# 克隆 LIBERO
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO && pip install -e . && cd ..
```

#### 4b. RL 训练

参考 `scripts/example_libero_robometer_wrapper.py`, 它提供了将 Robometer 接入 LIBERO 的 Gymnasium wrapper。

对于实验 2 中的每个模型变体, 在 2 个 LIBERO-90 任务上跑 online RL:

```bash
# 模板（需根据 example_libero_robometer_wrapper.py 适配）
uv run python scripts/example_libero_robometer_wrapper.py \
  --model-path ./logs/exp2_bt_struct/best_checkpoint \
  --task "LIVING_ROOM_SCENE2_put_the_black_bowl_on_the_plate" \
  --num-episodes 500 \
  --seed 42
```

> Robometer 原文选择了 2 个 sparse-reward 能学到 ~100% 的任务,
> 参见原文 Figure 4 和 Appendix。需要先确认哪些 LIBERO-90 任务符合此条件。

#### 4c. 多 seed 重复

对每个任务 x 每个模型, 跑 5 个 seed (42, 43, 44, 45, 46), 报告 mean +/- std。

#### 4d. 结果可视化

画 success rate vs training steps 曲线（参考 Robometer Figure 4 的风格）:
- x 轴: training steps
- y 轴: success rate (%)
- 4 条线: Pure BT / BT+L2 / BT+L_struct / Full Robometer
- 阴影: std across 5 seeds

> 注意: RL 实验最耗时, 建议放在最后做, 或者先只跑 1-2 个 seed 看趋势。

---

## Part 4: Standalone Evaluation Commands

训练完成后, 可以用 `run_baseline_eval.py` 对任意 checkpoint 做独立评估。

### 4.1 Reward Alignment (VOC)

```bash
uv run python robometer/evals/run_baseline_eval.py \
  reward_model=rbm \
  model_path=./logs/exp2_bt_struct/best_checkpoint \
  custom_eval.eval_types=[reward_alignment] \
  custom_eval.reward_alignment=[libero_pi0] \
  custom_eval.use_frame_steps=true \
  custom_eval.subsample_n_frames=5 \
  custom_eval.reward_alignment_max_trajectories=30 \
  max_frames=8 \
  model_config.batch_size=32
```

### 4.2 Policy Ranking (Kendall tau)

```bash
uv run python robometer/evals/run_baseline_eval.py \
  reward_model=rbm \
  model_path=./logs/exp2_bt_struct/best_checkpoint \
  custom_eval.eval_types=[policy_ranking] \
  custom_eval.policy_ranking=[libero_pi0] \
  custom_eval.use_frame_steps=false \
  custom_eval.num_examples_per_quality_pr=1000 \
  max_frames=8 \
  model_config.batch_size=32
```

### 4.3 Confusion Matrix

```bash
uv run python robometer/evals/run_baseline_eval.py \
  reward_model=rbm \
  model_path=./logs/exp2_bt_struct/best_checkpoint \
  custom_eval.eval_types=[confusion_matrix] \
  custom_eval.confusion_matrix=[[abraranwar_libero_rfm_libero256_90,ykorkmaz_libero_failure_rfm_libero_90_failure]] \
  max_frames=8 \
  model_config.batch_size=32
```

---

## Part 5: Time and Resource Estimates

| 实验 | 模型数量 | 每个训练时间 (单 A100) | 总时间 |
|------|---------|----------------------|--------|
| Exp 1: Diagnosis | 2 | ~2-3h | ~5h |
| Exp 2: Core Ablation | 4 (复用 Exp1) | ~2-3h | ~5h (新增 2 个) |
| Exp 3: Lambda Sweep | 5 | ~2-3h | ~12h |
| Exp 4: RL | 4 models x 2 tasks x 5 seeds | ~1-2h each | ~40-80h |
| Eval (standalone) | - | ~15min each | ~2h |

> 建议优先级: Exp 2 > Exp 1 > Exp 3 > Exp 4
> Exp 1 和 Exp 2 共享 2 个模型, 合起来只需训练 4 个模型。
> Exp 3 可以缩减到 3 个点 {0.01, 0.1, 1.0} 如果时间不够。
> Exp 4 可以先跑 1 个 seed 确认趋势, 再补全。

---

## Part 6: Checklist

- [ ] 环境安装成功（torch + transformers + robometer import OK）
- [ ] LIBERO 数据集下载并路径对齐
- [ ] Qwen3-VL-4B 和 Robometer-4B 模型下载
- [ ] 4 个文件代码修改完成（configs + trainer）
- [ ] 单次训练 dry run 成功（max_steps=10 快速验证）
- [ ] Exp 2 四个模型训练完成
- [ ] Exp 1 可视化脚本完成（potential curve + delta distribution）
- [ ] Exp 3 lambda sweep 完成
- [ ] 主结果表填写完成
- [ ] (可选) Exp 4 RL 实验完成

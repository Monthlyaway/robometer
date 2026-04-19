# Setup Guide

拿到这台机器（AutoDL）后的完整环境恢复指南。所有数据和模型已在磁盘上，无需网络。

---

## 磁盘布局

```
/root/autodl-tmp/
├── robometer/                                 # 代码仓库 (git repo)
│   ├── .venv/                                 # Python 3.10 虚拟环境
│   ├── train.py                               # 训练入口
│   ├── robometer/                             # 核心代码
│   ├── logs/                                  # 训练输出 (gitignored)
│   └── my_paper/                              # 论文相关文档和脚本
│
├── raw_datasets/                              # ROBOMETER_DATASET_PATH
│   ├── libero_rfm/                            # abraranwar/libero_rfm
│   │   ├── libero256_10/                      # 训练集 (成功)
│   │   ├── libero256_object/                  # 训练集 (成功)
│   │   ├── libero256_spatial/                 # 训练集 (成功)
│   │   ├── libero256_goal/                    # 训练集 (成功)
│   │   ├── libero256_90/                      # 评估集 (成功)
│   │   ├── libero_10/ ... libero_spatial/     # 非256版本 (未使用)
│   │   └── README.md
│   └── libero_failure_rfm/                    # ykorkmaz/libero_failure_rfm
│       ├── libero_10_failure/                 # 训练集 (失败)
│       ├── libero_object_failure/             # 训练集 (失败)
│       ├── libero_spatial_failure/            # 训练集 (失败)
│       ├── libero_goal_failure/               # 训练集 (失败)
│       ├── libero_90_failure/                 # 评估集 (失败)
│       └── README.md
│
├── processed_datasets/                        # ROBOMETER_PROCESSED_DATASETS_PATH
│   ├── abraranwar_libero_rfm_libero256_10/    # 预处理后 (frames/ + dataset_info.json)
│   ├── abraranwar_libero_rfm_libero256_object/
│   ├── abraranwar_libero_rfm_libero256_spatial/
│   ├── abraranwar_libero_rfm_libero256_goal/
│   ├── abraranwar_libero_rfm_libero256_90/
│   ├── ykorkmaz_libero_failure_rfm_libero_10_failure/
│   ├── ykorkmaz_libero_failure_rfm_libero_object_failure/
│   ├── ykorkmaz_libero_failure_rfm_libero_spatial_failure/
│   ├── ykorkmaz_libero_failure_rfm_libero_goal_failure/
│   └── ykorkmaz_libero_failure_rfm_libero_90_failure/
│
└── .cache/huggingface/hub/                    # ~/.cache/huggingface -> 此处 (symlink)
    ├── models--Qwen--Qwen3-VL-4B-Instruct/   # 8.3GB, 基座 VLM
    ├── models--unsloth--Qwen3-VL-4B-Instruct/ # symlink 到 Qwen 版本
    └── models--robometer--Robometer-4B/       # 原始 Robometer 权重 (未使用)
```

---

## 每次开机 / 重启后执行

```bash
cd /root/autodl-tmp/robometer
source .venv/bin/activate

# 环境变量 (每次开新 terminal 都要 export)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export ROBOMETER_DATASET_PATH=/root/autodl-tmp/raw_datasets
export ROBOMETER_PROCESSED_DATASETS_PATH=/root/autodl-tmp/processed_datasets
```

### 健康检查

```bash
# torchcodec 必须是 0.7 (torch 2.8.0 兼容版本)
pip show torchcodec | grep Version
# 如果显示 0.11.x，修复:
#   pip uninstall torchcodec -y && pip install torchcodec==0.7

# 数据完整性
ls $ROBOMETER_DATASET_PATH/libero_rfm/ | wc -l           # 11
ls $ROBOMETER_PROCESSED_DATASETS_PATH/ | grep -c "^[ay]"  # 10

# 模型缓存
ls ~/.cache/huggingface/hub/models--unsloth--Qwen3-VL-4B-Instruct/snapshots/*/model-00001-of-00002.safetensors
# 应该返回一个 symlink 路径，如果报 No such file:
#   见下方 Troubleshooting
```

---

## Troubleshooting

### 1. `torchcodec ... undefined symbol: torch_dtype_float4_e2m1fn_x2`

`uv sync` 会安装 torchcodec 0.11+，与 torch 2.8.0 不兼容。

```bash
source .venv/bin/activate
pip uninstall torchcodec -y
pip install torchcodec==0.7
```

### 2. `Read timed out (cas-bridge.xethub.hf.co)` / 网络超时

模型和数据都已本地缓存，不需要网络。确保设了离线变量:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

### 3. `unsloth/Qwen3-VL-4B-Instruct does not appear to have files`

Unsloth 内部把 model id 重定向到 `unsloth/Qwen3-VL-4B-Instruct`，但该缓存目录可能缺少权重文件。修复:

```bash
UNSLOTH_SNAP=$(ls -d ~/.cache/huggingface/hub/models--unsloth--Qwen3-VL-4B-Instruct/snapshots/*/)
QWEN_SNAP=$(ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/*/)

for f in "$QWEN_SNAP"/*; do
    fname=$(basename "$f")
    [ ! -e "$UNSLOTH_SNAP/$fname" ] && ln -s "$f" "$UNSLOTH_SNAP/$fname" && echo "linked: $fname"
done
```

### 4. `ROBOMETER_DATASET_PATH not set`

忘了 export 环境变量，见上方"每次开机后执行"。

### 5. `'list' object has no attribute 'replace'`（eval_datasets 报错）

`libero_pi0` 的 eval 配置含嵌套列表（paired 数据集）。使用 `libero` 代替:

```bash
data.eval_datasets=[libero]    # 正确
# data.eval_datasets=[libero_pi0]  # 会报错
```

### 6. 需要重新预处理数据集

正常情况下不需要，processed_datasets 已在磁盘上。如确实需要:

```bash
export ROBOMETER_DATASET_PATH=/root/autodl-tmp/raw_datasets
python -m robometer.data.scripts.preprocess_datasets \
  --config robometer/configs/preprocess_libero.yaml
```

预处理脚本已 patch 为优先从本地 parquet 文件加载，不走网络。

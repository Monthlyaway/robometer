# Experiment Guide v1: 执行实验与查看结果

前置条件: 已完成 [setup.md](setup.md) 中的环境恢复。

---

## Dry Run 验证

确认代码无误再跑正式实验:

```bash
bash my_paper/scripts/dry_run.sh
```

预期: 10 步完成 (~40s)，每步可见 `train/struct_loss: -1.9xxxxx`，最终输出 `Training complete!`

---

## 正式实验

### 共享参数

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

### Exp 1: BT-only (无进度头，无正则)

```bash
accelerate launch --config_file robometer/configs/distributed/fsdp.yaml --num_processes=1 \
  train.py $BASE_ARGS \
  model.train_progress_head=false \
  loss.struct_loss_enabled=false \
  training.output_dir=./logs/exp1_bt_only \
  training.exp_name=exp1_bt_only
```

### Exp 2: BT + Progress (无正则)

```bash
accelerate launch --config_file robometer/configs/distributed/fsdp.yaml --num_processes=1 \
  train.py $BASE_ARGS \
  loss.struct_loss_enabled=false \
  training.output_dir=./logs/exp2_bt_progress \
  training.exp_name=exp2_bt_progress
```

### Exp 3: BT + Progress + L2 Smooth (baseline 正则)

```bash
accelerate launch --config_file robometer/configs/distributed/fsdp.yaml --num_processes=1 \
  train.py $BASE_ARGS \
  loss.struct_loss_enabled=true \
  loss.struct_loss_type=l2_smooth \
  loss.struct_lambda=0.1 \
  training.output_dir=./logs/exp3_l2_smooth \
  training.exp_name=exp3_l2_smooth
```

### Exp 4: BT + Progress + Entropy Prior (本文方法)

```bash
accelerate launch --config_file robometer/configs/distributed/fsdp.yaml --num_processes=1 \
  train.py $BASE_ARGS \
  loss.struct_loss_enabled=true \
  loss.struct_loss_type=entropy \
  loss.struct_lambda=0.1 \
  training.output_dir=./logs/exp4_entropy \
  training.exp_name=exp4_entropy
```

### Lambda 敏感性 (Exp 4 变体)

```bash
for LAMBDA in 0.01 0.1 1.0; do
  accelerate launch --config_file robometer/configs/distributed/fsdp.yaml --num_processes=1 \
    train.py $BASE_ARGS \
    loss.struct_loss_enabled=true \
    loss.struct_loss_type=entropy \
    loss.struct_lambda=$LAMBDA \
    training.output_dir=./logs/exp_lambda_${LAMBDA} \
    training.exp_name=exp_lambda_${LAMBDA}
done
```

---

## 查看结果

### TensorBoard

```bash
tensorboard --logdir ./logs --bind_all --port 6006
```

AutoDL: SSH 隧道 `ssh -L 6006:localhost:6006 root@<地址>`，浏览器 `http://localhost:6006`

关键 metrics:

| Tag | 含义 |
|-----|------|
| `train/pref_bt_loss` | Bradley-Terry 偏好损失 |
| `train/struct_loss` | 结构正则损失 (仅 Exp 3/4) |
| `eval_p_rank/voc_r_*` | VOC r (Pearson correlation) |
| `eval_p_rank/kendall_last_*` | Kendall τ |
| `eval_p_rank/suc_fail_diff_*` | 成功-失败差异 |

### 本地日志

```bash
ls ./logs/exp4_entropy/
# trainer_state.json   → loss 历史
# checkpoint-*/        → 模型权重
# eval_results_*.json  → 评估指标
# runs/                → TensorBoard 事件文件
```

### 论文占位符映射

| 论文占位符 | 数据来源 |
|-----------|---------|
| Table 1: VOC r / Kendall τ / Suc-Fail Diff | TensorBoard 或 `eval_results_*.json` |
| Fig 1: 势函数曲线 | 训练后推理脚本 (加载 checkpoint，对 eval 轨迹逐帧 predict progress) |
| Fig 2: 增量分布直方图 | 同上，计算相邻帧 progress 差值分布 |
| Fig/Table 3: λ 敏感性 | Lambda 敏感性实验的 VOC r / Kendall τ 汇总 |

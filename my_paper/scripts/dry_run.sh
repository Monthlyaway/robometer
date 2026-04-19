#!/bin/bash
# Dry-run: verify L_struct implementation works (10 steps, small batch)
# Expected: train/struct_loss appears in logs, no crashes, finite values
set -e

cd /root/autodl-tmp/robometer

echo "=== Dry Run: BT + L_struct (entropy) ==="
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
  training.per_device_train_batch_size=2 \
  training.learning_rate=2e-5 \
  training.max_steps=10 \
  training.eval_steps=100 \
  training.custom_eval_steps=100 \
  training.predict_pref_progress=false \
  loss.struct_loss_enabled=true \
  loss.struct_loss_type=entropy \
  loss.struct_lambda=0.1 \
  training.output_dir=./logs/dry_run_entropy \
  training.exp_name=dry_run_entropy \
  training.overwrite_output_dir=True \
  logging.log_to=[]

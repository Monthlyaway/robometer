#!/bin/bash
# Run all 4 core ablation experiments sequentially (Route B)
# Estimated total time: ~8-12h on RTX 4080S
set -e

cd /root/autodl-tmp/robometer
source .venv/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export ROBOMETER_DATASET_PATH=/root/autodl-tmp/raw_datasets
export ROBOMETER_PROCESSED_DATASETS_PATH=/root/autodl-tmp/processed_datasets

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

# ======================================================================
# Exp A: Pure BT (progress head as Phi, no regularization)
#   Paper: Table 1 Row A — demonstrates the monotonicity trap
# ======================================================================
echo "========== [1/4] Exp A: Pure BT =========="
accelerate launch \
  --config_file robometer/configs/distributed/fsdp.yaml \
  --num_processes=1 \
  train.py $BASE_ARGS \
  model.train_preference_head=false \
  "data.sample_type_ratio=[1,0,0]" \
  training.predict_pref_progress=false \
  loss.pref_loss_type=bt_sum \
  loss.struct_loss_enabled=false \
  training.output_dir=./logs/exp_a_pure_bt \
  training.exp_name=exp_a_pure_bt \
  "logging.log_to=[tensorboard]"

# ======================================================================
# Exp B: BT + L2 Smooth (naive regularization baseline)
#   Paper: Table 1 Row B — L2 temporal smoothing is insufficient
# ======================================================================
echo "========== [2/4] Exp B: BT + L2 Smooth =========="
accelerate launch \
  --config_file robometer/configs/distributed/fsdp.yaml \
  --num_processes=1 \
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

# ======================================================================
# Exp C: BT + L_struct / Entropy Prior (OUR METHOD)
#   Paper: Table 1 Row C — Maximum Entropy Increment Prior
# ======================================================================
echo "========== [3/4] Exp C: BT + Entropy (Ours) =========="
accelerate launch \
  --config_file robometer/configs/distributed/fsdp.yaml \
  --num_processes=1 \
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

# ======================================================================
# Exp D: Full Robometer (supervised oracle upper bound)
#   Paper: Table 1 Row D — uses preference head + supervised progress labels
# ======================================================================
echo "========== [4/4] Exp D: Full Robometer =========="
accelerate launch \
  --config_file robometer/configs/distributed/fsdp.yaml \
  --num_processes=1 \
  train.py $BASE_ARGS \
  model.train_preference_head=true \
  "data.sample_type_ratio=[1,0,0]" \
  training.predict_pref_progress=true \
  loss.pref_loss_type=head \
  loss.struct_loss_enabled=false \
  training.output_dir=./logs/exp_d_robometer \
  training.exp_name=exp_d_robometer \
  "logging.log_to=[tensorboard]"

echo "========== All 4 experiments complete! =========="
echo "View results: tensorboard --logdir ./logs --bind_all --port 6006"

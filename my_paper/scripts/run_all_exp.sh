#!/bin/bash
# Run ablation experiments for Structure-Regularized Potential Inference
#
# Usage:
#   bash run_all_exp.sh <experiment> [suffix]
#
#   <experiment> = dry_run | a | b | c | d | e | all
#   [suffix]     = optional name suffix appended to output dir, e.g. "round3"
#
# Examples:
#   bash my_paper/scripts/run_all_exp.sh c              # → logs/exp_c_entropy/
#   bash my_paper/scripts/run_all_exp.sh c round3        # → logs/exp_c_entropy_round3/
#   bash my_paper/scripts/run_all_exp.sh a final         # → logs/exp_a_pure_bt_final/
#   bash my_paper/scripts/run_all_exp.sh all v2          # → logs/exp_a_pure_bt_v2/ etc.
#   bash my_paper/scripts/run_all_exp.sh e sweep1        # → logs/exp_e_lambda_0.01_sweep1/ etc.
#
# Timing: ~0.5s/step on RTX 4080S with 2B model.
#   dry_run (5 steps): ~10s
#   Single experiment (1250 steps): ~70min
#   All A-D: ~2.5h
set -e

EXP="${1:?Usage: bash run_all_exp.sh <dry_run|a|b|c|d|e|all> [suffix]}"
SUFFIX="${2:-}"

cd /root/autodl-tmp/robometer
source .venv/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export ROBOMETER_DATASET_PATH=/root/autodl-tmp/raw_datasets
export ROBOMETER_PROCESSED_DATASETS_PATH=/root/autodl-tmp/processed_datasets

BASE_ARGS="
  model.base_model_id=Qwen/Qwen3-VL-2B-Instruct
  model.use_unsloth=false
  model.use_peft=true
  model.train_progress_head=true
  model.train_success_head=false
  data.train_datasets=[libero_pi0]
  data.eval_datasets=[libero_pi0]
  data.max_frames=8
  training.per_device_train_batch_size=16
  training.gradient_accumulation_steps=1
  training.learning_rate=2e-5
  training.max_steps=1250
  training.do_eval=false
  training.evaluation_strategy=no
  training.save_strategy=steps
  training.save_steps=500
  training.logging_steps=50
  custom_eval.reward_alignment=[libero_pi0]
  custom_eval.policy_ranking=[libero_pi0]
"
EVAL_TYPES_ARG="custom_eval.eval_types=[policy_ranking,reward_alignment]"

LAUNCH="accelerate launch --config_file robometer/configs/distributed/fsdp.yaml --num_processes=1"

# Append _$SUFFIX to a base name if SUFFIX is set
name() { if [ -n "$SUFFIX" ]; then echo "${1}_${SUFFIX}"; else echo "$1"; fi; }

run_train() {
  $LAUNCH train.py $BASE_ARGS "$EVAL_TYPES_ARG" "$@"
}

# ------------------------------------------------------------------
run_dry_run() {
  local N=$(name dry_run)
  echo "========== Dry Run (5 steps + eval) → $N =========="
  run_train \
    model.train_preference_head=true \
    "data.sample_type_ratio=[1,0,0]" \
    training.predict_pref_progress=false \
    training.max_steps=5 \
    training.do_eval=true \
    training.evaluation_strategy=steps \
    training.eval_steps=5 \
    training.custom_eval_steps=5 \
    loss.struct_loss_enabled=true \
    loss.struct_loss_type=entropy \
    loss.struct_lambda=0.1 \
    loss.progress_loss_type=l2 \
    training.output_dir=./logs/$N \
    training.exp_name=$N \
    training.overwrite_output_dir=True \
    "logging.log_to=[]"
}

# ------------------------------------------------------------------
# Exp A: Pure BT (progress head as Phi, no regularization)
#   Table 1 Row A — demonstrates the monotonicity trap
# ------------------------------------------------------------------
run_a() {
  local N=$(name exp_a_pure_bt)
  echo "========== Exp A: Pure BT → $N =========="
  run_train \
    model.train_preference_head=false \
    model.progress_use_sigmoid=false \
    "data.sample_type_ratio=[1,0,0]" \
    training.predict_pref_progress=false \
    loss.pref_loss_type=bt_sum \
    loss.progress_loss_type=l2 \
    loss.struct_loss_enabled=false \
    training.output_dir=./logs/$N \
    training.exp_name=$N \
    "logging.log_to=[tensorboard]"
}

# ------------------------------------------------------------------
# Exp B: BT + L2 Smooth (naive regularization baseline)
#   Table 1 Row B — L2 temporal smoothing is insufficient
# ------------------------------------------------------------------
run_b() {
  local N=$(name exp_b_l2_smooth)
  echo "========== Exp B: BT + L2 Smooth → $N =========="
  run_train \
    model.train_preference_head=false \
    model.progress_use_sigmoid=false \
    "data.sample_type_ratio=[1,0,0]" \
    training.predict_pref_progress=false \
    loss.pref_loss_type=bt_sum \
    loss.progress_loss_type=l2 \
    loss.struct_loss_enabled=true \
    loss.struct_loss_type=l2_smooth \
    loss.struct_lambda=0.1 \
    training.output_dir=./logs/$N \
    training.exp_name=$N \
    "logging.log_to=[tensorboard]"
}

# ------------------------------------------------------------------
# Exp C: BT + L_struct / Entropy Prior (OUR METHOD)
#   Table 1 Row C — Maximum Entropy Increment Prior
# ------------------------------------------------------------------
run_c() {
  local N=$(name exp_c_entropy)
  echo "========== Exp C: BT + Entropy (Ours) → $N =========="
  run_train \
    model.train_preference_head=false \
    model.progress_use_sigmoid=false \
    "data.sample_type_ratio=[1,0,0]" \
    training.predict_pref_progress=false \
    loss.pref_loss_type=bt_sum \
    loss.progress_loss_type=l2 \
    loss.struct_loss_enabled=true \
    loss.struct_loss_type=entropy \
    loss.struct_lambda=0.1 \
    training.output_dir=./logs/$N \
    training.exp_name=$N \
    "logging.log_to=[tensorboard]"
}

# ------------------------------------------------------------------
# Exp D: Full Robometer (supervised oracle upper bound)
#   Table 1 Row D — uses preference head + supervised progress labels
# ------------------------------------------------------------------
run_d() {
  local N=$(name exp_d_robometer)
  echo "========== Exp D: Full Robometer → $N =========="
  run_train \
    model.train_preference_head=true \
    "data.sample_type_ratio=[1,0,0]" \
    training.predict_pref_progress=true \
    loss.pref_loss_type=head \
    loss.struct_loss_enabled=false \
    training.output_dir=./logs/$N \
    training.exp_name=$N \
    "logging.log_to=[tensorboard]"
}

# ------------------------------------------------------------------
# Exp E: Lambda sensitivity sweep
#   Section 5.3 hyperparameter analysis
# ------------------------------------------------------------------
run_e() {
  for LAMBDA in 0.01 0.1 1.0; do
    local N=$(name "exp_e_lambda_${LAMBDA}")
    echo "========== Exp E: Lambda=$LAMBDA → $N =========="
    run_train \
      model.train_preference_head=false \
      model.progress_use_sigmoid=false \
      "data.sample_type_ratio=[1,0,0]" \
      training.predict_pref_progress=false \
      loss.pref_loss_type=bt_sum \
      loss.progress_loss_type=l2 \
      loss.struct_loss_enabled=true \
      loss.struct_loss_type=entropy \
      loss.struct_lambda=$LAMBDA \
      training.output_dir=./logs/$N \
      training.exp_name=$N \
      "logging.log_to=[tensorboard]"
  done
}

# ------------------------------------------------------------------
# Dispatch
# ------------------------------------------------------------------
case "$EXP" in
  dry_run)  run_dry_run ;;
  a)        run_a ;;
  b)        run_b ;;
  c)        run_c ;;
  d)        run_d ;;
  e)        run_e ;;
  all)
    run_a
    run_b
    run_c
    run_d
    echo "========== All 4 experiments (A-D) complete! =========="
    ;;
  *)
    echo "Unknown experiment: $EXP"
    echo "Usage: bash run_all_exp.sh <dry_run|a|b|c|d|e|all> [suffix]"
    exit 1
    ;;
esac

echo "View results: tensorboard --logdir ./logs --bind_all --port 6006"

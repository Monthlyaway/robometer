#!/bin/bash
# Run RL evaluation comparing reward models A, C, and sparse baseline.
#
# Prerequisites:
#   pip install libero2gym stable-baselines3
#   # or: clone LIBERO into ../LIBERO
#
# Usage:
#   bash my_paper/scripts/run_rl_comparison.sh          # full (5 seeds × 3 models × 2 tasks)
#   bash my_paper/scripts/run_rl_comparison.sh quick     # quick test (1 seed × 3 models × 1 task)
#
# Training data context:
#   Our reward models are trained on libero_pi0:
#     Train: LIBERO-{10, Object, Spatial, Goal} + generated failures
#     Eval:  LIBERO-90 (held out)
#   So testing RL on LIBERO-90 tasks is zero-shot for the reward model.
#
# Tasks (from Robometer paper appendix):
#   Task 28: "close the top drawer of the cabinet"
#   Task 33: "close the microwave"
#   Both chosen because sparse RL can learn them, enabling sample-efficiency comparison.

set -e
cd /root/autodl-tmp/robometer
source .venv/bin/activate
export MUJOCO_GL=egl
export HF_HUB_OFFLINE=1

MODE="${1:-full}"

# Checkpoint paths (update after training completes)
EXP_A_CKPT="./logs/exp_a_pure_bt_round3a/exp_a_pure_bt_round3a/checkpoint-2500"
EXP_C_CKPT="./logs/exp_c_entropy_round3a/exp_c_entropy_round3a/checkpoint-2500"

OUTPUT_DIR="./logs/rl_eval"

if [ "$MODE" = "quick" ]; then
  SEEDS="0"
  TASKS="28"
  TIMESTEPS=20000
  echo "=== Quick mode: 1 seed, 1 task, 20k steps ==="
else
  SEEDS="0 1 2 3 4"
  TASKS="28 33"
  TIMESTEPS=100000
  echo "=== Full mode: 5 seeds, 2 tasks, 100k steps ==="
fi

for TASK_ID in $TASKS; do
  echo ""
  echo "================================================================"
  echo "Task $TASK_ID"
  echo "================================================================"

  for SEED in $SEEDS; do
    echo ""
    echo "--- Sparse reward (seed=$SEED, task=$TASK_ID) ---"
    python my_paper/scripts/run_rl_eval.py \
      --sparse \
      --task-id $TASK_ID \
      --seed $SEED \
      --total-timesteps $TIMESTEPS \
      --output-dir "$OUTPUT_DIR/task_${TASK_ID}"

    if [ -d "$EXP_A_CKPT" ]; then
      echo ""
      echo "--- Exp A: Pure BT (seed=$SEED, task=$TASK_ID) ---"
      python my_paper/scripts/run_rl_eval.py \
        --reward-model-path "$EXP_A_CKPT" \
        --reward-model-name exp_a_pure_bt \
        --task-id $TASK_ID \
        --seed $SEED \
        --total-timesteps $TIMESTEPS \
        --output-dir "$OUTPUT_DIR/task_${TASK_ID}"
    else
      echo "WARNING: Exp A checkpoint not found at $EXP_A_CKPT, skipping"
    fi

    if [ -d "$EXP_C_CKPT" ]; then
      echo ""
      echo "--- Exp C: BT + L_struct (seed=$SEED, task=$TASK_ID) ---"
      python my_paper/scripts/run_rl_eval.py \
        --reward-model-path "$EXP_C_CKPT" \
        --reward-model-name exp_c_entropy \
        --task-id $TASK_ID \
        --seed $SEED \
        --total-timesteps $TIMESTEPS \
        --output-dir "$OUTPUT_DIR/task_${TASK_ID}"
    else
      echo "WARNING: Exp C checkpoint not found at $EXP_C_CKPT, skipping"
    fi
  done
done

echo ""
echo "================================================================"
echo "All RL evaluations complete!"
echo "Results in: $OUTPUT_DIR/"
echo ""
echo "To plot: python my_paper/scripts/plot_rl_results.py"
echo "================================================================"

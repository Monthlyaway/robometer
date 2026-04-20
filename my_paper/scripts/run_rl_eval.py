"""
RL evaluation script: train SAC policies with different reward models on LIBERO-90 tasks.

Compares:
  - Sparse reward (env default)
  - Exp A checkpoint (Pure BT, no L_struct)
  - Exp C checkpoint (BT + L_struct, our method)

Following Robometer paper appendix (Section: RL with Ablated Reward Models):
  - SAC on LIBERO-90 Task 28 (close the top drawer) and Task 33 (close the microwave)
  - DINO-v2-small image features + proprioceptive state
  - Eval every 5000 steps, 25 episodes per eval
  - 5 seeds per reward model

Usage:
  python my_paper/scripts/run_rl_eval.py \
    --reward-model-path /path/to/checkpoint \
    --reward-model-name exp_c_entropy \
    --task-id 28 \
    --seed 0 \
    --total-timesteps 100000

  # Sparse reward baseline (no reward model):
  python my_paper/scripts/run_rl_eval.py \
    --sparse \
    --task-id 28 \
    --seed 0

Prerequisites:
  pip install libero2gym stable-baselines3 torch
  # LIBERO requires MuJoCo: pip install mujoco
"""

import argparse
import logging
import os
import sys
import json
from pathlib import Path
from collections import deque
from typing import Optional, Dict, Any, Sequence

import numpy as np
import gymnasium as gym

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# Suppress per-step debug spam from the reward model inference (loguru)
try:
    from loguru import logger as _loguru_logger
    _loguru_logger.disable("robometer.evals.eval_server")
    _loguru_logger.disable("robometer.evals")
    _loguru_logger.disable("robometer.utils.setup_utils")
except ImportError:
    pass
# Also suppress standard logging
logging.getLogger("robometer").setLevel(logging.WARNING)


def make_libero_env(task_suite_name: str, task_id: int, seed: int):
    """Create a single LIBERO environment with Gymnasium interface."""
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero import benchmark, get_libero_path

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)
    task_name = task.name
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )

    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": 256,
        "camera_widths": 256,
    }
    base_env = OffScreenRenderEnv(**env_args)
    base_env.seed(seed)

    from scripts.example_libero_robometer_wrapper import GymToGymnasiumWrapper
    env = GymToGymnasiumWrapper(base_env, time_limit=400)

    if not hasattr(env, "action_space"):
        env.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )

    return env, task_name


class DINOFeatureWrapper(gym.ObservationWrapper):
    """Extract DINO-v2-small features from agentview_image + concat proprioception.

    Following Robometer paper: DINO-v2-small featurized external image (384-dim)
    concatenated with proprioceptive joint positions (typically 7-dim for Franka).
    Output: flat vector suitable for MlpPolicy.
    """

    def __init__(self, env, device: str = "cuda"):
        super().__init__(env)
        import torch
        self.device = device
        self.dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.dino = self.dino.to(device).eval()

        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ])

        # DINO-v2-small: 384-dim; proprio: ~7-dim joint + 2-dim gripper
        self.dino_dim = 384
        self.proprio_keys = ["joint_states", "gripper_states", "ee_states"]
        self._proprio_dim = None
        self._obs_space_initialized = False
        # Placeholder obs space; updated on first observation
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.dino_dim,), dtype=np.float32
        )

    def _get_proprio(self, obs: dict) -> np.ndarray:
        parts = []
        for k in self.proprio_keys:
            if k in obs:
                v = np.asarray(obs[k], dtype=np.float32).flatten()
                parts.append(v)
        if not parts:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(parts)

    def observation(self, obs):
        import torch

        img = obs.get("agentview_image", None)
        if img is None:
            for k in obs:
                if "image" in k.lower() or "rgb" in k.lower():
                    img = obs[k]
                    break
        if img is None:
            raise ValueError(f"No image key found in obs. Keys: {list(obs.keys())}")

        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.dino(img_tensor).squeeze(0).cpu().numpy()

        proprio = self._get_proprio(obs)

        combined = np.concatenate([feat, proprio]).astype(np.float32)

        if not self._obs_space_initialized:
            self._proprio_dim = len(proprio)
            total_dim = self.dino_dim + self._proprio_dim
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32
            )
            self._obs_space_initialized = True

        return combined


class RewardModelWrapper(gym.Wrapper):
    """Replace env reward with reward model predictions (PBRS-style).

    Uses the Robometer reward wrapper internally but provides a simplified interface.
    """

    def __init__(
        self,
        env: gym.Env,
        model_path: str,
        device: str = "cuda",
        max_frames: int = 8,
        use_relative_rewards: bool = True,
        add_env_reward: bool = True,
        reward_freq: int = 1,
    ):
        super().__init__(env)
        from scripts.example_libero_robometer_wrapper import _RewardModelInferenceMixin

        class _Mixin(_RewardModelInferenceMixin):
            pass

        self._rm = _Mixin(model_path=model_path, device=device, max_frames=max_frames)
        self.reward_key = "agentview_image"
        self.use_relative_rewards = use_relative_rewards
        self.add_env_reward = add_env_reward
        self.reward_freq = reward_freq
        self._frames = []
        self._prev_reward = 0.0
        self._cached_pred_reward = 0.0
        self._step_count = 0
        self._language_instruction = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._language_instruction = getattr(self.env, "language_instruction", "")
        # Walk the wrapper chain to find language_instruction
        e = self.env
        while self._language_instruction in (None, "") and hasattr(e, "env"):
            e = e.env
            self._language_instruction = getattr(e, "language_instruction", "")
        self._frames = []
        self._prev_reward = 0.0
        if isinstance(obs, dict) and self.reward_key in obs:
            from robometer.utils.tensor_utils import t2n
            self._frames.append(t2n(obs[self.reward_key]))
        return obs, info

    def step(self, action):
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1

        if isinstance(obs, dict) and self.reward_key in obs:
            from robometer.utils.tensor_utils import t2n
            self._frames.append(t2n(obs[self.reward_key]))

        should_compute = (
            self._step_count % self.reward_freq == 0
            or terminated or truncated
        )

        if should_compute and self._frames:
            frames = np.stack(self._frames, axis=0)
            raw = dict(
                frames=frames,
                task=self._language_instruction or "",
                id=0,
                metadata=dict(subsequence_length=len(self._frames)),
                video_embeddings=None,
                text_embedding=None,
            )
            rewards, _ = self._rm._compute_rewards_batch([raw])
            full_reward = rewards[0] if rewards else 0.0

            if self.use_relative_rewards:
                self._cached_pred_reward = full_reward - self._prev_reward
                self._prev_reward = full_reward
            else:
                self._cached_pred_reward = full_reward

        pred_reward = self._cached_pred_reward
        env_reward_shifted = env_reward - 1.0

        if self.add_env_reward:
            out_reward = env_reward_shifted + pred_reward
        else:
            out_reward = pred_reward

        info["env_reward"] = env_reward
        info["predicted_reward"] = pred_reward
        info["success"] = info.get("success", terminated)

        if terminated or truncated:
            self._frames = []
            self._prev_reward = 0.0
            self._cached_pred_reward = 0.0
            self._step_count = 0

        return obs, out_reward, terminated, truncated, info


class SparseRewardWrapper(gym.Wrapper):
    """Sparse reward baseline: -1 per step, 0 on success."""

    def step(self, action):
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        info["success"] = info.get("success", terminated)
        sparse_reward = 0.0 if terminated else -1.0
        return obs, sparse_reward, terminated, truncated, info


class SuccessTracker(gym.Wrapper):
    """Track success rate over a rolling window for logging."""

    def __init__(self, env, window_size: int = 25):
        super().__init__(env)
        self.successes = deque(maxlen=window_size)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            self.successes.append(float(info.get("success", False)))
        return obs, reward, terminated, truncated, info

    @property
    def success_rate(self):
        if not self.successes:
            return 0.0
        return np.mean(self.successes)


def build_env(
    task_suite: str,
    task_id: int,
    seed: int,
    reward_model_path: Optional[str] = None,
    device: str = "cuda",
    max_frames: int = 8,
    reward_freq: int = 1,
):
    """Build the full environment pipeline."""
    env, task_name = make_libero_env(task_suite, task_id, seed)

    if reward_model_path:
        env = RewardModelWrapper(
            env,
            model_path=reward_model_path,
            device=device,
            max_frames=max_frames,
            use_relative_rewards=True,
            add_env_reward=True,
            reward_freq=reward_freq,
        )
    else:
        env = SparseRewardWrapper(env)

    env = DINOFeatureWrapper(env, device=device)
    env = SuccessTracker(env)

    print(f"Environment built: {task_name} (task_id={task_id})", flush=True)
    print(f"  Reward: {'model=' + reward_model_path if reward_model_path else 'sparse'}", flush=True)
    print(f"  Obs space: {env.observation_space}", flush=True)
    print(f"  Act space: {env.action_space}", flush=True)

    return env, task_name


def train_sac(
    env,
    task_name: str,
    reward_name: str,
    seed: int,
    total_timesteps: int,
    eval_freq: int,
    output_dir: str,
    learning_starts: int = 5000,
):
    """Train SAC and log success rates."""
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback

    class SuccessLogCallback(BaseCallback):
        def __init__(self, eval_freq: int, log_path: str, total_steps: int, progress_freq: int = 100):
            super().__init__()
            self.eval_freq = eval_freq
            self.log_path = log_path
            self.total_steps = total_steps
            self.progress_freq = progress_freq
            self.results = []
            self._tracker = None
            self._start_time = None

        def _find_tracker(self):
            """Walk the wrapper chain to find our SuccessTracker."""
            env = self.training_env.envs[0]
            while env is not None:
                if isinstance(env, SuccessTracker):
                    return env
                env = getattr(env, "env", None)
            return None

        def _on_step(self) -> bool:
            import time
            if self._start_time is None:
                self._start_time = time.time()

            if self.num_timesteps % self.progress_freq == 0:
                elapsed = time.time() - self._start_time
                sps = self.num_timesteps / max(elapsed, 1)
                remaining = (self.total_steps - self.num_timesteps) / max(sps, 0.01)
                print(
                    f"  step {self.num_timesteps:>6d}/{self.total_steps}  "
                    f"({sps:.1f} steps/s, ~{remaining:.0f}s left)",
                    flush=True,
                )

            if self.num_timesteps % self.eval_freq == 0:
                if self._tracker is None:
                    self._tracker = self._find_tracker()
                sr = self._tracker.success_rate if self._tracker else 0.0
                entry = {
                    "timestep": self.num_timesteps,
                    "success_rate": sr,
                    "reward_name": reward_name,
                    "seed": seed,
                }
                self.results.append(entry)
                print(
                    f"  >>> EVAL [{reward_name}] step={self.num_timesteps:>6d}  "
                    f"success_rate={sr:.3f}",
                    flush=True,
                )
                with open(self.log_path, "w") as f:
                    json.dump(self.results, f, indent=2)
            return True

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"rl_{reward_name}_seed{seed}.json")

    # SAC hyperparams from Robometer paper appendix Table 7
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=1e-5,
        batch_size=128,
        tau=0.005,
        gamma=0.99,
        learning_starts=learning_starts,
        train_freq=1,
        gradient_steps=1,
        seed=seed,
        verbose=0,
        device="cuda",
    )

    callback = SuccessLogCallback(
        eval_freq=eval_freq, log_path=log_path,
        total_steps=total_timesteps, progress_freq=100,
    )

    print(f"\nStarting SAC training: {reward_name} / seed={seed}", flush=True)
    print(f"  Total timesteps: {total_timesteps}", flush=True)
    print(f"  Learning starts: {learning_starts}", flush=True)
    print(f"  Log: {log_path}", flush=True)

    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(os.path.join(output_dir, f"sac_{reward_name}_seed{seed}"))

    return callback.results


def main():
    parser = argparse.ArgumentParser(description="RL evaluation with learned reward models")
    parser.add_argument(
        "--reward-model-path",
        type=str,
        default=None,
        help="Path to reward model checkpoint. If not set, uses sparse reward.",
    )
    parser.add_argument(
        "--reward-model-name",
        type=str,
        default="sparse",
        help="Name for this reward model (used in logs), e.g. exp_a_pure_bt, exp_c_entropy",
    )
    parser.add_argument("--sparse", action="store_true", help="Use sparse reward only (ignores --reward-model-path)")
    parser.add_argument("--task-suite", type=str, default="libero_90")
    parser.add_argument(
        "--task-id",
        type=int,
        default=28,
        help="LIBERO task id. Paper uses 28 (close top drawer) and 33 (close microwave).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--eval-freq", type=int, default=5000)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./logs/rl_eval",
        help="Directory for RL training logs and saved models",
    )
    parser.add_argument(
        "--learning-starts",
        type=int,
        default=5000,
        help="Number of random steps before SAC starts training",
    )
    parser.add_argument(
        "--reward-freq",
        type=int,
        default=1,
        help="Compute VLM reward every N steps (reuse last reward in between). "
             "Set to 20 for ~20x speedup.",
    )

    args = parser.parse_args()

    if args.sparse:
        args.reward_model_path = None
        args.reward_model_name = "sparse"

    env, task_name = build_env(
        task_suite=args.task_suite,
        task_id=args.task_id,
        seed=args.seed,
        reward_model_path=args.reward_model_path,
        device=args.device,
        max_frames=args.max_frames,
        reward_freq=args.reward_freq,
    )

    results = train_sac(
        env=env,
        task_name=task_name,
        reward_name=args.reward_model_name,
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        eval_freq=args.eval_freq,
        output_dir=args.output_dir,
        learning_starts=args.learning_starts,
    )

    env.close()
    print(f"\nDone. Final success rate: {results[-1]['success_rate']:.3f}", flush=True)


if __name__ == "__main__":
    main()

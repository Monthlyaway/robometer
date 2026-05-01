"""
Environment wrappers for LIBERO + Robometer RL training.

DINOv2FeatureWrapper: converts image observations into compact feature vectors.
make_libero_env(): factory that chains LIBERO -> GymAdapter -> RewardModel -> DINO -> RecordEpisodeStatistics.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import gymnasium as gym

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class DINOv2FeatureWrapper(gym.ObservationWrapper):
    """
    Extracts DINOv2-ViT-S/14 CLS features from ``agentview_image`` and
    concatenates them with low-dimensional proprioception, producing a flat
    ``Box(obs_dim,)`` observation suitable for MLP-based SAC.

    Proprioception = ``robot0_joint_pos`` (7) + ``robot0_gripper_qpos`` (2) = 9-dim.
    DINOv2 CLS token = 384-dim.  Total = 393.
    """

    PROPRIO_KEYS = ("robot0_joint_pos", "robot0_gripper_qpos")
    IMAGE_KEY = "agentview_image"
    DINO_DIM = 384
    PROPRIO_DIM = 9  # 7 + 2

    def __init__(self, env: gym.Env, device: str = "cuda"):
        super().__init__(env)
        self.device = torch.device(device)

        # --- Load DINOv2 from local torch-hub cache ---
        hub_repo = os.path.expanduser(
            "~/.cache/torch/hub/facebookresearch_dinov2_main"
        )
        sys.path.insert(0, hub_repo)
        from hubconf import dinov2_vits14  # type: ignore

        self._dino: torch.nn.Module = dinov2_vits14(pretrained=True)
        self._dino = self._dino.to(self.device).eval()
        for p in self._dino.parameters():
            p.requires_grad_(False)

        obs_dim = self.DINO_DIM + self.PROPRIO_DIM
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # ImageNet normalisation constants
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_image(self, img: np.ndarray) -> np.ndarray:
        """img: (H, W, 3) uint8 -> (384,) float32 features."""
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
        t = torch.nn.functional.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
        t = (t - self._mean) / self._std
        feat = self._dino(t)  # (1, 384)
        return feat.squeeze(0).cpu().numpy()

    def _get_proprio(self, obs: Dict[str, Any]) -> np.ndarray:
        parts = [np.asarray(obs[k], dtype=np.float32) for k in self.PROPRIO_KEYS]
        return np.concatenate(parts)

    def observation(self, obs):
        dino_feat = self._encode_image(obs[self.IMAGE_KEY])
        proprio = self._get_proprio(obs)
        return np.concatenate([dino_feat, proprio]).astype(np.float32)


class SparseRewardWrapper(gym.Env):
    """Gymnasium wrapper around raw LIBERO (robosuite, old gym API).

    Converts old-style gym (obs, rew, done, info) to gymnasium 5-tuple and
    provides a sparse reward: ``-1`` per step, ``0`` on the terminal success step.
    Manually constructs action_space / observation_space since robosuite envs
    don't expose them.
    """

    def __init__(self, env, time_limit: int = 400):
        super().__init__()
        self.env = env
        self.time_limit = time_limit
        self._step_count = 0

        act_dim = env.robots[0].action_dim
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
        )
        # observation_space will be overridden by DINOv2FeatureWrapper
        self.observation_space = gym.spaces.Dict({})

    def reset(self, *, seed=None, options=None):
        self._step_count = 0
        if seed is not None:
            self.env.seed(seed)
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        return obs, {}

    def step(self, action):
        result = self.env.step(action)
        obs, _reward, done, info = result[0], result[1], result[2], result[3]
        self._step_count += 1
        terminated = done
        truncated = (self._step_count >= self.time_limit) if self.time_limit else False
        if "success" not in info:
            info["success"] = terminated
        sparse_reward = 0.0 if info.get("success", False) else -1.0
        return obs, sparse_reward, terminated, truncated, info

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)


def make_libero_env(
    task_suite_name: str = "libero_90",
    task_id: int = 28,
    model_path: str = "/root/autodl-tmp/robometer/logs/exp_c_dirfix_v1/exp_c_dirfix_v1/checkpoint-1000",
    device: str = "cuda",
    seed: int = 42,
    time_limit: int = 400,
    camera_size: int = 256,
    max_frames: Optional[int] = None,
    no_reward_model: bool = False,
) -> gym.Env:
    """
    Build the full LIBERO environment with optional reward model and DINOv2 features.

    Pipeline (dense reward)::

        OffScreenRenderEnv
          -> LiberoRobometerRewardWrapper (VLM PBRS reward, internally wraps with GymToGymnasiumWrapper)
          -> DINOv2FeatureWrapper (image -> 393-dim flat obs)
          -> RecordEpisodeStatistics

    Pipeline (sparse reward, ``no_reward_model=True``)::

        OffScreenRenderEnv
          -> SparseRewardWrapper (GymToGymnasiumWrapper + sparse -1/step reward)
          -> DINOv2FeatureWrapper (image -> 393-dim flat obs)
          -> RecordEpisodeStatistics
    """
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero import benchmark, get_libero_path

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)
    bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )

    base_env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=camera_size,
        camera_widths=camera_size,
    )
    base_env.seed(seed)

    if no_reward_model:
        reward_env = SparseRewardWrapper(base_env, time_limit=time_limit)
    else:
        from scripts.example_libero_robometer_wrapper import (
            LiberoRobometerRewardWrapper,
        )
        import scripts.example_libero_robometer_wrapper as _wrapper_mod

        def _extract_rewards_unclamped(outputs):
            if outputs.get("outputs_progress") is None:
                raise ValueError("No progress outputs found")
            progress_pred = outputs["outputs_progress"].get("progress_pred", [])
            rewards = []
            for progress_list in progress_pred:
                if isinstance(progress_list, list) and len(progress_list) > 0:
                    rewards.append(float(progress_list[-1]))
                else:
                    rewards.append(0.0)
            return np.array(rewards, dtype=np.float32)

        _wrapper_mod.extract_rewards_from_output = _extract_rewards_unclamped

        reward_env = LiberoRobometerRewardWrapper(
            base_env,
            model_path=model_path,
            device=device,
            reward_relabeling_keys=["agentview_image"],
            use_relative_rewards=True,
            add_estimated_reward=True,
            max_frames=max_frames,
        )

    dino_env = DINOv2FeatureWrapper(reward_env, device=device)

    env = gym.wrappers.RecordEpisodeStatistics(dino_env)
    return env

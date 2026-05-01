"""
CleanRL-style SAC for LIBERO with Robometer PBRS reward.

Single-file implementation following https://github.com/vwxyzjn/cleanrl
Uses DINOv2 image features + proprioception as flat observation.
"""

from __future__ import annotations

import logging
import os
import random
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional

# ---- Silence noisy third-party loggers BEFORE any imports touch them ----
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["ROBOMETER_LOG_LEVEL"] = "ERROR"
os.environ["PYTHONUNBUFFERED"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

for _noisy in (
    "robometer", "robometer.evals", "robometer.evals.eval_server",
    "robometer.utils", "robometer.utils.save", "robometer.utils.setup_utils",
    "transformers", "accelerate", "unsloth", "tensorflow", "absl",
):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import tyro


# ============================================================
# CLI Arguments
# ============================================================

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__).rstrip(".py")
    """experiment name"""
    seed: int = 42
    """random seed"""
    torch_deterministic: bool = True
    """torch.backends.cudnn.deterministic"""
    cuda: bool = True
    """use CUDA"""
    track: bool = False
    """use wandb logging"""
    wandb_project_name: str = "robometer-rl"
    """wandb project"""
    wandb_entity: Optional[str] = None
    """wandb entity"""

    # --- Environment ---
    task_suite_name: str = "libero_90"
    """LIBERO task suite"""
    task_id: int = 28
    """LIBERO task index (28 = close top drawer)"""
    model_path: str = "/root/autodl-tmp/robometer/logs/exp_c_dirfix_v1/exp_c_dirfix_v1/checkpoint-1000"
    """path to trained reward model checkpoint"""
    time_limit: int = 400
    """max steps per episode"""
    camera_size: int = 256
    """LIBERO camera resolution"""
    no_reward_model: bool = False
    """skip VLM reward model; use only sparse env reward (0 on success, -1/step)"""

    # --- SAC hyperparameters ---
    total_timesteps: int = 100_000
    """total training steps"""
    buffer_size: int = 200_000
    """replay buffer capacity"""
    gamma: float = 0.99
    """discount factor"""
    tau: float = 0.005
    """target network EMA rate"""
    batch_size: int = 256
    """minibatch size"""
    learning_starts: int = 5_000
    """random exploration before training"""
    policy_lr: float = 3e-4
    """actor learning rate"""
    q_lr: float = 3e-4
    """critic learning rate"""
    alpha_lr: float = 3e-4
    """entropy coefficient learning rate"""
    autotune: bool = True
    """auto-tune entropy coefficient alpha"""
    alpha: float = 0.2
    """fixed entropy coefficient (used when autotune=False)"""

    # --- Network ---
    hidden_dim: int = 256
    """MLP hidden layer size"""

    # --- Evaluation ---
    eval_freq: int = 5_000
    """evaluate every N steps"""
    n_eval_episodes: int = 25
    """episodes per evaluation"""

    # --- Logging ---
    log_freq: int = 1_000
    """log training stats every N steps"""
    save_freq: int = 25_000
    """save checkpoint every N steps"""
    run_name: str = ""
    """auto-generated run name (set at runtime)"""


# ============================================================
# Replay Buffer
# ============================================================

class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, max_size: int, device: torch.device):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = device

        self.obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, act_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size,), dtype=np.float32)
        self.dones = np.zeros((max_size,), dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.obs[idx]).to(self.device),
            torch.from_numpy(self.actions[idx]).to(self.device),
            torch.from_numpy(self.rewards[idx]).to(self.device),
            torch.from_numpy(self.next_obs[idx]).to(self.device),
            torch.from_numpy(self.dones[idx]).to(self.device),
        )


# ============================================================
# Networks
# ============================================================

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, act_dim)
        self.log_std_head = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs):
        h = self.net(obs)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def get_action(self, obs):
        mean, log_std = self(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        # Enforcing action bound: log_prob with tanh squashing correction
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        mean_action = torch.tanh(mean)
        return action, log_prob, mean_action


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], dim=-1))


# ============================================================
# Evaluation
# ============================================================

def evaluate(env_fn, actor, device, n_episodes: int = 25):
    """
    Run *n_episodes* rollouts and return (mean_return, mean_length, success_rate).
    """
    returns, lengths, successes = [], [], []
    env = env_fn()
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_ret, ep_len, ep_success = 0.0, 0, False
        while not done:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action, _, _ = actor.get_action(obs_t)
                action = action.squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += reward
            ep_len += 1
            if info.get("success", False):
                ep_success = True
            done = terminated or truncated
        returns.append(ep_ret)
        lengths.append(ep_len)
        successes.append(float(ep_success))
    env.close()
    return np.mean(returns), np.mean(lengths), np.mean(successes)


# ============================================================
# Main
# ============================================================

def main():
    args = tyro.cli(Args)

    tag = "sparse" if args.no_reward_model else args.exp_name
    run_name = f"{args.task_suite_name}_t{args.task_id}__{tag}__{args.seed}__{int(time.time())}"
    args.run_name = run_name

    # --- Seeding ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"[SAC] device={device}, seed={args.seed}")

    # --- Logging ---
    from torch.utils.tensorboard import SummaryWriter

    log_dir = os.path.join("runs", run_name)
    writer = SummaryWriter(log_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n" + "\n".join(f"|{k}|{v}|" for k, v in vars(args).items()),
    )

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
            save_code=True,
        )

    # --- Environment ---
    from robometer.rl.wrappers import make_libero_env

    def make_env(seed: int):
        def _thunk():
            return make_libero_env(
                task_suite_name=args.task_suite_name,
                task_id=args.task_id,
                model_path=args.model_path,
                device=str(device),
                seed=seed,
                time_limit=args.time_limit,
                camera_size=args.camera_size,
                no_reward_model=args.no_reward_model,
            )
        return _thunk

    mode_str = "sparse-only (no VLM)" if args.no_reward_model else "VLM PBRS reward"
    print(f"[SAC] creating environment ({mode_str} + DINOv2) ...", flush=True)
    env = make_env(args.seed)()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_high = float(env.action_space.high[0])
    print(f"[SAC] env ready. obs_dim={obs_dim}, act_dim={act_dim}, act_high={act_high}", flush=True)
    print(f"[SAC] total_timesteps={args.total_timesteps}, learning_starts={args.learning_starts}", flush=True)
    print(f"[SAC] batch_size={args.batch_size}, gamma={args.gamma}, tau={args.tau}", flush=True)

    # --- Networks ---
    actor = Actor(obs_dim, act_dim, args.hidden_dim).to(device)
    qf1 = QNetwork(obs_dim, act_dim, args.hidden_dim).to(device)
    qf2 = QNetwork(obs_dim, act_dim, args.hidden_dim).to(device)
    qf1_target = QNetwork(obs_dim, act_dim, args.hidden_dim).to(device)
    qf2_target = QNetwork(obs_dim, act_dim, args.hidden_dim).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    actor_optimizer = optim.Adam(actor.parameters(), lr=args.policy_lr)
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr
    )

    # --- Auto-tune alpha ---
    if args.autotune:
        target_entropy = -float(act_dim)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        alpha_optimizer = optim.Adam([log_alpha], lr=args.alpha_lr)
    else:
        alpha = args.alpha

    # --- Replay buffer ---
    rb = ReplayBuffer(obs_dim, act_dim, args.buffer_size, device)

    # --- Training loop ---
    obs, _ = env.reset()
    episode_count = 0
    start_time = time.time()
    episode_return = 0.0
    episode_length = 0

    for global_step in range(1, args.total_timesteps + 1):
        # Action selection
        if global_step <= args.learning_starts:
            action = np.random.uniform(-act_high, act_high, size=(act_dim,)).astype(np.float32)
        else:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action, _, _ = actor.get_action(obs_t)
                action = action.squeeze(0).cpu().numpy()

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        rb.add(obs, action, reward, next_obs, float(terminated))

        episode_return += reward
        episode_length += 1

        obs = next_obs

        # Periodic progress (prints regardless of episode boundaries)
        if global_step % args.log_freq == 0:
            elapsed = time.time() - start_time
            sps = global_step / elapsed
            print(
                f"[step {global_step}/{args.total_timesteps}] "
                f"ep={episode_count} ep_ret={episode_return:.2f} "
                f"ep_len={episode_length} reward={reward:.4f} "
                f"SPS={sps:.2f} elapsed={elapsed:.0f}s"
            )

        if done:
            episode_count += 1
            success = info.get("success", False)
            writer.add_scalar("train/episode_return", episode_return, global_step)
            writer.add_scalar("train/episode_length", episode_length, global_step)
            writer.add_scalar("train/episode_success", float(success), global_step)
            sps = global_step / (time.time() - start_time)
            print(
                f"[episode {episode_count} done @ step {global_step}] "
                f"ret={episode_return:.2f} len={episode_length} "
                f"success={int(success)} SPS={sps:.2f}"
            )
            obs, _ = env.reset()
            episode_return = 0.0
            episode_length = 0

        # --- SAC update ---
        if global_step > args.learning_starts:
            s_obs, s_act, s_rew, s_next_obs, s_done = rb.sample(args.batch_size)

            # --- Critic loss ---
            with torch.no_grad():
                next_action, next_log_prob, _ = actor.get_action(s_next_obs)
                q1_next = qf1_target(s_next_obs, next_action)
                q2_next = qf2_target(s_next_obs, next_action)
                min_q_next = torch.min(q1_next, q2_next) - alpha * next_log_prob
                target_q = s_rew.unsqueeze(-1) + (1.0 - s_done.unsqueeze(-1)) * args.gamma * min_q_next

            q1_val = qf1(s_obs, s_act)
            q2_val = qf2(s_obs, s_act)
            qf_loss = F.mse_loss(q1_val, target_q) + F.mse_loss(q2_val, target_q)

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            # --- Actor loss ---
            new_action, log_prob, _ = actor.get_action(s_obs)
            q1_new = qf1(s_obs, new_action)
            q2_new = qf2(s_obs, new_action)
            min_q_new = torch.min(q1_new, q2_new)
            actor_loss = (alpha * log_prob - min_q_new).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # --- Alpha loss ---
            if args.autotune:
                with torch.no_grad():
                    _, new_log_prob, _ = actor.get_action(s_obs)
                alpha_loss = (-log_alpha * (new_log_prob + target_entropy)).mean()
                alpha_optimizer.zero_grad()
                alpha_loss.backward()
                alpha_optimizer.step()
                alpha = log_alpha.exp().item()

            # --- Target update ---
            for p, p_tgt in zip(qf1.parameters(), qf1_target.parameters()):
                p_tgt.data.mul_(1 - args.tau)
                p_tgt.data.add_(args.tau * p.data)
            for p, p_tgt in zip(qf2.parameters(), qf2_target.parameters()):
                p_tgt.data.mul_(1 - args.tau)
                p_tgt.data.add_(args.tau * p.data)

            # --- Logging ---
            if global_step % args.log_freq == 0:
                writer.add_scalar("losses/qf_loss", qf_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
                sps = global_step / (time.time() - start_time)
                writer.add_scalar("charts/SPS", sps, global_step)

        # --- Evaluation ---
        if global_step % args.eval_freq == 0:
            eval_seed = args.seed + 10000
            eval_ret, eval_len, eval_suc = evaluate(
                make_env(eval_seed), actor, device, args.n_eval_episodes
            )
            writer.add_scalar("eval/mean_return", eval_ret, global_step)
            writer.add_scalar("eval/mean_length", eval_len, global_step)
            writer.add_scalar("eval/success_rate", eval_suc, global_step)
            print(
                f"[eval @ {global_step}] return={eval_ret:.2f} "
                f"len={eval_len:.0f} success={eval_suc:.2%}"
            )

        # --- Save checkpoint ---
        if global_step % args.save_freq == 0:
            ckpt_dir = os.path.join(log_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt = {
                "global_step": global_step,
                "actor": actor.state_dict(),
                "qf1": qf1.state_dict(),
                "qf2": qf2.state_dict(),
                "qf1_target": qf1_target.state_dict(),
                "qf2_target": qf2_target.state_dict(),
                "actor_optimizer": actor_optimizer.state_dict(),
                "q_optimizer": q_optimizer.state_dict(),
            }
            if args.autotune:
                ckpt["log_alpha"] = log_alpha.detach().cpu()
                ckpt["alpha_optimizer"] = alpha_optimizer.state_dict()
            path = os.path.join(ckpt_dir, f"step_{global_step}.pt")
            torch.save(ckpt, path)
            print(f"[ckpt] saved {path}")

    env.close()
    writer.close()
    if args.track:
        wandb.finish()
    print("[SAC] training complete.")


if __name__ == "__main__":
    main()

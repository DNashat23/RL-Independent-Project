"""
PPO training on GridHealthEnv with learning-curve plotting and
baseline comparison.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — saves to file without needing a display
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Make sure the env module is importable ───────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from binded_scheme import GridHealthEnv

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.wrappers import TimeLimit

# ── Callback: record ep_rew_mean & ep_len_mean at each rollout ───────────────
class RolloutLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.timesteps   = []
        self.ep_rew_mean = []
        self.ep_len_mean = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> bool:
        log = self.model.ep_info_buffer
        if len(log) == 0:
            return True
        rews = [ep["r"] for ep in log]
        lens = [ep["l"] for ep in log]
        self.timesteps.append(self.num_timesteps)
        self.ep_rew_mean.append(float(np.mean(rews)))
        self.ep_len_mean.append(float(np.mean(lens)))
        return True


# ── Training ─────────────────────────────────────────────────────────────────
def train():
    print("=" * 60)
    print("  Training PPO on GridHealthEnv — 200 000 timesteps")
    print("=" * 60)

    # vec_env = make_vec_env(GridHealthEnv, n_envs=4)
    
    vec_env = make_vec_env(
        lambda: TimeLimit(GridHealthEnv(), max_episode_steps=300),
        n_envs=4
        )
    

    model = PPO(
        policy        = "MlpPolicy",
        env           = vec_env,
        n_steps       = 2048,
        batch_size    = 64,
        learning_rate = 3e-4,
        ent_coef      = 0.01,
        verbose       = 1,
    )

    logger = RolloutLogger()
    model.learn(total_timesteps=500_000, callback=logger, progress_bar=False)

    model.save(os.path.join(os.path.dirname(__file__), "ppo_gridhealthenv"))
    print("\nModel saved → ppo_gridhealthenv.zip")
    return model, logger


# ── Learning-curve plot ───────────────────────────────────────────────────────
def plot_learning_curve(logger: RolloutLogger, save_path: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    fig.suptitle("PPO — GridHealthEnv Learning Curves", fontsize=14, fontweight="bold")

    ts = np.array(logger.timesteps)

    # ep_rew_mean
    ax1.plot(ts, logger.ep_rew_mean, color="royalblue", linewidth=1.4, label="ep_rew_mean")
    ax1.set_ylabel("Mean Episode Reward")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.35)
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # ep_len_mean
    ax2.plot(ts, logger.ep_len_mean, color="darkorange", linewidth=1.4, label="ep_len_mean")
    ax2.set_xlabel("Timesteps")
    ax2.set_ylabel("Mean Episode Length")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.35)
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nLearning curve saved → {save_path}")


# ── Evaluation helpers ────────────────────────────────────────────────────────
def evaluate_agent(model, n_episodes: int = 20):
    """Return (mean_reward, std_reward, rewards_list, lengths_list)."""
    env = TimeLimit(GridHealthEnv(), max_episode_steps=300)
    rewards, lengths = [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done, ep_rew, ep_len = False, 0.0, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(action)
            ep_rew += r
            ep_len += 1
            done = terminated or truncated
        rewards.append(ep_rew)
        lengths.append(ep_len)
    env.close()
    return rewards, lengths


def evaluate_random(n_episodes: int = 20):
    """Same loop but sample actions uniformly at random."""
    env = TimeLimit(GridHealthEnv(), max_episode_steps=300)
    rewards, lengths = [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done, ep_rew, ep_len = False, 0.0, 0
        while not done:
            action = env.action_space.sample()
            obs, r, terminated, truncated, _ = env.step(action)
            ep_rew += r
            ep_len += 1
            done = terminated or truncated
        rewards.append(ep_rew)
        lengths.append(ep_len)
    env.close()
    return rewards, lengths


# ── Comparison plot ───────────────────────────────────────────────────────────
def plot_comparison(ppo_rews, rand_rews, ppo_lens, rand_lens, save_path: str):
    episodes = np.arange(1, len(ppo_rews) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("PPO vs Random Baseline — 20-Episode Evaluation",
                 fontsize=13, fontweight="bold")

    # Reward per episode
    ax1.plot(episodes, ppo_rews,  marker="o", color="royalblue",  label="PPO",    linewidth=1.4)
    ax1.plot(episodes, rand_rews, marker="s", color="tomato",     label="Random", linewidth=1.4)
    ax1.axhline(np.mean(ppo_rews),  color="royalblue", linestyle="--", alpha=0.7,
                label=f"PPO mean = {np.mean(ppo_rews):.2f}")
    ax1.axhline(np.mean(rand_rews), color="tomato",    linestyle="--", alpha=0.7,
                label=f"Random mean = {np.mean(rand_rews):.2f}")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.set_title("Episode Reward")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.35)

    # Episode length
    ax2.plot(episodes, ppo_lens,  marker="o", color="royalblue",  label="PPO",    linewidth=1.4)
    ax2.plot(episodes, rand_lens, marker="s", color="tomato",     label="Random", linewidth=1.4)
    ax2.axhline(np.mean(ppo_lens),  color="royalblue", linestyle="--", alpha=0.7,
                label=f"PPO mean = {np.mean(ppo_lens):.1f}")
    ax2.axhline(np.mean(rand_lens), color="tomato",    linestyle="--", alpha=0.7,
                label=f"Random mean = {np.mean(rand_lens):.1f}")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps")
    ax2.set_title("Episode Length")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.35)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Comparison plot saved → {save_path}")


# ── Summary table ─────────────────────────────────────────────────────────────
def print_summary(ppo_rews, rand_rews, ppo_lens, rand_lens):
    def stats(arr):
        return (np.mean(arr), np.std(arr), np.min(arr), np.max(arr))

    pr, ps, pmin, pmax = stats(ppo_rews)
    rr, rs, rmin, rmax = stats(rand_rews)
    pl, _,  _,    _    = stats(ppo_lens)
    rl, _,  _,    _    = stats(rand_lens)

    print("\n" + "=" * 60)
    print("  150-Episode Evaluation Summary")
    print("=" * 60)
    print(f"{'Metric':<28} {'PPO':>12} {'Random':>12}")
    print("-" * 54)
    print(f"{'Mean reward':<28} {pr:>12.2f} {rr:>12.2f}")
    print(f"{'Std  reward':<28} {ps:>12.2f} {rs:>12.2f}")
    print(f"{'Min  reward':<28} {pmin:>12.2f} {rmin:>12.2f}")
    print(f"{'Max  reward':<28} {pmax:>12.2f} {rmax:>12.2f}")
    print(f"{'Mean episode length':<28} {pl:>12.1f} {rl:>12.1f}")
    print("=" * 60)

    delta = pr - rr
    sign  = "+" if delta >= 0 else ""
    print(f"\n  PPO vs Random reward delta: {sign}{delta:.2f}")
    if delta > 0:
        print("  Result: PPO outperforms the random baseline.")
    else:
        print("  Result: PPO does not yet outperform the random baseline.")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)

    # 1. Train
    model, logger = train()

    # 2. Learning curve
    plot_learning_curve(
        logger,
        save_path=os.path.join(base_dir, "learning_curve.png"),
    )

    # 3. Evaluate
    print("\nEvaluating trained PPO agent (150 episodes, deterministic)…")
    ppo_rews,  ppo_lens  = evaluate_agent(model, n_episodes=150)

    print("Evaluating random baseline (150 episodes)…")
    rand_rews, rand_lens = evaluate_random(n_episodes=150)

    # 4. Comparison plot
    plot_comparison(
        ppo_rews, rand_rews, ppo_lens, rand_lens,
        save_path=os.path.join(base_dir, "eval_comparison.png"),
    )

    # 5. Summary table
    print_summary(ppo_rews, rand_rews, ppo_lens, rand_lens)

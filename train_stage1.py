"""
Stage 1 Curriculum Training — 3 healthy cells, no unhealthy cells, static board.

The agent learns to find and collect all 3 healthy cells efficiently.
Board is randomized each episode but fixed within it (no respawning).

Promotion requires:
  - ep_rew_mean >= PROMOTION_THRESHOLD (collecting ~all 3 cells on average)
  - Your visual confirmation via heatmap + replay

Usage:
    python train_stage1.py
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from binded_scheme import GridHealthEnv

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import TimeLimit

# ── Configuration ─────────────────────────────────────────────────────────────
TOTAL_TIMESTEPS     = 500_000
N_ENVS              = 4
MAX_EPISODE_STEPS   = 200
PROMOTION_THRESHOLD = 2.5        # near 3.0 = collecting ~all 3 cells per episode
MODEL_SAVE_PATH     = "ppo_stage1"

PPO_KWARGS = dict(
    policy        = "MlpPolicy",
    n_steps       = 2048,
    batch_size    = 64,
    learning_rate = 3e-4,
    ent_coef      = 0.01,
    verbose       = 1,
)


# ── Stage 1 environment ───────────────────────────────────────────────────────
class Stage1Env(GridHealthEnv):
    """3 healthy cells, no unhealthy cells. Static board within episode."""
    N_HEALTHY   = 3
    N_UNHEALTHY = 0


# ── Callback ──────────────────────────────────────────────────────────────────
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
        self.timesteps.append(self.num_timesteps)
        self.ep_rew_mean.append(float(np.mean([ep["r"] for ep in log])))
        self.ep_len_mean.append(float(np.mean([ep["l"] for ep in log])))
        return True


# ── Training ──────────────────────────────────────────────────────────────────
def train(base_dir):
    print("=" * 60)
    print("  STAGE 1: 3 healthy cells, static board, no unhealthy")
    print(f"  Timesteps:           {TOTAL_TIMESTEPS:,}")
    print(f"  Max steps/episode:   {MAX_EPISODE_STEPS}")
    print(f"  Promotion threshold: ep_rew_mean >= {PROMOTION_THRESHOLD}")
    print("=" * 60)

    vec_env = make_vec_env(
        lambda: TimeLimit(Stage1Env(), max_episode_steps=MAX_EPISODE_STEPS),
        n_envs=N_ENVS
    )

    model  = PPO(env=vec_env, **PPO_KWARGS)
    logger = RolloutLogger()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=logger, progress_bar=False)

    save_path = os.path.join(base_dir, MODEL_SAVE_PATH)
    model.save(save_path)
    print(f"\nModel saved → {save_path}.zip")

    return model, logger


# ── Learning curve ────────────────────────────────────────────────────────────
def plot_learning_curve(logger, save_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    fig.suptitle("Stage 1 — Learning Curves (3 Healthy Cells, Static Board)",
                 fontsize=13, fontweight="bold")
    ts = np.array(logger.timesteps)

    ax1.plot(ts, logger.ep_rew_mean, color="royalblue", linewidth=1.4)
    ax1.axhline(PROMOTION_THRESHOLD, color="green", linestyle="--", linewidth=1.2,
                label=f"Promotion threshold ({PROMOTION_THRESHOLD})")
    ax1.axhline(3.0, color="gold", linestyle=":", linewidth=1.2,
                label="Perfect score (3.0)")
    ax1.set_ylabel("Mean Episode Reward")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.35)
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    ax2.plot(ts, logger.ep_len_mean, color="darkorange", linewidth=1.4)
    ax2.set_xlabel("Timesteps")
    ax2.set_ylabel("Mean Episode Length")
    ax2.grid(True, alpha=0.35)
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Learning curve saved → {save_path}")


# ── Heatmap ───────────────────────────────────────────────────────────────────
def plot_heatmap(trajectory, save_path, label="Episode"):
    g = GridHealthEnv.GRID_SIZE
    visit_count = np.zeros((g, g), dtype=np.float32)
    for step in trajectory:
        r, c = step["pos"]
        visit_count[r, c] += 1

    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = LinearSegmentedColormap.from_list("visits", ["white", "royalblue", "navy"])
    im   = ax.imshow(visit_count, cmap=cmap, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Visit count")
    center = g // 2
    ax.scatter(center, center, marker="*", s=300, color="gold", zorder=5, label="Start")
    ax.set_title(f"Stage 1 — Heatmap — {label} ({len(trajectory)} steps)",
                 fontweight="bold")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Heatmap saved → {save_path}")


# ── Replay: every single step, 10 per row ────────────────────────────────────
def plot_replay(trajectory, save_path_prefix, label="Episode", cols=10):
    """One frame per step. 10 columns × 2 rows = 20 steps per PNG."""
    g              = GridHealthEnv.GRID_SIZE
    frames_per_png = cols * 2
    chunks = [
        trajectory[i:i+frames_per_png]
        for i in range(0, len(trajectory), frames_per_png)
    ]

    for png_idx, chunk in enumerate(chunks):
        n_rows = max(1, (len(chunk) + cols - 1) // cols)
        fig, axes = plt.subplots(n_rows, cols,
                                 figsize=(cols * 1.8, n_rows * 2.0))
        axes = np.array(axes).flatten()

        for plot_idx, step in enumerate(chunk):
            ax  = axes[plot_idx]
            rgb = np.ones((g, g, 3), dtype=np.float32) * 0.92
            rgb[step["grid"] ==  1.0] = [0.20, 0.75, 0.30]
            rgb[step["grid"] == -1.0] = [0.85, 0.20, 0.20]
            ax.imshow(rgb, interpolation="nearest")
            for x in np.arange(-0.5, g, 1):
                ax.axhline(x, color="black", linewidth=0.4, alpha=0.3)
                ax.axvline(x, color="black", linewidth=0.4, alpha=0.3)
            ar, ac = step["pos"]
            ax.add_patch(mpatches.Circle((ac, ar), 0.35, color="royalblue", zorder=5))
            step_num = png_idx * frames_per_png + plot_idx + 1
            r_str = f"+{step['reward']:.0f}" if step['reward'] > 0 else f"{step['reward']:.0f}"
            ax.set_title(f"t={step_num} hp={step['health']}\n{r_str}",
                         fontsize=6, fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])

        for ax in axes[len(chunk):]:
            ax.set_visible(False)

        fig.legend(
            handles=[
                mpatches.Patch(color=[0.20, 0.75, 0.30], label="Healthy"),
                mpatches.Patch(color=[0.92, 0.92, 0.92], label="Neutral"),
                mpatches.Patch(color="royalblue",         label="Agent"),
            ],
            loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.03)
        )
        start_t = png_idx * frames_per_png + 1
        end_t   = png_idx * frames_per_png + len(chunk)
        fig.suptitle(f"Stage 1 — {label}  (Steps {start_t}–{end_t})",
                     fontsize=11, fontweight="bold")
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        out = f"{save_path_prefix}_part{png_idx+1}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Replay [{label}] part {png_idx+1} saved → {out}")


# ── Run 50 episodes, return best and worst ────────────────────────────────────
def run_episodes(model, n_seeds=50, max_steps=200):
    all_results = []
    for seed in range(n_seeds):
        env = Stage1Env()
        obs, _ = env.reset(seed=seed)
        trajectory, total_reward, done = [], 0.0, False
        while not done and len(trajectory) < max_steps:
            grid_snap = env.grid.copy()
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(int(action))
            total_reward += r
            trajectory.append({
                "pos":    tuple(env.agent_pos),
                "grid":   grid_snap,
                "health": info["health"],
                "reward": r,
            })
            done = terminated or truncated
        env.close()
        all_results.append((total_reward, trajectory))

    # Sort by reward, tracking original seed index
    indexed = sorted(enumerate(all_results), key=lambda x: x[1][0])
    worst_seed, (worst_reward, worst_traj) = indexed[0]
    best_seed,  (best_reward,  best_traj)  = indexed[-1]

    print(f"Best  episode (seed {best_seed}):  "
          f"{len(best_traj)} steps, reward={best_reward:.2f}, "
          f"hp={best_traj[-1]['health']}")
    print(f"Worst episode (seed {worst_seed}): "
          f"{len(worst_traj)} steps, reward={worst_reward:.2f}, "
          f"hp={worst_traj[-1]['health']}")
    return best_traj, best_reward, worst_traj, worst_reward


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, n_episodes=50):
    final_healths, collected_counts = [], []
    for seed in range(n_episodes):
        env = Stage1Env()
        obs, _ = env.reset(seed=seed)
        done, steps = False, 0
        while not done and steps < MAX_EPISODE_STEPS:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            steps += 1
        final_healths.append(info["health"])
        collected_counts.append(
            Stage1Env.N_HEALTHY - int(np.sum(env.grid == 1.0))
        )
        env.close()

    print(f"\n  Evaluation over {n_episodes} episodes:")
    print(f"  Mean final health:    {np.mean(final_healths):.1f} "
          f"± {np.std(final_healths):.1f}")
    print(f"  Mean cells collected: {np.mean(collected_counts):.2f} "
          f"/ {Stage1Env.N_HEALTHY}")
    print(f"  All-collected rate:   "
          f"{np.mean(np.array(collected_counts)==Stage1Env.N_HEALTHY)*100:.0f}%")
    return final_healths, collected_counts


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Train
    model, logger = train(base_dir)

    # 2. Learning curve
    plot_learning_curve(
        logger,
        save_path=os.path.join(base_dir, "stage1_learning_curve.png")
    )

    # 3. Run 50 episodes, visualize best and worst
    print("\nRunning 50 episodes for visualization...")
    best_traj, best_rew, worst_traj, worst_rew = run_episodes(model, n_seeds=50)

    plot_heatmap(best_traj,
                 os.path.join(base_dir, "stage1_heatmap_best.png"),
                 label=f"Best (reward={best_rew:.1f})")
    plot_heatmap(worst_traj,
                 os.path.join(base_dir, "stage1_heatmap_worst.png"),
                 label=f"Worst (reward={worst_rew:.1f})")
    plot_replay(best_traj,
                os.path.join(base_dir, "stage1_replay_best"),
                label=f"Best Episode (reward={best_rew:.1f})")
    plot_replay(worst_traj,
                os.path.join(base_dir, "stage1_replay_worst"),
                label=f"Worst Episode (reward={worst_rew:.1f})")

    # 4. Evaluate
    final_healths, collected_counts = evaluate(model, n_episodes=50)

    # 5. Promotion check
    final_rew = logger.ep_rew_mean[-1] if logger.ep_rew_mean else -999
    print(f"\nFinal ep_rew_mean: {final_rew:.2f}  |  Threshold: {PROMOTION_THRESHOLD}")

    if final_rew < PROMOTION_THRESHOLD:
        print("\n✗ Threshold not met. Review visualizations to diagnose.")
        sys.exit(0)

    print("\n" + "=" * 60)
    print("  PROMOTION DECISION")
    print("=" * 60)
    print("  Please review:")
    print("    → stage1_heatmap_best.png / stage1_heatmap_worst.png")
    print("    → stage1_replay_best_part*.png")
    print("    → stage1_replay_worst_part*.png")
    print()
    while True:
        answer = input("  Advance to Stage 2? (y/n): ").strip().lower()
        if answer in ("y", "n"):
            break
        print("  Please enter y or n.")

    if answer == "y":
        print("\n✓ Promotion confirmed. Run train_stage2.py to continue.")
    else:
        print("\n✗ Promotion declined. Adjust and retrain Stage 1.")
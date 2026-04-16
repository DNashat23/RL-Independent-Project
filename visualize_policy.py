"""
Visualize a trained PPO agent's behavior on GridHealthEnv.

Produces two figures:
  1. visit_heatmap.png  — how often the agent visited each cell
  2. episode_replay.png — grid snapshots at evenly spaced steps
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(__file__))
from binded_scheme import GridHealthEnv
from stable_baselines3 import PPO


# ── Run one deterministic episode, record everything ─────────────────────────
def run_episode(model, seed=0):
    env = GridHealthEnv()
    obs, _ = env.reset(seed=seed)

    trajectory = []   # list of (agent_pos, grid_snapshot, reward, health)
    total_reward = 0.0

    done = False
    while not done:
        grid_snap  = env.grid.copy()
        action, _  = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, info = env.step(int(action))
        total_reward += r
        trajectory.append({
            "pos":    tuple(env.agent_pos),
            "grid":   grid_snap,
            "reward": r,
            "health": info["health"],
        })
        done = terminated or truncated

    env.close()
    return trajectory, total_reward


# ── Figure 1: visit heatmap ───────────────────────────────────────────────────
def plot_heatmap(trajectory, save_path):
    g = GridHealthEnv.GRID_SIZE
    visit_count = np.zeros((g, g), dtype=np.float32)
    for step in trajectory:
        r, c = step["pos"]
        visit_count[r, c] += 1

    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = LinearSegmentedColormap.from_list("visits", ["white", "royalblue", "navy"])
    im   = ax.imshow(visit_count, cmap=cmap, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Visit count")

    # Mark start
    center = g // 2
    ax.scatter(center, center, marker="*", s=300, color="gold",
               zorder=5, label="Start")

    ax.set_title(f"Agent Visit Heatmap  ({len(trajectory)} steps)", fontweight="bold")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Heatmap saved → {save_path}")


# ── Figure 2: episode replay snapshots ───────────────────────────────────────
def plot_replay(trajectory, save_path, n_snapshots=9):
    """Show n_snapshots evenly-spaced grid states side by side."""
    g       = GridHealthEnv.GRID_SIZE
    indices = np.linspace(0, len(trajectory) - 1, n_snapshots, dtype=int)

    cols = 3
    rows = (n_snapshots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    axes = axes.flatten()

    for plot_idx, step_idx in enumerate(indices):
        ax   = axes[plot_idx]
        step = trajectory[step_idx]

        # Build RGB grid
        rgb = np.ones((g, g, 3), dtype=np.float32) * 0.92  # neutral = light gray
        healthy_mask   = step["grid"] ==  1.0
        unhealthy_mask = step["grid"] == -1.0
        rgb[healthy_mask]   = [0.20, 0.75, 0.30]   # green
        rgb[unhealthy_mask] = [0.85, 0.20, 0.20]   # red

        ax.imshow(rgb, interpolation="nearest")

        # Draw grid lines
        for x in np.arange(-0.5, g, 1):
            ax.axhline(x, color="black", linewidth=0.4, alpha=0.4)
            ax.axvline(x, color="black", linewidth=0.4, alpha=0.4)

        # Agent marker
        ar, ac = step["pos"]
        circle = mpatches.Circle((ac, ar), 0.35, color="royalblue", zorder=5)
        ax.add_patch(circle)

        ax.set_title(
            f"Step {step_idx + 1}  |  hp={step['health']}",
            fontsize=8, fontweight="bold"
        )
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused subplots
    for ax in axes[n_snapshots:]:
        ax.set_visible(False)

    # Legend
    legend_elements = [
        mpatches.Patch(color=[0.20, 0.75, 0.30], label="Healthy"),
        mpatches.Patch(color=[0.85, 0.20, 0.20], label="Unhealthy"),
        mpatches.Patch(color=[0.92, 0.92, 0.92], label="Neutral"),
        mpatches.Patch(color="royalblue",         label="Agent"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("PPO Agent — Episode Replay", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Replay saved → {save_path}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    base_dir   = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "ppo_gridhealthenv")

    print("Loading model…")
    model = PPO.load(model_path)

    print("Running episode…")
    trajectory, total_reward = run_episode(model, seed=42)
    print(f"Episode finished — {len(trajectory)} steps, reward = {total_reward:.2f}")

    plot_heatmap(
        trajectory,
        save_path=os.path.join(base_dir, "visit_heatmap.png"),
    )
    plot_replay(
        trajectory,
        save_path=os.path.join(base_dir, "episode_replay.png"),
        n_snapshots=9,
    )
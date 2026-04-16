import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_stage1 import Stage1Env, plot_heatmap, plot_replay, run_episodes
from stable_baselines3 import PPO

base_dir = os.path.dirname(os.path.abspath(__file__))

print("Loading model...")
model = PPO.load(os.path.join(base_dir, "ppo_stage1"))

print("Running 50 episodes...")
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

print("Done.")
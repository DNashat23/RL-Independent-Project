import gymnasium as gym
from gymnasium import spaces
import numpy as np


class GridHealthEnv(gym.Env):
    """
    8x8 grid survival environment.

    Health and reward are DECOUPLED:
      - Reward: +1 for healthy cell, -1 for unhealthy cell, 0 otherwise
      - Health: survival counter that bleeds -1/step, restored by healthy cells

    Board is STATIC within an episode (no respawning).
    Each episode gets a new random layout.

    Observation: flattened grid (64) + normalized position (2) + normalized health (1) = 67
    Actions:     0=up, 1=down, 2=left, 3=right
    Termination: health <= 0 OR all healthy cells collected
    """

    metadata = {"render_modes": [], "render_fps": 4}

    GRID_SIZE   = 8
    N_HEALTHY   = 10
    N_UNHEALTHY = 10

    R_HEALTHY   =  1.0
    R_UNHEALTHY = -1.0

    H_INIT      = 100
    H_HEALTHY   =  15
    H_UNHEALTHY = -10
    H_NEUTRAL   =  -1

    def __init__(self):
        super().__init__()
        g = self.GRID_SIZE

        self.observation_space = spaces.Box(
            low  = np.array([-1.0] * (g * g) + [0.0, 0.0, 0.0], dtype=np.float32),
            high = np.array([ 1.0] * (g * g) + [1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

        self.grid      = None
        self.agent_pos = None
        self.health    = None

    # ------------------------------------------------------------------ #
    #  Reset                                                               #
    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        g = self.GRID_SIZE

        self.grid = np.zeros((g, g), dtype=np.float32)

        all_cells     = list(range(g * g))
        healthy_idx   = self.np_random.choice(all_cells, self.N_HEALTHY, replace=False)
        remaining     = list(set(all_cells) - set(healthy_idx))
        unhealthy_idx = self.np_random.choice(remaining, self.N_UNHEALTHY, replace=False)

        for idx in healthy_idx:
            self.grid[idx // g, idx % g] = 1.0
        for idx in unhealthy_idx:
            self.grid[idx // g, idx % g] = -1.0

        # Agent starts at center on a neutral cell
        center = g // 2
        self.agent_pos = np.array([center, center], dtype=np.int32)
        self.grid[center, center] = 0.0

        self.health = self.H_INIT

        return self._get_obs(), {}

    # ------------------------------------------------------------------ #
    #  Step                                                                #
    # ------------------------------------------------------------------ #
    def step(self, action):
        g        = self.GRID_SIZE
        row, col = self.agent_pos

        # Move (wall bumps: agent stays, pays neutral cost)
        if   action == 0 and row > 0:      row -= 1
        elif action == 1 and row < g - 1:  row += 1
        elif action == 2 and col > 0:      col -= 1
        elif action == 3 and col < g - 1:  col += 1

        self.agent_pos = np.array([row, col], dtype=np.int32)

        # Cell reward + health change
        cell_val = self.grid[row, col]
        if cell_val == 1.0:
            reward        = self.R_HEALTHY
            health_change = self.H_HEALTHY
            self.grid[row, col] = 0.0          # consume cell (no respawn)
        elif cell_val == -1.0:
            reward        = self.R_UNHEALTHY
            health_change = self.H_UNHEALTHY
            self.grid[row, col] = 0.0          # consume cell
        else:
            reward        = 0.0
            health_change = self.H_NEUTRAL

        self.health = int(np.clip(self.health + health_change, 0, 200))

        # Terminate on death OR all healthy cells collected
        all_collected = np.sum(self.grid == 1.0) == 0
        terminated    = bool(self.health <= 0 or all_collected)
        truncated     = False

        return self._get_obs(), float(reward), terminated, truncated, {
            "health":        self.health,
            "all_collected": all_collected,
        }

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #
    def _get_obs(self):
        normalized_pos    = self.agent_pos.astype(np.float32) / (self.GRID_SIZE - 1)
        normalized_health = np.array([self.health / 200.0], dtype=np.float32)
        return np.concatenate([
            self.grid.flatten(),
            normalized_pos,
            normalized_health
        ])

    def close(self):
        pass
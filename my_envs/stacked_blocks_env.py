import gym
import numpy as np
from gym import spaces
from collections import namedtuple

# Define a simple ActionSpec to mimic the expected API.
ActionSpec = namedtuple("ActionSpec", ["minimum", "maximum"])

class StackedBlocksEnv(gym.Env):
    """
    Custom environment for stacking a block on top of a fixed base block.

    The observation is a 6-dimensional vector:
      - The first 3 numbers are the current position of the movable block (block2).
      - The last 3 numbers are the target (goal) position for block2 (computed as block1 plus an offset).

    The base block (block1) is fixed. The action is a single scalar, which is broadcasted to all three dimensions.
    """
    def __init__(self, fixed_start_end=None):
        super(StackedBlocksEnv, self).__init__()
        # Define a 1-dimensional continuous action space.
        self._action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        # Define a 6-dimensional continuous observation space.
        self._observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        # Fixed base block position.
        self.block1 = np.array([0.5, 0.5, 0.3], dtype=np.float32)
        # Placeholder for the movable block (block2) and goal.
        self.block2 = None
        self.goal = None
        # Optionally set a fixed offset (used to compute the goal from block1).
        self.fixed_start_end = fixed_start_end
        self.reset()

    def reset(self):
        """
        Resets the environment:
          - block1 remains fixed.
          - block2 is initialized to a distinct starting position.
          - The goal is computed as block1 plus an offset.
        """
        self.block1 = np.array([0.5, 0.5, 0.3], dtype=np.float32)
        self.block2 = np.array([0.2, 0.2, 0.1], dtype=np.float32)
        if self.fixed_start_end is not None:
            offset = np.array(self.fixed_start_end, dtype=np.float32)
        else:
            offset = np.array([0.0, 0.0, 0.1], dtype=np.float32)
        self.goal = self.block1 + offset
        return self._get_obs()

    def step(self, action):
        """
        Applies the action to update block2.
          - The scalar action is broadcasted to a 3-dimensional update.
          - block2 is updated and clipped between 0 and 1.
          - The reward is 1 if block2 is within 0.05 units of goal, else 0.
        """
        action = np.array(action, dtype=np.float32)
        update = np.full((3,), action[0])
        self.block2 = np.clip(self.block2 + update, 0.0, 1.0)
        reward = self._compute_reward()
        done = self._check_done()
        info = {}
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        """
        Returns the observation:
          - A concatenation of block2 and goal.
        """
        return np.concatenate([self.block2.astype(np.float32), self.goal.astype(np.float32)])

    def _compute_reward(self):
        """
        Computes reward: 1 if block2 is within 0.05 units of goal, otherwise 0.
        """
        dist = np.linalg.norm(self.block2 - self.goal)
        return 1.0 if dist < 0.05 else 0.0

    def _check_done(self):
        return self._compute_reward() == 1.0

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def action_spec(self):
        return ActionSpec(minimum=self._action_space.low, maximum=self._action_space.high)
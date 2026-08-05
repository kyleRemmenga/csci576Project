"""Environment wrappers for training."""

import numpy as np
from gymnasium import ActionWrapper, spaces


class FlatAction(ActionWrapper):
    """Collapse MultiDiscrete into a single Discrete, preserving the joint mask.

    Index order matches ``ShapezBuildEnv.action_masks()``, which flattens
    ``(building, col, row, rotation)``: mask entry i corresponds
    to flat action i.
    """

    def __init__(self, env):
        super().__init__(env)
        if not isinstance(env.action_space, spaces.MultiDiscrete):
            raise TypeError(f"expected MultiDiscrete, got {type(env.action_space)}")

        self.nvec = np.asarray(env.action_space.nvec)
        self.action_space = spaces.Discrete(int(np.prod(self.nvec)))

    def action(self, action):
        return np.array(np.unravel_index(int(action), self.nvec))

    def flatten_action(self, multi_action):
        """Inverse of ``action`` - useful for scripted policies and imitation data."""
        return int(np.ravel_multi_index(tuple(int(v) for v in multi_action), self.nvec))

    def action_masks(self):
        """Joint mask, one entry per flat action."""
        mask = self.env.action_masks()
        if mask.shape != (self.action_space.n,):
            raise ValueError(
                f"mask length {mask.shape} does not match flat action space "
                f"{self.action_space.n}; env.action_masks() and the action space "
                "have diverged"
            )
        return mask

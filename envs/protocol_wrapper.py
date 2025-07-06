import gymnasium as gym
import numpy as np

class ProtocolWrapper(gym.Wrapper):
    def __init__(self, env, protocol):
        super().__init__(env)
        self.gravity = protocol.get("gravity", 9.8)
        self.mass_scale = protocol.get("mass_scale", 1.0)
        self.friction = protocol.get("friction", 1.0)
        self._apply_protocol()

    def _apply_protocol(self):
        if hasattr(self.env, "model"):
            # Gravity
            self.env.model.opt.gravity[:] = [0, 0, -self.gravity]

            # Mass scaling
            for i in range(len(self.env.model.body_mass)):
                self.env.model.body_mass[i] *= self.mass_scale

            # Friction scaling
            for i in range(len(self.env.model.geom_friction)):
                self.env.model.geom_friction[i] *= self.friction

    def reset(self, **kwargs):
        return super().reset(**kwargs)

    def step(self, action):
        return super().step(action)
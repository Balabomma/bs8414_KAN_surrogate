"""KAN-Attention-LSTM v7 — v6 + hard ambient output clamp.

Parent: model_v6 (physics-causal inputs: plume q(t)^(2/3), Q(t), anchor bump).
One change: the decoded output is hard-clamped at ambient (18 degC) in the
*scaled* space, i.e. out = max(out, (T_amb - mean)/scale) per sensor.

Why: v6 run2 physics validation flagged sub-ambient excursions to -69/-75 degC
on 4% of masked points (validate_physics_v6.py). Ground-truth facade temps are
never sub-ambient, so a soft penalty was leaving a real physics violation. A
hard floor enforces T >= T_amb exactly and zeros the gradient below it, so the
network stops spending capacity on the physically impossible region and can fit
the active range. Post-hoc clamp on v6 gave test R2 0.8921 -> 0.8932 (marginal,
within variance); training with the clamp is expected to match or slightly beat
that while removing the FLAG. Physics-causal channels are unchanged from v6.

The clamp threshold lives in a registered buffer `ambient_scaled` (per sensor),
set via set_ambient() before training and restored with the state_dict at eval.
"""
import torch

from config import N_SENSORS
from model_v6 import KANAttentionLSTMv6
from model import count_parameters

T_AMBIENT = 18.0


class KANAttentionLSTMv7(KANAttentionLSTMv6):
    def __init__(self, n_timesteps=181, **kw):
        super().__init__(n_timesteps=n_timesteps, **kw)
        n_sensors = kw.get("n_sensors", N_SENSORS)
        # default 0.0 -> harmless until set_ambient() / state_dict load overrides it
        self.register_buffer("ambient_scaled", torch.zeros(n_sensors))
        self._ambient_set = False

    def set_ambient(self, mean, scale, t_amb=T_AMBIENT):
        """Set the per-sensor ambient floor in scaled space from the fitted scaler."""
        mean_t = torch.as_tensor(mean, dtype=torch.float32)
        scale_t = torch.as_tensor(scale, dtype=torch.float32)
        amb = (t_amb - mean_t) / scale_t
        self.ambient_scaled.copy_(amb.to(self.ambient_scaled.device))
        self._ambient_set = True

    def forward(self, params, time_array, anchor):
        out, attn_w = super().forward(params, time_array, anchor)
        # hard floor at ambient (per-sensor, broadcast over B, T)
        out = torch.maximum(out, self.ambient_scaled)
        return out, attn_w


if __name__ == "__main__":
    import numpy as np
    from features_v6 import N_V6_FEATURES
    model = KANAttentionLSTMv7(n_continuous=2 + 13 + N_V6_FEATURES)
    model.set_ambient(np.full(N_SENSORS, 200.0, np.float32),
                      np.full(N_SENSORS, 150.0, np.float32))
    print(f"Params: {count_parameters(model):,}")
    p = torch.rand(4, 16 + N_V6_FEATURES); p[:, 0] = torch.randint(0, 4, (4,)).float()
    out, _ = model(p, torch.linspace(0, 1, 181), torch.rand(4) * 0.5 + 0.3)
    print(f"Output: {out.shape}  min(scaled)={out.min().item():.3f} "
          f"floor={model.ambient_scaled.min().item():.3f}")

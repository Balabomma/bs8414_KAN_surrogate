"""KAN-Attention-LSTM v5 — v4 + causal physics driver channels.

Adds to model_v4 two physically causal per-timestep input channels derived
from the FDS fire ramp (identical protocol across all BS8414 sims):

    q(t) = HRR_MW * F_ramp(t)        instantaneous burner HRR in MW
                                     (HRRPUA x burner surface area / 1000)
    Q(t) = int_0^t q dt (normalised) cumulative delivered energy

so the network sees the actual boundary condition driving the thermal
response instead of inferring it from positional encodings.
"""
import numpy as np
import torch

from model_v4 import KANAttentionLSTMv4
from model import count_parameters
from features_v4 import V4_FEATURES, HRR_MW_SCALE

# column of the normalised hrr_mw feature in the (B, 16+K) params tensor
HRR_MW_COL = 16 + V4_FEATURES.index("hrr_mw")

# Fire_RAMP_Q knots common to every simulation (seconds, fraction)
RAMP_KNOTS_T = [0.0, 300.0, 600.0, 706.31, 720.0, 1380.0, 1500.0, 1800.0]
RAMP_KNOTS_F = [0.0, 0.571, 0.95, 0.971, 1.0, 1.0, 0.571, 0.0]
T_END = 1800.0


def ramp_curve(n_timesteps=181):
    t = np.linspace(0.0, T_END, n_timesteps)
    return np.interp(t, RAMP_KNOTS_T, RAMP_KNOTS_F).astype(np.float32)


class KANAttentionLSTMv5(KANAttentionLSTMv4):
    N_TIME_FEATS = 4  # delta, bump, q(t), Q(t)

    def __init__(self, n_timesteps=181, **kw):
        super().__init__(**kw)
        f = torch.from_numpy(ramp_curve(n_timesteps))
        self.register_buffer("ramp_f", f)
        cum = torch.cumsum(f, dim=0)
        self.register_buffer("ramp_cum", cum / cum[-1])

    def forward(self, params, time_array, anchor):
        B, T = params.shape[0], len(time_array)

        param_embed = self.param_encoder(params, anchor)
        time_embed = self.time_encoding(time_array)

        delta = time_array.unsqueeze(0) - anchor.unsqueeze(1)
        bump = torch.exp(-(delta / 0.08) ** 2)

        # physical burner HRR in MW (denormalise the hrr_mw feature)
        hrr_mw = params[:, HRR_MW_COL:HRR_MW_COL + 1] * HRR_MW_SCALE  # (B, 1)
        q = hrr_mw * self.ramp_f.unsqueeze(0)      # (B, T) instantaneous MW
        Q = hrr_mw * self.ramp_cum.unsqueeze(0)    # (B, T) cumulative energy

        combined = torch.cat([
            param_embed.unsqueeze(1).expand(-1, T, -1),
            time_embed.unsqueeze(0).expand(B, -1, -1),
            delta.unsqueeze(-1),
            bump.unsqueeze(-1),
            q.unsqueeze(-1),
            Q.unsqueeze(-1),
        ], dim=-1)

        x = self.input_proj(combined)
        x = x + self.multi_scale(x)

        lstm_out, _ = self.lstm(x)
        attn_out, attn_w = self.attention(lstm_out, lstm_out, lstm_out)
        features = self.attn_norm(lstm_out + attn_out)

        preds = torch.cat([
            self.decoder_ext_lv1(features),
            self.decoder_ext_lv2(features),
            self.decoder_ins_lv2(features),
        ], dim=-1)

        skip = self.skip_proj(param_embed).unsqueeze(1).expand(-1, T, -1)
        return preds + skip, attn_w


if __name__ == "__main__":
    model = KANAttentionLSTMv5()
    print(f"Params: {count_parameters(model):,}")
    p = torch.randn(4, 35); p[:, 0] = torch.randint(0, 4, (4,)).float()
    out, _ = model(p, torch.linspace(0, 1, 181), torch.rand(4) * 0.5 + 0.3)
    print(f"Output: {out.shape}")

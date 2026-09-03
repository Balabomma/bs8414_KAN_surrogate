"""KAN-Attention-LSTM v6 — v5 + plume-theory causal channel.

Adds a fifth per-timestep channel q(t)^(2/3): plume correlations (Heskestad,
McCaffrey) give gas temperature rise ~ HRR^(2/3), so the network receives the
physically expected burner-driven temperature shape and only has to learn the
cladding contribution on top of it.
"""
import torch

from model_v5 import KANAttentionLSTMv5, HRR_MW_COL
from features_v4 import HRR_MW_SCALE
from model import count_parameters


class KANAttentionLSTMv6(KANAttentionLSTMv5):
    N_TIME_FEATS = 5  # delta, bump, q(t), Q(t), q(t)^(2/3)

    def forward(self, params, time_array, anchor):
        B, T = params.shape[0], len(time_array)

        param_embed = self.param_encoder(params, anchor)
        time_embed = self.time_encoding(time_array)

        delta = time_array.unsqueeze(0) - anchor.unsqueeze(1)
        bump = torch.exp(-(delta / 0.08) ** 2)

        hrr_mw = params[:, HRR_MW_COL:HRR_MW_COL + 1] * HRR_MW_SCALE
        q = hrr_mw * self.ramp_f.unsqueeze(0)
        Q = hrr_mw * self.ramp_cum.unsqueeze(0)
        q23 = q.clamp(min=0.0) ** (2.0 / 3.0)   # plume DT ~ HRR^(2/3)

        combined = torch.cat([
            param_embed.unsqueeze(1).expand(-1, T, -1),
            time_embed.unsqueeze(0).expand(B, -1, -1),
            delta.unsqueeze(-1),
            bump.unsqueeze(-1),
            q.unsqueeze(-1),
            Q.unsqueeze(-1),
            q23.unsqueeze(-1),
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
    from features_v6 import N_V6_FEATURES
    model = KANAttentionLSTMv6(n_continuous=2 + 13 + N_V6_FEATURES)
    print(f"Params: {count_parameters(model):,}")
    p = torch.rand(4, 16 + N_V6_FEATURES); p[:, 0] = torch.randint(0, 4, (4,)).float()
    out, _ = model(p, torch.linspace(0, 1, 181), torch.rand(4) * 0.5 + 0.3)
    print(f"Output: {out.shape}")

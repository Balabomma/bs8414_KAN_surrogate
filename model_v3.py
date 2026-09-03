"""KAN-Attention-LSTM v3 — adds ignition-anchor time channels.

Identical to model.KANAttentionLSTM except forward() takes a per-sample
normalised anchor time (expected ignition-peak time / 1800 s) and feeds two
extra per-timestep channels into the temporal backbone:

    delta = t - t_anchor          (signed time since expected ignition)
    bump  = exp(-(delta/0.08)^2)  (soft window around expected ignition)

This lets the network render sharp ignition transients at a *hinted* time
instead of having to extrapolate spike timing from the parameter embedding —
the failure mode behind Test_1_PE_PIR HRR1333's R² collapse on valid/test.
"""
import torch
import torch.nn as nn

from model import (
    KANLinear, KANParameterEncoder, TimeEncoding, MultiScaleConv,
    KANSensorDecoder, count_parameters, kan_regularization,
)

BUMP_WIDTH = 0.08  # in normalised time units (0.08 * 1800 s = 144 s)


class KANAttentionLSTMv3(nn.Module):
    N_TIME_FEATS = 2  # delta, bump

    def __init__(self, n_sensors=24, hidden_size=96, embedding_dim=24,
                 n_heads=4, dropout=0.08, num_knots=8):
        super().__init__()

        self.param_encoder = KANParameterEncoder(output_dim=embedding_dim, num_knots=num_knots)
        self.time_encoding = TimeEncoding(d_model=embedding_dim)

        self.input_proj = nn.Sequential(
            nn.Linear(embedding_dim * 2 + self.N_TIME_FEATS, hidden_size),
            nn.GELU(), nn.Dropout(dropout),
        )

        self.multi_scale = MultiScaleConv(hidden_size, hidden_size)

        self.lstm = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size,
            num_layers=2, batch_first=True, bidirectional=True, dropout=dropout,
        )

        lstm_out = hidden_size * 2
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_out, num_heads=n_heads, dropout=dropout, batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(lstm_out)

        self.decoder_ext_lv1 = KANSensorDecoder(lstm_out, 8, num_knots=num_knots)
        self.decoder_ext_lv2 = KANSensorDecoder(lstm_out, 8, num_knots=num_knots)
        self.decoder_ins_lv2 = KANSensorDecoder(lstm_out, 8, num_knots=num_knots)

        self.skip_proj = nn.Linear(embedding_dim, n_sensors)

    def forward(self, params, time_array, anchor):
        """params: (B, 16); time_array: (T,); anchor: (B,) normalised anchor times."""
        B, T = params.shape[0], len(time_array)

        param_embed = self.param_encoder(params)
        time_embed = self.time_encoding(time_array)

        delta = time_array.unsqueeze(0) - anchor.unsqueeze(1)          # (B, T)
        bump = torch.exp(-(delta / BUMP_WIDTH) ** 2)                   # (B, T)

        combined = torch.cat([
            param_embed.unsqueeze(1).expand(-1, T, -1),
            time_embed.unsqueeze(0).expand(B, -1, -1),
            delta.unsqueeze(-1),
            bump.unsqueeze(-1),
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
    model = KANAttentionLSTMv3()
    print(f"Params: {count_parameters(model):,}")
    p = torch.randn(4, 16); p[:, 0] = torch.randint(0, 4, (4,)).float()
    out, _ = model(p, torch.linspace(0, 1, 181), torch.rand(4) * 0.5 + 0.3)
    print(f"Output: {out.shape}")

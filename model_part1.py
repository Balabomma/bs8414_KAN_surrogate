"""KAN-Attention-LSTM for the BS 8414 Part1 geometry-variant corpus.

Parent: `model.py` (`KANAttentionLSTM`, the 24-sensor / 60-sim architecture).
`KANLinear` is imported from it rather than re-declared, so the B-spline edge
block under test is byte-identical to the one the 60-sim results were produced
with — the KAN-vs-MLP ablation contract survives the corpus change.

Three changes, all forced by the Part1 corpus:

1. **Conditioning.** HRR and mesh are constants in Part1 and carried no signal,
   so they are gone. In their place the encoder takes three categorical axes —
   cladding (12), insulation (5) and geometry (8) — each with its own embedding.
   Geometry is an 8-way embedding over the observed flag combinations rather
   than three separate booleans: the user's call, and it lets the model learn an
   arbitrary interaction between removing the cavity, the gaps and the barriers
   without assuming those effects compose additively. The cost, stated because
   it constrains what the model may be asked afterwards: a combination absent
   from training has no embedding row and cannot be predicted at all.

2. **Two thermocouple groups, not three.** The Insulation Level 2 decoder is
   gone with the channels it predicted.

3. **An HRR head.** A second decoder predicts the global energy budget
   (HRR, Q_RADI, Q_CONV, Q_COND, Q_TOTAL) from the same temporal features. The
   burner ramp is identical in every deck, so what this head has to learn is the
   cladding/insulation combustion contribution — the same physics that drives
   the thermocouples, which is why it shares the backbone instead of being a
   separate model.

Physics enforced on the output rather than penalised in the loss (soft penalties
were not holding on the 60-sim corpus): thermocouples are clamped at ambient and
total HRR at zero.

Run `python model_part1.py` for the parameter count and a forward-pass check.
"""
import math

import torch
import torch.nn as nn

from config_part1 import (
    N_CLADDING, N_INSULATION, N_GEOMETRY, N_MATERIAL_FEATURES,
    CLADDING_EMBED_DIM, INSULATION_EMBED_DIM, GEOMETRY_EMBED_DIM,
    N_SENSORS, GROUP_SIZES, N_HRR_CHANNELS, N_TIMESTEPS,
    EMBEDDING_DIM, LSTM_HIDDEN_SIZE, ATTENTION_HEADS, DROPOUT, NUM_KNOTS,
    T_AMBIENT, COL_CLADDING, COL_INSULATION, COL_GEOM,
)
from layers_part1 import (KANLinear, TimeEncoding, MultiScaleConv, KANSensorDecoder,
                   kan_regularization)


class Part1ParameterEncoder(nn.Module):
    """[cladding_id, insulation_id, geom_id, 13 material] -> embedding."""

    def __init__(self, output_dim=EMBEDDING_DIM, num_knots=NUM_KNOTS):
        super().__init__()
        self.cladding_embedding = nn.Embedding(N_CLADDING, CLADDING_EMBED_DIM)
        self.insulation_embedding = nn.Embedding(N_INSULATION, INSULATION_EMBED_DIM)
        self.geometry_embedding = nn.Embedding(N_GEOMETRY, GEOMETRY_EMBED_DIM)

        input_dim = (CLADDING_EMBED_DIM + INSULATION_EMBED_DIM
                     + GEOMETRY_EMBED_DIM + N_MATERIAL_FEATURES)
        self.kan1 = KANLinear(input_dim, 48, num_knots=num_knots)
        self.kan2 = KANLinear(48, output_dim, num_knots=num_knots)
        self.output_dim = output_dim

    def forward(self, params):
        clad = self.cladding_embedding(params[:, COL_CLADDING].long())
        ins = self.insulation_embedding(params[:, COL_INSULATION].long())
        geom = self.geometry_embedding(params[:, COL_GEOM].long())
        material = params[:, 3:]
        x = torch.cat([clad, ins, geom, material], dim=-1)
        return self.kan2(self.kan1(x))


class Part1KANAttentionLSTM(nn.Module):
    """Shared temporal backbone, two output heads (thermocouples and HRR)."""

    def __init__(self, n_sensors=N_SENSORS, n_hrr_channels=N_HRR_CHANNELS,
                 hidden_size=LSTM_HIDDEN_SIZE, embedding_dim=EMBEDDING_DIM,
                 n_heads=ATTENTION_HEADS, dropout=DROPOUT, num_knots=NUM_KNOTS):
        super().__init__()
        self.n_sensors = n_sensors
        self.n_hrr_channels = n_hrr_channels

        self.param_encoder = Part1ParameterEncoder(output_dim=embedding_dim,
                                                   num_knots=num_knots)
        self.time_encoding = TimeEncoding(d_model=embedding_dim)

        self.input_proj = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_size), nn.GELU(), nn.Dropout(dropout),
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

        # One decoder per thermocouple group: External LV1 (0-7), LV2 (8-15).
        self.sensor_decoders = nn.ModuleList([
            KANSensorDecoder(lstm_out, size, num_knots=num_knots)
            for size in GROUP_SIZES
        ])
        self.hrr_decoder = KANSensorDecoder(lstm_out, n_hrr_channels,
                                            num_knots=num_knots)

        self.skip_proj = nn.Linear(embedding_dim, n_sensors)
        self.hrr_skip_proj = nn.Linear(embedding_dim, n_hrr_channels)

        # Output-space physics floors, in standardised units. Populated by
        # set_output_scaling(); zeros until then, which is inert for the HRR
        # floor only if the scaler says 0 kW maps to 0 — so training must call it.
        self.register_buffer("ambient_scaled", torch.full((n_sensors,), -1e9))
        self.register_buffer("hrr_floor_scaled", torch.full((n_hrr_channels,), -1e9))

    def set_output_scaling(self, tc_scaler, hrr_scaler, hrr_nonnegative_idx=(0,),
                           t_ambient=T_AMBIENT):
        """Install the hard physical floors, expressed in standardised space.

        Called after the scalers are fitted on the training split. Channels not
        listed in `hrr_nonnegative_idx` keep an effectively infinite floor: the
        Q_* terms are net budget contributions and are legitimately negative.
        """
        amb = (t_ambient - torch.as_tensor(tc_scaler.mean, dtype=torch.float32)) \
            / torch.as_tensor(tc_scaler.scale, dtype=torch.float32)
        self.ambient_scaled.copy_(amb.to(self.ambient_scaled.device))

        floor = torch.full((self.n_hrr_channels,), -1e9)
        mean = torch.as_tensor(hrr_scaler.mean, dtype=torch.float32)
        scale = torch.as_tensor(hrr_scaler.scale, dtype=torch.float32)
        for i in hrr_nonnegative_idx:
            floor[i] = (0.0 - mean[i]) / scale[i]
        self.hrr_floor_scaled.copy_(floor.to(self.hrr_floor_scaled.device))

    def forward(self, params, time_array):
        B, T = params.shape[0], len(time_array)

        param_embed = self.param_encoder(params)
        time_embed = self.time_encoding(time_array)

        combined = torch.cat([
            param_embed.unsqueeze(1).expand(-1, T, -1),
            time_embed.unsqueeze(0).expand(B, -1, -1),
        ], dim=-1)

        x = self.input_proj(combined)
        x = x + self.multi_scale(x)

        lstm_out, _ = self.lstm(x)
        attn_out, attn_w = self.attention(lstm_out, lstm_out, lstm_out)
        features = self.attn_norm(lstm_out + attn_out)

        tc = torch.cat([d(features) for d in self.sensor_decoders], dim=-1)
        tc = tc + self.skip_proj(param_embed).unsqueeze(1).expand(-1, T, -1)
        tc = torch.maximum(tc, self.ambient_scaled)

        hrr = self.hrr_decoder(features)
        hrr = hrr + self.hrr_skip_proj(param_embed).unsqueeze(1).expand(-1, T, -1)
        hrr = torch.maximum(hrr, self.hrr_floor_scaled)

        return tc, hrr, attn_w


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────────────────
# Uniform interface
# ──────────────────────────────────────────────────────────────────────
# `train_part1.py` and `evaluate_part1.py` are byte-identical across the KAN,
# MLP and Attention-LSTM projects and bind to these three names only. Adding a
# Part1 variant means writing a `model_part1.py` that exports them — nothing in
# the training or scoring path forks per architecture, which is what keeps the
# three-way comparison attributable to the architecture.
MODEL_NAME = "KAN-Attention-LSTM (Part1)"
Part1Surrogate = Part1KANAttentionLSTM

# Spline L2 + adjacent-knot smoothness. The MLP control has no counterpart and
# sets this to 0.0 — the one disclosed difference in that ablation.
LAMBDA_REG = 2e-3


def regularization(model):
    """Architecture-specific weight penalty, added to the training loss."""
    return kan_regularization(model)


if __name__ == "__main__":
    from config_part1 import N_INPUT_PARAMS
    from data_loader_part1 import ChannelScaler
    import numpy as np

    model = Part1Surrogate()
    print(f"Part1 KAN-Attention-LSTM parameters: {count_parameters(model):,}")

    tc_scaler = ChannelScaler().load_state_dict(
        {"mean": np.full(N_SENSORS, 200.0, np.float32),
         "scale": np.full(N_SENSORS, 150.0, np.float32)})
    hrr_scaler = ChannelScaler().load_state_dict(
        {"mean": np.full(N_HRR_CHANNELS, 800.0, np.float32),
         "scale": np.full(N_HRR_CHANNELS, 600.0, np.float32)})
    model.set_output_scaling(tc_scaler, hrr_scaler)

    p = torch.rand(4, N_INPUT_PARAMS)
    p[:, COL_CLADDING] = torch.randint(0, N_CLADDING, (4,)).float()
    p[:, COL_INSULATION] = torch.randint(0, N_INSULATION, (4,)).float()
    p[:, COL_GEOM] = torch.randint(0, N_GEOMETRY, (4,)).float()

    tc, hrr, attn = model(p, torch.linspace(0, 1, N_TIMESTEPS))
    print(f"tc  {tuple(tc.shape)}  min(scaled)={tc.min():.3f} "
          f"floor={model.ambient_scaled.min():.3f}")
    print(f"hrr {tuple(hrr.shape)}  min(scaled)={hrr.min():.3f} "
          f"floor={model.hrr_floor_scaled[0]:.3f}")

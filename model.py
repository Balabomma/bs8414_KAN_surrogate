"""
KAN-Attention-LSTM Surrogate Model for BS8414 Facade Fire Test.

Replaces FC layers with Kolmogorov-Arnold Network (KAN) layers using
learnable B-spline activation functions. Retains BiLSTM + self-attention
temporal backbone.

KAN: Instead of fixed activations on nodes, KAN uses learnable activations
on edges (connections). Each edge has a B-spline function that is learned
during training, enabling superior function approximation with fewer params.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class KANLinear(nn.Module):
    """KAN Layer: replaces nn.Linear with learnable B-spline activations on edges.

    Instead of: y = activation(W @ x + b)
    KAN does:   y_j = sum_i(spline_ij(x_i))

    Each connection (i->j) has its own learnable activation function
    parameterized as a linear combination of B-spline basis functions.
    """

    def __init__(self, in_features, out_features, num_knots=8, residual=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_knots = num_knots
        self.residual = residual

        # Learnable B-spline coefficients for each edge (out x in x num_knots)
        self.spline_weights = nn.Parameter(
            torch.randn(out_features, in_features, num_knots) * (1.0 / math.sqrt(in_features * num_knots))
        )

        # Learnable per-feature input scale (helps map inputs into basis support)
        self.input_scale = nn.Parameter(torch.ones(in_features))

        # RBF basis centers in [-1, 1] (inputs are squashed to this range via tanh)
        centers = torch.linspace(-1.0, 1.0, num_knots)
        self.register_buffer('centers', centers)
        # Slightly wider than uniform spacing for smooth overlap
        self.width = 2.0 / (num_knots - 1) * 1.2

        # Residual linear connection (SiLU-weighted)
        if residual:
            self.residual_weight = nn.Parameter(
                torch.randn(out_features, in_features) * (1.0 / math.sqrt(in_features))
            )
            self.residual_bias = nn.Parameter(torch.zeros(out_features))

        # LayerNorm for stability (works with small batches and seq data)
        self.ln = nn.LayerNorm(out_features)

    def compute_basis(self, x):
        """Compute RBF basis values. x: (..., in_features) -> (..., in_features, num_knots)

        Inputs are tanh-squashed into [-1, 1] to guarantee basis coverage,
        with a learnable per-feature scale applied before the squash.
        """
        x_scaled = torch.tanh(x * self.input_scale)
        x_expanded = x_scaled.unsqueeze(-1)  # (..., in_features, 1)
        basis = torch.exp(-0.5 * ((x_expanded - self.centers) / self.width) ** 2)
        return basis  # (..., in_features, num_knots)

    def forward(self, x):
        """x: (batch, in_features) or (batch, seq, in_features)"""
        orig_shape = x.shape
        if x.dim() == 3:
            batch, seq, feat = x.shape
            x_flat = x.reshape(-1, feat)
        else:
            x_flat = x
            batch = x.shape[0]

        # Compute basis functions for each input
        basis = self.compute_basis(x_flat)  # (batch*, in_features, num_knots)

        # Spline output: sum over input features and knots
        out = torch.einsum('bik,oik->bo', basis, self.spline_weights)

        # Add residual connection
        if self.residual:
            residual = F.silu(x_flat) @ self.residual_weight.t() + self.residual_bias
            out = out + residual

        # LayerNorm over feature dim (safe for any batch/seq shape)
        out = self.ln(out)

        if len(orig_shape) == 3:
            out = out.reshape(batch, seq, -1)

        return out

    def spline_regularization(self):
        """L2 + smoothness penalty on spline weights, for use in training loss."""
        l2 = self.spline_weights.pow(2).mean()
        # Smoothness: penalize differences between adjacent knot coefficients
        diff = self.spline_weights[..., 1:] - self.spline_weights[..., :-1]
        smooth = diff.pow(2).mean()
        return l2 + smooth


class KANParameterEncoder(nn.Module):
    """KAN-based parameter encoder with learnable activations."""

    def __init__(self, n_cladding=4, cladding_embed_dim=12, n_material=13,
                 output_dim=24, num_knots=8):
        super().__init__()
        self.cladding_embedding = nn.Embedding(n_cladding, cladding_embed_dim)

        # KAN layers for material + continuous features
        input_dim = cladding_embed_dim + 2 + n_material  # embed + hrr_mesh + material
        self.kan1 = KANLinear(input_dim, 32, num_knots=num_knots)
        self.kan2 = KANLinear(32, output_dim, num_knots=num_knots)
        self.output_dim = output_dim

    def forward(self, params):
        cladding_idx = params[:, 0].long()
        continuous = params[:, 1:]  # hrr, mesh, + 13 material
        clad_embed = self.cladding_embedding(cladding_idx)
        combined = torch.cat([clad_embed, continuous], dim=-1)
        x = self.kan1(combined)
        return self.kan2(x)


class TimeEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, time_array):
        pe = torch.zeros(len(time_array), self.d_model, device=time_array.device)
        pos = time_array.unsqueeze(1) * 1000
        div = torch.exp(torch.arange(0, self.d_model, 2, device=time_array.device).float()
                        * -(math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe


class MultiScaleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv3 = nn.Conv1d(in_ch, out_ch // 3, kernel_size=3, padding=1)
        self.conv9 = nn.Conv1d(in_ch, out_ch // 3, kernel_size=9, padding=4)
        self.conv27 = nn.Conv1d(in_ch, out_ch - 2 * (out_ch // 3), kernel_size=27, padding=13)
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x):
        xt = x.transpose(1, 2)
        return self.norm(torch.cat([self.conv3(xt), self.conv9(xt), self.conv27(xt)], dim=1).transpose(1, 2))


class KANSensorDecoder(nn.Module):
    """KAN-based sensor decoder with interpretable activations."""

    def __init__(self, input_dim, n_sensors, num_knots=8, dropout=0.15):
        super().__init__()
        hidden = max(32, n_sensors * 2)
        self.kan1 = KANLinear(input_dim, hidden, num_knots=num_knots)
        self.dropout1 = nn.Dropout(dropout)
        self.kan2 = KANLinear(hidden, n_sensors, num_knots=num_knots)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        return self.kan2(self.dropout1(self.kan1(self.dropout2(x))))


class KANAttentionLSTM(nn.Module):
    """
    KAN-Attention-LSTM: Kolmogorov-Arnold Networks replace FC layers.

    Architecture:
        KAN Parameter Encoder -> Time Encoding -> Multi-Scale Conv
        -> BiLSTM -> Self-Attention -> KAN Sensor Decoders

    KAN layers used in: parameter encoder + sensor decoders
    Standard layers: Conv1D, LSTM, Attention (temporal backbone)
    """

    def __init__(self, n_sensors=24, hidden_size=96, embedding_dim=24,
                 n_heads=4, dropout=0.08, num_knots=8):
        super().__init__()

        self.param_encoder = KANParameterEncoder(output_dim=embedding_dim, num_knots=num_knots)
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

        # KAN decoders instead of FC decoders
        self.decoder_ext_lv1 = KANSensorDecoder(lstm_out, 8, num_knots=num_knots)
        self.decoder_ext_lv2 = KANSensorDecoder(lstm_out, 8, num_knots=num_knots)
        self.decoder_ins_lv2 = KANSensorDecoder(lstm_out, 8, num_knots=num_knots)

        self.skip_proj = nn.Linear(embedding_dim, n_sensors)

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

        preds = torch.cat([
            self.decoder_ext_lv1(features),
            self.decoder_ext_lv2(features),
            self.decoder_ins_lv2(features),
        ], dim=-1)

        skip = self.skip_proj(param_embed).unsqueeze(1).expand(-1, T, -1)
        return preds + skip, attn_w


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def kan_regularization(model):
    """Sum spline L2 + smoothness penalty over all KANLinear layers in `model`."""
    reg = 0.0
    n = 0
    for m in model.modules():
        if isinstance(m, KANLinear):
            reg = reg + m.spline_regularization()
            n += 1
    if n == 0:
        return torch.tensor(0.0)
    return reg / n


if __name__ == "__main__":
    model = KANAttentionLSTM()
    print(f"KAN-Attention-LSTM Parameters: {count_parameters(model):,}")
    print(model)
    params = torch.randn(4, 16)
    params[:, 0] = torch.randint(0, 4, (4,)).float()
    time = torch.linspace(0, 1, 181)
    out, attn = model(params, time)
    print(f"\nInput: {params.shape} -> Output: {out.shape}")

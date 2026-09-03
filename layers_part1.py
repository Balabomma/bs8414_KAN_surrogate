"""Layer primitives for the Part1 surrogate, vendored verbatim.

Extracted from this project's `model.py` (the 60-sim corpus model) so the Part1
pipeline stands alone: the published repository is the Part1 geometry-corpus
surrogate and carries no 60-sim code.

VERBATIM, not re-implemented. The blocks below are byte-for-byte the ones every
existing result was produced with — a re-implementation differing in an
initialisation constant or a basis width would silently invalidate comparison
against those runs. Same reasoning, and the same pattern, as
`kan_layers_part1.py` in the FunDiff-KAN project.

Verify a symbol against the original with:

    python -c "import inspect, hashlib, layers_part1 as L; \
               print(hashlib.sha256(inspect.getsource(L.KANLinear).encode()).hexdigest()[:16])"
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

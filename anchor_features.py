"""Ignition-anchor time extraction and prediction for the v3 surrogate.

The anchor is the peak time of the smoothed mean external-LV1 curve (sensors
0-7), normalised by T_END. For unseen parameter combinations it is predicted
per cladding by a mesh-localised weighted least-squares fit in 1/hrr_norm —
ignition delay scales roughly with inverse heat flux, and the mesh kernel
keeps the strongly mesh-dependent timing trends separate.

Only TRAIN-split sims go into the bank; training uses leave-one-out
predictions so the network sees realistic anchor errors.
"""
import numpy as np

SIGMA_MESH = 0.04   # kernel width in mesh_norm units (adjacent mesh ~0.1 apart)
RIDGE = 2e-3


def anchor_time_from_curve(curve_2d, t_end_norm=1.0):
    """curve_2d: (T, n_sensors) degC. Returns normalised anchor time in [0, 1]."""
    c = curve_2d[:, :8].mean(axis=1)
    cs = np.convolve(c, np.ones(5) / 5, mode="same")
    return (cs.argmax() / (len(c) - 1)) * t_end_norm


def predict_anchor(clad_id, hrr_norm, mesh_norm, bank, exclude_idx=None):
    """Weighted LSQ of anchor ~ a + b*(1/hrr_norm), mesh-kernel weighted.

    bank: dict with arrays 'clad', 'hrr', 'mesh', 'anchor' (train sims only).
    exclude_idx: bank index to leave out (for LOO on train sims).
    """
    sel = bank["clad"] == clad_id
    if exclude_idx is not None:
        sel = sel & (np.arange(len(bank["clad"])) != exclude_idx)
    if sel.sum() == 0:
        return float(bank["anchor"].mean())

    x = 1.0 / bank["hrr"][sel]
    y = bank["anchor"][sel]
    w = np.exp(-(((bank["mesh"][sel] - mesh_norm) / SIGMA_MESH) ** 2))
    w = w + 1e-6

    X = np.column_stack([np.ones(len(x)), x])
    W = np.diag(w)
    A = X.T @ W @ X + RIDGE * np.diag([0.0, 1.0]) * w.sum()
    b = X.T @ W @ y
    try:
        coef = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return float(np.average(y, weights=w))
    pred = coef[0] + coef[1] / hrr_norm
    # clamp to plausible range
    return float(np.clip(pred, 0.05, 0.95))


def build_bank(params, outputs, train_idx):
    """Anchor bank from TRAIN sims only."""
    train_idx = np.asarray(train_idx)
    return {
        "clad": params[train_idx, 0].astype(int),
        "hrr": params[train_idx, 1].astype(np.float64),
        "mesh": params[train_idx, 2].astype(np.float64),
        "anchor": np.array([anchor_time_from_curve(outputs[i]) for i in train_idx]),
    }


def anchors_for(params, bank, loo_bank_pos=None):
    """Predict anchors for a params array (N, 16).

    loo_bank_pos: optional array of bank positions for LOO (train sims);
    None entries / -1 mean no exclusion.
    """
    out = np.zeros(len(params), dtype=np.float32)
    for i in range(len(params)):
        ex = None
        if loo_bank_pos is not None and loo_bank_pos[i] >= 0:
            ex = int(loo_bank_pos[i])
        out[i] = predict_anchor(int(params[i, 0]), float(params[i, 1]),
                                float(params[i, 2]), bank, exclude_idx=ex)
    return out

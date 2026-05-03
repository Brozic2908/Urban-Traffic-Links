"""
Branch B method module: DMFM-Gt for XT forecasting.

This file is designed to be loaded by:
    06B_branchB_run_xt_forecast_topk_gt.py

Correct Branch-B interpretation
--------------------------------
DMFM is used to forecast the directed graph matrix G_t, not to directly forecast
traffic speed X_t by itself.

For each origin t and horizon h:
    1) Fit a Dynamic Matrix Factor Model on the training graph series G_t:
          G_t = M + U1 F_t U2^T + E_t
    2) Forecast latent factors by either:
          dmfm_lse_gt  : matrix MAR-LSE, F_t = A1 F_{t-1} A2^T + noise
          dmfm_vlse_gt : vector VAR-LSE on vec(F_t)
    3) Reconstruct:
          G_hat[t+h|t] = M + U1 F_hat[t+h|t] U2^T
    4) The shared runner then performs the downstream Branch-B forecast:
          X_hat[t+h] = A_h X_t + B_h(TopK(G_hat[t+h|t]) X_t)

This fixes the older 06D_dmfm_paper_xt_forecast.py behavior where DMFM predicted
X directly and did not use G_t before predicting X.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ElasticNet settings used by the shared XT runner.
ALPHA = float(os.environ.get("BRANCHB_XT_ALPHA", "0.001"))
L1_RATIO = float(os.environ.get("BRANCHB_XT_L1_RATIO", "0.5"))
MAX_ITER = int(os.environ.get("BRANCHB_XT_MAX_ITER", "500"))
TOL = float(os.environ.get("BRANCHB_XT_TOL", "1e-2"))
SELECTION = os.environ.get("BRANCHB_XT_SELECTION", "random")
RANDOM_STATE = int(os.environ.get("BRANCHB_RANDOM_STATE", "42"))
EPS = 1e-8


def _env_rank() -> Tuple[int, int]:
    s = os.environ.get("DMFM_GT_RANK", "8,8").replace("x", ",").replace("X", ",")
    vals = [int(v.strip()) for v in s.split(",") if v.strip()]
    if len(vals) == 1:
        return vals[0], vals[0]
    if len(vals) >= 2:
        return vals[0], vals[1]
    return 8, 8


DMFM_RANK = _env_rank()
DMFM_CENTER = int(os.environ.get("DMFM_GT_CENTER", "1")) == 1
DMFM_RIDGE = float(os.environ.get("DMFM_GT_RIDGE", "1e-4"))
DMFM_ALS_ITERS = int(os.environ.get("DMFM_GT_ALS_ITERS", "30"))
DMFM_STABILIZE = int(os.environ.get("DMFM_GT_STABILIZE", "1")) == 1
DMFM_RHO = float(os.environ.get("DMFM_GT_RHO", "0.98"))
DMFM_MAX_TRAIN_MATS = int(os.environ.get("DMFM_GT_MAX_TRAIN_MATS", "0"))  # 0 = all


# =============================================================================
# Prepared-data helpers: same contract as other Branch-B method modules
# =============================================================================
def check_branchB_common_dir_ready(common_dir: Path) -> None:
    common_dir = Path(common_dir)
    required = []
    for split in ["train", "val", "test"]:
        d = common_dir / split
        for name in [
            "z.npy",
            "segment_ids.npy",
            "timestamps.npy",
            "G_series_meta.csv",
            "G_weight_series.npy",
            "G_best_lag_series.npy",
        ]:
            required.append(d / name)
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing standard Gt prepared files:\n" + "\n".join(map(str, missing)))


def _safe_datetime(arr):
    arr = np.asarray(arr).astype(str)
    s = pd.Series(arr).str.replace("__", " ", regex=False).str.replace("Slot_", "", regex=False)
    s = s.str.replace(r"(\d{4}-\d{2}-\d{2})\s+(\d{2})(\d{2})", r"\1 \2:\3:00", regex=True)
    return pd.to_datetime(s, errors="coerce")


def load_gt_split(common_dir: Path, split: str) -> Dict[str, Any]:
    common_dir = Path(common_dir)
    d = common_dir / split
    check_branchB_common_dir_ready(common_dir)

    z = np.load(d / "z.npy", mmap_mode="r")
    G = np.load(d / "G_weight_series.npy", mmap_mode="r")
    L = np.load(d / "G_best_lag_series.npy", mmap_mode="r")
    segment_ids = np.asarray(np.load(d / "segment_ids.npy"), dtype=np.int64)
    timestamps_raw = np.asarray(np.load(d / "timestamps.npy", allow_pickle=True)).astype(str)
    meta = pd.read_csv(d / "G_series_meta.csv")

    m = int(min(len(meta), z.shape[0], G.shape[0], L.shape[0], len(timestamps_raw)))
    if m <= 0:
        raise ValueError(f"Empty split after alignment: {split}")
    if len(meta) != m or z.shape[0] != m or G.shape[0] != m:
        print(f"[WARN] Aligning split={split}: meta={len(meta)}, z={z.shape[0]}, G={G.shape[0]} -> {m}", flush=True)

    meta = meta.iloc[:m].reset_index(drop=True)
    if "timestamp_local" in meta.columns:
        meta["timestamp_local"] = pd.to_datetime(meta["timestamp_local"], errors="coerce")
    if "session_id" not in meta.columns:
        if "date_key" in meta.columns:
            meta["session_id"] = meta["date_key"].astype(str)
        else:
            meta["session_id"] = "session_0"

    return {
        "z": z[:m],
        "G_weight_series": G[:m],
        "G_best_lag_series": L[:m],
        "segment_ids": segment_ids,
        "timestamps": _safe_datetime(timestamps_raw[:m]),
        "meta": meta,
        "common_dir": common_dir,
        "split_name": split,
    }


def _session_groups(meta: pd.DataFrame) -> List[np.ndarray]:
    if "session_id" in meta.columns:
        groups = []
        for _, sub in meta.groupby("session_id", sort=False):
            idx = sub.index.to_numpy(dtype=np.int64)
            if len(idx):
                groups.append(idx)
        return groups
    return [np.arange(len(meta), dtype=np.int64)]


def iter_eval_pairs(meta: pd.DataFrame, horizon: int):
    h = int(horizon)
    for idx in _session_groups(meta):
        if len(idx) <= h:
            continue
        for pos in range(0, len(idx) - h):
            origin = int(idx[pos])
            target = int(idx[pos + h])
            if "can_predict_granger" in meta.columns:
                if not bool(meta.loc[origin, "can_predict_granger"]):
                    continue
            elif "can_predict" in meta.columns:
                if not bool(meta.loc[origin, "can_predict"]):
                    continue
            yield origin, target


def batch_vector_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    diff = y_pred - y_true
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))
    return {"mae": mae, "mse": mse, "rmse": rmse}


# =============================================================================
# DMFM internals
# =============================================================================
def _take_train_indices(T: int, max_train: int) -> np.ndarray:
    if max_train <= 0 or max_train >= T:
        return np.arange(T, dtype=np.int64)
    # Deterministic evenly-spaced subsample; avoids biasing to early timestamps.
    return np.linspace(0, T - 1, max_train).round().astype(np.int64)


def _mean_matrix(G: np.ndarray, idx: np.ndarray) -> np.ndarray:
    N = int(G.shape[1])
    M = np.zeros((N, N), dtype=np.float64)
    for t in idx:
        M += np.asarray(G[int(t)], dtype=np.float64)
    M /= max(1, len(idx))
    return M.astype(np.float32)


def _eigh_top(mat: np.ndarray, rank: int) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float64)
    mat = 0.5 * (mat + mat.T)
    vals, vecs = np.linalg.eigh(mat)
    order = np.argsort(vals)[::-1]
    rank = max(1, min(int(rank), mat.shape[0]))
    U = vecs[:, order[:rank]]
    return U.astype(np.float32)


def _estimate_loadings(G: np.ndarray, idx: np.ndarray, mean_G: np.ndarray, r1: int, r2: int) -> Tuple[np.ndarray, np.ndarray]:
    N = int(G.shape[1])
    C1 = np.zeros((N, N), dtype=np.float64)
    C2 = np.zeros((N, N), dtype=np.float64)
    denom = max(1, len(idx) * N)

    for t in idx:
        X = np.asarray(G[int(t)], dtype=np.float64)
        if DMFM_CENTER:
            X = X - mean_G
        C1 += X @ X.T
        C2 += X.T @ X

    C1 /= denom
    C2 /= denom
    U1 = _eigh_top(C1, r1)
    U2 = _eigh_top(C2, r2)
    return U1, U2


def _project_factors(G: np.ndarray, idx: np.ndarray, mean_G: np.ndarray, U1: np.ndarray, U2: np.ndarray) -> np.ndarray:
    r1, r2 = int(U1.shape[1]), int(U2.shape[1])
    F = np.zeros((len(idx), r1, r2), dtype=np.float32)
    U1T = U1.T.astype(np.float32)
    U2f = U2.astype(np.float32)
    for k, t in enumerate(idx):
        X = np.asarray(G[int(t)], dtype=np.float32)
        if DMFM_CENTER:
            X = X - mean_G
        F[k] = U1T @ X @ U2f
    return F


def _ridge_solve_right(X: np.ndarray, Y: np.ndarray, ridge: float) -> np.ndarray:
    """Return B minimizing ||X B - Y||^2 + ridge ||B||^2."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    p = X.shape[1]
    return np.linalg.solve(X.T @ X + float(ridge) * np.eye(p), X.T @ Y)


def _fit_vlse(F: np.ndarray, ridge: float) -> np.ndarray:
    # vec in C-order is okay as long as prediction/reconstruction use same order.
    V = F.reshape(F.shape[0], -1).astype(np.float64)
    X = V[:-1]
    Y = V[1:]
    B = _ridge_solve_right(X, Y, ridge)  # f_next = f_current @ B
    Phi = B.T.astype(np.float64)         # column-vector form f_next_col = Phi @ f_col
    if DMFM_STABILIZE:
        Phi = _stabilize_matrix(Phi, DMFM_RHO)
    return Phi.astype(np.float32)


def _fit_mar_lse(F: np.ndarray, r1: int, r2: int, ridge: float, max_iter: int) -> Tuple[np.ndarray, np.ndarray]:
    A1 = np.eye(r1, dtype=np.float64)
    A2 = np.eye(r2, dtype=np.float64)
    Y = F[1:].astype(np.float64)
    Xlag = F[:-1].astype(np.float64)

    for _ in range(max(1, int(max_iter))):
        # Given A2, solve Ft = A1 (Ftm1 A2^T)
        Z_blocks = []
        Y_blocks = []
        for Xt, Yt in zip(Xlag, Y):
            Z_blocks.append(Xt @ A2.T)  # r1 x r2
            Y_blocks.append(Yt)
        Z = np.concatenate(Z_blocks, axis=1)  # r1 x (T*r2)
        YY = np.concatenate(Y_blocks, axis=1)
        A1 = (YY @ Z.T) @ np.linalg.pinv(Z @ Z.T + ridge * np.eye(r1))

        # Given A1, solve Ft^T = A2 (A1 Ftm1)^T
        W_blocks = []
        Yt_blocks = []
        for Xt, Yt in zip(Xlag, Y):
            W_blocks.append((A1 @ Xt).T)  # r2 x r1
            Yt_blocks.append(Yt.T)
        W = np.concatenate(W_blocks, axis=1)  # r2 x (T*r1)
        YYt = np.concatenate(Yt_blocks, axis=1)
        A2 = (YYt @ W.T) @ np.linalg.pinv(W @ W.T + ridge * np.eye(r2))

        if DMFM_STABILIZE:
            A1 = _stabilize_matrix(A1, DMFM_RHO)
            A2 = _stabilize_matrix(A2, DMFM_RHO)

    return A1.astype(np.float32), A2.astype(np.float32)


def _stabilize_matrix(A: np.ndarray, rho: float) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    try:
        vals = np.linalg.eigvals(A)
        sr = float(np.max(np.abs(vals))) if len(vals) else 0.0
        if np.isfinite(sr) and sr > float(rho) and sr > EPS:
            A = A * (float(rho) / sr)
    except Exception:
        pass
    return A


def _predict_factor_mar(F0: np.ndarray, A1: np.ndarray, A2: np.ndarray, h: int) -> np.ndarray:
    Fp = np.asarray(F0, dtype=np.float32)
    A1f = np.asarray(A1, dtype=np.float32)
    A2f = np.asarray(A2, dtype=np.float32)
    for _ in range(int(h)):
        Fp = A1f @ Fp @ A2f.T
    return Fp


def _predict_factor_vlse(F0: np.ndarray, Phi: np.ndarray, h: int) -> np.ndarray:
    shape = F0.shape
    f = np.asarray(F0, dtype=np.float32).reshape(-1)
    Phif = np.asarray(Phi, dtype=np.float32)
    for _ in range(int(h)):
        f = Phif @ f
    return f.reshape(shape)


def _reconstruct_G(mean_G: np.ndarray, U1: np.ndarray, Fp: np.ndarray, U2: np.ndarray) -> np.ndarray:
    Gh = U1 @ Fp @ U2.T
    if DMFM_CENTER:
        Gh = Gh + mean_G
    return np.nan_to_num(Gh, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def build_g_model(method_name: str, train: Dict[str, Any], val: Dict[str, Any], test: Dict[str, Any]) -> Dict[str, Any]:
    if method_name not in {"dmfm_lse_gt", "dmfm_vlse_gt"}:
        raise ValueError(f"Unsupported DMFM method: {method_name}")

    G_train = train["G_weight_series"]
    T, N, N2 = G_train.shape
    if N != N2:
        raise ValueError(f"G_weight_series must be T x N x N, got {G_train.shape}")
    if T < 3:
        raise ValueError("Need at least 3 train matrices for DMFM-Gt")

    r1, r2 = DMFM_RANK
    r1 = max(1, min(int(r1), int(N)))
    r2 = max(1, min(int(r2), int(N)))

    train_idx = _take_train_indices(int(T), DMFM_MAX_TRAIN_MATS)
    print(
        f"[DMFM-Gt] fitting method={method_name}, T={T}, N={N}, rank=({r1},{r2}), "
        f"train_mats={len(train_idx)}, center={DMFM_CENTER}, ridge={DMFM_RIDGE}",
        flush=True,
    )

    mean_G = _mean_matrix(G_train, train_idx) if DMFM_CENTER else np.zeros((N, N), dtype=np.float32)
    U1, U2 = _estimate_loadings(G_train, train_idx, mean_G, r1, r2)
    F_train = _project_factors(G_train, train_idx, mean_G, U1, U2)

    # The factor dynamics are learned on the selected chronological train samples.
    # If max_train_mats subsamples evenly, this is an approximation; for final runs keep max=0.
    params: Dict[str, Any] = {}
    if method_name == "dmfm_lse_gt":
        A1, A2 = _fit_mar_lse(F_train, r1, r2, DMFM_RIDGE, DMFM_ALS_ITERS)
        params.update({"A1": A1, "A2": A2})
    elif method_name == "dmfm_vlse_gt":
        Phi = _fit_vlse(F_train, DMFM_RIDGE)
        params.update({"Phi": Phi})

    return {
        "method": method_name,
        "mean_G": mean_G.astype(np.float32),
        "U1": U1.astype(np.float32),
        "U2": U2.astype(np.float32),
        "rank": (r1, r2),
        "params": params,
        "config": {
            "DMFM_GT_RANK": (r1, r2),
            "DMFM_GT_CENTER": DMFM_CENTER,
            "DMFM_GT_RIDGE": DMFM_RIDGE,
            "DMFM_GT_ALS_ITERS": DMFM_ALS_ITERS,
            "DMFM_GT_STABILIZE": DMFM_STABILIZE,
            "DMFM_GT_RHO": DMFM_RHO,
            "DMFM_GT_MAX_TRAIN_MATS": DMFM_MAX_TRAIN_MATS,
        },
    }


def _safe_idx(i: int, length: int) -> int:
    if length <= 0:
        return 0
    return int(max(0, min(int(i), int(length) - 1)))


def predict_G_method(
    method_name: str,
    g_model: Dict[str, Any],
    split_name: str,
    split_data: Dict[str, Any],
    origin_idx: int,
    target_idx: int,
    horizon: int,
) -> np.ndarray:
    G = split_data["G_weight_series"]
    origin_idx = _safe_idx(origin_idx, G.shape[0])

    mean_G = g_model["mean_G"]
    U1 = g_model["U1"]
    U2 = g_model["U2"]
    X0 = np.asarray(G[origin_idx], dtype=np.float32)
    if DMFM_CENTER:
        X0 = X0 - mean_G
    F0 = U1.T @ X0 @ U2

    params = g_model["params"]
    h = int(horizon)
    if method_name == "dmfm_lse_gt":
        Fp = _predict_factor_mar(F0, params["A1"], params["A2"], h)
    elif method_name == "dmfm_vlse_gt":
        Fp = _predict_factor_vlse(F0, params["Phi"], h)
    else:
        raise ValueError(f"Unsupported DMFM method: {method_name}")

    return _reconstruct_G(mean_G, U1, Fp, U2)

# ml_core/src/models/ML_BranchA/scripts/10_debug_corr_equal_one_branchA.py
"""
Debug vì sao Branch A có rất nhiều cặp |corr| = 1.

Script kiểm tra:
1. Top-K có vô tình lấy đường chéo i == j không.
2. R_series lưu sẵn có bao nhiêu cặp off-diagonal |corr| == 1.
3. Recompute Pearson bằng float64 từ z-window để xem có phải do float16/rounding không.
4. Kiểm tra hai chuỗi tốc độ trong rolling window có giống hệt nhau không.
5. Kiểm tra hai OSM edge trong Top corr=1 có dùng cùng tomtom_segment_ids không.
6. Xuất CSV debug để đưa vào báo cáo/giải thích.

Run:
    python -u ml_core/src/models/ML_BranchA/scripts/10_debug_corr_equal_one_branchA.py \
      --splits val,test \
      --lags 1-9 \
      --samples-per-split-lag 10 \
      --topk 50 \
      2>&1 | tee logs_A10_debug_corr_equal_one.txt
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


WINDOW = 10
EPS = 1e-12
SEED = 42


# ============================================================
# Helpers
# ============================================================

def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def print_stage(title: str) -> None:
    print("\n" + "=" * 100)
    print(f"{now_str()} | {title}")
    print("=" * 100)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def find_project_root() -> Path:
    cwd = Path.cwd().resolve()
    candidates = [cwd, *cwd.parents, Path("/kaggle/working/UTraffic-ML"), Path("/kaggle/working")]
    for p in candidates:
        if p.name == "UTraffic-ML" and (p / "ml_core").exists():
            return p
        if (p / "UTraffic-ML").exists() and (p / "UTraffic-ML" / "ml_core").exists():
            return p / "UTraffic-ML"
        if (p / "ml_core").exists():
            return p
    return cwd


def parse_str_list(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def parse_int_list(s: str) -> List[int]:
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def decode_arr(arr: np.ndarray) -> np.ndarray:
    out = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return np.array(out)


def safe_float(x: Any, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default


# ============================================================
# Load data
# ============================================================

def load_branchA_split(common_dir: Path, split: str) -> Dict[str, Any]:
    split_dir = common_dir / split

    required = [
        split_dir / "R_series.npy",
        split_dir / "z.npy",
        split_dir / "segment_ids.npy",
        split_dir / "timestamps.npy",
        split_dir / "R_series_meta.csv",
    ]

    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing files:\n" + "\n".join(str(p) for p in missing))

    R = np.load(split_dir / "R_series.npy", mmap_mode="r")
    z = np.load(split_dir / "z.npy", mmap_mode="r")
    segment_ids = np.load(split_dir / "segment_ids.npy").astype(np.int64)
    timestamps = pd.to_datetime(np.load(split_dir / "timestamps.npy"))
    meta = pd.read_csv(split_dir / "R_series_meta.csv")

    if "timestamp_local" in meta.columns:
        meta["timestamp_local"] = pd.to_datetime(meta["timestamp_local"])

    return {
        "split_dir": split_dir,
        "R": R,
        "z": z,
        "segment_ids": segment_ids,
        "timestamps": timestamps,
        "meta": meta,
    }


def get_timestamp(data: Dict[str, Any], idx: int) -> str:
    meta = data["meta"]
    if "timestamp_local" in meta.columns and idx < len(meta):
        return str(meta.loc[idx, "timestamp_local"])
    timestamps = data["timestamps"]
    if idx < len(timestamps):
        return str(timestamps[idx])
    return ""


def iter_eval_indices(data: Dict[str, Any], lag: int) -> List[Tuple[int, int]]:
    """
    origin_idx -> target_idx.
    R_true dùng target_idx.
    """
    meta = data["meta"]
    T = data["R"].shape[0]
    out = []

    sess = meta["session_id"].to_numpy() if "session_id" in meta.columns else None

    for origin_idx in range(T - lag):
        target_idx = origin_idx + lag
        if sess is not None:
            if origin_idx >= len(sess) or target_idx >= len(sess):
                continue
            if sess[origin_idx] != sess[target_idx]:
                continue
        out.append((origin_idx, target_idx))

    return out


# ============================================================
# Correlation functions
# ============================================================

def pearson64(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan

    sx = np.std(x)
    sy = np.std(y)

    if sx <= EPS or sy <= EPS:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def affine_fit_r2(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit y = a*x + b. Return a, b, r2.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan, np.nan, np.nan

    vx = np.var(x)
    if vx <= EPS:
        return np.nan, np.nan, np.nan

    a = float(np.cov(x, y, bias=True)[0, 1] / vx)
    b = float(np.mean(y) - a * np.mean(x))
    yhat = a * x + b

    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))

    if ss_tot <= EPS:
        r2 = np.nan
    else:
        r2 = 1.0 - ss_res / ss_tot

    return a, b, float(r2)


def get_window_z(z: np.ndarray, target_idx: int, i: int, j: int, window: int = WINDOW) -> Tuple[np.ndarray, np.ndarray]:
    start = max(0, target_idx - window + 1)
    end = target_idx + 1

    xi = np.asarray(z[start:end, i], dtype=np.float64)
    xj = np.asarray(z[start:end, j], dtype=np.float64)

    return xi, xj


def topk_offdiag_abs(R: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Chỉ lấy upper triangle k=1, loại đường chéo.
    """
    R = np.asarray(R)
    n = R.shape[0]

    iu = np.triu_indices(n, k=1)
    vals = R[iu].astype(np.float64)
    abs_vals = np.abs(vals)

    finite = np.isfinite(abs_vals)
    ii = iu[0][finite]
    jj = iu[1][finite]
    vals = vals[finite]
    abs_vals = abs_vals[finite]

    if len(abs_vals) == 0:
        return np.array([]), np.array([]), np.array([])

    kk = min(k, len(abs_vals))
    idx = np.argpartition(-abs_vals, kk - 1)[:kk]
    idx = idx[np.argsort(-abs_vals[idx])]

    return ii[idx], jj[idx], vals[idx]


def matrix_corr_saturation_stats(R: np.ndarray) -> Dict[str, Any]:
    """
    Đếm số lượng cặp off-diagonal có |corr| sát 1.
    """
    R = np.asarray(R)
    n = R.shape[0]
    iu = np.triu_indices(n, k=1)
    vals = np.asarray(R[iu], dtype=np.float64)
    abs_vals = np.abs(vals)
    abs_vals = abs_vals[np.isfinite(abs_vals)]

    total = len(abs_vals)
    if total == 0:
        return {}

    return {
        "offdiag_pair_count": int(total),
        "count_abs_eq_1_exact": int(np.sum(abs_vals == 1.0)),
        "ratio_abs_eq_1_exact": float(np.mean(abs_vals == 1.0)),
        "count_abs_ge_099999": int(np.sum(abs_vals >= 0.99999)),
        "ratio_abs_ge_099999": float(np.mean(abs_vals >= 0.99999)),
        "count_abs_ge_0999": int(np.sum(abs_vals >= 0.999)),
        "ratio_abs_ge_0999": float(np.mean(abs_vals >= 0.999)),
        "count_abs_ge_099": int(np.sum(abs_vals >= 0.99)),
        "ratio_abs_ge_099": float(np.mean(abs_vals >= 0.99)),
        "max_abs_corr": float(np.max(abs_vals)),
        "p999_abs_corr": float(np.quantile(abs_vals, 0.999)),
        "p99_abs_corr": float(np.quantile(abs_vals, 0.99)),
        "p95_abs_corr": float(np.quantile(abs_vals, 0.95)),
    }


def boundary_gap_stats(R: np.ndarray) -> Dict[str, Any]:
    R = np.asarray(R)
    n = R.shape[0]
    iu = np.triu_indices(n, k=1)
    vals = np.abs(np.asarray(R[iu], dtype=np.float64))
    vals = vals[np.isfinite(vals)]

    if len(vals) < 51:
        return {}

    vals = np.sort(vals)[::-1]

    return {
        "abs_corr_rank_1": float(vals[0]),
        "abs_corr_rank_10": float(vals[9]),
        "abs_corr_rank_11": float(vals[10]),
        "gap_rank10_rank11": float(vals[9] - vals[10]),
        "abs_corr_rank_50": float(vals[49]),
        "abs_corr_rank_51": float(vals[50]),
        "gap_rank50_rank51": float(vals[49] - vals[50]),
    }


# ============================================================
# TomTom metadata debug
# ============================================================

def find_matched_edge_metadata(project_root: Path) -> Optional[Path]:
    candidates = [
        project_root / "ml_core/src/data_processing/outputs/branchA/match_summary/matched_osm_edge_metadata.csv",
        project_root / "ml_core/src/data_processing/outputs/branchA/matched_osm_edge_metadata.csv",
        project_root / "ml_core/src/data_processing/outputs/branchA/osm_edge_forecasting_dataset/tables/node_quality.csv",
    ]

    for p in candidates:
        if p.exists():
            return p

    hits = list((project_root / "ml_core/src/data_processing/outputs").glob("**/matched_osm_edge_metadata.csv"))
    if hits:
        return hits[0]

    return None


def parse_segment_set(x: Any) -> set:
    """
    Parse cột tomtom_segment_ids/tomtom_unique_segments dạng string/list.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return set()

    if isinstance(x, (list, tuple, set)):
        return set(map(str, x))

    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return set()

    # Try JSON/list literal
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, (list, tuple, set)):
            return set(map(str, obj))
    except Exception:
        pass

    # Fallback split
    s = s.replace("[", "").replace("]", "").replace("{", "").replace("}", "")
    parts = re.split(r"[,\s;|]+", s)
    return set(p.strip("'\" ") for p in parts if p.strip("'\" "))


def load_edge_metadata(project_root: Path) -> Tuple[Optional[pd.DataFrame], str]:
    p = find_matched_edge_metadata(project_root)
    if p is None:
        print("[WARN] Cannot find matched_osm_edge_metadata.csv")
        return None, ""

    df = pd.read_csv(p)
    print(f"[OK] Loaded edge metadata: {p}")
    print("Columns:", list(df.columns)[:30], "...")

    id_col = None
    for c in ["model_node_id", "segment_id", "osm_edge_id", "edge_id"]:
        if c in df.columns:
            id_col = c
            break

    if id_col is None:
        print("[WARN] Cannot identify edge id column.")
        return None, ""

    df[id_col] = df[id_col].astype(np.int64)
    if id_col != "model_node_id":
        df["model_node_id"] = df[id_col].astype(np.int64)

    return df, str(p)


def find_tomtom_col(edge_meta: Optional[pd.DataFrame]) -> Optional[str]:
    if edge_meta is None:
        return None

    candidates = [
        "tomtom_segment_ids",
        "tomtom_unique_segments",
        "tomtom_segments",
        "matched_tomtom_segment_ids",
        "source_segment_ids",
    ]

    for c in candidates:
        if c in edge_meta.columns:
            return c

    # fuzzy
    for c in edge_meta.columns:
        lc = c.lower()
        if "tomtom" in lc and "segment" in lc and "id" in lc:
            return c

    return None


def tomtom_overlap_info(
    edge_meta: Optional[pd.DataFrame],
    sid_i: int,
    sid_j: int,
) -> Dict[str, Any]:
    out = {
        "tomtom_col": "",
        "tomtom_count_i": np.nan,
        "tomtom_count_j": np.nan,
        "tomtom_intersection_count": np.nan,
        "tomtom_union_count": np.nan,
        "tomtom_jaccard": np.nan,
        "tomtom_same_set": np.nan,
    }

    if edge_meta is None:
        return out

    tomtom_col = find_tomtom_col(edge_meta)
    if tomtom_col is None:
        return out

    meta = edge_meta.set_index("model_node_id", drop=False)

    if sid_i not in meta.index or sid_j not in meta.index:
        return out

    si = parse_segment_set(meta.loc[sid_i, tomtom_col])
    sj = parse_segment_set(meta.loc[sid_j, tomtom_col])

    inter = si & sj
    union = si | sj

    out.update({
        "tomtom_col": tomtom_col,
        "tomtom_count_i": int(len(si)),
        "tomtom_count_j": int(len(sj)),
        "tomtom_intersection_count": int(len(inter)),
        "tomtom_union_count": int(len(union)),
        "tomtom_jaccard": float(len(inter) / len(union)) if len(union) else np.nan,
        "tomtom_same_set": bool(si == sj) if len(union) else np.nan,
    })

    return out


# ============================================================
# Main
# ============================================================

def run(args):
    project_root = find_project_root()

    common_dir = Path(args.common_dir)
    if not common_dir.is_absolute():
        common_dir = project_root / common_dir

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    ensure_dir(out_dir)

    splits = parse_str_list(args.splits)
    lags = parse_int_list(args.lags)
    rng = np.random.default_rng(args.seed)

    print_stage("CONFIG")
    print("PROJECT_ROOT:", project_root)
    print("COMMON_DIR  :", common_dir)
    print("OUT_DIR     :", out_dir)
    print("Splits      :", splits)
    print("Lags        :", lags)
    print("TopK        :", args.topk)

    print_stage("LOAD EDGE METADATA")
    edge_meta, edge_meta_path = load_edge_metadata(project_root)

    all_pair_rows = []
    all_matrix_rows = []

    for split in splits:
        print_stage(f"LOAD SPLIT {split}")
        data = load_branchA_split(common_dir, split)

        R_series = data["R"]
        z = data["z"]
        segment_ids = data["segment_ids"]

        print("R shape:", R_series.shape, "dtype:", R_series.dtype)
        print("z shape:", z.shape, "dtype:", z.dtype)
        print("N nodes:", len(segment_ids))

        for lag in lags:
            pairs = iter_eval_indices(data, lag)
            if not pairs:
                print(f"[WARN] No pairs split={split}, lag={lag}")
                continue

            if args.samples_per_split_lag > 0 and len(pairs) > args.samples_per_split_lag:
                chosen = rng.choice(len(pairs), size=args.samples_per_split_lag, replace=False)
                selected_pairs = [pairs[int(c)] for c in chosen]
            else:
                selected_pairs = pairs

            print(f"[{split}] lag={lag}: selected {len(selected_pairs)}/{len(pairs)} samples")

            for sample_id, (origin_idx, target_idx) in enumerate(selected_pairs):
                R = np.asarray(R_series[target_idx], dtype=np.float32)

                mat_stats = {
                    "split": split,
                    "lag": lag,
                    "sample_id": sample_id,
                    "origin_idx": origin_idx,
                    "target_idx": target_idx,
                    "timestamp": get_timestamp(data, target_idx),
                    "R_dtype": str(R_series.dtype),
                    "z_dtype": str(z.dtype),
                }
                mat_stats.update(matrix_corr_saturation_stats(R))
                mat_stats.update(boundary_gap_stats(R))
                all_matrix_rows.append(mat_stats)

                ii, jj, vals = topk_offdiag_abs(R, args.topk)

                for rank, (i, j, corr_stored) in enumerate(zip(ii, jj, vals), start=1):
                    i = int(i)
                    j = int(j)

                    sid_i = int(segment_ids[i])
                    sid_j = int(segment_ids[j])

                    xi, xj = get_window_z(z, target_idx, i, j, WINDOW)
                    corr64 = pearson64(xi, xj)
                    a, b, r2 = affine_fit_r2(xi, xj)

                    diff = xj - xi

                    row = {
                        "split": split,
                        "lag": lag,
                        "sample_id": sample_id,
                        "rank": rank,
                        "origin_idx": origin_idx,
                        "target_idx": target_idx,
                        "timestamp": get_timestamp(data, target_idx),

                        "i": i,
                        "j": j,
                        "segment_i": sid_i,
                        "segment_j": sid_j,
                        "is_diagonal": bool(i == j),

                        "corr_stored": float(corr_stored),
                        "abs_corr_stored": float(abs(corr_stored)),
                        "corr_recomputed_float64": corr64,
                        "abs_corr_recomputed_float64": abs(corr64) if np.isfinite(corr64) else np.nan,
                        "stored_minus_recomputed": float(corr_stored - corr64) if np.isfinite(corr64) else np.nan,

                        "x_i_values": json.dumps([float(v) for v in xi]),
                        "x_j_values": json.dumps([float(v) for v in xj]),

                        "x_i_std": float(np.nanstd(xi)),
                        "x_j_std": float(np.nanstd(xj)),
                        "x_i_mean": float(np.nanmean(xi)),
                        "x_j_mean": float(np.nanmean(xj)),
                        "x_i_unique_count": int(len(np.unique(np.round(xi[np.isfinite(xi)], 8)))),
                        "x_j_unique_count": int(len(np.unique(np.round(xj[np.isfinite(xj)], 8)))),

                        "vectors_exact_equal": bool(np.array_equal(xi, xj)),
                        "vectors_allclose_1e_8": bool(np.allclose(xi, xj, atol=1e-8, rtol=1e-8)),
                        "max_abs_diff_xj_minus_xi": float(np.nanmax(np.abs(diff))) if len(diff) else np.nan,
                        "mean_abs_diff_xj_minus_xi": float(np.nanmean(np.abs(diff))) if len(diff) else np.nan,

                        "affine_a": a,
                        "affine_b": b,
                        "affine_r2": r2,
                        "is_almost_perfect_affine": bool(np.isfinite(r2) and r2 >= 0.999999),
                    }

                    row.update(tomtom_overlap_info(edge_meta, sid_i, sid_j))

                    # Diagnostic label
                    reasons = []
                    if row["is_diagonal"]:
                        reasons.append("BUG_DIAGONAL_PAIR")
                    if row["vectors_exact_equal"] or row["vectors_allclose_1e_8"]:
                        reasons.append("IDENTICAL_Z_WINDOW")
                    if row["is_almost_perfect_affine"]:
                        reasons.append("PERFECT_AFFINE_RELATION")
                    if row.get("tomtom_same_set") is True:
                        reasons.append("SAME_TOMTOM_SEGMENT_SET")
                    elif safe_float(row.get("tomtom_jaccard")) >= 0.8:
                        reasons.append("HIGH_TOMTOM_SEGMENT_OVERLAP")
                    if np.isfinite(row["corr_recomputed_float64"]) and abs(row["corr_stored"]) == 1.0 and abs(row["corr_recomputed_float64"]) < 0.999999:
                        reasons.append("POSSIBLE_FLOAT16_ROUNDING")
                    if row["x_i_std"] <= EPS or row["x_j_std"] <= EPS:
                        reasons.append("ZERO_OR_TINY_STD")

                    row["diagnostic_reasons"] = "|".join(reasons)

                    all_pair_rows.append(row)

    print_stage("SAVE OUTPUTS")

    pair_df = pd.DataFrame(all_pair_rows)
    matrix_df = pd.DataFrame(all_matrix_rows)

    pair_path = out_dir / "debug_top_pairs_corr_equal_one_details.csv"
    matrix_path = out_dir / "debug_matrix_corr_saturation_stats.csv"

    pair_df.to_csv(pair_path, index=False, encoding="utf-8-sig")
    matrix_df.to_csv(matrix_path, index=False, encoding="utf-8-sig")

    print("Saved:", pair_path)
    print("Saved:", matrix_path)

    print_stage("SUMMARY")

    if not matrix_df.empty:
        summary_mat = (
            matrix_df
            .groupby(["split", "lag"])[
                [
                    "ratio_abs_eq_1_exact",
                    "ratio_abs_ge_099999",
                    "ratio_abs_ge_0999",
                    "gap_rank10_rank11",
                    "gap_rank50_rank51",
                    "abs_corr_rank_10",
                    "abs_corr_rank_50",
                ]
            ]
            .agg(["mean", "median", "min", "max"])
            .reset_index()
        )
        summary_mat.columns = ["_".join([str(x) for x in c if str(x) != ""]) for c in summary_mat.columns]
        summary_mat_path = out_dir / "summary_matrix_corr_saturation_by_split_lag.csv"
        summary_mat.to_csv(summary_mat_path, index=False, encoding="utf-8-sig")
        print("Saved:", summary_mat_path)

    if not pair_df.empty:
        reason_counts = (
            pair_df["diagnostic_reasons"]
            .fillna("")
            .str.get_dummies(sep="|")
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        reason_counts.columns = ["diagnostic_reason", "count"]
        reason_path = out_dir / "summary_diagnostic_reason_counts.csv"
        reason_counts.to_csv(reason_path, index=False, encoding="utf-8-sig")
        print("Saved:", reason_path)

        summary_pair = {
            "total_top_pairs_checked": int(len(pair_df)),
            "diagonal_pair_count": int(pair_df["is_diagonal"].sum()),
            "stored_abs_corr_eq_1_count": int((pair_df["abs_corr_stored"] == 1.0).sum()),
            "recomputed_abs_corr_eq_1_count": int((pair_df["abs_corr_recomputed_float64"] >= 0.999999999).sum()),
            "identical_window_count": int(pair_df["vectors_allclose_1e_8"].sum()),
            "perfect_affine_count": int(pair_df["is_almost_perfect_affine"].sum()),
            "same_tomtom_set_count": int((pair_df["tomtom_same_set"] == True).sum()) if "tomtom_same_set" in pair_df.columns else -1,
            "high_tomtom_overlap_count": int((pair_df["tomtom_jaccard"] >= 0.8).sum()) if "tomtom_jaccard" in pair_df.columns else -1,
        }

        summary_json_path = out_dir / "summary_debug_corr_equal_one.json"
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary_pair, f, ensure_ascii=False, indent=2)

        print("Saved:", summary_json_path)
        print(json.dumps(summary_pair, ensure_ascii=False, indent=2))

    print_stage("HOW TO READ")
    print("""
Đọc kết quả như sau:

1. Nếu diagonal_pair_count > 0:
   => Có bug lấy nhầm đường chéo i == j trong Top-K.

2. Nếu corr_stored = 1 nhưng corr_recomputed_float64 < 0.999999:
   => Có thể R_series dtype float16/rounding làm tròn lên 1.

3. Nếu vectors_allclose_1e_8 = True:
   => Hai OSM edge có cùng chuỗi z trong rolling window, nên corr = 1 là đúng về toán học.

4. Nếu tomtom_same_set = True hoặc tomtom_jaccard cao:
   => Hai OSM edge dùng cùng nguồn TomTom segment sau map-match.

5. Nếu affine_r2 >= 0.999999:
   => Hai chuỗi có quan hệ tuyến tính gần hoàn hảo, Pearson ra 1 là hợp lý.

6. Nếu gap_rank10_rank11 = 0:
   => Top-10 không có ranh giới rõ, Overlap@10 rất nhạy.
""")


def build_argparser():
    project_root = find_project_root()

    p = argparse.ArgumentParser()

    p.add_argument(
        "--common-dir",
        type=str,
        default=str(project_root / "ml_core/src/models/ML_BranchA/data/05_branchA_prepare_segment_segment_rt"),
    )

    p.add_argument(
        "--out-dir",
        type=str,
        default=str(project_root / "ml_core/src/models/ML_BranchA/results/10_debug_corr_equal_one"),
    )

    p.add_argument("--splits", type=str, default="val,test")
    p.add_argument("--lags", type=str, default="1-9")
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--samples-per-split-lag", type=int, default=10)
    p.add_argument("--seed", type=int, default=SEED)

    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run(args)
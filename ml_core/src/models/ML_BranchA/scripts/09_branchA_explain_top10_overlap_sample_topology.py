# ml_core/src/models/ML_BranchA/scripts/09_branchA_explain_top10_overlap_sample_topology.py
"""
Branch A — Explain unstable Overlap@10.

Script này phục vụ giải thích góp ý của thầy:
- Kiểm tra sample_size của các cặp nằm trong Top-10 tương quan mạnh.
- Vẽ phân phối giá trị tương quan của Top-10 và Top-50.
- Kiểm tra topology OSM của các cặp Top-10.

Input chính:
    ml_core/src/models/ML_BranchA/data/05_branchA_prepare_segment_segment_rt/
        train/val/test/R_series.npy
        train/val/test/segment_ids.npy
        train/val/test/R_series_meta.csv

Input phụ:
    ml_core/src/data_processing/outputs/branchA/osm_edge_forecasting_dataset/
        osm_edge_tensor.npz
        tables/node_quality.csv

Output:
    ml_core/src/models/ML_BranchA/results/09_explain_top10_overlap/

Chạy nhanh:
    python -u ml_core/src/models/ML_BranchA/scripts/09_branchA_explain_top10_overlap_sample_topology.py \
      --splits val,test \
      --lags 1-9 \
      --samples-per-split-lag 20 \
      --topks 10,50 \
      2>&1 | tee logs_A09_explain_top10.txt

Chạy nhẹ hơn:
    python -u ml_core/src/models/ML_BranchA/scripts/09_branchA_explain_top10_overlap_sample_topology.py \
      --splits val,test \
      --lags 1,3,6,9 \
      --samples-per-split-lag 10 \
      --topks 10,50 \
      2>&1 | tee logs_A09_explain_top10_light.txt
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


WINDOW = 10
SEED = 42
EPS = 1e-8


# ============================================================
# Basic helpers
# ============================================================

def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def print_stage(title: str) -> None:
    print("\n" + "=" * 96)
    print(f"{now_str()} | {title}")
    print("=" * 96)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_int_list(s: str) -> List[int]:
    out: List[int] = []
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


def parse_str_list(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def find_project_root() -> Path:
    cwd = Path.cwd().resolve()
    candidates = [cwd, *cwd.parents, Path("/kaggle/working/UTraffic-ML"), Path("/kaggle/working")]
    for p in candidates:
        if (p / "ml_core").exists():
            if p.name == "UTraffic-ML":
                return p
            if (p / "dataset").exists():
                return p
        if (p / "UTraffic-ML").exists():
            pp = p / "UTraffic-ML"
            if (pp / "ml_core").exists():
                return pp
    return cwd


def decode_str_array(arr: np.ndarray) -> np.ndarray:
    out = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return np.array(out)


def get_npz_key(data: np.lib.npyio.NpzFile, candidates: List[str]) -> Optional[str]:
    for k in candidates:
        if k in data.files:
            return k
    return None


# ============================================================
# Load Branch A data
# ============================================================

def load_rt_split(common_dir: Path, split: str, mmap_mode: str = "r") -> Dict[str, Any]:
    split_dir = common_dir / split
    required = [
        split_dir / "R_series.npy",
        split_dir / "segment_ids.npy",
        split_dir / "timestamps.npy",
        split_dir / "R_series_meta.csv",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing Branch A files in {split_dir}:\n" +
            "\n".join(str(p) for p in missing)
        )

    R_series = np.load(split_dir / "R_series.npy", mmap_mode=mmap_mode)
    segment_ids = np.load(split_dir / "segment_ids.npy").astype(np.int64)
    timestamps = pd.to_datetime(np.load(split_dir / "timestamps.npy"))
    meta = pd.read_csv(split_dir / "R_series_meta.csv")

    if "timestamp_local" in meta.columns:
        meta["timestamp_local"] = pd.to_datetime(meta["timestamp_local"])

    raw_meta = None
    raw_meta_path = split_dir / "raw_meta.csv"
    if raw_meta_path.exists():
        raw_meta = pd.read_csv(raw_meta_path)
        if "timestamp_local" in raw_meta.columns:
            raw_meta["timestamp_local"] = pd.to_datetime(raw_meta["timestamp_local"])

    return {
        "R_series": R_series,
        "segment_ids": segment_ids,
        "timestamps": timestamps,
        "meta": meta,
        "raw_meta": raw_meta,
    }


def get_R(split_data: Dict[str, Any], t_idx: int) -> np.ndarray:
    R = np.asarray(split_data["R_series"][int(t_idx)], dtype=np.float32)
    np.nan_to_num(R, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
    np.clip(R, -1.0, 1.0, out=R)
    return R


def get_timestamp_for_r(split_data: Dict[str, Any], r_idx: int) -> Optional[pd.Timestamp]:
    meta = split_data.get("meta")
    timestamps = split_data.get("timestamps")

    if meta is not None and len(meta) > r_idx:
        if "timestamp_local" in meta.columns:
            return pd.Timestamp(meta.loc[r_idx, "timestamp_local"])
        if "timestamp" in meta.columns:
            return pd.Timestamp(meta.loc[r_idx, "timestamp"])

    if timestamps is not None and len(timestamps) > r_idx:
        return pd.Timestamp(timestamps[r_idx])

    return None


def iter_eval_pairs(split_data: Dict[str, Any], horizon: int) -> List[Tuple[int, int]]:
    """
    Trả về các cặp origin_idx -> target_idx cùng session nếu meta có session_id.
    Top-K sẽ lấy tại target_idx.
    """
    meta = split_data["meta"]
    T = len(meta)
    pairs = []

    sess = meta["session_id"].to_numpy() if "session_id" in meta.columns else None

    for origin_idx in range(T - horizon):
        target_idx = origin_idx + horizon
        if sess is not None and sess[origin_idx] != sess[target_idx]:
            continue
        pairs.append((origin_idx, target_idx))

    return pairs


# ============================================================
# Load OSM-edge tensor and sample_size
# ============================================================

def load_osm_tensor_dataset(osm_dataset_dir: Path) -> Dict[str, Any]:
    """
    Cố gắng đọc sample_size theo thời gian từ osm_edge_tensor.npz.
    Nếu không đủ key thì vẫn chạy bằng node_quality.csv.
    """
    tensor_path = osm_dataset_dir / "osm_edge_tensor.npz"
    out: Dict[str, Any] = {
        "available": False,
        "sample_size": None,
        "timestamps": None,
        "model_node_ids": None,
        "feature_names": None,
    }

    if not tensor_path.exists():
        print(f"[WARN] Cannot find tensor NPZ: {tensor_path}")
        return out

    data = np.load(tensor_path, allow_pickle=True)

    x_key = get_npz_key(data, ["X_raw", "X_filled", "X", "traffic_tensor", "tensor"])
    feature_key = get_npz_key(data, ["feature_names", "features"])
    time_key = get_npz_key(data, ["timestamps", "timestamp_local", "times"])
    node_key = get_npz_key(data, ["model_node_ids", "segment_ids", "osm_edge_ids", "edge_ids"])

    if x_key is None or feature_key is None or time_key is None or node_key is None:
        print("[WARN] osm_edge_tensor.npz lacks one of required keys.")
        print("Available keys:", data.files)
        return out

    feature_names = decode_str_array(data[feature_key])
    if "sample_size" not in feature_names:
        print("[WARN] sample_size not found in feature_names.")
        return out

    sidx = int(np.where(feature_names == "sample_size")[0][0])

    X = data[x_key]
    sample_size = np.asarray(X[:, :, sidx], dtype=np.float32)
    timestamps = pd.to_datetime(data[time_key])
    model_node_ids = np.asarray(data[node_key]).astype(np.int64)

    out.update({
        "available": True,
        "sample_size": sample_size,
        "timestamps": timestamps,
        "model_node_ids": model_node_ids,
        "feature_names": feature_names,
        "x_key": x_key,
    })

    print(f"[OK] Loaded sample_size from {tensor_path}")
    print(f"     X key        : {x_key}")
    print(f"     sample_size  : {sample_size.shape}")
    print(f"     timestamps   : {len(timestamps)}")
    print(f"     model nodes  : {len(model_node_ids)}")
    return out


def load_node_quality(osm_dataset_dir: Path) -> Optional[pd.DataFrame]:
    candidates = [
        osm_dataset_dir / "tables" / "node_quality.csv",
        osm_dataset_dir / "node_quality.csv",
    ]

    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            print(f"[OK] Loaded node_quality: {p}")
            return df

    print("[WARN] node_quality.csv not found.")
    return None


def build_quality_lookup(node_quality: Optional[pd.DataFrame]) -> Dict[int, Dict[str, Any]]:
    if node_quality is None or node_quality.empty:
        return {}

    id_col = None
    for c in ["model_node_id", "segment_id", "osm_edge_id", "edge_id"]:
        if c in node_quality.columns:
            id_col = c
            break

    if id_col is None:
        print("[WARN] Cannot identify node id column in node_quality.")
        return {}

    out: Dict[int, Dict[str, Any]] = {}
    for _, row in node_quality.iterrows():
        try:
            node_id = int(row[id_col])
            out[node_id] = row.to_dict()
        except Exception:
            continue
    return out


def map_common_nodes_to_tensor_nodes(
    common_segment_ids: np.ndarray,
    tensor_node_ids: Optional[np.ndarray],
) -> Dict[int, int]:
    """
    Return: common matrix index -> tensor node index.
    """
    if tensor_node_ids is None:
        return {}

    tensor_pos = {int(s): i for i, s in enumerate(tensor_node_ids)}
    mapping = {}

    for common_idx, sid in enumerate(common_segment_ids):
        sid = int(sid)
        if sid in tensor_pos:
            mapping[common_idx] = int(tensor_pos[sid])

    return mapping


def build_time_lookup(timestamps: Optional[pd.DatetimeIndex]) -> Dict[pd.Timestamp, int]:
    if timestamps is None:
        return {}
    return {pd.Timestamp(t): i for i, t in enumerate(timestamps)}


# ============================================================
# Top-K extraction
# ============================================================

def topk_offdiag_pairs_abs(R: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return top-k off-diagonal pairs by absolute correlation.
    Vì R đối xứng, chỉ lấy upper triangle k=1 để tránh duplicate.
    """
    n = R.shape[0]
    iu = np.triu_indices(n, k=1)
    vals = np.asarray(R[iu], dtype=np.float32)
    abs_vals = np.abs(vals)

    finite = np.isfinite(abs_vals)
    pair_i = iu[0][finite]
    pair_j = iu[1][finite]
    vals = vals[finite]
    abs_vals = abs_vals[finite]

    if len(abs_vals) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    kk = min(k, len(abs_vals))
    idx = np.argpartition(-abs_vals, kk - 1)[:kk]
    idx = idx[np.argsort(-abs_vals[idx])]

    return pair_i[idx].astype(np.int64), pair_j[idx].astype(np.int64), vals[idx].astype(np.float32)


def corr_boundary_stats(R: np.ndarray, k_small: int = 10, k_large: int = 50) -> Dict[str, float]:
    """
    Kiểm tra khoảng cách giá trị quanh rank 10 và rank 50.
    Nếu rank10, rank11 rất gần nhau thì Overlap@10 rất nhạy.
    """
    n = R.shape[0]
    iu = np.triu_indices(n, k=1)
    vals = np.abs(np.asarray(R[iu], dtype=np.float32))
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {}

    vals_sorted = np.sort(vals)[::-1]

    def safe_rank(rank_1based: int) -> float:
        idx = rank_1based - 1
        if 0 <= idx < len(vals_sorted):
            return float(vals_sorted[idx])
        return float("nan")

    return {
        "abs_corr_rank_1": safe_rank(1),
        "abs_corr_rank_10": safe_rank(k_small),
        "abs_corr_rank_11": safe_rank(k_small + 1),
        "gap_rank10_rank11": safe_rank(k_small) - safe_rank(k_small + 1),
        "abs_corr_rank_50": safe_rank(k_large),
        "abs_corr_rank_51": safe_rank(k_large + 1),
        "gap_rank50_rank51": safe_rank(k_large) - safe_rank(k_large + 1),
    }


# ============================================================
# Sample size analysis
# ============================================================

def summarize_pair_sample_size(
    pair_i: int,
    pair_j: int,
    split_data: Dict[str, Any],
    target_r_idx: int,
    osm_data: Dict[str, Any],
    common_to_tensor: Dict[int, int],
    time_lookup: Dict[pd.Timestamp, int],
    quality_lookup: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    segment_ids = split_data["segment_ids"]
    sid_i = int(segment_ids[pair_i])
    sid_j = int(segment_ids[pair_j])

    row: Dict[str, Any] = {
        "i": int(pair_i),
        "j": int(pair_j),
        "segment_i": sid_i,
        "segment_j": sid_j,
    }

    # Node-level quality fallback
    qi = quality_lookup.get(sid_i, {})
    qj = quality_lookup.get(sid_j, {})

    for prefix, q in [("i", qi), ("j", qj)]:
        for col in [
            "valid_ratio",
            "missing_ratio",
            "recommended_keep",
            "sample_size_mean",
            "sample_size_sum",
            "average_speed_mean",
            "average_speed_std",
        ]:
            row[f"{prefix}_{col}"] = q.get(col, np.nan)

    # Time-specific sample_size from tensor, nếu có
    row.update({
        "sample_size_window_mean_i": np.nan,
        "sample_size_window_mean_j": np.nan,
        "sample_size_window_min_i": np.nan,
        "sample_size_window_min_j": np.nan,
        "sample_size_target_i": np.nan,
        "sample_size_target_j": np.nan,
        "sample_size_pair_window_min": np.nan,
        "sample_size_pair_window_mean": np.nan,
    })

    if not osm_data.get("available", False):
        return row

    tstamp = get_timestamp_for_r(split_data, target_r_idx)
    if tstamp is None or tstamp not in time_lookup:
        return row

    if pair_i not in common_to_tensor or pair_j not in common_to_tensor:
        return row

    tensor_t = time_lookup[tstamp]
    tensor_i = common_to_tensor[pair_i]
    tensor_j = common_to_tensor[pair_j]

    S = osm_data["sample_size"]

    start = max(0, tensor_t - WINDOW + 1)
    end = tensor_t + 1

    si_win = np.asarray(S[start:end, tensor_i], dtype=np.float32)
    sj_win = np.asarray(S[start:end, tensor_j], dtype=np.float32)

    pair_min_series = np.minimum(si_win, sj_win)

    row.update({
        "sample_size_window_mean_i": float(np.nanmean(si_win)),
        "sample_size_window_mean_j": float(np.nanmean(sj_win)),
        "sample_size_window_min_i": float(np.nanmin(si_win)),
        "sample_size_window_min_j": float(np.nanmin(sj_win)),
        "sample_size_target_i": float(S[tensor_t, tensor_i]),
        "sample_size_target_j": float(S[tensor_t, tensor_j]),
        "sample_size_pair_window_min": float(np.nanmin(pair_min_series)),
        "sample_size_pair_window_mean": float(np.nanmean(pair_min_series)),
    })

    return row


# ============================================================
# Topology analysis
# ============================================================

def normalize_edge_meta(node_quality: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Dùng node_quality làm edge metadata nếu có đủ cột.
    Cần cột định danh model_node_id và ideally u/v hoặc tọa độ.
    """
    if node_quality is None or node_quality.empty:
        return None

    df = node_quality.copy()

    if "model_node_id" not in df.columns:
        for c in ["segment_id", "osm_edge_id", "edge_id"]:
            if c in df.columns:
                df["model_node_id"] = df[c]
                break

    if "model_node_id" not in df.columns:
        return None

    df["model_node_id"] = df["model_node_id"].astype(np.int64)
    return df


def find_endpoint_columns(edge_meta: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    Tìm cột u/v node OSM nếu tồn tại.
    """
    candidates = [
        ("u", "v"),
        ("from_node", "to_node"),
        ("osm_u", "osm_v"),
        ("u_osm", "v_osm"),
        ("source_osm_node", "target_osm_node"),
        ("start_node", "end_node"),
    ]
    for u_col, v_col in candidates:
        if u_col in edge_meta.columns and v_col in edge_meta.columns:
            return u_col, v_col
    return None, None


def build_line_graph_adjacency(edge_meta: Optional[pd.DataFrame], segment_ids: np.ndarray) -> Dict[int, List[int]]:
    """
    Xây line graph adjacency:
    Hai OSM directed edges được xem là lân cận nếu share endpoint u/v.
    Output index là matrix index 0..N-1.
    """
    adjacency: Dict[int, List[int]] = {i: [] for i in range(len(segment_ids))}

    if edge_meta is None:
        print("[WARN] No edge_meta; topology hop distance will be unknown.")
        return adjacency

    u_col, v_col = find_endpoint_columns(edge_meta)
    if u_col is None or v_col is None:
        print("[WARN] Cannot find OSM endpoint columns u/v in node_quality.")
        print("[WARN] Topology hop distance will fallback to geometric distance if coordinates exist.")
        return adjacency

    meta_by_id = edge_meta.set_index("model_node_id", drop=False)
    available_ids = set(meta_by_id.index.astype(np.int64).tolist())

    endpoint_to_edges: Dict[Any, List[int]] = defaultdict(list)

    for idx, sid in enumerate(segment_ids.astype(np.int64)):
        sid = int(sid)
        if sid not in available_ids:
            continue

        row = meta_by_id.loc[sid]
        u = row[u_col]
        v = row[v_col]

        endpoint_to_edges[u].append(idx)
        endpoint_to_edges[v].append(idx)

    adj_sets = {i: set() for i in range(len(segment_ids))}

    for _, edge_indices in endpoint_to_edges.items():
        if len(edge_indices) <= 1:
            continue
        for e in edge_indices:
            adj_sets[e].update(edge_indices)

    for i in range(len(segment_ids)):
        adj_sets[i].discard(i)
        adjacency[i] = sorted(adj_sets[i])

    edge_count = sum(len(v) for v in adjacency.values())
    print(f"[OK] Built line-graph adjacency. Directed edge-neighbor links: {edge_count}")
    return adjacency


def shortest_hop(adjacency: Dict[int, List[int]], src: int, dst: int, max_hop: int = 6) -> Optional[int]:
    if src == dst:
        return 0

    q = deque([(src, 0)])
    seen = {src}

    while q:
        node, dist = q.popleft()
        if dist >= max_hop:
            continue
        for nb in adjacency.get(node, []):
            if nb == dst:
                return dist + 1
            if nb not in seen:
                seen.add(nb)
                q.append((nb, dist + 1))

    return None


def hop_bin(hop: Optional[int]) -> str:
    if hop is None:
        return ">6/unreachable"
    if hop <= 4:
        return f"{hop}-hop"
    if hop <= 6:
        return "5-6-hop"
    return ">6/unreachable"


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    lat1 = math.radians(float(lat1))
    lon1 = math.radians(float(lon1))
    lat2 = math.radians(float(lat2))
    lon2 = math.radians(float(lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return float(2 * R * math.asin(math.sqrt(a)))


def add_topology_info(
    row: Dict[str, Any],
    edge_meta: Optional[pd.DataFrame],
    adjacency: Dict[int, List[int]],
    segment_ids: np.ndarray,
    i: int,
    j: int,
) -> Dict[str, Any]:
    hop = shortest_hop(adjacency, i, j, max_hop=6)
    row["hop_distance"] = np.nan if hop is None else int(hop)
    row["hop_bin"] = hop_bin(hop)

    row["geo_midpoint_distance_m"] = np.nan

    if edge_meta is not None:
        required = {"model_node_id", "mid_lat", "mid_lon"}
        if required.issubset(edge_meta.columns):
            meta_by_id = edge_meta.set_index("model_node_id", drop=False)
            sid_i = int(segment_ids[i])
            sid_j = int(segment_ids[j])
            if sid_i in meta_by_id.index and sid_j in meta_by_id.index:
                ri = meta_by_id.loc[sid_i]
                rj = meta_by_id.loc[sid_j]
                try:
                    row["geo_midpoint_distance_m"] = haversine_m(
                        ri["mid_lat"], ri["mid_lon"], rj["mid_lat"], rj["mid_lon"]
                    )
                except Exception:
                    pass

    return row


# ============================================================
# Plotting
# ============================================================

def plot_corr_distribution(corr_df: pd.DataFrame, out_dir: Path) -> None:
    if corr_df.empty:
        return

    for split in sorted(corr_df["split"].unique()):
        sub_split = corr_df[corr_df["split"] == split].copy()

        plt.figure(figsize=(11, 6))
        for topk, g in sub_split.groupby("topk"):
            plt.hist(
                g["abs_corr"].dropna().to_numpy(),
                bins=40,
                alpha=0.45,
                label=f"Top-{topk}",
                density=True,
            )
        plt.title(f"Distribution of |correlation| for Top-10 vs Top-50 | {split}")
        plt.xlabel("|correlation|")
        plt.ylabel("Density")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        path = out_dir / f"corr_distribution_top10_top50_{split}.png"
        plt.savefig(path, dpi=170)
        plt.close()
        print("Saved:", path)

        plt.figure(figsize=(10, 6))
        box_data = []
        labels = []
        for topk in sorted(sub_split["topk"].unique()):
            vals = sub_split[sub_split["topk"] == topk]["abs_corr"].dropna().to_numpy()
            if len(vals) > 0:
                box_data.append(vals)
                labels.append(f"Top-{topk}")
        if box_data:
            plt.boxplot(box_data, labels=labels, showfliers=False)
            plt.title(f"Boxplot of |correlation| for Top-10 vs Top-50 | {split}")
            plt.ylabel("|correlation|")
            plt.grid(True, axis="y", alpha=0.3)
            plt.tight_layout()
            path = out_dir / f"corr_boxplot_top10_top50_{split}.png"
            plt.savefig(path, dpi=170)
            plt.close()
            print("Saved:", path)


def plot_top10_sample_size(sample_df: pd.DataFrame, out_dir: Path) -> None:
    if sample_df.empty:
        return

    metrics = [
        "sample_size_pair_window_mean",
        "sample_size_pair_window_min",
        "i_sample_size_mean",
        "j_sample_size_mean",
        "i_valid_ratio",
        "j_valid_ratio",
    ]

    for metric in metrics:
        if metric not in sample_df.columns:
            continue
        vals = sample_df[metric].replace([np.inf, -np.inf], np.nan).dropna()
        if vals.empty:
            continue

        for split in sorted(sample_df["split"].unique()):
            sub = sample_df[sample_df["split"] == split].copy()
            vals = sub[metric].replace([np.inf, -np.inf], np.nan).dropna()
            if vals.empty:
                continue

            plt.figure(figsize=(10, 5))
            plt.hist(vals.to_numpy(), bins=40, alpha=0.75)
            plt.title(f"Top-10 pair {metric} | {split}")
            plt.xlabel(metric)
            plt.ylabel("Count")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            path = out_dir / f"top10_{metric}_{split}.png"
            plt.savefig(path, dpi=170)
            plt.close()
            print("Saved:", path)


def plot_topology_summary(topology_df: pd.DataFrame, out_dir: Path) -> None:
    if topology_df.empty:
        return

    for split in sorted(topology_df["split"].unique()):
        sub = topology_df[topology_df["split"] == split].copy()
        if sub.empty:
            continue

        order = ["0-hop", "1-hop", "2-hop", "3-hop", "4-hop", "5-6-hop", ">6/unreachable"]
        counts = sub["hop_bin"].value_counts().reindex(order).fillna(0)

        plt.figure(figsize=(10, 5))
        plt.bar(counts.index.astype(str), counts.values)
        plt.title(f"Topology hop distance of Top-10 true-correlation pairs | {split}")
        plt.xlabel("OSM line-graph hop distance")
        plt.ylabel("Number of Top-10 pairs")
        plt.xticks(rotation=20)
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        path = out_dir / f"top10_topology_hop_distribution_{split}.png"
        plt.savefig(path, dpi=170)
        plt.close()
        print("Saved:", path)

        if "geo_midpoint_distance_m" in sub.columns:
            vals = sub["geo_midpoint_distance_m"].replace([np.inf, -np.inf], np.nan).dropna()
            if not vals.empty:
                plt.figure(figsize=(10, 5))
                plt.hist(vals.to_numpy(), bins=40, alpha=0.75)
                plt.title(f"Geographic distance of Top-10 pairs | {split}")
                plt.xlabel("Midpoint distance between two OSM edges (meters)")
                plt.ylabel("Count")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                path = out_dir / f"top10_geo_distance_distribution_{split}.png"
                plt.savefig(path, dpi=170)
                plt.close()
                print("Saved:", path)


# ============================================================
# Main analysis
# ============================================================

def run_analysis(args) -> None:
    project_root = find_project_root()

    common_dir = Path(args.common_dir)
    if not common_dir.is_absolute():
        common_dir = project_root / common_dir

    osm_dataset_dir = Path(args.osm_dataset_dir)
    if not osm_dataset_dir.is_absolute():
        osm_dataset_dir = project_root / osm_dataset_dir

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    ensure_dir(out_dir)

    print_stage("CONFIG")
    print("PROJECT_ROOT    :", project_root)
    print("COMMON_DIR      :", common_dir)
    print("OSM_DATASET_DIR :", osm_dataset_dir)
    print("OUT_DIR         :", out_dir)

    splits = parse_str_list(args.splits)
    lags = parse_int_list(args.lags)
    topks = parse_int_list(args.topks)
    rng = np.random.default_rng(args.seed)

    print("Splits:", splits)
    print("Lags  :", lags)
    print("TopKs :", topks)

    print_stage("LOAD OSM SAMPLE_SIZE + NODE QUALITY")
    osm_data = load_osm_tensor_dataset(osm_dataset_dir)
    node_quality = load_node_quality(osm_dataset_dir)
    quality_lookup = build_quality_lookup(node_quality)
    edge_meta = normalize_edge_meta(node_quality)

    all_sample_rows = []
    all_corr_rows = []
    all_topology_rows = []
    all_boundary_rows = []

    print_stage("PROCESS SPLITS")

    for split in splits:
        print_stage(f"SPLIT = {split}")
        split_data = load_rt_split(common_dir, split, mmap_mode="r")

        segment_ids = split_data["segment_ids"]
        adjacency = build_line_graph_adjacency(edge_meta, segment_ids)

        common_to_tensor = map_common_nodes_to_tensor_nodes(
            segment_ids,
            osm_data.get("model_node_ids") if osm_data.get("available") else None,
        )
        time_lookup = build_time_lookup(osm_data.get("timestamps") if osm_data.get("available") else None)

        print(f"R_series shape : {split_data['R_series'].shape}")
        print(f"Nodes          : {len(segment_ids)}")
        print(f"Tensor mapping : {len(common_to_tensor)}/{len(segment_ids)} nodes mapped")

        for lag in lags:
            pairs = iter_eval_pairs(split_data, lag)
            if not pairs:
                print(f"[WARN] No eval pairs for split={split}, lag={lag}")
                continue

            if args.samples_per_split_lag > 0 and len(pairs) > args.samples_per_split_lag:
                chosen_idx = rng.choice(len(pairs), size=args.samples_per_split_lag, replace=False)
                sample_pairs = [pairs[int(x)] for x in chosen_idx]
            else:
                sample_pairs = pairs

            print(f"[{split}] lag={lag}: using {len(sample_pairs)}/{len(pairs)} samples")

            for sample_id, (origin_idx, target_idx) in enumerate(sample_pairs):
                R_true = get_R(split_data, target_idx)
                tstamp = get_timestamp_for_r(split_data, target_idx)

                boundary = corr_boundary_stats(R_true, k_small=10, k_large=50)
                boundary.update({
                    "split": split,
                    "lag": int(lag),
                    "sample_id": int(sample_id),
                    "origin_idx": int(origin_idx),
                    "target_idx": int(target_idx),
                    "timestamp": str(tstamp) if tstamp is not None else "",
                })
                all_boundary_rows.append(boundary)

                for topk in topks:
                    pi, pj, vals = topk_offdiag_pairs_abs(R_true, topk)
                    for rank, (i, j, corr) in enumerate(zip(pi, pj, vals), start=1):
                        base_row = {
                            "split": split,
                            "lag": int(lag),
                            "sample_id": int(sample_id),
                            "origin_idx": int(origin_idx),
                            "target_idx": int(target_idx),
                            "timestamp": str(tstamp) if tstamp is not None else "",
                            "topk": int(topk),
                            "rank": int(rank),
                            "i": int(i),
                            "j": int(j),
                            "segment_i": int(segment_ids[i]),
                            "segment_j": int(segment_ids[j]),
                            "corr": float(corr),
                            "abs_corr": float(abs(corr)),
                        }
                        all_corr_rows.append(base_row)

                        # Chỉ phân tích sample_size + topology cho Top-10
                        if topk == 10:
                            srow = dict(base_row)
                            srow.update(
                                summarize_pair_sample_size(
                                    int(i),
                                    int(j),
                                    split_data,
                                    int(target_idx),
                                    osm_data,
                                    common_to_tensor,
                                    time_lookup,
                                    quality_lookup,
                                )
                            )
                            all_sample_rows.append(srow)

                            trow = dict(base_row)
                            trow = add_topology_info(
                                trow,
                                edge_meta,
                                adjacency,
                                segment_ids,
                                int(i),
                                int(j),
                            )
                            all_topology_rows.append(trow)

    print_stage("SAVE CSV OUTPUTS")

    sample_df = pd.DataFrame(all_sample_rows)
    corr_df = pd.DataFrame(all_corr_rows)
    topology_df = pd.DataFrame(all_topology_rows)
    boundary_df = pd.DataFrame(all_boundary_rows)

    sample_path = out_dir / "top10_pairs_sample_size_quality.csv"
    corr_path = out_dir / "top10_top50_correlation_values.csv"
    topology_path = out_dir / "top10_pairs_topology.csv"
    boundary_path = out_dir / "top10_top50_boundary_gap_stats.csv"

    sample_df.to_csv(sample_path, index=False, encoding="utf-8-sig")
    corr_df.to_csv(corr_path, index=False, encoding="utf-8-sig")
    topology_df.to_csv(topology_path, index=False, encoding="utf-8-sig")
    boundary_df.to_csv(boundary_path, index=False, encoding="utf-8-sig")

    print("Saved:", sample_path)
    print("Saved:", corr_path)
    print("Saved:", topology_path)
    print("Saved:", boundary_path)

    print_stage("SAVE SUMMARY TABLES")

    summary_rows = []

    if not sample_df.empty:
        group_cols = ["split", "lag"]
        agg_cols = [
            "sample_size_pair_window_mean",
            "sample_size_pair_window_min",
            "i_sample_size_mean",
            "j_sample_size_mean",
            "i_valid_ratio",
            "j_valid_ratio",
        ]

        available_agg = [c for c in agg_cols if c in sample_df.columns]

        if available_agg:
            summary_sample = (
                sample_df
                .groupby(group_cols)[available_agg]
                .agg(["mean", "median", "min", "max"])
                .reset_index()
            )
            summary_sample.columns = [
                "_".join([str(x) for x in col if str(x) != ""])
                for col in summary_sample.columns.values
            ]
            summary_sample_path = out_dir / "summary_top10_sample_size_by_split_lag.csv"
            summary_sample.to_csv(summary_sample_path, index=False, encoding="utf-8-sig")
            print("Saved:", summary_sample_path)

    if not topology_df.empty:
        topo_summary = (
            topology_df
            .groupby(["split", "lag", "hop_bin"])
            .size()
            .reset_index(name="count")
        )
        topo_summary_path = out_dir / "summary_top10_topology_hop_by_split_lag.csv"
        topo_summary.to_csv(topo_summary_path, index=False, encoding="utf-8-sig")
        print("Saved:", topo_summary_path)

    if not boundary_df.empty:
        boundary_summary = (
            boundary_df
            .groupby(["split", "lag"])[
                ["gap_rank10_rank11", "gap_rank50_rank51", "abs_corr_rank_10", "abs_corr_rank_50"]
            ]
            .agg(["mean", "median", "min", "max"])
            .reset_index()
        )
        boundary_summary.columns = [
            "_".join([str(x) for x in col if str(x) != ""])
            for col in boundary_summary.columns.values
        ]
        boundary_summary_path = out_dir / "summary_boundary_gap_by_split_lag.csv"
        boundary_summary.to_csv(boundary_summary_path, index=False, encoding="utf-8-sig")
        print("Saved:", boundary_summary_path)

    print_stage("PLOTS")
    plot_corr_distribution(corr_df, out_dir)
    plot_top10_sample_size(sample_df, out_dir)
    plot_topology_summary(topology_df, out_dir)

    print_stage("DONE")
    print("Main outputs:")
    print("  1)", sample_path)
    print("  2)", corr_path)
    print("  3)", topology_path)
    print("  4)", boundary_path)
    print()
    print("Gợi ý đọc kết quả:")
    print("- Nếu sample_size_pair_window_mean thấp ở nhiều cặp Top-10, có thể giải thích Top-10 bị nhiễu dữ liệu.")
    print("- Nếu gap_rank10_rank11 nhỏ, Top-10 rất nhạy vì rank 10 và rank 11 gần như ngang nhau.")
    print("- Nếu nhiều cặp Top-10 nằm >6-hop/unreachable hoặc khoảng cách địa lý xa, có thể là tương quan ngắn hạn/ngẫu nhiên.")


def build_argparser() -> argparse.ArgumentParser:
    project_root = find_project_root()

    p = argparse.ArgumentParser()

    p.add_argument(
        "--common-dir",
        type=str,
        default=str(project_root / "ml_core/src/models/ML_BranchA/data/05_branchA_prepare_segment_segment_rt"),
        help="Branch A common data dir containing train/val/test/R_series.npy",
    )

    p.add_argument(
        "--osm-dataset-dir",
        type=str,
        default=str(project_root / "ml_core/src/data_processing/outputs/branchA/osm_edge_forecasting_dataset"),
        help="OSM edge forecasting dataset dir containing osm_edge_tensor.npz and tables/node_quality.csv",
    )

    p.add_argument(
        "--out-dir",
        type=str,
        default=str(project_root / "ml_core/src/models/ML_BranchA/results/09_explain_top10_overlap"),
        help="Output directory",
    )

    p.add_argument(
        "--splits",
        type=str,
        default="val,test",
        help="Comma-separated splits, e.g. val,test",
    )

    p.add_argument(
        "--lags",
        type=str,
        default="1-9",
        help="Comma-separated lags or ranges, e.g. 1,3,6,9 or 1-9",
    )

    p.add_argument(
        "--topks",
        type=str,
        default="10,50",
        help="Top-K values for correlation distribution",
    )

    p.add_argument(
        "--samples-per-split-lag",
        type=int,
        default=20,
        help="Number of samples per split-lag. Use 0 for all samples.",
    )

    p.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run_analysis(args)
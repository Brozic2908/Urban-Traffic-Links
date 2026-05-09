from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional, List

import numpy as np
import pandas as pd


THIS_FILE = Path(__file__).resolve()
DATA_PROCESSING_DIR = THIS_FILE.parent
PROJECT_ROOT = DATA_PROCESSING_DIR.parents[2]

DEFAULT_SOURCE_NPZ = (
    DATA_PROCESSING_DIR
    / "outputs"
    / "branchA"
    / "osm_edge_forecasting_dataset"
    / "train_val_test_split.npz"
)

DEFAULT_OUTPUT_DIR = (
    DATA_PROCESSING_DIR
    / "outputs"
    / "branchB"
    / "osm_edge_base_like_branchA"
)


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now_str()}] {msg}", flush=True)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_npz(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing source NPZ: {path}")

    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def find_key(keys: List[str], candidates: List[str]) -> Optional[str]:
    lower_map = {k.lower(): k for k in keys}

    for c in candidates:
        if c in keys:
            return c
        if c.lower() in lower_map:
            return lower_map[c.lower()]

    return None


def get_split_array(obj: Dict[str, Any], split: str, names: List[str]) -> Optional[np.ndarray]:
    keys = list(obj.keys())

    candidates = []
    for name in names:
        candidates.extend([
            f"{name}_{split}",
            f"{split}_{name}",
            f"{name}{split}",
            f"{split}{name}",
        ])

    key = find_key(keys, candidates)
    if key is None:
        return None

    return obj[key]


def decode_strings(arr: np.ndarray) -> np.ndarray:
    out = []
    for x in np.asarray(arr).tolist():
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return np.asarray(out)


def get_feature_index(obj: Dict[str, Any], primary_feature: str) -> int:
    keys = list(obj.keys())
    feature_key = find_key(keys, ["feature_names", "features", "feature_cols", "columns"])

    if feature_key is None:
        log("[WARN] Cannot find feature_names. Use feature index 0.")
        return 0

    names = decode_strings(obj[feature_key]).tolist()

    if primary_feature in names:
        return int(names.index(primary_feature))

    log(f"[WARN] primary_feature={primary_feature} not found. Use feature index 0.")
    log(f"[WARN] Available features: {names}")
    return 0


def get_segment_ids(obj: Dict[str, Any], n_nodes: int) -> np.ndarray:
    keys = list(obj.keys())
    key = find_key(keys, ["segment_ids", "node_ids", "edge_ids", "osm_edge_ids", "model_node_ids"])

    if key is not None:
        ids = np.asarray(obj[key])
        if len(ids) == n_nodes:
            try:
                return ids.astype(np.int64)
            except Exception:
                return np.arange(n_nodes, dtype=np.int64)

    return np.arange(n_nodes, dtype=np.int64)


def resolve_node_indices(segment_ids: np.ndarray, max_nodes: int, node_sample: str, seed: int) -> Optional[np.ndarray]:
    n = int(len(segment_ids))

    if int(max_nodes) <= 0 or int(max_nodes) >= n:
        return None

    if node_sample == "first":
        return np.arange(int(max_nodes), dtype=np.int64)

    if node_sample == "random":
        rng = np.random.default_rng(int(seed))
        return np.sort(rng.choice(n, size=int(max_nodes), replace=False).astype(np.int64))

    raise ValueError("--node-sample must be first or random")


def normalize_timestamp_value(x: Any) -> str:
    s = str(x)

    if "__" in s:
        s = s.replace("__", " ")

    ts = pd.to_datetime(s, errors="coerce")
    if pd.notna(ts):
        return ts.strftime("%Y-%m-%d %H:%M:%S")

    return s


def parse_time_minutes(value: Any) -> int:
    s = str(value)

    m = re.search(r"(?<!\d)([0-2]?\d):([0-5]\d)(?!\d)", s)
    if m:
        hh, mm = int(m.group(1)), int(m.group(2))
        if 0 <= hh <= 23:
            return hh * 60 + mm

    m = re.search(r"(?<!\d)([0-2]\d)([0-5]\d)(?!\d)", s)
    if m:
        hh, mm = int(m.group(1)), int(m.group(2))
        if 0 <= hh <= 23:
            return hh * 60 + mm

    ts = pd.to_datetime(s, errors="coerce")
    if pd.notna(ts):
        return int(ts.hour) * 60 + int(ts.minute)

    return -1


def make_meta(timestamps: np.ndarray, split: str) -> pd.DataFrame:
    ts_norm = np.asarray([normalize_timestamp_value(x) for x in np.asarray(timestamps).tolist()])
    ts = pd.to_datetime(pd.Series(ts_norm), errors="coerce")

    date_key = ts.dt.strftime("%Y-%m-%d").fillna("unknown_date")
    time_minutes = []

    for raw, parsed in zip(ts_norm, ts):
        if pd.notna(parsed):
            time_minutes.append(int(parsed.hour) * 60 + int(parsed.minute))
        else:
            time_minutes.append(parse_time_minutes(raw))

    meta = pd.DataFrame({
        "split": split,
        "timestamp_local": ts_norm,
        "date_key": date_key.astype(str),
        "time_minutes": np.asarray(time_minutes, dtype=np.int32),
    })

    meta["session_id"] = meta["date_key"].astype(str)
    meta["slot_index"] = meta.groupby("session_id").cumcount().astype(int)

    return meta


def make_synthetic_timestamps(n: int, split: str) -> np.ndarray:
    base_date = {
        "train": "2000-01-01",
        "val": "2000-02-01",
        "test": "2000-03-01",
    }.get(split, "2000-01-01")

    out = []
    start = pd.Timestamp(base_date + " 06:00:00")

    for i in range(n):
        out.append(str(start + pd.Timedelta(minutes=15 * i)))

    return np.asarray(out)


def extract_split(obj: Dict[str, Any], split: str, primary_feature: str) -> Dict[str, Any]:
    X = get_split_array(obj, split, ["X", "x", "data", "tensor", "z"])
    timestamps = get_split_array(obj, split, ["timestamps", "timestamp", "times", "time"])

    if X is None:
        raise KeyError(
            f"Cannot find X array for split={split}. "
            f"Expected keys like X_{split}, {split}_X, z_{split}. "
            f"NPZ keys={list(obj.keys())}"
        )

    X = np.asarray(X)

    if X.ndim == 3:
        fidx = get_feature_index(obj, primary_feature)
        z = X[:, :, fidx].astype(np.float32)
    elif X.ndim == 2:
        z = X.astype(np.float32)
    else:
        raise ValueError(f"Unsupported X shape for split={split}: {X.shape}")

    if timestamps is None:
        log(f"[WARN] Missing timestamps for split={split}. Use synthetic timestamps.")
        timestamps = make_synthetic_timestamps(z.shape[0], split)

    return {
        "z": z,
        "timestamps": np.asarray(timestamps),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-npz", type=str, default=str(DEFAULT_SOURCE_NPZ))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--primary-feature", type=str, default="average_speed")
    parser.add_argument("--max-nodes", type=int, default=0)
    parser.add_argument("--node-sample", type=str, default="first", choices=["first", "random"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_npz = Path(args.source_npz)
    if not source_npz.is_absolute():
        source_npz = PROJECT_ROOT / source_npz

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    log("PREPARE BRANCH B BASE LIKE BRANCH A")
    log(f"PROJECT_ROOT: {PROJECT_ROOT}")
    log(f"SOURCE_NPZ  : {source_npz}")
    log(f"OUTPUT_DIR  : {output_dir}")

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output_dir}. Use --overwrite or choose a new output dir.")
        log(f"[CLEAN] Removing old output: {output_dir}")
        shutil.rmtree(output_dir)

    obj = load_npz(source_npz)
    log(f"NPZ keys: {list(obj.keys())}")

    splits = {}
    for split in ["train", "val", "test"]:
        splits[split] = extract_split(obj, split, args.primary_feature)

    n_nodes = int(splits["train"]["z"].shape[1])
    segment_ids = get_segment_ids(obj, n_nodes)

    node_idx = resolve_node_indices(
        segment_ids=segment_ids,
        max_nodes=int(args.max_nodes),
        node_sample=str(args.node_sample),
        seed=int(args.seed),
    )

    if node_idx is None:
        segment_ids_out = segment_ids
        log(f"NODE_MODE: full graph, N={len(segment_ids_out)}")
    else:
        segment_ids_out = segment_ids[node_idx]
        log(f"NODE_MODE: subset, N={len(segment_ids_out)}")

    ensure_dir(output_dir)

    summary = {
        "source_npz": str(source_npz),
        "output_dir": str(output_dir),
        "primary_feature": args.primary_feature,
        "max_nodes": int(args.max_nodes),
        "node_sample": args.node_sample,
        "splits": {},
    }

    for split in ["train", "val", "test"]:
        split_dir = ensure_dir(output_dir / split)

        z = splits[split]["z"]
        timestamps = splits[split]["timestamps"]

        if node_idx is not None:
            z = z[:, node_idx]

        meta = make_meta(timestamps, split)

        np.save(split_dir / "z.npy", z.astype(np.float32))
        np.save(split_dir / "segment_ids.npy", np.asarray(segment_ids_out, dtype=np.int64))
        np.save(split_dir / "timestamps.npy", np.asarray([normalize_timestamp_value(x) for x in timestamps]))
        meta.to_csv(split_dir / "G_series_meta.csv", index=False)

        summary["splits"][split] = {
            "z_shape": list(map(int, z.shape)),
            "segment_ids_shape": list(map(int, np.asarray(segment_ids_out).shape)),
            "timestamps_shape": list(map(int, np.asarray(timestamps).shape)),
            "meta_rows": int(len(meta)),
            "n_dates": int(meta["date_key"].nunique()),
        }

        log(f"[{split}] z={z.shape}, segment_ids={len(segment_ids_out)}, meta={len(meta)}")

    with open(output_dir / "branchB_base_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log("DONE.")
    log(f"Saved base dir: {output_dir}")


if __name__ == "__main__":
    main()
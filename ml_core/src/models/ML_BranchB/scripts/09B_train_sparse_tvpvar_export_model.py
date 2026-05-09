# ml_core/src/models/ML_BranchB/scripts/09B_train_sparse_tvpvar_export_model.py
"""
Train/export Sparse TVP-VAR graph model for Branch B backend inference.

Goal
----
Offline step:
    train/val/test prepared Branch-B data
        -> build_g_model("sparse_tvpvar_gt", train, val, test)
        -> save model artifact

Backend step is handled by:
    10B_sparse_tvpvar_predict_service_backend.py

Run from project root:
    python -u ml_core/src/models/ML_BranchB/scripts/09B_train_sparse_tvpvar_export_model.py \
      --data-dir ml_core/src/data_processing/outputs/branchB/osm_edge_granger_series_like_branchA \
      --lags 1-9 \
      --overwrite
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd


METHOD_NAME = "sparse_tvpvar_gt"
STABLE_METHOD_MODULE_NAME = "branchB_sparse_tvpvar_method"


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now_str()}] {msg}", flush=True)


def find_project_root() -> Path:
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents, Path("/kaggle/working/UTraffic-ML"), Path("/kaggle/working")]:
        if (p / "ml_core").exists():
            return p if p.name == "UTraffic-ML" or (p / "ml_core").exists() else p / "UTraffic-ML"
        if (p / "UTraffic-ML").exists() and (p / "UTraffic-ML" / "ml_core").exists():
            return p / "UTraffic-ML"
    return cwd


def parse_int_list(s: Optional[str]) -> List[int]:
    if s is None or str(s).strip() == "":
        return []
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


def load_method_module(script_path: Path):
    """
    Load the original Sparse TVP-VAR method script under a stable module name.

    This matters for pickle: if the exported g_model contains objects/classes
    from the method module, loading it later must use the same module name.
    """
    if not script_path.exists():
        raise FileNotFoundError(f"Cannot find Sparse TVP-VAR method script: {script_path}")

    spec = importlib.util.spec_from_file_location(STABLE_METHOD_MODULE_NAME, str(script_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import method module from: {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[STABLE_METHOD_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def resolve_node_indices(
    segment_ids: np.ndarray,
    max_nodes: int = 0,
    node_sample: str = "first",
    seed: int = 42,
) -> Optional[np.ndarray]:
    N = int(len(segment_ids))
    max_nodes = int(max_nodes)
    if max_nodes <= 0 or max_nodes >= N:
        return None

    if node_sample == "first":
        return np.arange(max_nodes, dtype=np.int64)

    if node_sample == "random":
        rng = np.random.default_rng(int(seed))
        return np.sort(rng.choice(N, size=max_nodes, replace=False).astype(np.int64))

    raise ValueError("--node-sample must be first or random")


def subset_split_data(data: Dict[str, Any], node_idx: Optional[np.ndarray]) -> Dict[str, Any]:
    if node_idx is None:
        return data

    idx = np.asarray(node_idx, dtype=np.int64)
    out = dict(data)
    out["_node_idx"] = idx

    if "z" in out:
        out["z"] = np.asarray(out["z"])[:, idx]

    if "G_weight_series" in out:
        G = np.asarray(out["G_weight_series"])
        out["G_weight_series"] = G[:, idx, :][:, :, idx].astype(np.float32, copy=False)

    if "G_best_lag_series" in out:
        L = np.asarray(out["G_best_lag_series"])
        out["G_best_lag_series"] = L[:, idx, :][:, :, idx]

    if "segment_ids" in out:
        out["segment_ids"] = np.asarray(out["segment_ids"], dtype=np.int64)[idx]

    return out


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return str(o)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=default)


def write_matrix_axis(segment_ids: np.ndarray, out_path: Path) -> None:
    df = pd.DataFrame({
        "matrix_index": np.arange(len(segment_ids), dtype=np.int64),
        "segment_id": np.asarray(segment_ids, dtype=np.int64),
        "role": "row_target_and_column_source",
        "contract": "G[target_index, source_index] = directed influence from source to target",
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def parse_args() -> argparse.Namespace:
    project_root = find_project_root()
    default_data_dir = project_root / "ml_core" / "src" / "data_processing" / "outputs" / "branchB" / "osm_edge_granger_series_like_branchA"
    default_scripts_dir = project_root / "ml_core" / "src" / "models" / "ML_BranchB" / "scripts"
    default_method_script = default_scripts_dir / "06_branchB_run_xt_forecast_sparse_tvpvar_gt.py"
    default_out_dir = project_root / "ml_core" / "src" / "models" / "ML_BranchB" / "artifacts" / "sparse_tvpvar_gt_model"

    ap = argparse.ArgumentParser(description="Train/export Branch-B Sparse TVP-VAR model for backend inference.")
    ap.add_argument("--data-dir", type=str, default=str(default_data_dir))
    ap.add_argument("--method-script", type=str, default=str(default_method_script))
    ap.add_argument("--out-dir", type=str, default=str(default_out_dir))
    ap.add_argument("--lags", type=str, default="1-9", help="Horizons, e.g. 1-9 or 1,2,3.")
    ap.add_argument("--max-nodes", type=int, default=0, help="0 means full graph. Use 512 for smoke test only.")
    ap.add_argument("--node-sample", type=str, default="first", choices=["first", "random"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    project_root = find_project_root()

    common_dir = Path(args.data_dir)
    if not common_dir.is_absolute():
        common_dir = project_root / common_dir

    method_script = Path(args.method_script)
    if not method_script.is_absolute():
        method_script = project_root / method_script

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir

    log("=" * 96)
    log("BRANCH-B SPARSE TVP-VAR OFFLINE TRAIN/EXPORT")
    log("=" * 96)
    log(f"PROJECT_ROOT : {project_root}")
    log(f"DATA_DIR     : {common_dir}")
    log(f"METHOD_SCRIPT: {method_script}")
    log(f"OUT_DIR      : {out_dir}")
    log(f"LAGS         : {args.lags}")
    log(f"MAX_NODES    : {args.max_nodes}")

    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output dir exists: {out_dir}. Pass --overwrite to replace.")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    horizons = parse_int_list(args.lags)
    if not horizons:
        raise ValueError("--lags cannot be empty.")

    module = load_method_module(method_script)

    if hasattr(module, "check_branchB_common_dir_ready"):
        module.check_branchB_common_dir_ready(common_dir)

    log("Loading train/val/test splits...")
    train = module.load_gt_split(common_dir, "train")
    val = module.load_gt_split(common_dir, "val")
    test = module.load_gt_split(common_dir, "test")

    if not (np.array_equal(train["segment_ids"], val["segment_ids"]) and np.array_equal(train["segment_ids"], test["segment_ids"])):
        raise ValueError("segment_ids mismatch across train/val/test.")

    full_segment_ids = np.asarray(train["segment_ids"], dtype=np.int64)
    node_idx = resolve_node_indices(
        segment_ids=full_segment_ids,
        max_nodes=args.max_nodes,
        node_sample=args.node_sample,
        seed=args.seed,
    )

    train = subset_split_data(train, node_idx)
    val = subset_split_data(val, node_idx)
    test = subset_split_data(test, node_idx)

    segment_ids = np.asarray(train["segment_ids"], dtype=np.int64)
    module.HORIZONS = list(map(int, horizons))

    log(f"n_segments   : {len(segment_ids)}")
    log("Fitting Sparse TVP-VAR g_model offline...")
    t0 = time.perf_counter()
    g_model = module.build_g_model(METHOD_NAME, train, val, test)
    offline_fit_time_sec = time.perf_counter() - t0
    log(f"Offline fit completed in {offline_fit_time_sec:.4f} sec.")

    model_path = out_dir / "sparse_tvpvar_g_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(g_model, f, protocol=pickle.HIGHEST_PROTOCOL)

    np.save(out_dir / "segment_ids.npy", segment_ids.astype(np.int64))
    if node_idx is not None:
        np.save(out_dir / "node_idx.npy", node_idx.astype(np.int64))
    else:
        np.save(out_dir / "node_idx.npy", np.array([], dtype=np.int64))

    write_matrix_axis(segment_ids, out_dir / "matrix_axis.csv")

    config = {
        "method": METHOD_NAME,
        "method_label": "Sparse TVP-VAR-Gt",
        "data_dir": str(common_dir),
        "method_script": str(method_script),
        "model_path": str(model_path),
        "stable_method_module_name": STABLE_METHOD_MODULE_NAME,
        "horizons": list(map(int, horizons)),
        "n_segments": int(len(segment_ids)),
        "max_nodes": int(args.max_nodes),
        "node_sample": args.node_sample,
        "seed": int(args.seed),
        "used_node_subset": bool(node_idx is not None),
        "offline_fit_time_sec": float(offline_fit_time_sec),
        "matrix_contract": "G_pred[target, source] = predicted directed influence from source road to target road",
        "backend_rule": "load fitted artifact, predict only on test split, do not refit online",
        "created_at": now_str(),
    }
    save_json(config, out_dir / "sparse_tvpvar_config.json")

    log("[SAVED] model       : " + str(model_path))
    log("[SAVED] config      : " + str(out_dir / "sparse_tvpvar_config.json"))
    log("[SAVED] segment_ids : " + str(out_dir / "segment_ids.npy"))
    log("[SAVED] matrix_axis : " + str(out_dir / "matrix_axis.csv"))
    log("DONE")


if __name__ == "__main__":
    main()

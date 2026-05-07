# ml_core/src/models/ML_BranchB/scripts/08B_benchmark_gt_runtime_only.py
"""
Benchmark runtime only for Branch B graph prediction.

Mục tiêu:
1) Đo thời gian tính G_weight_series trung bình, nhưng KHÔNG lưu G_weight_series.
2) Đo offline time của từng method:
      G_train -> build_g_model(...)
3) Đo online time của từng method:
      G_t -> predict_G_method(...) -> G_hat[t+h|t]
4) Plot thời gian giữa các method.

Không làm:
- Không dự báo X_t.
- Không tính TopK(G) @ X_t.
- Không so sánh G_hat với G_true.
- Không tính MAE/RMSE/Overlap.
- Không tính practical_score.
- Không chọn method tốt nhất.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHOD_TO_SCRIPT = {
    "true_gt": "06_branchB_run_xt_forecast_true_gt.py",
    "persistence_gt": "06_branchB_run_xt_forecast_persistence_gt.py",
    "ewma_gt": "06_branchB_run_xt_forecast_ewma_gt.py",
    "sparse_var_gt": "06_branchB_run_xt_forecast_sparse_var_gt.py",
    "sparse_tvpvar_gt": "06_branchB_run_xt_forecast_sparse_tvpvar_gt.py",
    "dmfm_lse_gt": "06_branchB_run_xt_forecast_dmfm_gt.py",
    "dmfm_vlse_gt": "06_branchB_run_xt_forecast_dmfm_gt.py",
}

PRACTICAL_METHODS = [
    "persistence_gt",
    "ewma_gt",
    "sparse_var_gt",
    "sparse_tvpvar_gt",
    "dmfm_lse_gt",
    "dmfm_vlse_gt",
]

METHOD_LABELS = {
    "true_gt": "True-Gt",
    "persistence_gt": "Persistence-Gt",
    "ewma_gt": "EWMA-Gt",
    "sparse_var_gt": "Sparse VAR-Gt",
    "sparse_tvpvar_gt": "Sparse TVP-VAR-Gt",
    "dmfm_lse_gt": "DMFM-LSE-Gt",
    "dmfm_vlse_gt": "DMFM-VLSE-Gt",
}


# =============================================================================
# Basic utilities
# =============================================================================

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def log(msg: str) -> None:
    print(f"[{now_str()}] {msg}", flush=True)


def find_project_root() -> Path:
    cwd = Path.cwd().resolve()
    candidates = [
        cwd,
        *cwd.parents,
        Path("/kaggle/working/UTraffic-ML"),
        Path("/kaggle/working"),
    ]
    for p in candidates:
        if (p / "ml_core").exists() and (p / "dataset").exists():
            return p
        if p.name == "UTraffic-ML" and (p / "ml_core").exists():
            return p
        if (p / "UTraffic-ML").exists():
            pp = p / "UTraffic-ML"
            if (pp / "ml_core").exists():
                return pp
    return cwd


def parse_str_list(s: Optional[str]) -> List[str]:
    if s is None:
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]


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


def expand_methods(methods_arg: str, include_true: bool = False) -> List[str]:
    tokens = parse_str_list(methods_arg) or ["all"]
    out: List[str] = []
    for t in tokens:
        if t == "all":
            out.extend(PRACTICAL_METHODS)
        elif t == "all_with_true":
            out.extend(["true_gt"] + PRACTICAL_METHODS)
        elif t in METHOD_TO_SCRIPT:
            out.append(t)
        else:
            raise ValueError(f"Unknown method={t}")

    if include_true and "true_gt" not in out:
        out = ["true_gt"] + out

    seen = set()
    final = []
    for m in out:
        if m not in seen:
            final.append(m)
            seen.add(m)
    return final


def safe_output_dir(project_root: Path, out_dir_arg: Optional[str]) -> Path:
    """
    Không ghi đè.
    Nếu user không truyền out-dir thì tạo folder timestamp.
    Nếu user truyền out-dir nhưng folder đã tồn tại thì tự thêm suffix timestamp.
    """
    default_base = (
        project_root
        / "ml_core"
        / "src"
        / "models"
        / "ML_BranchB"
        / "results"
        / f"08_gt_runtime_full_{stamp()}"
    )

    if out_dir_arg is None or str(out_dir_arg).strip() == "":
        out_dir = default_base
    else:
        out_dir = Path(out_dir_arg)
        if not out_dir.is_absolute():
            out_dir = project_root / out_dir
        if out_dir.exists():
            out_dir = out_dir.parent / f"{out_dir.name}_{stamp()}"

    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def load_module(script_path: Path):
    module_name = f"runtime_module_{script_path.stem}_{abs(hash(str(script_path))) % 10**8}"
    spec = importlib.util.spec_from_file_location(module_name, str(script_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_node_indices(
    segment_ids: np.ndarray,
    max_nodes: int = 0,
    node_sample: str = "first",
    seed: int = 42,
) -> Optional[np.ndarray]:
    """
    max_nodes=0 hoặc max_nodes >= N nghĩa là full graph.
    """
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

    if "z" in out:
        out["z"] = np.asarray(out["z"])[:, idx]

    if "G_weight_series" in out:
        G = out["G_weight_series"]
        out["G_weight_series"] = np.asarray(G[:, idx][:, :, idx], dtype=np.float32)

    if "G_best_lag_series" in out:
        L = out["G_best_lag_series"]
        out["G_best_lag_series"] = np.asarray(L[:, idx][:, :, idx])

    if "segment_ids" in out:
        out["segment_ids"] = np.asarray(out["segment_ids"], dtype=np.int64)[idx]

    return out


def get_peak_ram_mb() -> float:
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux/Kaggle: KB
        return float(rss) / 1024.0
    except Exception:
        return float("nan")


# =============================================================================
# Runtime benchmark for methods
# =============================================================================

def collect_pairs(module, meta: pd.DataFrame, horizon: int, max_samples: int) -> List[Tuple[int, int]]:
    if hasattr(module, "iter_eval_pairs"):
        pairs = list(module.iter_eval_pairs(meta, int(horizon)))
    else:
        pairs = []
        for i in range(0, len(meta) - int(horizon)):
            pairs.append((i, i + int(horizon)))

    if int(max_samples) > 0:
        pairs = pairs[: int(max_samples)]

    return [(int(a), int(b)) for a, b in pairs]


def benchmark_one_method(
    method_name: str,
    scripts_dir: Path,
    common_dir: Path,
    lags: List[int],
    splits: List[str],
    node_idx: Optional[np.ndarray],
    max_samples_per_lag: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    script_path = scripts_dir / METHOD_TO_SCRIPT[method_name]
    module = load_module(script_path)

    if hasattr(module, "check_branchB_common_dir_ready"):
        module.check_branchB_common_dir_ready(common_dir)

    train = module.load_gt_split(common_dir, "train")
    val = module.load_gt_split(common_dir, "val")
    test = module.load_gt_split(common_dir, "test")

    train = subset_split_data(train, node_idx)
    val = subset_split_data(val, node_idx)
    test = subset_split_data(test, node_idx)

    split_map = {
        "train": train,
        "val": val,
        "test": test,
    }

    n_segments = int(len(train["segment_ids"]))

    # Một số module đọc biến global HORIZONS
    module.HORIZONS = list(map(int, lags))

    log(f"[{method_name}] n_segments={n_segments}")
    log(f"[{method_name}] building G model / offline fit...")

    t0 = time.perf_counter()
    g_model = module.build_g_model(method_name, train, val, test)
    offline_fit_time_sec = time.perf_counter() - t0

    method_meta = {
        "method": method_name,
        "method_label": METHOD_LABELS.get(method_name, method_name),
        "n_segments": n_segments,
        "offline_fit_time_sec": float(offline_fit_time_sec),
        "peak_ram_mb_after_fit": float(get_peak_ram_mb()),
    }

    log(f"[{method_name}] offline_fit_time_sec={offline_fit_time_sec:.4f}")

    rows: List[Dict[str, Any]] = []

    for split_name in splits:
        if split_name not in split_map:
            raise ValueError(f"Unknown split={split_name}. Use train,val,test.")

        split_data = split_map[split_name]
        meta = split_data["meta"]

        for h in lags:
            pairs = collect_pairs(module, meta, int(h), int(max_samples_per_lag))
            predict_total_sec = 0.0
            n_pairs = len(pairs)

            log(f"[{method_name}] split={split_name} lag={h} predict samples={n_pairs}")

            for origin_idx, target_idx in pairs:
                t1 = time.perf_counter()
                G_pred = module.predict_G_method(
                    method_name,
                    g_model,
                    split_name,
                    split_data,
                    int(origin_idx),
                    int(target_idx),
                    int(h),
                )
                predict_total_sec += time.perf_counter() - t1

                # Không lưu G_pred, chỉ đo runtime.
                del G_pred

            avg_ms = predict_total_sec * 1000.0 / max(n_pairs, 1)

            rows.append({
                "method": method_name,
                "method_label": METHOD_LABELS.get(method_name, method_name),
                "split": split_name,
                "lag": int(h),
                "n_segments": int(n_segments),
                "n_samples": int(n_pairs),
                "offline_fit_time_sec": float(offline_fit_time_sec),
                "online_predict_time_sec": float(predict_total_sec),
                "avg_online_ms_per_sample": float(avg_ms),
                "peak_ram_mb": float(get_peak_ram_mb()),
            })

    return rows, method_meta


# =============================================================================
# Benchmark G_weight compute time without saving
# =============================================================================

def load_prepare_module(project_root: Path):
    prep_path = (
        project_root
        / "ml_core"
        / "src"
        / "data_processing"
        / "prepare_branchB_osm_edge_granger_series_like_branchA.py"
    )
    return load_module(prep_path)


def benchmark_gweight_compute_time(
    project_root: Path,
    source_dir: Path,
    max_nodes: int,
    node_sample: str,
    seed: int,
    granger_p: int,
    bucket_minutes: int,
    min_bucket_samples: int,
    max_candidates: int,
    candidate_block_size: int,
    ridge: float,
    min_improvement: float,
    fit_intercept: bool,
    unsigned: bool,
    max_buckets: int,
) -> pd.DataFrame:
    """
    Đo thời gian tính các graph G_weight kiểu Granger-series nhưng không lưu G_weight_series.npy.

    Lưu ý:
    - prepare Granger-series không tính một G riêng cho từng timestamp.
    - Nó học graph global/bucket trên train, rồi expand graph đó cho các timestamp.
    - Vì vậy script này đo:
        1) time tính global graph
        2) time tính từng bucket graph
        3) trung bình theo unique graph và ước lượng theo valid timestamp.
    """
    prep = load_prepare_module(project_root)

    log("[G_WEIGHT] loading base train split...")
    train = prep.load_base_split(source_dir, "train", mmap=False)
    train = prep.sort_split_by_day_time(train)

    node_idx = prep.resolve_node_indices(
        train["segment_ids"],
        max_nodes=int(max_nodes),
        node_indices_arg=None,
        node_ids_arg=None,
        node_sample=str(node_sample),
        seed=int(seed),
    )

    train = prep.subset_nodes(train, node_idx)
    train = prep.add_history_and_bucket_columns(
        train,
        p=int(granger_p),
        bucket_minutes=int(bucket_minutes),
    )

    z = np.asarray(train["z"], dtype=np.float32)
    meta = train["meta"]
    N = int(z.shape[1])

    log(f"[G_WEIGHT] benchmark N={N}, T={len(meta)}, granger_p={granger_p}")

    rows: List[Dict[str, Any]] = []

    def compute_one(bucket_id: Optional[int], label: str):
        X_lags, Y, origins = prep.build_training_samples_for_bucket(
            z=z,
            meta=meta,
            p=int(granger_p),
            bucket_id=bucket_id,
        )

        if X_lags.shape[0] <= max(2 * int(granger_p) + 2, 4):
            log(f"[G_WEIGHT] skip {label}: too few samples={X_lags.shape[0]}")
            return

        t0 = time.perf_counter()
        G, L, summary = prep.compute_granger_graph(
            X_lags=X_lags,
            Y=Y,
            p=int(granger_p),
            max_candidates=int(max_candidates),
            candidate_block_size=int(candidate_block_size),
            ridge=float(ridge),
            fit_intercept=bool(fit_intercept),
            signed=not bool(unsigned),
            min_improvement=float(min_improvement),
            dtype="float32",
            lag_dtype="int16",
            label=label,
        )
        elapsed = time.perf_counter() - t0

        # Không lưu G/L.
        del G, L

        if bucket_id is None:
            valid_ts_count = int(meta["can_predict_granger"].sum())
        else:
            valid_ts_count = int(
                (
                    (meta["can_predict_granger"].astype(bool))
                    & (meta["bucket_id"].astype(int) == int(bucket_id))
                ).sum()
            )

        rows.append({
            "graph_kind": "global" if bucket_id is None else "bucket",
            "bucket_id": -1 if bucket_id is None else int(bucket_id),
            "label": label,
            "n_segments": int(N),
            "n_train_samples_for_graph": int(X_lags.shape[0]),
            "n_valid_timestamps_using_this_graph": int(valid_ts_count),
            "compute_G_weight_time_sec": float(elapsed),
            "avg_ms_per_train_sample": float(elapsed * 1000.0 / max(int(X_lags.shape[0]), 1)),
            "avg_ms_per_valid_timestamp_est": float(elapsed * 1000.0 / max(valid_ts_count, 1)),
            "peak_ram_mb": float(get_peak_ram_mb()),
        })

    # Global graph
    compute_one(None, "global")

    # Bucket graphs
    bucket_ids = sorted([int(x) for x in meta["bucket_id"].dropna().unique() if int(x) >= 0])
    if int(max_buckets) > 0:
        bucket_ids = bucket_ids[: int(max_buckets)]

    for bid in bucket_ids:
        n_bucket = int(
            (
                (meta["can_predict_granger"].astype(bool))
                & (meta["bucket_id"].astype(int) == bid)
            ).sum()
        )

        if n_bucket < int(min_bucket_samples):
            log(
                f"[G_WEIGHT] skip bucket={bid}: "
                f"valid samples={n_bucket} < min_bucket_samples={min_bucket_samples}"
            )
            continue

        compute_one(bid, f"bucket_{bid}")

    return pd.DataFrame(rows)


# =============================================================================
# Plots
# =============================================================================

def save_bar_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    ylabel: str,
    out_path: Path,
    rotate: int = 25,
):
    if df.empty:
        return

    plt.figure(figsize=(12, 6))
    plt.bar(df[x].astype(str), df[y].astype(float))
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotate, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_online_by_lag_plot(df: pd.DataFrame, out_path: Path):
    if df.empty:
        return

    plt.figure(figsize=(12, 6))

    for method_label, sub in df.groupby("method_label"):
        sub = sub.sort_values("lag")
        plt.plot(
            sub["lag"],
            sub["avg_online_ms_per_sample"],
            marker="o",
            label=str(method_label),
        )

    plt.title("Online predict time by horizon")
    plt.xlabel("Horizon / lag")
    plt.ylabel("Avg online predict time (ms/sample)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def make_plots(out_dir: Path, detail_df: pd.DataFrame, gweight_df: pd.DataFrame):
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    if not detail_df.empty:
        # Offline: one value per method
        offline_df = (
            detail_df
            .groupby(["method", "method_label"], as_index=False)
            .agg(offline_fit_time_sec=("offline_fit_time_sec", "max"))
            .sort_values("offline_fit_time_sec", ascending=False)
        )

        save_bar_plot(
            offline_df,
            x="method_label",
            y="offline_fit_time_sec",
            title="Offline fit time by method",
            ylabel="Offline fit time (seconds)",
            out_path=plot_dir / "plot_offline_fit_time_by_method.png",
        )

        # Online: average over splits/lags
        online_df = (
            detail_df
            .groupby(["method", "method_label"], as_index=False)
            .agg(avg_online_ms_per_sample=("avg_online_ms_per_sample", "mean"))
            .sort_values("avg_online_ms_per_sample", ascending=False)
        )

        save_bar_plot(
            online_df,
            x="method_label",
            y="avg_online_ms_per_sample",
            title="Average online predict time by method",
            ylabel="Avg online predict time (ms/sample)",
            out_path=plot_dir / "plot_online_avg_time_by_method.png",
        )

        # Online by lag
        online_lag_df = (
            detail_df
            .groupby(["method", "method_label", "lag"], as_index=False)
            .agg(avg_online_ms_per_sample=("avg_online_ms_per_sample", "mean"))
        )

        save_online_by_lag_plot(
            online_lag_df,
            out_path=plot_dir / "plot_online_time_by_lag.png",
        )

    if not gweight_df.empty:
        gplot_df = gweight_df.copy()
        gplot_df["graph_label"] = gplot_df["label"].astype(str)

        save_bar_plot(
            gplot_df,
            x="graph_label",
            y="compute_G_weight_time_sec",
            title="G_weight graph compute time without saving",
            ylabel="Compute time (seconds)",
            out_path=plot_dir / "plot_g_weight_compute_time.png",
        )


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Prepared Branch-B G_weight full data dir.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output dir. If exists, timestamp suffix is added automatically.",
    )
    parser.add_argument("--methods", type=str, default="all")
    parser.add_argument("--splits", type=str, default="val,test")
    parser.add_argument("--lags", type=str, default="1-9")

    parser.add_argument(
        "--max-nodes",
        type=int,
        default=0,
        help="0 means full graph. Use only for smoke test.",
    )
    parser.add_argument(
        "--node-sample",
        type=str,
        default="first",
        choices=["first", "random"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-samples-per-lag",
        type=int,
        default=0,
        help="0 means all samples.",
    )

    # G_weight compute benchmark
    parser.add_argument("--measure-gweight-compute", action="store_true")
    parser.add_argument(
        "--gweight-source-dir",
        type=str,
        default=None,
        help="Usually outputs/branchB/osm_edge_base_like_branchA",
    )
    parser.add_argument("--granger-p", type=int, default=3)
    parser.add_argument("--bucket-minutes", type=int, default=60)
    parser.add_argument("--min-bucket-samples", type=int, default=20)
    parser.add_argument("--max-candidates", type=int, default=50)
    parser.add_argument("--candidate-block-size", type=int, default=256)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--min-improvement", type=float, default=0.0)
    parser.add_argument("--fit-intercept", action="store_true")
    parser.add_argument("--unsigned", action="store_true")
    parser.add_argument(
        "--max-buckets",
        type=int,
        default=0,
        help="0 means all buckets. Use small number for smoke test.",
    )

    args = parser.parse_args()

    project_root = find_project_root()
    scripts_dir = project_root / "ml_core" / "src" / "models" / "ML_BranchB" / "scripts"

    common_dir = Path(args.data_dir)
    if not common_dir.is_absolute():
        common_dir = project_root / common_dir

    out_dir = safe_output_dir(project_root, args.out_dir)

    methods = expand_methods(args.methods)
    splits = parse_str_list(args.splits) or ["val", "test"]
    lags = parse_int_list(args.lags) or list(range(1, 10))

    log("=" * 88)
    log("BRANCH B GT RUNTIME BENCHMARK ONLY")
    log("=" * 88)
    log(f"PROJECT_ROOT : {project_root}")
    log(f"DATA_DIR     : {common_dir}")
    log(f"OUT_DIR      : {out_dir}")
    log(f"METHODS      : {methods}")
    log(f"SPLITS       : {splits}")
    log(f"LAGS         : {lags}")
    log(f"MAX_NODES    : {args.max_nodes} | 0 means full")
    log(f"MAX_SAMPLES  : {args.max_samples_per_lag} | 0 means all")

    # Save config
    config = vars(args).copy()
    config["project_root"] = str(project_root)
    config["data_dir_resolved"] = str(common_dir)
    config["out_dir_resolved"] = str(out_dir)

    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # Determine node_idx using first method module
    first_module = load_module(scripts_dir / METHOD_TO_SCRIPT[methods[0]])
    first_train = first_module.load_gt_split(common_dir, "train")

    node_idx = resolve_node_indices(
        first_train["segment_ids"],
        max_nodes=int(args.max_nodes),
        node_sample=str(args.node_sample),
        seed=int(args.seed),
    )

    del first_train

    node_mode = "full" if node_idx is None else f"subset_{len(node_idx)}"
    log(f"NODE_MODE    : {node_mode}")

    # 1) Method offline/online runtime
    all_rows: List[Dict[str, Any]] = []
    method_rows: List[Dict[str, Any]] = []

    for method_name in methods:
        log("\n" + "=" * 88)
        log(f"METHOD: {method_name}")
        log("=" * 88)

        rows, meta_row = benchmark_one_method(
            method_name=method_name,
            scripts_dir=scripts_dir,
            common_dir=common_dir,
            lags=lags,
            splits=splits,
            node_idx=node_idx,
            max_samples_per_lag=int(args.max_samples_per_lag),
        )

        all_rows.extend(rows)
        method_rows.append(meta_row)

        part_df = pd.DataFrame(rows)
        part_path = out_dir / f"runtime_detail_{method_name}.csv"
        part_df.to_csv(part_path, index=False)
        log(f"Saved method detail: {part_path}")

    detail_df = pd.DataFrame(all_rows)
    method_df = pd.DataFrame(method_rows)

    detail_path = out_dir / "gt_runtime_detail.csv"
    method_path = out_dir / "gt_runtime_methods.csv"

    detail_df.to_csv(detail_path, index=False)
    method_df.to_csv(method_path, index=False)

    log(f"Saved detail CSV : {detail_path}")
    log(f"Saved method CSV : {method_path}")

    # 2) G_weight compute benchmark without saving
    gweight_df = pd.DataFrame()

    if args.measure_gweight_compute:
        if args.gweight_source_dir is None:
            g_source = (
                project_root
                / "ml_core"
                / "src"
                / "data_processing"
                / "outputs"
                / "branchB"
                / "osm_edge_base_like_branchA"
            )
        else:
            g_source = Path(args.gweight_source_dir)
            if not g_source.is_absolute():
                g_source = project_root / g_source

        log("\n" + "=" * 88)
        log("G_WEIGHT COMPUTE TIME BENCHMARK WITHOUT SAVING")
        log("=" * 88)
        log(f"G_WEIGHT_SOURCE_DIR: {g_source}")

        gweight_df = benchmark_gweight_compute_time(
            project_root=project_root,
            source_dir=g_source,
            max_nodes=int(args.max_nodes),
            node_sample=str(args.node_sample),
            seed=int(args.seed),
            granger_p=int(args.granger_p),
            bucket_minutes=int(args.bucket_minutes),
            min_bucket_samples=int(args.min_bucket_samples),
            max_candidates=int(args.max_candidates),
            candidate_block_size=int(args.candidate_block_size),
            ridge=float(args.ridge),
            min_improvement=float(args.min_improvement),
            fit_intercept=bool(args.fit_intercept),
            unsigned=bool(args.unsigned),
            max_buckets=int(args.max_buckets),
        )

        gweight_path = out_dir / "gweight_compute_runtime_without_saving.csv"
        gweight_df.to_csv(gweight_path, index=False)
        log(f"Saved G_weight runtime CSV: {gweight_path}")

    # 3) Plots
    make_plots(out_dir, detail_df, gweight_df)

    log("\n" + "=" * 88)
    log("DONE")
    log("=" * 88)
    log(f"Output dir: {out_dir}")
    log(f"Plots dir : {out_dir / 'plots'}")

    if not detail_df.empty:
        log("\nRuntime preview:")
        print(detail_df.head(20).to_string(index=False), flush=True)

    if not gweight_df.empty:
        log("\nG_weight compute preview:")
        print(gweight_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
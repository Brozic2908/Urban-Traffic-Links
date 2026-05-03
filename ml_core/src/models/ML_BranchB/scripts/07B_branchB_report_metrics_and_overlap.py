"""
Branch B unified report: XT metrics + Top-K graph overlap.

What it does
------------
1) Reads all per-lag XT metric CSV files produced by 06B_branchB_run_xt_forecast_topk_gt.py.
2) Filters by selected methods/splits/lags if requested.
3) Saves summary tables for MAE/MSE/RMSE.
4) Plots MAE, MSE, RMSE by horizon.
5) Optionally computes directed Top-K overlap between predicted G_hat[t+h|t]
   and True G[t+h] for selected graph methods.

Example
-------
python -u ml_core/src/models/ML_BranchB/scripts/07B_branchB_report_metrics_and_overlap.py \
  --data-dir ml_core/src/data_processing/outputs/branchB/osm_edge_granger_series_like_branchA \
  --results-dir ml_core/src/models/ML_BranchB/results/06_branchB_gt_pipeline \
  --out-dir ml_core/src/models/ML_BranchB/results/branchB_report \
  --methods true_gt,persistence_gt,ewma_gt,sparse_tvpvar_gt,sparse_var_gt,dmfm_lse_gt,dmfm_vlse_gt \
  --splits val,test \
  --lags 1-9 \
  --topk-values 5,10,20,50 \
  --samples-per-split-lag 10
"""
from __future__ import annotations

import argparse
import json
import os
import re
import types
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EPS = 1e-8

METHOD_TO_SCRIPT = {
    "no_gt": "06_branchB_run_xt_forecast_no_gt.py",
    "true_gt": "06_branchB_run_xt_forecast_true_gt.py",
    "persistence_gt": "06_branchB_run_xt_forecast_persistence_gt.py",
    "ewma_gt": "06_branchB_run_xt_forecast_ewma_gt.py",
    "sparse_tvpvar_gt": "06_branchB_run_xt_forecast_sparse_tvpvar_gt.py",
    "sparse_var_gt": "06_branchB_run_xt_forecast_sparse_var_gt.py",
    "dmfm_lse_gt": "06_branchB_run_xt_forecast_dmfm_gt.py",
    "dmfm_vlse_gt": "06_branchB_run_xt_forecast_dmfm_gt.py",
}

METHOD_LABELS = {
    "no_gt": "No-Graph",
    "true_gt": "True-Gt",
    "persistence_gt": "Persistence-Gt",
    "ewma_gt": "EWMA-Gt",
    "sparse_tvpvar_gt": "Sparse TVP-VAR-Gt",
    "sparse_var_gt": "Sparse VAR-Gt",
    "dmfm_lse_gt": "DMFM-LSE-Gt",
    "dmfm_vlse_gt": "DMFM-VLSE-Gt",
}

PRACTICAL_ALL = [
    "true_gt",
    "persistence_gt",
    "ewma_gt",
    "sparse_tvpvar_gt",
    "sparse_var_gt",
    "dmfm_lse_gt",
    "dmfm_vlse_gt",
]


def find_project_root() -> Path:
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents, Path("/kaggle/working/UTraffic-ML"), Path("/kaggle/working")]:
        if (p / "ml_core").exists() and (p / "dataset").exists():
            return p
        if p.name == "UTraffic-ML":
            return p
        if (p / "UTraffic-ML").exists():
            pp = p / "UTraffic-ML"
            if (pp / "ml_core").exists():
                return pp
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


def parse_str_list(s: Optional[str]) -> List[str]:
    if not s:
        return []
    out = []
    for x in str(s).split(","):
        x = x.strip()
        if x:
            out.append(x)
    return out


def expand_methods(s: str) -> List[str]:
    tokens = parse_str_list(s) or ["all"]
    out: List[str] = []
    for t in tokens:
        if t == "all":
            out.extend(PRACTICAL_ALL)
        elif t == "baselines":
            out.extend(["no_gt", "true_gt"])
        elif t in METHOD_TO_SCRIPT:
            out.append(t)
        else:
            raise ValueError(f"Unknown method={t}. Valid: {sorted(METHOD_TO_SCRIPT)} plus all/baselines")
    seen = set()
    final = []
    for x in out:
        if x not in seen:
            seen.add(x)
            final.append(x)
    return final


def base_method_from_run_name(method: str) -> str:
    if method.startswith("no_gt"):
        return "no_gt"
    for base in sorted(METHOD_TO_SCRIPT, key=len, reverse=True):
        if method == base or method.startswith(base + "_"):
            return base
    return str(method)


def pretty_method(method: str) -> str:
    base = base_method_from_run_name(method)
    label = METHOD_LABELS.get(base, base)
    m = re.search(r"topk(\d+)", method)
    if m:
        label += f" TopK-{m.group(1)}"
    m = re.search(r"nodes(\d+)", method)
    if m:
        label += f" [{m.group(1)} nodes]"
    if "nogamma" in method:
        label += " (no gamma)"
    return label


def load_metric_files(results_dir: Path) -> pd.DataFrame:
    files = sorted(results_dir.rglob("*_xt_per_lag_metrics.csv"))
    if not files:
        raise FileNotFoundError(f"No *_xt_per_lag_metrics.csv found under {results_dir}")
    frames = []
    for p in files:
        try:
            df = pd.read_csv(p)
            if df.empty:
                continue
            df["source_file"] = str(p)
            if "base_method" not in df.columns:
                df["base_method"] = df["method"].map(base_method_from_run_name)
            df["method_label"] = df["method"].map(pretty_method)
            frames.append(df)
        except Exception as e:
            print(f"[WARN] failed reading {p}: {e}", flush=True)
    if not frames:
        raise ValueError(f"Metric files found, but no readable non-empty CSV under {results_dir}")
    out = pd.concat(frames, ignore_index=True)
    if "horizon" in out.columns and "lag" not in out.columns:
        out = out.rename(columns={"horizon": "lag"})
    return out


def filter_metrics(df: pd.DataFrame, methods: List[str], splits: List[str], lags: List[int]) -> pd.DataFrame:
    out = df.copy()
    if methods:
        out = out[out["base_method"].isin(methods) | out["method"].isin(methods)].copy()
    if splits:
        out = out[out["split"].astype(str).isin(splits)].copy()
    if lags:
        out = out[out["lag"].astype(int).isin(lags)].copy()
    if out.empty:
        raise ValueError("No metrics left after filtering. Check --methods/--splits/--lags.")
    return out


def plot_metric_lines(df: pd.DataFrame, out_dir: Path) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for split in sorted(df["split"].unique()):
        df_s = df[df["split"] == split]
        for metric in ["mae", "mse", "rmse"]:
            plt.figure(figsize=(12, 5.5))
            for method, g in df_s.groupby("method_label"):
                g = g.sort_values("lag")
                plt.plot(g["lag"], g[metric], marker="o", label=method)
            plt.title(f"Branch B XT Forecast - {metric.upper()} by Horizon ({split})")
            plt.xlabel("Forecast horizon / lag")
            plt.ylabel(metric.upper())
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=8)
            plt.tight_layout()
            path = plot_dir / f"branchB_xt_{metric}_by_lag_{split}.png"
            plt.savefig(path, dpi=160, bbox_inches="tight")
            plt.close()
            print("Saved:", path, flush=True)


def load_module_definitions_only(path: Path):
    text = path.read_text(encoding="utf-8")
    cut_positions = []
    for marker in ["\n# -------------------------\n# Run", "\nMETHOD_NAME =", "\nPROJECT_ROOT = find_project_root()"]:
        idx = text.find(marker)
        if idx >= 0:
            cut_positions.append(idx)
    if cut_positions:
        text = text[: min(cut_positions)]
    module = types.ModuleType(path.stem)
    module.__file__ = str(path)
    exec(compile(text, str(path), "exec"), module.__dict__)
    return module


def topk_source_indices(G: np.ndarray, k: int, remove_self_loop: bool = True) -> np.ndarray:
    G = np.asarray(G, dtype=np.float32)
    N = G.shape[0]
    kk = min(max(1, int(k)), max(1, N - 1 if remove_self_loop else N))
    A = np.abs(G).copy()
    if remove_self_loop:
        np.fill_diagonal(A, -np.inf)
    idx = np.argpartition(A, -kk, axis=1)[:, -kk:]
    return idx.astype(np.int64)


def rowwise_overlap_at_k(G_pred: np.ndarray, G_true: np.ndarray, k: int, remove_self_loop: bool = True) -> float:
    P = topk_source_indices(G_pred, k, remove_self_loop=remove_self_loop)
    T = topk_source_indices(G_true, k, remove_self_loop=remove_self_loop)
    vals = []
    for i in range(P.shape[0]):
        vals.append(len(set(P[i].tolist()).intersection(T[i].tolist())) / max(1, int(k)))
    return float(np.mean(vals)) if vals else float("nan")


def subset_split_data(data: Dict[str, Any], node_idx: Optional[np.ndarray]) -> Dict[str, Any]:
    if node_idx is None:
        return data
    idx = np.asarray(node_idx, dtype=np.int64)
    out = dict(data)
    out["_node_idx"] = idx
    out["segment_ids"] = np.asarray(data["segment_ids"])[idx].astype(np.int64)
    out["z"] = np.asarray(data["z"], dtype=np.float32)[:, idx]
    if "G_weight_series" in data:
        out["G_weight_series"] = np.asarray(data["G_weight_series"][:, idx, :][:, :, idx], dtype=np.float32)
    if "G_best_lag_series" in data:
        out["G_best_lag_series"] = np.asarray(data["G_best_lag_series"][:, idx, :][:, :, idx])
    return out


def resolve_node_indices(common_dir: Path, max_nodes: int, seed: int, node_sample: str) -> Optional[np.ndarray]:
    if int(max_nodes) <= 0:
        return None
    seg = np.asarray(np.load(common_dir / "train" / "segment_ids.npy"), dtype=np.int64)
    N = len(seg)
    m = min(int(max_nodes), N)
    if node_sample == "random":
        rng = np.random.default_rng(int(seed))
        return np.sort(rng.choice(np.arange(N), size=m, replace=False)).astype(np.int64)
    return np.arange(m, dtype=np.int64)


def sample_pairs(pairs: List[Tuple[int, int]], n: int, seed: int) -> List[Tuple[int, int]]:
    if n <= 0 or n >= len(pairs):
        return pairs
    # Evenly spaced gives stable coverage across the day and deterministic logs.
    idx = np.linspace(0, len(pairs) - 1, n).round().astype(int)
    return [pairs[int(i)] for i in idx]


def compute_overlap(
    project_root: Path,
    data_dir: Path,
    out_dir: Path,
    methods: List[str],
    splits: List[str],
    lags: List[int],
    topk_values: List[int],
    samples_per_split_lag: int,
    max_nodes: int,
    node_sample: str,
    seed: int,
    remove_self_loop: bool,
) -> pd.DataFrame:
    scripts_dir = project_root / "ml_core" / "src" / "models" / "ML_BranchB" / "scripts"
    node_idx = resolve_node_indices(data_dir, max_nodes=max_nodes, seed=seed, node_sample=node_sample)
    graph_methods = [m for m in methods if m != "no_gt"]
    rows: List[Dict[str, Any]] = []

    # Load true module once to get data and true target G.
    true_mod = load_module_definitions_only(scripts_dir / METHOD_TO_SCRIPT["true_gt"])
    base_splits = {s: subset_split_data(true_mod.load_gt_split(data_dir, s), node_idx) for s in splits}

    for method in graph_methods:
        script = scripts_dir / METHOD_TO_SCRIPT[method]
        if not script.exists():
            print(f"[WARN] missing method script for overlap: {method} -> {script}", flush=True)
            continue
        print(f"[OVERLAP] method={method}", flush=True)
        mod = load_module_definitions_only(script)
        train = subset_split_data(mod.load_gt_split(data_dir, "train"), node_idx)
        val = subset_split_data(mod.load_gt_split(data_dir, "val"), node_idx)
        test = subset_split_data(mod.load_gt_split(data_dir, "test"), node_idx)
        g_model = mod.build_g_model(method, train, val, test)

        split_map = {"train": train, "val": val, "test": test}
        for split in splits:
            split_data = split_map[split]
            true_data = base_splits[split]
            meta = split_data["meta"]
            for h in lags:
                pairs = list(mod.iter_eval_pairs(meta, int(h)))
                pairs = sample_pairs(pairs, int(samples_per_split_lag), int(seed))
                if not pairs:
                    continue
                acc = {int(k): [] for k in topk_values}
                for origin_idx, target_idx in pairs:
                    G_pred = mod.predict_G_method(method, g_model, split, split_data, origin_idx, target_idx, int(h))
                    G_true = np.asarray(true_data["G_weight_series"][target_idx], dtype=np.float32)
                    for k in topk_values:
                        acc[int(k)].append(rowwise_overlap_at_k(G_pred, G_true, int(k), remove_self_loop=remove_self_loop))
                for k in topk_values:
                    vals = acc[int(k)]
                    rows.append({
                        "method": method,
                        "method_label": METHOD_LABELS.get(method, method),
                        "split": split,
                        "lag": int(h),
                        "topk_overlap": int(k),
                        "samples": int(len(vals)),
                        "overlap_mean": float(np.mean(vals)) if vals else np.nan,
                        "overlap_std": float(np.std(vals)) if vals else np.nan,
                        "max_nodes": 0 if node_idx is None else int(len(node_idx)),
                    })

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(out_dir / "branchB_overlap_by_method_split_lag_topk.csv", index=False)
        plot_overlap(df, out_dir)
    return df


def plot_overlap(df: pd.DataFrame, out_dir: Path) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for split in sorted(df["split"].unique()):
        for k in sorted(df["topk_overlap"].unique()):
            sub = df[(df["split"] == split) & (df["topk_overlap"] == k)]
            plt.figure(figsize=(12, 5.5))
            for method, g in sub.groupby("method_label"):
                g = g.sort_values("lag")
                plt.plot(g["lag"], g["overlap_mean"], marker="o", label=method)
            plt.title(f"Branch B Directed Graph Top-{k} Overlap ({split})")
            plt.xlabel("Forecast horizon / lag")
            plt.ylabel(f"Overlap@{k}")
            plt.ylim(0, 1.02)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=8)
            plt.tight_layout()
            path = plot_dir / f"branchB_overlap_top{k}_{split}.png"
            plt.savefig(path, dpi=160, bbox_inches="tight")
            plt.close()
            print("Saved:", path, flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="ml_core/src/data_processing/outputs/branchB/osm_edge_granger_series_like_branchA")
    p.add_argument("--results-dir", type=str, default="ml_core/src/models/ML_BranchB/results/06_branchB_gt_pipeline")
    p.add_argument("--out-dir", type=str, default="ml_core/src/models/ML_BranchB/results/branchB_report")
    p.add_argument("--methods", type=str, default="all")
    p.add_argument("--splits", type=str, default="val,test")
    p.add_argument("--lags", type=str, default="1-9")
    p.add_argument("--topk-values", type=str, default="5,10,20,50")
    p.add_argument("--samples-per-split-lag", type=int, default=10, help="0 = use all pairs; 10 is quick and stable for report.")
    p.add_argument("--max-nodes", type=int, default=0, help="Only for overlap recomputation. 0 = full prepared data.")
    p.add_argument("--node-sample", type=str, default="first", choices=["first", "random"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-overlap", action="store_true")
    p.add_argument("--keep-self-loop", action="store_true")
    args = p.parse_args()

    root = find_project_root()
    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    if not data_dir.is_absolute():
        data_dir = root / data_dir
    if not results_dir.is_absolute():
        results_dir = root / results_dir
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = expand_methods(args.methods)
    splits = parse_str_list(args.splits) or ["val", "test"]
    lags = parse_int_list(args.lags) or list(range(1, 10))
    topk_values = parse_int_list(args.topk_values) or [20]

    print("PROJECT_ROOT:", root, flush=True)
    print("DATA_DIR    :", data_dir, flush=True)
    print("RESULTS_DIR :", results_dir, flush=True)
    print("OUT_DIR     :", out_dir, flush=True)
    print("METHODS     :", methods, flush=True)
    print("SPLITS      :", splits, flush=True)
    print("LAGS        :", lags, flush=True)

    all_metrics = load_metric_files(results_dir)
    all_metrics.to_csv(out_dir / "branchB_all_xt_per_lag_metrics.csv", index=False)
    metrics = filter_metrics(all_metrics, methods, splits, lags)
    metrics.to_csv(out_dir / "branchB_filtered_xt_per_lag_metrics.csv", index=False)

    summary = (
        metrics.groupby(["split", "base_method", "method", "method_label"], as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            mse_mean=("mse", "mean"),
            rmse_mean=("rmse", "mean"),
            mae_best=("mae", "min"),
            rmse_best=("rmse", "min"),
            n_lags=("lag", "nunique"),
            n_samples=("n_samples", "sum"),
        )
        .sort_values(["split", "rmse_mean", "mae_mean"])
    )
    summary.to_csv(out_dir / "branchB_xt_summary_by_method.csv", index=False)
    plot_metric_lines(metrics, out_dir)

    if not args.skip_overlap:
        overlap = compute_overlap(
            project_root=root,
            data_dir=data_dir,
            out_dir=out_dir,
            methods=methods,
            splits=splits,
            lags=lags,
            topk_values=topk_values,
            samples_per_split_lag=int(args.samples_per_split_lag),
            max_nodes=int(args.max_nodes),
            node_sample=str(args.node_sample),
            seed=int(args.seed),
            remove_self_loop=not bool(args.keep_self_loop),
        )
        print("Overlap rows:", overlap.shape if overlap is not None else None, flush=True)

    config = vars(args).copy()
    config.update({"resolved_data_dir": str(data_dir), "resolved_results_dir": str(results_dir), "resolved_out_dir": str(out_dir)})
    with open(out_dir / "report_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("\nDONE.", flush=True)
    print("Metrics summary:", out_dir / "branchB_xt_summary_by_method.csv", flush=True)
    print("Plots:", out_dir / "plots", flush=True)


if __name__ == "__main__":
    main()

# ml_core/src/models/ML_BranchB/scripts/06B_branchB_run_xt_forecast_topk_gt.py
"""
Branch B — XT forecast with Top-K Gt graph signal for selected methods.

Main changes:
- Apply Top-K graph signal consistently to every Rt/Gt-based method.
- Keep No-Rt baseline for comparison.
- Remove gamma from the velocity/XT forecast formula:
      X_hat[t+h] = A_h X_t + B_h(TopK(G_hat[t,h]) X_t)
  No graph-signal gamma scaling is used. By default, the downstream ElasticNet
  also uses fit_intercept=False, so there is no additive gamma/bias term.
- Support full run or a selected node subset for fast testing.
- Support multi-CPU execution by method-level process parallelism or
  horizon-level thread parallelism.

Recommended quick test:
    python ml_core/src/models/ML_BranchB/scripts/06B_branchB_run_xt_forecast_topk_gt.py \
      --methods no_gt,true_gt,persistence_gt,ewma_gt \
      --topk 20 --lags 1-3 --max-nodes 512 --parallel-level horizon --n-jobs 4

Recommended full run:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    python ml_core/src/models/ML_BranchB/scripts/06B_branchB_run_xt_forecast_topk_gt.py \
      --methods all --topk 20 --lags 1-9 --parallel-level method --n-jobs 4

Dense TVP-VAR is intentionally excluded from --methods all because it is usually
infeasible for full N. Add --include-dense or use --methods all_with_dense for a
small --max-nodes test.
"""

from __future__ import annotations

# Keep BLAS/OpenMP threads at 1 per process to avoid CPU oversubscription when
# running several methods in parallel.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import types
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import MultiTaskElasticNet


EPS = 1e-8

METHOD_TO_SCRIPT = {
    "no_gt": "06_branchB_run_xt_forecast_no_gt.py",
    "true_gt": "06_branchB_run_xt_forecast_true_gt.py",
    "granger_gt": "06_branchB_run_xt_forecast_granger_gt.py",
    "granger_dynamic_gt": "06_branchB_run_xt_forecast_granger_dynamic_gt.py",
    "persistence_gt": "06_branchB_run_xt_forecast_persistence_gt.py",
    "ewma_gt": "06_branchB_run_xt_forecast_ewma_gt.py",
    "sparse_tvpvar_gt": "06_branchB_run_xt_forecast_sparse_tvpvar_gt.py",
    "sparse_var_gt": "06_branchB_run_xt_forecast_sparse_var_gt.py",
    "dmfm_lse_gt": "06_branchB_run_xt_forecast_dmfm_gt.py",
    "dmfm_vlse_gt": "06_branchB_run_xt_forecast_dmfm_gt.py",
    "factorized_var_gt": "06_branchB_run_xt_forecast_factorized_var_gt.py",
    "factorized_mar_gt": "06_branchB_run_xt_forecast_factorized_mar_gt.py",
    "factorized_tvpvar_gt": "06_branchB_run_xt_forecast_factorized_tvpvar_gt.py",
    "dense_tvpvar_gt": "06_branchB_run_xt_forecast_dense_tvpvar_gt.py",
}

PRACTICAL_ALL_METHODS = [
    "true_gt",
    "persistence_gt",
    "ewma_gt",
    "sparse_tvpvar_gt",
    "sparse_var_gt",
    "dmfm_lse_gt",
    "dmfm_vlse_gt",
]

ALL_WITH_DENSE_METHODS = PRACTICAL_ALL_METHODS + ["dense_tvpvar_gt"]

BASE_LABELS = {
    "no_gt": "No-Graph",
    "true_gt": "True-Gt",
    "granger_gt": "Granger-GT",
    "granger_dynamic_gt": "Granger-Dynamic-GT",
    "persistence_gt": "Persistence-Gt",
    "ewma_gt": "EWMA-Gt",
    "sparse_tvpvar_gt": "Sparse TVP-VAR-Gt",
    "sparse_var_gt": "Sparse VAR-Gt",
    "dmfm_lse_gt": "DMFM-LSE-Gt",
    "dmfm_vlse_gt": "DMFM-VLSE-Gt",
    "factorized_var_gt": "Factorized VAR-Gt",
    "factorized_mar_gt": "Factorized MAR-Gt",
    "factorized_tvpvar_gt": "Factorized TVP-VAR-Gt",
    "dense_tvpvar_gt": "Dense TVP-VAR-Gt",
}


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
        if p.name == "UTraffic-ML":
            return p
        if (p / "UTraffic-ML").exists():
            pp = p / "UTraffic-ML"
            if (pp / "ml_core").exists():
                return pp
    return cwd


def parse_int_list(s: Optional[str]) -> List[int]:
    if s is None:
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
    if s is None:
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]


def expand_methods(methods_arg: str, include_dense: bool = False) -> List[str]:
    tokens = parse_str_list(methods_arg)
    if not tokens:
        tokens = ["all"]

    out: List[str] = []
    for token in tokens:
        token = token.strip()
        if token == "all":
            out.extend(PRACTICAL_ALL_METHODS)
        elif token == "all_with_dense":
            out.extend(ALL_WITH_DENSE_METHODS)
        elif token == "baselines":
            out.extend(["no_gt", "true_gt"])
        elif token in METHOD_TO_SCRIPT:
            out.append(token)
        else:
            raise ValueError(
                f"Unknown method '{token}'. Valid: {sorted(METHOD_TO_SCRIPT)} plus all, all_with_dense, baselines"
            )

    if include_dense and "dense_tvpvar_gt" not in out:
        out.append("dense_tvpvar_gt")

    # preserve order while removing duplicates
    seen = set()
    final = []
    for m in out:
        if m not in seen:
            seen.add(m)
            final.append(m)
    return final


def load_module_definitions_only(path: Path):
    """
    Load function/class definitions from the old standalone scripts without
    executing their bottom run block.

    The uploaded base scripts contain a bottom block that runs immediately at
    import time. This loader cuts the file before METHOD_NAME/PROJECT_ROOT run
    configuration to avoid accidental duplicate runs.
    """
    text = path.read_text(encoding="utf-8")
    cut_positions = []
    for marker in [
        "\n# -------------------------\n# Run",
        "\nMETHOD_NAME =",
        "\nPROJECT_ROOT = find_project_root()",
    ]:
        idx = text.find(marker)
        if idx >= 0:
            cut_positions.append(idx)
    if cut_positions:
        text = text[: min(cut_positions)]

    module = types.ModuleType(path.stem)
    module.__file__ = str(path)
    code = compile(text, str(path), "exec")
    exec(code, module.__dict__)
    return module


def topk_graph_signal(
    G: np.ndarray,
    x_t: np.ndarray,
    topk: int = 20,
    normalize: bool = True,
    remove_self_loop: bool = True,
) -> np.ndarray:
    """
    Compute Top-K graph signal without gamma.

    G shape convention: (N, N) = target x source.

    Full graph signal:
        gx = G @ x_t

    Top-K signal for each target row j:
        gx[j] = sum_{i in TopKAbs(G[j, :])} G[j, i] * x_t[i]

    If normalize=True:
        gx[j] = gx[j] / (sum_i |G[j, i]| + eps)

    No multiplicative gamma and no additive gamma/bias term are used here.
    """
    G = np.asarray(G, dtype=np.float32)
    x_t = np.asarray(x_t, dtype=np.float32)

    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError(f"G must be square matrix, got shape={G.shape}")

    N = int(G.shape[0])
    if x_t.shape[0] != N:
        raise ValueError(f"x_t shape mismatch: x_t={x_t.shape}, G={G.shape}")

    k = int(topk)

    if k <= 0 or k >= N:
        G_work = G.astype(np.float32, copy=True)
        if remove_self_loop:
            diag = np.arange(N)
            G_work[diag, diag] = 0.0
        gx = G_work @ x_t
        if normalize:
            denom = np.sum(np.abs(G_work), axis=1) + EPS
            gx = gx / denom
        return np.nan_to_num(gx, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # Use absolute value to choose strongest sources per target row.
    absG = np.abs(G).astype(np.float32, copy=True)
    if remove_self_loop:
        diag = np.arange(N)
        absG[diag, diag] = -np.inf

    idx = np.argpartition(absG, -k, axis=1)[:, -k:]  # (N, k)
    rows = np.arange(N)[:, None]
    weights = G[rows, idx].astype(np.float32, copy=False)
    x_src = x_t[idx].astype(np.float32, copy=False)

    gx = np.sum(weights * x_src, axis=1)
    if normalize:
        denom = np.sum(np.abs(weights), axis=1) + EPS
        gx = gx / denom

    return np.nan_to_num(gx, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def subset_split_data(data: Dict[str, Any], node_idx: Optional[np.ndarray]) -> Dict[str, Any]:
    if node_idx is None:
        return data

    idx = np.asarray(node_idx, dtype=np.int64)
    out = dict(data)
    out["_node_idx"] = idx
    out["segment_ids"] = np.asarray(data["segment_ids"])[idx].astype(np.int64)
    out["z"] = np.asarray(data["z"], dtype=np.float32)[:, idx]

    # Standard Rt/Gt methods store G as T x N x N or H x N x N in split_data.
    # Dynamic methods may store graph banks outside split_data; for those, keep
    # only _node_idx so the method module can subset the graph lazily.
    if "G_weight_series" in data:
        out["G_weight_series"] = np.asarray(data["G_weight_series"][:, idx, :][:, :, idx], dtype=np.float32)

    if "G_best_lag_series" in data:
        out["G_best_lag_series"] = np.asarray(data["G_best_lag_series"][:, idx, :][:, :, idx])

    return out


def resolve_node_indices(
    common_dir: Path,
    max_nodes: int = 0,
    node_indices_arg: Optional[str] = None,
    node_ids_arg: Optional[str] = None,
    node_sample: str = "first",
    seed: int = 42,
) -> Optional[np.ndarray]:
    """Return selected node indices, or None for full run."""
    if not node_indices_arg and not node_ids_arg and int(max_nodes) <= 0:
        return None

    seg_path = common_dir / "train" / "segment_ids.npy"
    if not seg_path.exists():
        raise FileNotFoundError(f"Missing segment_ids for node selection: {seg_path}")
    segment_ids = np.asarray(np.load(seg_path), dtype=np.int64)
    N = int(len(segment_ids))

    selected: Optional[np.ndarray] = None

    if node_indices_arg:
        idx = np.asarray(parse_int_list(node_indices_arg), dtype=np.int64)
        if len(idx) == 0:
            raise ValueError("--node-indices was provided but no valid index was parsed.")
        if idx.min() < 0 or idx.max() >= N:
            raise ValueError(f"node index out of range. N={N}, min={idx.min()}, max={idx.max()}")
        selected = idx

    if node_ids_arg:
        requested_ids = np.asarray(parse_int_list(node_ids_arg), dtype=np.int64)
        id_to_pos = {int(v): i for i, v in enumerate(segment_ids)}
        missing = [int(x) for x in requested_ids if int(x) not in id_to_pos]
        if missing:
            raise ValueError(f"Some --node-ids are not in train/segment_ids.npy: {missing[:20]}")
        idx = np.asarray([id_to_pos[int(x)] for x in requested_ids], dtype=np.int64)
        selected = idx if selected is None else np.intersect1d(selected, idx)

    if selected is None:
        max_nodes = int(max_nodes)
        max_nodes = min(max_nodes, N)
        if max_nodes <= 0 or max_nodes >= N:
            return None
        if node_sample == "first":
            selected = np.arange(max_nodes, dtype=np.int64)
        elif node_sample == "random":
            rng = np.random.default_rng(int(seed))
            selected = np.sort(rng.choice(N, size=max_nodes, replace=False).astype(np.int64))
        else:
            raise ValueError("--node-sample must be first or random")
    else:
        selected = np.asarray(sorted(set(map(int, selected.tolist()))), dtype=np.int64)
        if int(max_nodes) > 0 and len(selected) > int(max_nodes):
            selected = selected[: int(max_nodes)]

    if len(selected) == 0:
        raise ValueError("Node selection is empty.")
    return selected


def make_method_output_name(
    method_name: str,
    topk: int,
    normalize: bool,
    remove_self_loop: bool,
    fit_intercept: bool,
    node_idx: Optional[np.ndarray],
) -> str:
    """Create stable method/output label."""
    intercept_tag = "withbias" if fit_intercept else "nogamma"
    node_tag = "full" if node_idx is None else f"nodes{len(node_idx)}"

    if method_name == "no_gt":
        return f"no_gt_{intercept_tag}_{node_tag}"

    norm_tag = "norm" if normalize else "nonorm"
    self_tag = "noself" if remove_self_loop else "self"
    return f"{method_name}_topk{int(topk)}_{norm_tag}_{self_tag}_{intercept_tag}_{node_tag}"


def fit_direct_xt_model_custom(
    module,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    fit_intercept: bool = False,
) -> MultiTaskElasticNet:
    model = MultiTaskElasticNet(
        alpha=float(getattr(module, "ALPHA", 0.001)),
        l1_ratio=float(getattr(module, "L1_RATIO", 0.5)),
        fit_intercept=bool(fit_intercept),
        max_iter=int(getattr(module, "MAX_ITER", 500)),
        tol=float(getattr(module, "TOL", 1e-2)),
        selection=str(getattr(module, "SELECTION", "random")),
        random_state=int(getattr(module, "RANDOM_STATE", 42)),
    )
    model.fit(X_train, Y_train)
    return model


def build_xt_dataset_for_horizon_topk(
    module,
    method_name: str,
    g_model: Dict[str, object],
    train_data: Dict[str, object],
    split_name: str,
    split_data: Dict[str, object],
    horizon: int,
    use_gt: bool,
    topk: int,
    normalize: bool,
    remove_self_loop: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    z = np.asarray(split_data["z"], dtype=np.float32)
    meta = split_data["meta"]

    X_rows: List[np.ndarray] = []
    Y_rows: List[np.ndarray] = []

    for origin_idx, target_idx in module.iter_eval_pairs(meta, int(horizon)):
        x_t = np.asarray(z[origin_idx], dtype=np.float32)
        y_true = np.asarray(z[target_idx], dtype=np.float32)

        if use_gt:
            G_used = module.predict_G_method(
                method_name,
                g_model,
                split_name,
                split_data,
                origin_idx,
                target_idx,
                int(horizon),
            )
            gx = topk_graph_signal(
                G=G_used,
                x_t=x_t,
                topk=int(topk),
                normalize=bool(normalize),
                remove_self_loop=bool(remove_self_loop),
            )
            feat = np.concatenate([x_t, gx], axis=0)
        else:
            feat = x_t

        X_rows.append(feat.astype(np.float32))
        Y_rows.append(y_true.astype(np.float32))

    if not X_rows:
        n = int(z.shape[1])
        return (
            np.empty((0, n * (2 if use_gt else 1)), dtype=np.float32),
            np.empty((0, n), dtype=np.float32),
        )

    return (
        np.stack(X_rows, axis=0).astype(np.float32),
        np.stack(Y_rows, axis=0).astype(np.float32),
    )


def run_one_horizon(
    module,
    method_name: str,
    display_method_name: str,
    g_model: Dict[str, object],
    train: Dict[str, object],
    val: Dict[str, object],
    test: Dict[str, object],
    horizon: int,
    use_gt: bool,
    topk: int,
    normalize: bool,
    remove_self_loop: bool,
    fit_intercept: bool,
    n_segments: int,
) -> List[Dict[str, Any]]:
    print(f"\n[HORIZON {horizon}] building train features...", flush=True)
    X_train, Y_train = build_xt_dataset_for_horizon_topk(
        module=module,
        method_name=method_name,
        g_model=g_model,
        train_data=train,
        split_name="train",
        split_data=train,
        horizon=horizon,
        use_gt=use_gt,
        topk=topk,
        normalize=normalize,
        remove_self_loop=remove_self_loop,
    )

    if len(X_train) == 0:
        print("No train samples; skip horizon", horizon, flush=True)
        return []

    print("X_train:", X_train.shape, "Y_train:", Y_train.shape, flush=True)
    model = fit_direct_xt_model_custom(module, X_train, Y_train, fit_intercept=fit_intercept)

    rows: List[Dict[str, Any]] = []
    for split_name, split_data in [("val", val), ("test", test)]:
        print(f"[HORIZON {horizon}] evaluating {split_name}...", flush=True)
        X_split, Y_split = build_xt_dataset_for_horizon_topk(
            module=module,
            method_name=method_name,
            g_model=g_model,
            train_data=train,
            split_name=split_name,
            split_data=split_data,
            horizon=horizon,
            use_gt=use_gt,
            topk=topk,
            normalize=normalize,
            remove_self_loop=remove_self_loop,
        )
        if len(X_split) == 0:
            continue
        Y_pred = model.predict(X_split).astype(np.float32)
        metrics = module.batch_vector_metrics(Y_split, Y_pred)
        row = {
            "method": display_method_name,
            "base_method": method_name,
            "split": split_name,
            "lag": int(horizon),
            "n_samples": int(len(X_split)),
            "n_segments": int(n_segments),
            "topk": int(topk) if use_gt else 0,
            "normalize": bool(normalize) if use_gt else False,
            "remove_self_loop": bool(remove_self_loop) if use_gt else False,
            "fit_intercept": bool(fit_intercept),
            "gamma_removed": not bool(fit_intercept),
            **metrics,
        }
        print(split_name, metrics, flush=True)
        rows.append(row)

    del X_train, Y_train, model
    return rows


def run_one_method(config: Dict[str, Any]) -> Dict[str, Any]:
    method_name = config["method_name"]
    project_root = Path(config["project_root"])
    common_dir = Path(config["common_dir"])
    scripts_dir = Path(config["scripts_dir"])
    base_results_dir = Path(config["base_results_dir"])
    topk = int(config["topk"])
    normalize = bool(config["normalize"])
    remove_self_loop = bool(config["remove_self_loop"])
    fit_intercept = bool(config["fit_intercept"])
    horizons = list(map(int, config["horizons"]))
    node_idx = None
    if config.get("node_idx") is not None:
        node_idx = np.asarray(config["node_idx"], dtype=np.int64)
    horizon_workers = max(1, int(config.get("horizon_workers", 1)))
    skip_existing = bool(config.get("skip_existing", False))

    script_path = scripts_dir / METHOD_TO_SCRIPT[method_name]
    if not script_path.exists():
        raise FileNotFoundError(f"Missing base script for {method_name}: {script_path}")

    display_method_name = make_method_output_name(
        method_name=method_name,
        topk=topk,
        normalize=normalize,
        remove_self_loop=remove_self_loop,
        fit_intercept=fit_intercept,
        node_idx=node_idx,
    )
    out_dir = base_results_dir / display_method_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{display_method_name}_xt_per_lag_metrics.csv"

    if skip_existing and out_path.exists():
        print(f"[SKIP] Existing metrics: {out_path}", flush=True)
        return {"method": display_method_name, "ok": True, "skipped": True, "out_path": str(out_path)}

    print("\n" + "=" * 88, flush=True)
    print(f"RUN METHOD: {method_name} -> {display_method_name}", flush=True)
    print("=" * 88, flush=True)
    print("PROJECT_ROOT :", project_root, flush=True)
    print("COMMON_DIR   :", common_dir, flush=True)
    print("SCRIPT_PATH  :", script_path, flush=True)
    print("OUT_DIR      :", out_dir, flush=True)
    print("TOPK         :", topk if method_name != "no_gt" else "N/A for No-Rt", flush=True)
    print("FIT_INTERCEPT:", fit_intercept, "| gamma_removed:", not fit_intercept, flush=True)
    print("NODE MODE    :", "full" if node_idx is None else f"subset n={len(node_idx)}", flush=True)
    print("HORIZONS     :", horizons, flush=True)
    print("HORIZON WORKERS:", horizon_workers, flush=True)

    module = load_module_definitions_only(script_path)
    if hasattr(module, "check_branchB_common_dir_ready"):
        module.check_branchB_common_dir_ready(common_dir)

    train = module.load_gt_split(common_dir, "train")
    val = module.load_gt_split(common_dir, "val")
    test = module.load_gt_split(common_dir, "test")

    train = subset_split_data(train, node_idx)
    val = subset_split_data(val, node_idx)
    test = subset_split_data(test, node_idx)

    assert np.array_equal(train["segment_ids"], val["segment_ids"])
    assert np.array_equal(train["segment_ids"], test["segment_ids"])

    n_segments = int(len(train["segment_ids"]))
    use_gt = method_name != "no_gt"

    print("n_segments:", n_segments, flush=True)
    print("method:", method_name, "| use_gt:", use_gt, flush=True)

    # Keep base script horizon constants aligned in case any helper uses them.
    module.HORIZONS = horizons

    print("Building G model...", flush=True)
    g_model = module.build_g_model(method_name, train, val, test)
    print("G model built.", flush=True)

    all_rows: List[Dict[str, Any]] = []
    if horizon_workers <= 1 or len(horizons) <= 1:
        for h in horizons:
            all_rows.extend(
                run_one_horizon(
                    module,
                    method_name,
                    display_method_name,
                    g_model,
                    train,
                    val,
                    test,
                    h,
                    use_gt,
                    topk,
                    normalize,
                    remove_self_loop,
                    fit_intercept,
                    n_segments,
                )
            )
    else:
        # Thread-based horizon parallelism shares the loaded memmaps/model in one process.
        with ThreadPoolExecutor(max_workers=horizon_workers) as ex:
            futs = {
                ex.submit(
                    run_one_horizon,
                    module,
                    method_name,
                    display_method_name,
                    g_model,
                    train,
                    val,
                    test,
                    h,
                    use_gt,
                    topk,
                    normalize,
                    remove_self_loop,
                    fit_intercept,
                    n_segments,
                ): h
                for h in horizons
            }
            for fut in as_completed(futs):
                h = futs[fut]
                try:
                    all_rows.extend(fut.result())
                except Exception as e:
                    raise RuntimeError(f"Horizon {h} failed for method {method_name}: {e}") from e

    per_lag = pd.DataFrame(all_rows)
    if not per_lag.empty:
        per_lag = per_lag.sort_values(["method", "split", "lag"]).reset_index(drop=True)
    per_lag.to_csv(out_path, index=False)

    # Save run config for reproducibility.
    config_to_save = dict(config)
    if config_to_save.get("node_idx") is not None:
        config_to_save["node_idx"] = list(map(int, config_to_save["node_idx"]))
    config_to_save["display_method_name"] = display_method_name
    config_to_save["out_path"] = str(out_path)
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config_to_save, f, ensure_ascii=False, indent=2)

    print("\n[DONE]", display_method_name, flush=True)
    print("saved:", out_path, flush=True)
    print(per_lag, flush=True)

    return {"method": display_method_name, "ok": True, "skipped": False, "out_path": str(out_path)}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods",
        type=str,
        default="all",
        help=(
            "Comma-separated methods. Use all, all_with_dense, baselines, or explicit names. "
            "Default all runs: true_gt, persistence_gt, ewma_gt, sparse_tvpvar_gt, sparse_var_gt, dmfm_lse_gt, dmfm_vlse_gt. Use explicit methods for a smaller run."
        ),
    )
    parser.add_argument(
        "--include-dense",
        action="store_true",
        help="Add dense_tvpvar_gt. Recommended only for small --max-nodes tests.",
    )
    parser.add_argument("--topk", type=int, default=20, help="Top-K sources per target row for Rt/Gt methods.")
    parser.add_argument("--lags", type=str, default="1-9", help="Horizons, e.g. 1-9 or 1,2,3.")
    parser.add_argument("--data-dir", type=str, default=None, help="Prepared Branch-B data dir.")
    parser.add_argument("--results-dir", type=str, default=None, help="Output results directory for this run. Default: BranchB/results/06_branchB_run_xt_forecast")
    parser.add_argument("--no-normalize", action="store_true", help="Disable row L1 normalization after Top-K.")
    parser.add_argument("--keep-self-loop", action="store_true", help="Keep diagonal/self-loop edges. Default removes self-loop.")
    parser.add_argument(
        "--fit-intercept",
        action="store_true",
        help="Use ElasticNet intercept. Default False removes additive gamma/bias term.",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=0,
        help="Use first/random N nodes for a quick test. 0 means full node set.",
    )
    parser.add_argument(
        "--node-indices",
        type=str,
        default=None,
        help="Explicit node positions, e.g. 0-511 or 0,10,20. Applied before --max-nodes truncation.",
    )
    parser.add_argument(
        "--node-ids",
        type=str,
        default=None,
        help="Explicit model node IDs from segment_ids.npy, e.g. 12,45,78.",
    )
    parser.add_argument(
        "--node-sample",
        type=str,
        default="first",
        choices=["first", "random"],
        help="When --max-nodes is used without explicit nodes, choose first or random nodes.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of CPU workers. Use with --parallel-level method or horizon.",
    )
    parser.add_argument(
        "--parallel-level",
        type=str,
        default="method",
        choices=["method", "horizon", "none"],
        help="Parallelize across methods with processes, or across horizons with threads inside each method.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately on the first method failure. Default logs and continues.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    project_root = find_project_root()
    scripts_dir = project_root / "ml_core" / "src" / "models" / "ML_BranchB" / "scripts"
    branchb_root = project_root / "ml_core" / "src" / "models" / "ML_BranchB"
    if args.results_dir is None:
        base_results_dir = branchb_root / "results" / "06_branchB_run_xt_forecast"
    else:
        base_results_dir = Path(args.results_dir)
        if not base_results_dir.is_absolute():
            base_results_dir = project_root / base_results_dir
    base_results_dir.mkdir(parents=True, exist_ok=True)
    base_results_dir.mkdir(parents=True, exist_ok=True)

    if args.data_dir is None:
        common_dir = project_root / "ml_core" / "src" / "data_processing" / "outputs" / "branchB" / "osm_edge_gt_like_branchA"
    else:
        common_dir = Path(args.data_dir)
        if not common_dir.is_absolute():
            common_dir = project_root / common_dir

    methods = expand_methods(args.methods, include_dense=bool(args.include_dense))
    horizons = parse_int_list(args.lags)
    if not horizons:
        raise ValueError("No horizons parsed from --lags")

    node_idx = resolve_node_indices(
        common_dir=common_dir,
        max_nodes=int(args.max_nodes),
        node_indices_arg=args.node_indices,
        node_ids_arg=args.node_ids,
        node_sample=args.node_sample,
        seed=int(args.seed),
    )

    normalize = not bool(args.no_normalize)
    remove_self_loop = not bool(args.keep_self_loop)
    n_jobs = max(1, int(args.n_jobs))

    print("PROJECT_ROOT :", project_root)
    print("SCRIPTS_DIR  :", scripts_dir)
    print("COMMON_DIR   :", common_dir)
    print("RESULTS_DIR  :", base_results_dir)
    print("METHODS      :", methods)
    print("HORIZONS     :", horizons)
    print("TOPK         :", args.topk)
    print("NORMALIZE    :", normalize)
    print("REMOVE_SELF  :", remove_self_loop)
    print("FIT_INTERCEPT:", bool(args.fit_intercept), "| gamma_removed:", not bool(args.fit_intercept))
    print("NODE MODE    :", "full" if node_idx is None else f"subset n={len(node_idx)}")
    print("N_JOBS       :", n_jobs)
    print("PARALLEL     :", args.parallel_level)

    base_config = {
        "project_root": str(project_root),
        "common_dir": str(common_dir),
        "scripts_dir": str(scripts_dir),
        "base_results_dir": str(base_results_dir),
        "topk": int(args.topk),
        "normalize": normalize,
        "remove_self_loop": remove_self_loop,
        "fit_intercept": bool(args.fit_intercept),
        "horizons": horizons,
        "node_idx": None if node_idx is None else list(map(int, node_idx.tolist())),
        "skip_existing": bool(args.skip_existing),
    }

    results: List[Dict[str, Any]] = []

    if args.parallel_level == "method" and n_jobs > 1 and len(methods) > 1:
        # Process-level method parallelism: fastest when running many methods, but
        # each worker loads its own data/model. Use carefully for full large N.
        configs = []
        for m in methods:
            cfg = dict(base_config)
            cfg["method_name"] = m
            cfg["horizon_workers"] = 1
            configs.append(cfg)

        with ProcessPoolExecutor(max_workers=min(n_jobs, len(configs))) as ex:
            futs = {ex.submit(run_one_method, cfg): cfg["method_name"] for cfg in configs}
            for fut in as_completed(futs):
                method = futs[fut]
                try:
                    results.append(fut.result())
                except Exception as e:
                    tb = traceback.format_exc()
                    print(f"\n[ERROR] method={method}: {e}\n{tb}", flush=True)
                    results.append({"method": method, "ok": False, "error": str(e)})
                    if args.stop_on_error:
                        raise
    else:
        horizon_workers = n_jobs if args.parallel_level == "horizon" else 1
        for m in methods:
            cfg = dict(base_config)
            cfg["method_name"] = m
            cfg["horizon_workers"] = horizon_workers
            try:
                results.append(run_one_method(cfg))
            except Exception as e:
                tb = traceback.format_exc()
                print(f"\n[ERROR] method={m}: {e}\n{tb}", flush=True)
                results.append({"method": m, "ok": False, "error": str(e)})
                if args.stop_on_error:
                    raise

    summary_path = base_results_dir / "topk_nogamma_run_summary.csv"
    pd.DataFrame(results).to_csv(summary_path, index=False)

    print("\n" + "=" * 88)
    print("RUN SUMMARY")
    print("=" * 88)
    print(pd.DataFrame(results))
    print("Saved summary:", summary_path)
    print("\nNext: run plot script, for example:")
    print(
        "python ml_core/src/models/ML_BranchB/scripts/07_branchB_plot_xt_forecast_results.py "
        f"--topk {int(args.topk)} --nogamma-only"
    )


if __name__ == "__main__":
    main()

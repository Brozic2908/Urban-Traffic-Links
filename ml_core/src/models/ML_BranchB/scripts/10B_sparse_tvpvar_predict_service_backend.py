# ml_core/src/models/ML_BranchB/scripts/10B_sparse_tvpvar_predict_service_backend.py
"""
Backend on-demand prediction service for Branch B Sparse TVP-VAR G matrices.

Default behavior:
- Load fitted Sparse TVP-VAR artifact.
- Accept only TEST timestamps.
- Predict G_pred[t+h|t] quickly.
- Do NOT save the predicted matrix unless --save-npz is explicitly passed.

Python usage in backend:
    from pathlib import Path
    import importlib.util

    service_path = Path("ml_core/src/models/ML_BranchB/scripts/10B_sparse_tvpvar_predict_service_backend.py")
    spec = importlib.util.spec_from_file_location("b10_service", service_path)
    service = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(service)

    result = service.predict_by_datetime(date="2024-08-28", time_in_day="09:00", step=3)
    G_pred = result["G_pred"]        # numpy array [N, N], target x source
    segment_ids = result["segment_ids"]

CLI smoke test:
    python -u ml_core/src/models/ML_BranchB/scripts/10B_sparse_tvpvar_predict_service_backend.py \
      --date 2024-08-28 --time 09:00 --step 3
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd


METHOD_NAME = "sparse_tvpvar_gt"
STABLE_METHOD_MODULE_NAME = "branchB_sparse_tvpvar_method"
ALLOWED_INFERENCE_SPLITS = ("test",)


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now_str()}] {msg}", flush=True)


def find_project_root() -> Path:
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents, Path("/kaggle/working/UTraffic-ML"), Path("/kaggle/working")]:
        if (p / "ml_core").exists():
            return p
        if (p / "UTraffic-ML").exists() and (p / "UTraffic-ML" / "ml_core").exists():
            return p / "UTraffic-ML"
    return cwd


def default_paths() -> Dict[str, Path]:
    project_root = find_project_root()
    branch_b_root = project_root / "ml_core" / "src" / "models" / "ML_BranchB"
    return {
        "project_root": project_root,
        "data_dir": project_root / "ml_core" / "src" / "data_processing" / "outputs" / "branchB" / "osm_edge_granger_series_like_branchA",
        "artifact_dir": branch_b_root / "artifacts" / "sparse_tvpvar_gt_model",
        "method_script": branch_b_root / "scripts" / "06_branchB_run_xt_forecast_sparse_tvpvar_gt.py",
        "output_dir": branch_b_root / "artifacts" / "backend_predictions_sparse_tvpvar",
    }


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_method_module(script_path: Path):
    if not script_path.exists():
        raise FileNotFoundError(f"Cannot find Sparse TVP-VAR method script: {script_path}")

    spec = importlib.util.spec_from_file_location(STABLE_METHOD_MODULE_NAME, str(script_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import method module from: {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[STABLE_METHOD_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def load_artifact(artifact_dir: Path, method_script_override: Optional[Path] = None) -> Dict[str, Any]:
    config_path = artifact_dir / "sparse_tvpvar_config.json"
    model_path = artifact_dir / "sparse_tvpvar_g_model.pkl"
    segment_ids_path = artifact_dir / "segment_ids.npy"
    node_idx_path = artifact_dir / "node_idx.npy"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}. Run 09B_train_sparse_tvpvar_export_model.py first.")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}. Run 09B_train_sparse_tvpvar_export_model.py first.")

    config = load_json(config_path)
    method_script = Path(method_script_override) if method_script_override is not None else Path(config.get("method_script", ""))
    if not method_script.is_absolute():
        method_script = find_project_root() / method_script

    # Important: load module before pickle.load, using the same stable module name.
    module = load_method_module(method_script)

    with open(model_path, "rb") as f:
        g_model = pickle.load(f)

    segment_ids = np.load(segment_ids_path).astype(np.int64)
    node_idx = np.load(node_idx_path).astype(np.int64) if node_idx_path.exists() else np.array([], dtype=np.int64)
    if len(node_idx) == 0:
        node_idx = None

    return {
        "config": config,
        "module": module,
        "g_model": g_model,
        "segment_ids": segment_ids,
        "node_idx": node_idx,
        "artifact_dir": artifact_dir,
    }


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


_TIME_H_RE = re.compile(r"^\s*(\d{1,2})\s*[hH]\s*(\d{1,2})?\s*$")
_TIME_COLON_RE = re.compile(r"^\s*(\d{1,2})\s*:\s*(\d{1,2})(?::\d{1,2})?\s*$")
_TIME_COMPACT_RE = re.compile(r"^\s*(\d{2})(\d{2})\s*$")


def normalize_time_to_hhmm(time_in_day: str) -> str:
    s = str(time_in_day).strip()

    m = _TIME_H_RE.match(s)
    if m:
        hh = int(m.group(1))
        mm = int(m.group(2) or 0)
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            return f"{hh:02d}:{mm:02d}"

    m = _TIME_COLON_RE.match(s)
    if m:
        hh = int(m.group(1))
        mm = int(m.group(2))
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            return f"{hh:02d}:{mm:02d}"

    m = _TIME_COMPACT_RE.match(s)
    if m:
        hh = int(m.group(1))
        mm = int(m.group(2))
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            return f"{hh:02d}:{mm:02d}"

    raise ValueError(f"Cannot parse time={time_in_day!r}. Use 09:00, 9h, 9h15, or 0900.")


def parse_time_minutes_from_any(value: Any) -> Optional[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None

    # direct minutes
    try:
        if isinstance(value, (int, np.integer)):
            v = int(value)
            if 0 <= v <= 24 * 60:
                return v
    except Exception:
        pass

    s = str(value).strip()
    if not s:
        return None

    m = re.search(r"(?<!\d)([0-2]?\d):([0-5]\d)(?!\d)", s)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))

    m = re.search(r"(?<!\d)([0-2]\d)([0-5]\d)(?!\d)", s)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))

    try:
        v = int(float(s))
        if 0 <= v <= 24 * 60:
            return v
    except Exception:
        pass

    return None


def standardize_meta(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Build robust columns:
        timestamp_local, _date, _time_hhmm, sample_id

    Supports Branch-B meta columns such as:
        timestamp_local, date, tod_minutes, time_minutes, slot_index, time_set, time_set_id
    """
    meta = meta.copy().reset_index(drop=True)

    if "sample_id" not in meta.columns:
        meta["sample_id"] = np.arange(len(meta), dtype=np.int64)

    ts = None
    if "timestamp_local" in meta.columns:
        ts = pd.to_datetime(meta["timestamp_local"], errors="coerce")

    date_col = None
    for c in ["date", "date_local", "day", "ngay"]:
        if c in meta.columns:
            date_col = c
            break

    minute_col = None
    for c in ["tod_minutes", "time_minutes", "minute_of_day", "start_minutes"]:
        if c in meta.columns:
            minute_col = c
            break

    if ts is None or ts.isna().all() or (ts.dt.year.fillna(1970).astype(int) == 1970).all():
        # Rebuild timestamp from date + minute/time fields when possible.
        if date_col is not None:
            dates = pd.to_datetime(meta[date_col].astype(str), errors="coerce")
            minutes = None
            if minute_col is not None:
                minutes = pd.to_numeric(meta[minute_col], errors="coerce")
            else:
                for c in ["time", "time_set", "time_set_id", "timestamp", "timestamp_label"]:
                    if c in meta.columns:
                        minutes = meta[c].apply(parse_time_minutes_from_any)
                        break

            if minutes is not None:
                ts = dates + pd.to_timedelta(pd.Series(minutes).fillna(0).astype(int), unit="m")
            else:
                ts = dates

    if ts is None:
        raise ValueError("Cannot build timestamp_local from metadata.")

    meta["timestamp_local"] = pd.to_datetime(ts, errors="coerce")
    meta = meta.dropna(subset=["timestamp_local"]).reset_index(drop=True)
    meta["_date"] = meta["timestamp_local"].dt.date.astype(str)
    meta["_time_hhmm"] = meta["timestamp_local"].dt.strftime("%H:%M")
    return meta


def load_split_date_set(split_dir: Path, meta: pd.DataFrame) -> set:
    raw_meta_path = split_dir / "raw_meta.csv"
    if raw_meta_path.exists():
        raw = pd.read_csv(raw_meta_path)
        if "date" in raw.columns:
            return set(raw["date"].astype(str).dropna().unique().tolist())
        if "timestamp_local" in raw.columns:
            ts = pd.to_datetime(raw["timestamp_local"], errors="coerce")
            return set(ts.dropna().dt.date.astype(str).unique().tolist())
    return set(meta["_date"].astype(str).unique().tolist())


def load_common_metadata(data_dir: Path) -> Dict[str, Any]:
    splits: Dict[str, Dict[str, Any]] = {}
    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        meta_path = split_dir / "G_series_meta.csv"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing metadata: {meta_path}")

        meta = standardize_meta(pd.read_csv(meta_path))
        splits[split] = {
            "split": split,
            "dir": split_dir,
            "meta": meta,
            "date_set": load_split_date_set(split_dir, meta),
        }
    return {"data_dir": data_dir, "splits": splits}


def available_dates_from_common(common_data: Dict[str, Any]) -> List[pd.Timestamp]:
    dates: List[pd.Timestamp] = []
    for split_obj in common_data["splits"].values():
        for d in split_obj["date_set"]:
            dt = pd.to_datetime(str(d), errors="coerce")
            if pd.notna(dt):
                dates.append(pd.Timestamp(dt).normalize())
    return sorted(set(dates))


def parse_request_date(date: str, common_data: Dict[str, Any], default_year: Optional[int] = None) -> pd.Timestamp:
    s = str(date).strip()
    available_dates = available_dates_from_common(common_data)
    if not available_dates:
        raise ValueError("Cannot infer date because no available split dates were found.")

    if re.search(r"\d{4}", s):
        dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
        if pd.isna(dt):
            raise ValueError(f"Cannot parse date={date!r}.")
        return pd.Timestamp(dt).normalize()

    m = re.match(r"^\s*(\d{1,2})\s*[/-]\s*(\d{1,2})\s*$", s)
    if m:
        day = int(m.group(1))
        month = int(m.group(2))
        candidates = [d for d in available_dates if d.day == day and d.month == month]
        if candidates:
            return pd.Timestamp(candidates[0]).normalize()

        if default_year is None:
            years = sorted(set(d.year for d in available_dates))
            if len(years) == 1:
                default_year = years[0]
            else:
                raise ValueError(f"Pass --year because date={date!r} has no year and available years={years}.")

        return pd.Timestamp(year=int(default_year), month=month, day=day)

    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if pd.isna(dt):
        raise ValueError(f"Cannot parse date={date!r}. Use 2024-08-28 or 28/8.")
    return pd.Timestamp(dt).normalize()


def parse_request_timestamp(
    date: str,
    time_in_day: str,
    common_data: Dict[str, Any],
    default_year: Optional[int] = None,
) -> pd.Timestamp:
    d = parse_request_date(date, common_data, default_year=default_year)
    hhmm = normalize_time_to_hhmm(time_in_day)
    hh, mm = map(int, hhmm.split(":"))
    return pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=hh, minute=mm)


def find_available_times_message(meta: pd.DataFrame, query_date: str, max_items: int = 30) -> str:
    same_day = meta[meta["_date"] == query_date].copy()
    if same_day.empty:
        return f"No available timestamp on {query_date}."
    times = same_day["_time_hhmm"].drop_duplicates().tolist()
    head = ", ".join(times[:max_items])
    more = "" if len(times) <= max_items else f", ... ({len(times)} times total)"
    return f"Available times on {query_date}: {head}{more}"


def locate_test_origin(
    query_ts: pd.Timestamp,
    common_data: Dict[str, Any],
    step: int,
) -> Tuple[int, int, Dict[str, Any], Dict[str, Any]]:
    query_date = str(query_ts.date())

    train_dates = common_data["splits"]["train"]["date_set"]
    val_dates = common_data["splits"]["val"]["date_set"]
    test_dates = common_data["splits"]["test"]["date_set"]

    if query_date in train_dates:
        raise PermissionError(f"Rejected: {query_ts} belongs to TRAIN. Backend inference allows TEST only.")
    if query_date in val_dates:
        raise PermissionError(f"Rejected: {query_ts} belongs to VAL. Backend inference allows TEST only.")
    if query_date not in test_dates:
        raise LookupError(f"Date {query_date} is not in TEST split.")

    meta = common_data["splits"]["test"]["meta"]
    hit = meta[meta["timestamp_local"] == query_ts]
    if hit.empty:
        raise LookupError(
            f"No test G_t found for {query_ts}. "
            + find_available_times_message(meta, query_date)
        )

    row = hit.iloc[0].to_dict()
    origin_idx = int(row.get("sample_id", int(hit.index[0])))
    if origin_idx < 0 or origin_idx >= len(meta):
        origin_idx = int(hit.index[0])

    target_idx = origin_idx + int(step)
    if target_idx >= len(meta):
        raise LookupError(f"target_idx={target_idx} is outside test split length={len(meta)}.")

    target_row = meta.iloc[target_idx].to_dict()
    if str(target_row["_date"]) != query_date:
        raise LookupError(
            f"step={step} crosses day boundary: origin={query_ts}, "
            f"target={target_row.get('timestamp_local')}. Use an earlier origin time."
        )

    return origin_idx, target_idx, row, target_row


def save_prediction_bundle(result: Dict[str, Any], output_path: Path, dtype: str = "float32") -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "G_pred": result["G_pred"].astype(dtype),
        "segment_ids": result["segment_ids"].astype(np.int64),
        "matrix_index": np.arange(len(result["segment_ids"]), dtype=np.int64),
        "split": np.array(str(result["split"])),
        "source_sample_id": np.array([int(result["source_sample_id"])], dtype=np.int64),
        "target_sample_id": np.array([int(result["target_sample_id"])], dtype=np.int64),
        "step": np.array([int(result["step"])], dtype=np.int64),
        "request_timestamp": np.array(str(result["request_timestamp"])),
        "target_timestamp": np.array(str(result["target_timestamp"])),
        "matrix_contract": np.array(str(result["matrix_contract"])),
    }
    np.savez_compressed(output_path, **payload)

    meta = {k: v for k, v in result.items() if k not in {"G_pred", "segment_ids"}}
    with open(output_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
    return output_path


def build_default_output_path(output_dir: Path, result: Dict[str, Any]) -> Path:
    ts = pd.Timestamp(result["request_timestamp"])
    return output_dir / f"sparse_tvpvar_backend_pred_test_{ts:%Y%m%d_%H%M}_h{int(result['step'])}.npz"


def predict_by_datetime(
    date: str,
    time_in_day: str,
    step: int,
    data_dir: Path | str | None = None,
    artifact_dir: Path | str | None = None,
    method_script: Path | str | None = None,
    default_year: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Backend-friendly function.

    Returns G_pred in memory; it does not save anything.
    """
    if int(step) <= 0:
        raise ValueError(f"step must be positive, got {step}.")

    paths = default_paths()

    data_dir = Path(data_dir) if data_dir is not None else paths["data_dir"]
    artifact_dir = Path(artifact_dir) if artifact_dir is not None else paths["artifact_dir"]
    method_script_path = Path(method_script) if method_script is not None else None

    if not data_dir.is_absolute():
        data_dir = paths["project_root"] / data_dir
    if not artifact_dir.is_absolute():
        artifact_dir = paths["project_root"] / artifact_dir
    if method_script_path is not None and not method_script_path.is_absolute():
        method_script_path = paths["project_root"] / method_script_path

    artifact = load_artifact(artifact_dir, method_script_override=method_script_path)
    module = artifact["module"]
    g_model = artifact["g_model"]
    segment_ids = artifact["segment_ids"]
    node_idx = artifact["node_idx"]

    common_meta = load_common_metadata(data_dir)
    query_ts = parse_request_timestamp(date, time_in_day, common_meta, default_year=default_year)
    origin_idx, target_idx, origin_row, target_row = locate_test_origin(query_ts, common_meta, int(step))

    test = module.load_gt_split(data_dir, "test")
    test = subset_split_data(test, node_idx)

    if len(np.asarray(test["segment_ids"])) != len(segment_ids):
        raise ValueError(
            f"segment_ids length mismatch: artifact={len(segment_ids)}, test={len(test['segment_ids'])}."
        )

    t0 = time.perf_counter()
    G_pred = module.predict_G_method(
        METHOD_NAME,
        g_model,
        "test",
        test,
        int(origin_idx),
        int(target_idx),
        int(step),
    )
    online_predict_time_sec = time.perf_counter() - t0

    G_pred = np.asarray(G_pred, dtype=np.float32)
    if G_pred.ndim != 2 or G_pred.shape[0] != G_pred.shape[1]:
        raise ValueError(f"G_pred must be square, got shape={G_pred.shape}")
    if G_pred.shape[0] != len(segment_ids):
        raise ValueError(f"G_pred shape mismatch: G_pred={G_pred.shape}, segment_ids={len(segment_ids)}")

    return {
        "G_pred": G_pred,
        "segment_ids": segment_ids,
        "split": "test",
        "source_sample_id": int(origin_idx),
        "target_sample_id": int(target_idx),
        "step": int(step),
        "request_timestamp": str(query_ts),
        "target_timestamp": str(target_row.get("timestamp_local")),
        "origin_row": {k: str(v) for k, v in origin_row.items() if k not in {"G_weight_series"}},
        "target_row": {k: str(v) for k, v in target_row.items() if k not in {"G_weight_series"}},
        "online_predict_time_sec": float(online_predict_time_sec),
        "matrix_shape": list(map(int, G_pred.shape)),
        "matrix_contract": "G_pred[target, source] = predicted directed influence from source road to target road",
        "saved": False,
    }


def parse_args() -> argparse.Namespace:
    paths = default_paths()
    ap = argparse.ArgumentParser(description="Backend on-demand Sparse TVP-VAR predictor for Branch-B G matrices.")
    ap.add_argument("--data-dir", type=str, default=str(paths["data_dir"]))
    ap.add_argument("--artifact-dir", type=str, default=str(paths["artifact_dir"]))
    ap.add_argument("--method-script", type=str, default=None)
    ap.add_argument("--date", type=str, required=True, help="Date, e.g. 2024-08-28 or 28/8.")
    ap.add_argument("--time", type=str, required=True, help="Time, e.g. 09:00, 9h, 9h15.")
    ap.add_argument("--step", type=int, required=True, help="Forecast horizon h, e.g. 3.")
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--print-topk", type=int, default=0, help="Print top-k strongest directed edges as preview. 0 disables.")
    ap.add_argument("--save-npz", action="store_true", help="Optional only. Default does NOT save predicted matrix.")
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--output-dir", type=str, default=str(paths["output_dir"]))
    ap.add_argument("--dtype", type=str, default="float32", choices=["float16", "float32"])
    return ap.parse_args()


def topk_edges_preview(G: np.ndarray, segment_ids: np.ndarray, k: int) -> List[Dict[str, Any]]:
    G = np.asarray(G)
    N = G.shape[0]
    k = int(k)
    if k <= 0:
        return []
    mask = np.ones((N, N), dtype=bool)
    np.fill_diagonal(mask, False)
    flat_idx = np.argpartition(np.abs(G[mask]), -min(k, mask.sum()))[-min(k, mask.sum()):]
    coords = np.argwhere(mask)
    chosen = coords[flat_idx]
    chosen = chosen[np.argsort(-np.abs(G[chosen[:, 0], chosen[:, 1]]))]
    rows = []
    for rank, (target_idx, source_idx) in enumerate(chosen, start=1):
        rows.append({
            "rank": int(rank),
            "target_index": int(target_idx),
            "source_index": int(source_idx),
            "target_segment_id": int(segment_ids[target_idx]),
            "source_segment_id": int(segment_ids[source_idx]),
            "weight": float(G[target_idx, source_idx]),
        })
    return rows


def main() -> None:
    args = parse_args()

    log("=" * 96)
    log("BRANCH-B SPARSE TVP-VAR BACKEND PREDICT")
    log("=" * 96)
    log(f"date/time : {args.date} {args.time}")
    log(f"step      : {args.step}")
    log("rule      : TEST split only; no online refit; default no matrix saving")

    result = predict_by_datetime(
        date=args.date,
        time_in_day=args.time,
        step=args.step,
        data_dir=args.data_dir,
        artifact_dir=args.artifact_dir,
        method_script=args.method_script,
        default_year=args.year,
    )

    print("\n[OK] split              :", result["split"])
    print("[OK] source_sample_id   :", result["source_sample_id"])
    print("[OK] target_sample_id   :", result["target_sample_id"])
    print("[OK] request_timestamp  :", result["request_timestamp"])
    print("[OK] target_timestamp   :", result["target_timestamp"])
    print("[OK] G_pred shape       :", tuple(result["matrix_shape"]))
    print("[OK] segment_ids shape  :", result["segment_ids"].shape)
    print("[OK] online_time_sec    :", f"{result['online_predict_time_sec']:.6f}")
    print("[OK] contract           :", result["matrix_contract"])
    print("[OK] saved              :", result["saved"])

    if args.print_topk > 0:
        preview = topk_edges_preview(result["G_pred"], result["segment_ids"], args.print_topk)
        print("\n[PREVIEW] top directed edges:")
        print(json.dumps(preview, ensure_ascii=False, indent=2))

    if args.save_npz or args.output:
        output_path = Path(args.output) if args.output else build_default_output_path(Path(args.output_dir), result)
        if not output_path.is_absolute():
            output_path = find_project_root() / output_path
        save_prediction_bundle(result, output_path, dtype=args.dtype)
        print("[SAVED] bundle:", output_path)
        print("[SAVED] meta  :", output_path.with_suffix(".json"))

    log("DONE")


if __name__ == "__main__":
    main()

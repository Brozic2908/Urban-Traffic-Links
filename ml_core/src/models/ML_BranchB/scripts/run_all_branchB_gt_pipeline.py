"""
One-command Branch B Granger-Gt pipeline.

This replaces the long manual Kaggle sequence in the notebook.
It can prepare data, check files, run selected methods, and generate MAE/MSE/RMSE
plots plus directed Top-K overlap plots.

Recommended quick test on Kaggle:
    python -u ml_core/src/models/ML_BranchB/scripts/run_all_branchB_gt_pipeline.py \
      --max-nodes 512 \
      --methods true_gt,persistence_gt,ewma_gt,dmfm_lse_gt,dmfm_vlse_gt \
      --lags 1-3 \
      --topk 20 \
      --n-jobs 2 \
      --overwrite

Recommended fuller run after the quick test:
    DMFM_GT_RANK=8,8 DMFM_GT_RIDGE=0.001 DMFM_GT_STABILIZE=1 DMFM_GT_RHO=0.98 \
    python -u ml_core/src/models/ML_BranchB/scripts/run_all_branchB_gt_pipeline.py \
      --max-nodes 0 \
      --methods all \
      --lags 1-9 \
      --topk 20 \
      --n-jobs 2 \
      --overwrite
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List


def find_project_root() -> Path:
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents, Path('/kaggle/working/UTraffic-ML'), Path('/kaggle/working')]:
        if (p / 'ml_core').exists() and (p / 'dataset').exists():
            return p
        if p.name == 'UTraffic-ML':
            return p
        if (p / 'UTraffic-ML').exists():
            pp = p / 'UTraffic-ML'
            if (pp / 'ml_core').exists():
                return pp
    return cwd


def run(cmd: List[str], stop_on_error: bool = True) -> int:
    print('\n' + '=' * 100, flush=True)
    print('RUN:', ' '.join(cmd), flush=True)
    print('=' * 100, flush=True)
    proc = subprocess.run(cmd)
    if proc.returncode != 0 and stop_on_error:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return int(proc.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-nodes', type=int, default=512, help='0 means full nodes. Use 512 first for quick test.')
    parser.add_argument('--topk', type=int, default=20)
    parser.add_argument('--lags', type=str, default='1-9')
    parser.add_argument('--methods', type=str, default='all', help='all or comma-list, e.g. true_gt,ewma_gt,dmfm_lse_gt,dmfm_vlse_gt')
    parser.add_argument('--splits', type=str, default='val,test')
    parser.add_argument('--granger-p', type=int, default=3)
    parser.add_argument('--granger-horizon', type=int, default=1)
    parser.add_argument('--bucket-minutes', type=int, default=60)
    parser.add_argument('--max-candidates', type=int, default=50)
    parser.add_argument('--n-jobs', type=int, default=2)
    parser.add_argument('--parallel-level', type=str, default='method', choices=['method', 'horizon', 'none'])
    parser.add_argument('--results-dir', type=str, default='ml_core/src/models/ML_BranchB/results/06_branchB_gt_pipeline')
    parser.add_argument('--report-dir', type=str, default='ml_core/src/models/ML_BranchB/results/branchB_report')
    parser.add_argument('--data-dir', type=str, default='ml_core/src/data_processing/outputs/branchB/osm_edge_granger_series_like_branchA')
    parser.add_argument('--topk-values', type=str, default='5,10,20,50')
    parser.add_argument('--samples-per-split-lag', type=int, default=10)
    parser.add_argument('--skip-prepare', action='store_true', help='Use existing prepared data-dir.')
    parser.add_argument('--skip-check', action='store_true')
    parser.add_argument('--skip-run', action='store_true')
    parser.add_argument('--skip-report', action='store_true')
    parser.add_argument('--skip-overlap', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--skip-existing', action='store_true')
    parser.add_argument('--stop-on-error', action='store_true')
    args = parser.parse_args()

    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

    root = find_project_root()
    os.chdir(root)
    py = sys.executable

    data_dir = args.data_dir
    results_dir = args.results_dir
    report_dir = args.report_dir

    if not args.skip_prepare:
        prepare_cmd = [
            py, '-u', 'ml_core/src/data_processing/prepare_branchB_osm_edge_granger_series_like_branchA.py',
            '--granger-horizon', str(args.granger_horizon),
            '--granger-p', str(args.granger_p),
            '--bucket-minutes', str(args.bucket_minutes),
            '--max-candidates', str(args.max_candidates),
        ]
        if args.max_nodes > 0:
            prepare_cmd += ['--max-nodes', str(args.max_nodes)]
        if args.overwrite:
            prepare_cmd += ['--overwrite']
        run(prepare_cmd, stop_on_error=True)

    if not args.skip_check:
        check_cmd = [
            py, '-u', 'ml_core/src/models/ML_BranchB/scripts/00_check_branchB_prepared_data.py',
            '--data-dir', data_dir,
        ]
        run(check_cmd, stop_on_error=True)

    if not args.skip_run:
        run_cmd = [
            py, '-u', 'ml_core/src/models/ML_BranchB/scripts/06B_branchB_run_xt_forecast_topk_gt.py',
            '--data-dir', data_dir,
            '--results-dir', results_dir,
            '--methods', str(args.methods),
            '--topk', str(args.topk),
            '--lags', str(args.lags),
            '--parallel-level', str(args.parallel_level),
            '--n-jobs', str(args.n_jobs),
        ]
        if args.max_nodes > 0:
            # Safe even if prepare already reduced nodes; this just keeps runner subset consistent.
            run_cmd += ['--max-nodes', str(args.max_nodes)]
        if args.skip_existing:
            run_cmd += ['--skip-existing']
        if args.stop_on_error:
            run_cmd += ['--stop-on-error']
        run(run_cmd, stop_on_error=bool(args.stop_on_error))

    if not args.skip_report:
        report_cmd = [
            py, '-u', 'ml_core/src/models/ML_BranchB/scripts/07B_branchB_report_metrics_and_overlap.py',
            '--data-dir', data_dir,
            '--results-dir', results_dir,
            '--out-dir', report_dir,
            '--methods', str(args.methods),
            '--splits', str(args.splits),
            '--lags', str(args.lags),
            '--topk-values', str(args.topk_values),
            '--samples-per-split-lag', str(args.samples_per_split_lag),
        ]
        if args.max_nodes > 0:
            report_cmd += ['--max-nodes', str(args.max_nodes)]
        if args.skip_overlap:
            report_cmd += ['--skip-overlap']
        run(report_cmd, stop_on_error=False)

    print('\nDONE.', flush=True)
    print('Prepared data:', root / data_dir, flush=True)
    print('Method results:', root / results_dir, flush=True)
    print('Report:', root / report_dir, flush=True)


if __name__ == '__main__':
    main()

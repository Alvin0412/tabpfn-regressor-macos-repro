# TabPFN Regressor macOS Repro

This repo isolates an intermittent native crash observed while fitting `TabPFNRegressor`
repeatedly on a small tabular regression task on macOS arm64.

Current status:

- Environment:
  - macOS 15.5 arm64
  - Python 3.13.11
  - `tabpfn==6.4.1`
  - `torch==2.10.0`
  - `scipy==1.17.1`
  - `scikit-learn==1.8.0`
- The crash was observed in a larger Kaggle training pipeline while repeatedly fitting
  `TabPFNRegressor` inside a sklearn `Pipeline`.
- The failure mode is not a normal Python exception. macOS generated `.ips` crash reports
  with `EXC_BAD_ACCESS` / `SIGSEGV`.
- The repro appears intermittent. Single `fit/predict` runs may succeed. Repeated runs are
  more likely to surface the problem.

Included assets:

- `data/men_fold1_train_aug.csv`
- `data/men_fold1_valid.csv`
- `artifacts/data_summary.json`
- `artifacts/crash_summary.txt`

These data files are a small exported regression fold from the upstream project. They are
only used to make the crash easier to exercise without depending on the original codebase.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Repro

Single run:

```bash
python scripts/repro_real_fold.py --loops 1
```

Stress loop:

```bash
python scripts/repro_real_fold.py --loops 20
```

The script prints progress after each `fit/predict` cycle. On the failing machine, the
process may terminate abruptly without a Python traceback.

## Notes

- This repo intentionally does not claim a perfectly deterministic repro yet.
- The goal is to give upstream maintainers a compact environment, a real fold sample, and
  crash-report context.

### Describe the bug

I observed an intermittent native crash while fitting `TabPFNRegressor` repeatedly on macOS arm64.

This is **not** a normal Python exception. On the failing machine, Python sometimes terminates abruptly and macOS writes `.ips` crash reports with `EXC_BAD_ACCESS` / `SIGSEGV`.

I have a small public repro repo here:

- https://github.com/Alvin0412/tabpfn-regressor-macos-repro

Important nuance: I do **not** currently have a perfectly deterministic repro. Single `fit/predict` runs often succeed. The crash originally showed up inside a larger training loop, and the best minimized version I have so far is a repeated-fit stress script using a real small regression fold.

### Steps/Code to Reproduce

From the repro repo:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
python scripts/print_env.py
python scripts/repro_real_fold.py --loops 20
```

Repro repo contents:

- `data/men_fold1_train_aug.csv`
- `data/men_fold1_valid.csv`
- `scripts/repro_real_fold.py`
- `artifacts/crash_summary.txt`

The script repeatedly does:

```python
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", TabPFNRegressor(
        device="cpu",
        n_estimators=1,
        fit_mode="fit_preprocessors",
        n_preprocessing_jobs=1,
        ignore_pretraining_limits=True,
        random_state=42,
    )),
])
model.fit(X_train, y_train)
model.predict(X_valid)
```

### Expected Results

The loop should complete without the Python process terminating.

### Actual Results

On the failing machine, the process sometimes terminates abruptly during the repeated fit/predict loop. When it happens, there is no useful Python traceback, but macOS writes crash reports showing:

- `EXC_BAD_ACCESS`
- `SIGSEGV`

In the crash reports I also repeatedly saw references to:

- `libomp`
- `libtorch_cpu`
- `libarrow`

I included a short extracted summary in:

- `artifacts/crash_summary.txt`

### Additional context

- I already checked issue #175, but that seems to be about `scipy < 1.11`. My environment uses `scipy==1.17.1`.
- I also noticed PR #802 mentions macOS CPU regression/regressor testing, which is why I thought this might still be worth reporting even though my repro is currently intermittent.
- The crash first appeared while repeatedly fitting `TabPFNRegressor` inside a sklearn `Pipeline` during a small tabular regression workload.

### Versions

```text
python 3.13.11
platform macOS-15.5-arm64-arm-64bit-Mach-O
tabpfn 6.4.1
torch 2.10.0
numpy 2.4.3
scipy 1.17.1
pandas 2.3.3
scikit-learn 1.8.0
pyarrow 23.0.1
```

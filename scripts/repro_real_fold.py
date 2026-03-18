#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from tabpfn import TabPFNRegressor


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "men_fold1_train_aug.csv"
VALID_PATH = ROOT / "data" / "men_fold1_valid.csv"


def build_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                TabPFNRegressor(
                    device="cpu",
                    n_estimators=1,
                    fit_mode="fit_preprocessors",
                    n_preprocessing_jobs=1,
                    ignore_pretraining_limits=True,
                    random_state=42,
                ),
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loops", type=int, default=20)
    args = parser.parse_args()

    train_df = pd.read_csv(TRAIN_PATH)
    valid_df = pd.read_csv(VALID_PATH)
    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]
    X_valid = valid_df

    print(f"train shape={X_train.shape} valid shape={X_valid.shape}")
    for i in range(1, args.loops + 1):
        print(f"iter {i}: build")
        model = build_model()
        print(f"iter {i}: fit")
        model.fit(X_train, y_train)
        print(f"iter {i}: predict")
        pred = model.predict(X_valid)
        print(f"iter {i}: ok {pred.shape}")


if __name__ == "__main__":
    main()

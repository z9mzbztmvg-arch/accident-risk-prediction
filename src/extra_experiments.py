"""Дополнительные эксперименты к отчёту: важность по выигрышу, абляции
групп признаков и сравнение моделей при фиксированном бюджете тревог.

Скрипт не перезаписывает основные артефакты конвейера. Результаты
сохраняются в extra_results.json в корне репозитория.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

import real_data
from real_data import FEATURES as REAL_FEATURES
from nir_pipeline import RANDOM_STATE, best_f1_threshold, statistical_baseline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INFRA = ["road_density", "intersection_count", "avg_lanes", "speed_limit",
         "dist_to_center", "poi_density", "is_school_zone", "pavement_code"]
CAL = ["month", "day_of_week", "is_weekend", "is_holiday", "is_winter",
       "doy_sin", "doy_cos"]
WEATHER = ["temperature", "precipitation", "wind_speed", "cloud_cover",
           "snowfall", "is_snow"]
LAGS = ["lag_acc_1", "roll_acc_7", "neigh_acc_1"]
ALL = list(REAL_FEATURES)


def metrics(y_true, score, threshold):
    pred = (score >= threshold).astype(int)
    return {
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, score)),
        "pr_auc": float(average_precision_score(y_true, score)),
    }


def main() -> None:
    t0 = time.time()
    df = real_data.build_dataset()
    print(f"dataset {len(df):,} rows, {time.time() - t0:.1f} s", flush=True)

    tr = df[df["date"] < "2023-10-01"]
    va = df[(df["date"] >= "2023-10-01") & (df["date"] < "2024-01-01")]
    te = df[df["date"] >= "2024-01-01"]

    with open(os.path.join(ROOT, "results.json"), encoding="utf-8") as f:
        saved = json.load(f)
    best_params = saved["best_params"]
    spw = float((tr["target"] == 0).sum()) / float((tr["target"] == 1).sum())

    out: dict = {"ablation": {}, "budgets": {}, "importance": {},
                 "training_seconds": {}}

    def fit_lgb(features):
        model = LGBMClassifier(scale_pos_weight=spw, random_state=RANDOM_STATE,
                               n_jobs=-1, verbose=-1, **best_params)
        model.fit(tr[features].to_numpy(), tr["target"].to_numpy())
        return model

    # 1. Абляции групп признаков.
    sets = {
        "infra": INFRA,
        "infra_cal": INFRA + CAL,
        "infra_cal_weather": INFRA + CAL + WEATHER,
        "all": ALL,
    }
    for name, features in sets.items():
        t = time.time()
        model = fit_lgb(features)
        pv = model.predict_proba(va[features].to_numpy())[:, 1]
        thr = best_f1_threshold(va["target"].to_numpy(), pv)
        pt = model.predict_proba(te[features].to_numpy())[:, 1]
        out["ablation"][name] = metrics(te["target"].to_numpy(), pt, thr)
        out["ablation"][name]["threshold"] = thr
        out["ablation"][name]["n_features"] = len(features)
        out["training_seconds"][f"lgb_{name}"] = round(time.time() - t, 1)
        print(name, out["ablation"][name], flush=True)

    # Важность по выигрышу для полного набора.
    full = fit_lgb(ALL)
    gain = full.booster_.feature_importance(importance_type="gain")
    split = full.booster_.feature_importance(importance_type="split")
    order = np.argsort(gain)[::-1]
    out["importance"] = [
        {"feature": ALL[i], "gain": float(gain[i]), "split": int(split[i])}
        for i in order
    ]

    # 2. Модели при одинаковом бюджете тревог: доля отмеченных строк.
    models = {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000,
                               random_state=RANDOM_STATE)),
        "rf": RandomForestClassifier(
            n_estimators=200, max_depth=16, min_samples_leaf=3,
            class_weight="balanced_subsample", n_jobs=-1,
            random_state=RANDOM_STATE),
        "lightgbm": LGBMClassifier(scale_pos_weight=spw,
                                   random_state=RANDOM_STATE, n_jobs=-1,
                                   verbose=-1, **best_params),
        "xgboost": XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.06,
            scale_pos_weight=spw, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, eval_metric="auc",
            tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE),
    }
    scores = {}
    for name, model in models.items():
        t = time.time()
        model.fit(tr[ALL].to_numpy(), tr["target"].to_numpy())
        pv = model.predict_proba(va[ALL].to_numpy())[:, 1]
        pt = model.predict_proba(te[ALL].to_numpy())[:, 1]
        scores[name] = pt
        thr = best_f1_threshold(va["target"].to_numpy(), pv)
        out.setdefault("point_metrics", {})[name] = metrics(
            te["target"].to_numpy(), pt, thr)
        out["point_metrics"][name]["threshold"] = thr
        out["training_seconds"][name] = round(time.time() - t, 1)
        print(name, out["point_metrics"][name], flush=True)

    yb, sb, _ = statistical_baseline(tr, va, te)
    scores["statistical"] = sb
    _, sv, _ = statistical_baseline(tr, va, va)
    thr_stat = best_f1_threshold(va["target"].to_numpy(), sv)
    out["point_metrics"]["statistical"] = metrics(yb, sb, thr_stat)
    out["point_metrics"]["statistical"]["threshold"] = thr_stat

    y = te["target"].to_numpy()
    n = len(y)
    for frac in (0.01, 0.05, 0.10):
        k = int(round(frac * n))
        row = {}
        for name, score in scores.items():
            cut = np.partition(score, n - k)[n - k]
            chosen = score >= cut
            row[name] = {
                "alerts": int(chosen.sum()),
                "precision": float(y[chosen].mean()),
                "recall": float(y[chosen].sum() / y.sum()),
            }
        out["budgets"][f"{int(frac * 100)}pct"] = row

    out["n_test"] = n
    out["n_test_positive"] = int(y.sum())
    out["total_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(ROOT, "extra_results.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

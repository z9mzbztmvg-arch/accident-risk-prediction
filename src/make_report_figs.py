"""Перегенерация отдельных рисунков отчёта в едином стиле шаблона практики.

Скрипт полезен, когда нужно обновить только профиль происшествий и важность
признаков, не запуская полный конвейер с подбором гиперпараметров.
"""
from __future__ import annotations

import os
import sys

import joblib
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import nir_pipeline as npipeline
import real_data

OUT = os.environ.get("FIG_DIR", os.path.join(os.path.dirname(HERE), "fig"))
os.makedirs(OUT, exist_ok=True)
npipeline.OUT_DIR = OUT

FEATURE_NAMES_RU = {
    "road_density": "плотность УДС",
    "intersection_count": "число перекрёстков",
    "avg_lanes": "число полос",
    "speed_limit": "ограничение скорости",
    "dist_to_center": "удалённость от центра",
    "poi_density": "плотность объектов притяжения",
    "is_school_zone": "школьная зона",
    "pavement_code": "тип покрытия",
    "temperature": "температура",
    "precipitation": "осадки",
    "wind_speed": "скорость ветра",
    "cloud_cover": "облачность",
    "snowfall": "слой снега",
    "is_snow": "снегопад",
    "month": "месяц",
    "day_of_week": "день недели",
    "is_weekend": "выходной день",
    "is_holiday": "праздник",
    "is_winter": "зимний сезон",
    "doy_sin": "сезонная составляющая (синус)",
    "doy_cos": "сезонная составляющая (косинус)",
    "lag_acc_1": "ДТП за предыдущие сутки",
    "roll_acc_7": "ДТП за неделю",
    "neigh_acc_1": "ДТП у соседних ячеек",
}


def temporal() -> None:
    print("Загрузка реальных данных для временного профиля ...")
    df = npipeline.clean_dataset(real_data.build_dataset())
    npipeline.fig_temporal(df)


def importance() -> None:
    bundle = joblib.load(os.path.join(os.path.dirname(HERE), "model.joblib"))
    model = bundle["model"]
    names = bundle["features"]
    pairs = sorted(zip(names, model.feature_importances_), key=lambda p: p[1])[-15:]
    labels = [FEATURE_NAMES_RU.get(name, name) for name, _ in pairs]
    values = [float(value) for _, value in pairs]
    fig, ax = plt.subplots(figsize=(npipeline.LINE_WIDTH_IN, 6.2))
    ax.barh(labels, values, color="#55A868", edgecolor="black", linewidth=0.6)
    ax.set_xlabel("Важность признака")
    npipeline._black_axes(ax)
    fig.tight_layout()
    npipeline._save(fig, "feature_importance")


if __name__ == "__main__":
    temporal()
    importance()
    print("Готово:", OUT)

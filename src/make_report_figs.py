"""Перегенерация отдельных рисунков отчёта в едином стиле шаблона практики.

Скрипт полезен, когда нужно обновить только профиль происшествий, важность
признаков и карту риска, не запуская полный конвейер с подбором гиперпараметров.
"""
from __future__ import annotations

import json
import os
import sys

import joblib
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

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
    gain = model.booster_.feature_importance(importance_type="gain")
    pairs = sorted(zip(names, gain), key=lambda p: p[1])[-15:]
    labels = [FEATURE_NAMES_RU.get(name, name) for name, _ in pairs]
    values = [float(value) / 1000.0 for _, value in pairs]
    fig, ax = plt.subplots(figsize=(npipeline.LINE_WIDTH_IN, 6.2))
    ax.barh(labels, values, color="#55A868", edgecolor="black", linewidth=0.6)
    ax.set_xlabel("Выигрыш, тыс.")
    npipeline._black_axes(ax)
    fig.tight_layout()
    npipeline._save(fig, "feature_importance")


def risk_map() -> None:
    """Карта риска в географических координатах с отметкой центра."""
    path = os.path.join(os.path.dirname(HERE), "risk_grids.json")
    with open(path, encoding="utf-8") as f:
        risk = np.asarray(json.load(f)["all"], dtype=float)
    lat0, lat1 = 55.57, 55.92
    lon0, lon1 = 37.35, 37.90
    fig, ax = plt.subplots(figsize=(npipeline.LINE_WIDTH_IN, 5.8))
    im = ax.imshow(risk.T, origin="lower", cmap="YlOrRd",
                   extent=[lon0, lon1, lat0, lat1],
                   aspect=1.0 / np.cos(np.radians(55.75)))
    from numpy.lib.stride_tricks import sliding_window_view
    padded = np.pad(np.nan_to_num(risk, nan=float(np.nanmean(risk))), 2,
                    mode="edge")
    smooth = sliding_window_view(padded, (5, 5)).mean(axis=(-1, -2))
    level = float(np.percentile(smooth, 90))
    ax.contour(np.linspace(lon0, lon1, risk.shape[1]),
               np.linspace(lat0, lat1, risk.shape[0]),
               smooth, levels=[level], colors="black", linewidths=1.1)
    ax.plot(37.6178, 55.7520, marker="*", color="black", markersize=9,
            markeredgecolor="white", markeredgewidth=0.6)
    ax.annotate("Кремль", xy=(37.6178, 55.7520), xytext=(4, -12),
                textcoords="offset points", fontsize=8)
    ax.set_xlabel("Долгота, градусы")
    ax.set_ylabel("Широта, градусы")
    ax.set_xticks(np.arange(37.4, 37.91, 0.1))
    ax.set_yticks(np.arange(55.6, 55.93, 0.1))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Средний прогнозный риск")
    proxy = mlines.Line2D([], [], color="black", linewidth=1.1,
                          label="90-й процентиль сглаженного риска")
    ax.legend(handles=[proxy], loc="lower right", fontsize=8, frameon=True)
    npipeline._black_axes(ax)
    fig.tight_layout()
    npipeline._save(fig, "risk_map")


if __name__ == "__main__":
    temporal()
    importance()
    risk_map()
    print("Готово:", OUT)

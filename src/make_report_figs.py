"""Перегенерация рисунков отчёта в едином стиле шаблона практики."""
from __future__ import annotations

import json
import os
import sys
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import nir_pipeline as npipeline

OUT = os.environ.get("FIG_DIR", os.path.join(os.path.dirname(HERE), "fig"))
os.makedirs(OUT, exist_ok=True)
npipeline.OUT_DIR = OUT

# Tempora в matplotlib некорректно строит часть подписей при экспорте,
# поэтому для перегенерируемых рисунков используется близкий Times New Roman.
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]


def save(fig, stem):
    """Сохранение без bbox_inches='tight' (иначе ломаются метрики шрифта)."""
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{stem}.{ext}"), dpi=200)
    plt.close(fig)


def thousands(value, _pos=None):
    return f"{value:,.0f}".replace(",", " ")


def temporal():
    print("Генерация набора данных для временного профиля ...")
    df = npipeline.generate_dataset()
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek + 1
    fmt = FuncFormatter(thousands)
    fig, ax = plt.subplots(1, 2, figsize=(npipeline.LINE_WIDTH_IN, 3.5))
    m = df.groupby("month")["accident_count"].sum()
    ax[0].bar(m.index, m.values, width=0.72, color="#4C72B0",
              edgecolor="black", linewidth=0.6)
    ax[0].set_xlabel("Месяц")
    ax[0].set_ylabel("Число ДТП")
    ax[0].set_xticks(range(1, 13))
    ax[0].set_xlim(0.4, 12.6)
    ax[0].yaxis.set_major_formatter(fmt)
    d = df.groupby("day_of_week")["accident_count"].sum()
    ax[1].bar(range(1, 8), d.values, width=0.72, color="#DD8452",
              edgecolor="black", linewidth=0.6)
    ax[1].set_xlabel("День недели")
    ax[1].set_ylabel("Число ДТП")
    ax[1].set_xticks(range(1, 8))
    ax[1].set_xticklabels(["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"])
    ax[1].set_xlim(0.4, 7.6)
    ax[1].yaxis.set_major_formatter(fmt)
    for a in ax:
        npipeline._black_axes(a)
    fig.subplots_adjust(left=0.13, right=0.985, top=0.97, bottom=0.19, wspace=0.34)
    save(fig, "temporal_profile")


FEATURE_NAMES_RU = {
    "road_density": "плотность улично-дорожной сети",
    "precipitation": "осадки",
    "roll_acc_7": "ДТП за неделю",
    "intersection_count": "число перекрёстков",
    "centrality": "центральность",
    "temperature": "температура",
    "is_snow": "снегопад",
    "visibility": "видимость",
    "dist_to_center": "удалённость от центра",
    "is_school_zone": "близость школы",
    "poi_density": "плотность объектов притяжения",
    "day_of_week": "день недели",
    "doy_cos": "сезонная составляющая (косинус)",
    "traffic_density": "плотность потока",
    "wind_speed": "скорость ветра",
}


def importance():
    with open(os.path.join(os.path.dirname(HERE), "feature_importance.json"),
              encoding="utf-8") as f:
        data = json.load(f)
    data = data[:15]
    labels = [textwrap.fill(FEATURE_NAMES_RU.get(row["feature"], row["feature"]), 26)
              for row in data]
    values = [row["importance"] for row in data]
    fig, ax = plt.subplots(figsize=(npipeline.LINE_WIDTH_IN, 6.2))
    ax.barh(labels[::-1], values[::-1], color="#55A868", edgecolor="black",
            linewidth=0.6)
    ax.set_xlabel("Важность признака")
    npipeline._black_axes(ax)
    fig.subplots_adjust(left=0.40, right=0.985, top=0.985, bottom=0.11)
    save(fig, "feature_importance")


if __name__ == "__main__":
    temporal()
    importance()
    print("Готово:", OUT)

"""Модуль визуализации зон риска и сервис прогнозирования.

Формирует интерактивную карту риска по сохранённым прогнозам обученной модели
и предоставляет модель прогнозирования через HTTP-интерфейс.
Запуск:  uvicorn src.app:app --reload
"""
from __future__ import annotations

import json
import os

import folium
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Границы условного мегаполиса (Волгоград) для привязки регулярной сетки
LAT_MIN, LAT_MAX = 48.65, 48.74
LON_MIN, LON_MAX = 44.45, 44.56
GRID = 30
MODEL_PATH = os.environ.get("NIR_MODEL", "model.joblib")
RISK_PATH = os.environ.get("NIR_RISK", "risk_grids.json")

app = FastAPI(title="Прогнозирование риска дорожно-транспортных происшествий")

_model = None
_risk_grids = None


def get_model():
    """Ленивая загрузка обученной модели градиентного бустинга."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                status_code=503,
                detail="Модель не найдена. Запустите nir_pipeline.py.",
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def get_risk_grids() -> dict:
    """Ленивая загрузка помесячных карт риска, рассчитанных моделью."""
    global _risk_grids
    if _risk_grids is None:
        if not os.path.exists(RISK_PATH):
            raise HTTPException(
                status_code=503,
                detail="Карты риска не найдены. Запустите nir_pipeline.py.",
            )
        with open(RISK_PATH, encoding="utf-8") as f:
            _risk_grids = json.load(f)
    return _risk_grids


class Observation(BaseModel):
    """Признаки наблюдения «ячейка --- сутки»."""

    features: list[float]


@app.get("/health")
def health() -> dict:
    """Проверка доступности сервиса."""
    return {"status": "ok"}


@app.post("/predict")
def predict(item: Observation) -> dict:
    """Возвращает вероятность возникновения происшествия в ячейке."""
    bundle = get_model()
    model = bundle["model"] if isinstance(bundle, dict) else bundle
    x = np.asarray(item.features, dtype=float).reshape(1, -1)
    risk = float(model.predict_proba(x)[0, 1])
    return {"risk": round(risk, 4), "high_risk": bool(risk >= 0.5)}


def render_risk_map(risk_grid: np.ndarray) -> folium.Map:
    """Строит интерактивную карту риска по регулярной сетке ячеек."""
    risk_grid = np.asarray(risk_grid, dtype=float)
    lat_step = (LAT_MAX - LAT_MIN) / GRID
    lon_step = (LON_MAX - LON_MIN) / GRID
    fmap = folium.Map(location=[(LAT_MIN + LAT_MAX) / 2, (LON_MIN + LON_MAX) / 2],
                      zoom_start=13, tiles="OpenStreetMap")
    for i in range(GRID):
        for j in range(GRID):
            value = float(risk_grid[i, j])
            if not np.isfinite(value):
                continue
            bounds = [
                [LAT_MIN + i * lat_step, LON_MIN + j * lon_step],
                [LAT_MIN + (i + 1) * lat_step, LON_MIN + (j + 1) * lon_step],
            ]
            folium.Rectangle(
                bounds=bounds,
                color=None,
                fill=True,
                fill_color="#d73027",
                fill_opacity=min(0.9, max(0.0, value)),
                tooltip=f"Индекс ячейки: {i}, {j}; риск: {value:.2f}",
            ).add_to(fmap)
    return fmap


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Страница выбора периода карты риска."""
    grids = get_risk_grids()
    options = "".join(
        f'<option value="{key}">{key}</option>' for key in sorted(grids)
    )
    return (
        "<html><head><meta charset='utf-8'><title>Риск ДТП</title></head>"
        "<body><h1>Карта риска дорожно-транспортных происшествий</h1>"
        "<form action='/map' method='get'>"
        f"<label>Период: <select name='period'>{options}</select></label> "
        "<button type='submit'>Показать</button></form>"
        "<p><a href='/docs'>Программный интерфейс</a></p></body></html>"
    )


@app.get("/map", response_class=HTMLResponse)
def risk_map(period: str = "all") -> str:
    """Отдаёт HTML-страницу с интерактивной картой риска за выбранный период."""
    grids = get_risk_grids()
    if period not in grids:
        period = "all"
    fmap = render_risk_map(np.asarray(grids[period], dtype=float))
    options = "".join(
        f'<option value="{key}"{" selected" if key == period else ""}>{key}</option>'
        for key in sorted(grids)
    )
    return (
        "<html><head><meta charset='utf-8'><title>Карта риска</title></head><body>"
        "<form action='/map' method='get'>"
        f"<label>Период: <select name='period'>{options}</select></label> "
        "<button type='submit'>Показать</button></form>"
        f"{fmap._repr_html_()}</body></html>"
    )

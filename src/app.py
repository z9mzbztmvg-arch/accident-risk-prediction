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

# Границы Москвы, для которых построена регулярная сетка
LAT_MIN, LAT_MAX = 55.57, 55.92
LON_MIN, LON_MAX = 37.35, 37.90
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
    """Строит интерактивную карту риска: ячейки сетки задаются полигонами
    Shapely и объединяются в GeoDataFrame GeoPandas."""
    import geopandas as gpd
    from shapely.geometry import box

    risk_grid = np.asarray(risk_grid, dtype=float)
    lat_step = (LAT_MAX - LAT_MIN) / GRID
    lon_step = (LON_MAX - LON_MIN) / GRID
    records, geometries = [], []
    for i in range(GRID):
        for j in range(GRID):
            value = float(risk_grid[i, j])
            if not np.isfinite(value):
                continue
            south = LAT_MIN + i * lat_step
            west = LON_MIN + j * lon_step
            geometries.append(box(west, south, west + lon_step, south + lat_step))
            records.append({"cell_i": i, "cell_j": j, "risk": round(value, 3)})
    cells = gpd.GeoDataFrame(records, geometry=geometries, crs="EPSG:4326")
    fmap = folium.Map(location=[(LAT_MIN + LAT_MAX) / 2, (LON_MIN + LON_MAX) / 2],
                      zoom_start=11, tiles="OpenStreetMap")
    folium.GeoJson(
        cells.__geo_interface__,
        style_function=lambda feature: {
            "fillColor": "#d73027",
            "fillOpacity": min(0.9, max(0.0, feature["properties"]["risk"])),
            "color": None,
        },
        tooltip=folium.GeoJsonTooltip(fields=["cell_i", "cell_j", "risk"],
                                      aliases=["Строка", "Столбец", "Риск"]),
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

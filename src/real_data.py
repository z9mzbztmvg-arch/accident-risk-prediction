"""Загрузка и подготовка реальных открытых данных о ДТП, погоде и УДС Москвы.

Источники:
  * ДТП -- выгрузка проекта "Карта ДТП" (dtp-stat.ru), первоисточник -- ГИБДД МВД РФ;
  * погода -- архив Open-Meteo (ERA5), суточные агрегаты;
  * улично-дорожная сеть -- извлечение OpenStreetMap для Москвы (BBBike).

Все функции кешируют результат в каталоге ``data/`` и при повторном запуске
не обращаются к сети. Сырые выгрузки складываются в ``data/raw`` и в
репозиторий не попадают.
"""

from __future__ import annotations

import gzip
import json
import math
import pathlib
import urllib.parse
import urllib.request
import xml.sax
import zipfile

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Параметры сетки и периода
# --------------------------------------------------------------------------- #
GRID = 30
LAT0, LAT1 = 55.57, 55.92
LON0, LON1 = 37.35, 37.90
START, END = "2022-01-01", "2024-12-31"
CENTER_LAT, CENTER_LON = 55.7522, 37.6156  # Кремль

ACCIDENTS_URL = "https://dtp-stat.ru/media/opendata/moskva.geojson.zip"
OSM_URL = "https://download.bbbike.org/osm/bbbike/Moscow/Moscow.osm.gz"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"

DRIVABLE = {
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "unclassified", "residential", "living_street", "service",
    "motorway_link", "trunk_link", "primary_link", "secondary_link",
    "tertiary_link",
}
UNPAVED = {"unpaved", "ground", "gravel", "dirt", "earth", "grass", "sand", "cobblestone"}
PAVED = {"asphalt", "concrete", "paved", "concrete:plates", "concrete:lanes"}


def _ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def _download(url: str, dest: pathlib.Path) -> pathlib.Path:
    """Скачивает файл, если его ещё нет в кеше."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    _ensure_dirs()
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "nir-research/1.0"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=300) as src, tmp.open("wb") as dst:
        while True:
            chunk = src.read(1 << 20)
            if not chunk:
                break
            dst.write(chunk)
    tmp.replace(dest)
    return dest


def _cell_index(lat: float, lon: float) -> tuple[int, int]:
    i = min(GRID - 1, max(0, int((lat - LAT0) / (LAT1 - LAT0) * GRID)))
    j = min(GRID - 1, max(0, int((lon - LON0) / (LON1 - LON0) * GRID)))
    return i, j


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# --------------------------------------------------------------------------- #
# 1. ДТП
# --------------------------------------------------------------------------- #
def load_accidents(force: bool = False) -> pd.DataFrame:
    """Возвращает таблицу ДТП за нужный период в границах сетки."""
    cached = DATA_DIR / "accidents.csv.gz"
    if cached.exists() and not force:
        return pd.read_csv(cached, parse_dates=["date"])

    archive = _download(ACCIDENTS_URL, RAW_DIR / "moskva.geojson.zip")
    with zipfile.ZipFile(archive) as zf:
        member = next(n for n in zf.namelist() if n.endswith(".geojson"))
        with zf.open(member) as fh:
            data = json.load(fh)

    rows = []
    for feature in data.get("features", []):
        geometry = feature.get("geometry") or {}
        coords = geometry.get("coordinates")
        if not coords or len(coords) < 2:
            continue
        lon, lat = coords[0], coords[1]
        if not (LAT0 <= lat <= LAT1 and LON0 <= lon <= LON1):
            continue
        props = feature.get("properties") or {}
        stamp = (props.get("datetime") or "")[:19]
        if not (START <= stamp[:10] <= END):
            continue
        rows.append({
            "datetime": stamp,
            "lat": lat,
            "lon": lon,
            "severity": props.get("severity"),
            "category": props.get("category"),
            "dead_count": props.get("dead_count") or 0,
            "injured_count": props.get("injured_count") or 0,
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["datetime"]).dt.normalize()
    df.to_csv(cached, index=False)
    return df


def accident_grid(force: bool = False) -> pd.DataFrame:
    """Сетка "ячейка x день" с числом ДТП и бинарной целью."""
    cached = DATA_DIR / "accident_grid.csv.gz"
    if cached.exists() and not force:
        return pd.read_csv(cached, parse_dates=["date"])

    accidents = load_accidents(force=force)
    counts = accidents.groupby(["date", "lat", "lon"]).size().reset_index(name="n")
    counts[["cell_i", "cell_j"]] = counts.apply(
        lambda r: pd.Series(_cell_index(r["lat"], r["lon"])), axis=1)
    counts = counts.groupby(["date", "cell_i", "cell_j"], as_index=False)["n"].sum()
    counts = counts.rename(columns={"n": "accident_count"})

    dates = pd.date_range(START, END, freq="D")
    cells = pd.MultiIndex.from_product(
        [range(GRID), range(GRID)], names=["cell_i", "cell_j"]).to_frame(index=False)
    full = cells.merge(pd.DataFrame({"date": dates}), how="cross")
    full = full.merge(counts, on=["date", "cell_i", "cell_j"], how="left")
    full["accident_count"] = full["accident_count"].fillna(0).astype(int)
    full["target"] = (full["accident_count"] > 0).astype(int)
    full.to_csv(cached, index=False)
    return full


# --------------------------------------------------------------------------- #
# 2. Погода
# --------------------------------------------------------------------------- #
def load_weather(force: bool = False) -> pd.DataFrame:
    cached = DATA_DIR / "weather.csv"
    if cached.exists() and not force:
        return pd.read_csv(cached, parse_dates=["date"])

    params = {
        "latitude": CENTER_LAT,
        "longitude": CENTER_LON,
        "start_date": START,
        "end_date": END,
        "daily": "temperature_2m_mean,precipitation_sum,wind_speed_10m_max,"
                 "cloud_cover_mean,snowfall_sum",
        "timezone": "Europe/Moscow",
    }
    query = urllib.parse.urlencode(params)
    req = urllib.request.Request(f"{WEATHER_URL}?{query}",
                                 headers={"User-Agent": "nir-research/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.load(resp)
    daily = payload["daily"]
    df = pd.DataFrame({
        "date": pd.to_datetime(daily["time"]),
        "temperature": daily["temperature_2m_mean"],
        "precipitation": daily["precipitation_sum"],
        "wind_speed": daily["wind_speed_10m_max"],
        "cloud_cover": daily["cloud_cover_mean"],
        "snowfall": daily["snowfall_sum"],
    })
    df.to_csv(cached, index=False)
    return df


# --------------------------------------------------------------------------- #
# 3. Улично-дорожная сеть OpenStreetMap
# --------------------------------------------------------------------------- #
class _WayCollector(xml.sax.ContentHandler):
    """Первый проход: собирает автомобильные дороги и точечные POI."""

    def __init__(self) -> None:
        super().__init__()
        self.ways: list[dict] = []
        self.current: dict | None = None
        self.refs: set[int] = set()
        self.poi_refs: set[int] = set()

    def startElement(self, name, attrs) -> None:  # noqa: N802
        if name == "way":
            self.current = {"id": attrs.get("id"), "tags": {}, "refs": []}
        elif name == "nd" and self.current is not None:
            ref = attrs.get("ref")
            if ref:
                self.current["refs"].append(int(ref))
        elif name == "tag" and self.current is not None:
            self.current["tags"][attrs.get("k")] = attrs.get("v")

    def endElement(self, name) -> None:  # noqa: N802
        if name != "way" or self.current is None:
            return
        tags = self.current["tags"]
        highway = tags.get("highway")
        is_road = highway in DRIVABLE
        is_poi = tags.get("amenity") in {"school", "kindergarten"} or \
            tags.get("shop") is not None or tags.get("tourism") in {"attraction", "museum"}
        if is_road or is_poi:
            self.current["kind"] = "road" if is_road else "poi"
            self.ways.append(self.current)
            self.refs.update(self.current["refs"])
            if is_poi:
                self.poi_refs.update(self.current["refs"])
        self.current = None


class _NodeCollector(xml.sax.ContentHandler):
    """Второй проход: координаты нужных узлов и точечные POI."""

    POI_AMENITY = {
        "cafe", "restaurant", "fast_food", "bank", "pharmacy", "hospital",
        "fuel", "marketplace", "cinema", "theatre", "museum", "library",
        "university", "college", "kindergarten",
    }

    def __init__(self, wanted: set[int]) -> None:
        super().__init__()
        self.wanted = wanted
        self.coords: dict[int, tuple[float, float]] = {}
        self.pois: list[tuple[float, float, str]] = []
        self.current: dict | None = None

    def startElement(self, name, attrs) -> None:  # noqa: N802
        if name == "node":
            node_id = attrs.get("id")
            lat, lon = attrs.get("lat"), attrs.get("lon")
            if node_id and lat and lon and int(node_id) in self.wanted:
                self.coords[int(node_id)] = (float(lat), float(lon))
            self.current = {"id": node_id, "lat": lat, "lon": lon, "tags": {}}
        elif name == "tag" and self.current is not None:
            self.current["tags"][attrs.get("k")] = attrs.get("v")

    def endElement(self, name) -> None:  # noqa: N802
        if name != "node" or not self.current:
            return
        node, tags = self.current, self.current["tags"]
        if node["lat"] and node["lon"]:
            amenity = tags.get("amenity")
            if amenity in {"school", "kindergarten"}:
                kind = "school"
            elif tags.get("shop") or amenity in self.POI_AMENITY or \
                    tags.get("tourism") in {"attraction", "museum"}:
                kind = "poi"
            else:
                kind = None
            if kind:
                self.pois.append((float(node["lat"]), float(node["lon"]), kind))
        self.current = None

    def startElementNS(self, name, qname, attrs):  # pragma: no cover
        pass


def _parse_speed(value: str | None) -> float | None:
    if not value:
        return None
    text = value.strip().lower()
    if text.startswith("ru:"):
        return {"ru:urban": 60, "ru:rural": 90, "ru:motorway": 110}.get(text)
    if text in {"walk", "foot"}:
        return 20
    if text in {"none", "signals"}:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    return float(digits) if digits else None


def load_infra(force: bool = False) -> pd.DataFrame:
    """Признаки УДС по ячейкам сетки, рассчитанные из OSM."""
    cached = DATA_DIR / "infra.csv"
    if cached.exists() and not force:
        return pd.read_csv(cached)

    archive = _download(OSM_URL, RAW_DIR / "Moscow.osm.gz")

    ways = _WayCollector()
    with gzip.open(archive, "rb") as fh:
        xml.sax.parse(fh, ways)
    nodes = _NodeCollector(ways.refs)
    with gzip.open(archive, "rb") as fh:
        xml.sax.parse(fh, nodes)

    coords = nodes.coords
    road_len = np.zeros((GRID, GRID))
    lane_weight = np.zeros((GRID, GRID))
    lane_len = np.zeros((GRID, GRID))
    speed_weight = np.zeros((GRID, GRID))
    speed_len = np.zeros((GRID, GRID))
    surface = [[{} for _ in range(GRID)] for _ in range(GRID)]
    node_ways: dict[int, set[str]] = {}
    poi_cnt = np.zeros((GRID, GRID))
    school_cnt = np.zeros((GRID, GRID))

    for way in ways.ways:
        refs = [r for r in way["refs"] if r in coords]
        if len(refs) < 2:
            continue
        tags = way["tags"]
        if way["kind"] == "road":
            lanes = tags.get("lanes")
            lanes = float(lanes) if lanes and lanes.replace(".", "", 1).isdigit() else np.nan
            speed = _parse_speed(tags.get("maxspeed"))
            surf = tags.get("surface", "")
            for a, b in zip(refs, refs[1:]):
                (lat1, lon1), (lat2, lon2) = coords[a], coords[b]
                length = _haversine(lat1, lon1, lat2, lon2)
                i, j = _cell_index((lat1 + lat2) / 2, (lon1 + lon2) / 2)
                road_len[i, j] += length
                if not np.isnan(lanes):
                    lane_weight[i, j] += lanes * length
                    lane_len[i, j] += length
                if speed:
                    speed_weight[i, j] += speed * length
                    speed_len[i, j] += length
                if surf:
                    surface[i][j][surf] = surface[i][j].get(surf, 0) + length
                node_ways.setdefault(a, set()).add(way["id"])
                node_ways.setdefault(b, set()).add(way["id"])
        else:
            lat = np.mean([coords[r][0] for r in refs])
            lon = np.mean([coords[r][1] for r in refs])
            i, j = _cell_index(lat, lon)
            if tags.get("amenity") in {"school", "kindergarten"}:
                school_cnt[i, j] += 1
            else:
                poi_cnt[i, j] += 1

    for lat, lon, kind in nodes.pois:
        i, j = _cell_index(lat, lon)
        if kind == "school":
            school_cnt[i, j] += 1
        else:
            poi_cnt[i, j] += 1

    intersections = np.zeros((GRID, GRID))
    for node_id, way_ids in node_ways.items():
        if len(way_ids) >= 3:
            i, j = _cell_index(*coords[node_id])
            intersections[i, j] += 1

    cell_lat = (LAT1 - LAT0) / GRID * 111.32
    cell_lon = (LON1 - LON0) / GRID * 111.32 * math.cos(math.radians(55.75))
    area = cell_lat * cell_lon

    rows = []
    for i in range(GRID):
        for j in range(GRID):
            lat = LAT0 + (i + 0.5) * (LAT1 - LAT0) / GRID
            lon = LON0 + (j + 0.5) * (LON1 - LON0) / GRID
            lanes = lane_weight[i, j] / lane_len[i, j] if lane_len[i, j] else 2.0
            speed = speed_weight[i, j] / speed_len[i, j] if speed_len[i, j] else 60.0
            counts = surface[i][j]
            if counts:
                top = max(counts, key=counts.get)
                code = 0 if top in PAVED else (2 if top in UNPAVED else 1)
            else:
                code = 1
            rows.append({
                "cell_i": i,
                "cell_j": j,
                "road_density": road_len[i, j] / area,
                "intersection_count": intersections[i, j],
                "avg_lanes": lanes,
                "speed_limit": speed,
                "dist_to_center": _haversine(lat, lon, CENTER_LAT, CENTER_LON),
                "poi_density": poi_cnt[i, j] / area,
                "is_school_zone": int(school_cnt[i, j] > 0),
                "pavement_code": code,
                "poi_count": poi_cnt[i, j],
            })

    df = pd.DataFrame(rows).sort_values(["cell_i", "cell_j"]).reset_index(drop=True)
    df.to_csv(cached, index=False)
    return df


# --------------------------------------------------------------------------- #
# 4. Сборка единого набора
# --------------------------------------------------------------------------- #
FEATURES = [
    "road_density", "intersection_count", "avg_lanes", "speed_limit",
    "dist_to_center", "poi_density", "is_school_zone", "pavement_code",
    "temperature", "precipitation", "wind_speed", "cloud_cover", "snowfall",
    "is_snow",
    "month", "day_of_week", "is_weekend", "is_holiday", "is_winter",
    "doy_sin", "doy_cos",
    "lag_acc_1", "roll_acc_7", "neigh_acc_1",
]

HOLIDAYS_MD = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8),
               (2, 23), (3, 8), (5, 1), (5, 9), (6, 12), (11, 4)]


def _temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    d = df["date"].dt
    df["month"] = d.month
    df["day_of_week"] = d.dayofweek
    df["is_weekend"] = (d.dayofweek >= 5).astype(int)
    df["is_winter"] = df["month"].isin([12, 1, 2]).astype(int)
    df["doy_sin"] = np.sin(2 * np.pi * d.dayofyear / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * d.dayofyear / 365.25)
    md = list(zip(d.month, d.day))
    df["is_holiday"] = np.array([int(item in HOLIDAYS_MD) for item in md])
    # видимость в архиве Open-Meteo отсутствует, поэтому туман не выделяется:
    # состояние "снег" определяется по суточному слою осадков и температуре
    df["is_snow"] = ((df["snowfall"] > 0) |
                     ((df["temperature"] < 0) & (df["precipitation"] > 0))).astype(int)
    return df


def _lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["cell_i", "cell_j", "date"]).copy()
    grouped = df.groupby(["cell_i", "cell_j"])["accident_count"]
    df["lag_acc_1"] = grouped.shift(1).fillna(0.0)
    df["roll_acc_7"] = grouped.transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).mean()).fillna(0.0)

    prev = df.pivot_table(index="date", columns=["cell_i", "cell_j"],
                          values="lag_acc_1", aggfunc="mean")
    neighbour = []
    for i in range(GRID):
        for j in range(GRID):
            values = []
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < GRID and 0 <= nj < GRID:
                        values.append(prev[(ni, nj)])
            neighbour.append(pd.concat(values, axis=1).mean(axis=1))
    matrix = pd.concat(neighbour, axis=1)
    matrix.columns = pd.MultiIndex.from_tuples(
        [(i, j) for i in range(GRID) for j in range(GRID)],
        names=["cell_i", "cell_j"])
    long = matrix.stack(["cell_i", "cell_j"], future_stack=True).rename("neigh_acc_1")
    df = df.merge(long.reset_index(), on=["date", "cell_i", "cell_j"], how="left")
    df["neigh_acc_1"] = df["neigh_acc_1"].fillna(0.0)
    return df


def build_dataset() -> pd.DataFrame:
    """Реальный набор "ячейка x день" с признаками УДС, погоды и календаря."""
    df = accident_grid()
    df = df.merge(load_weather(), on="date", how="left")
    df = df.merge(load_infra(), on=["cell_i", "cell_j"], how="left")
    df = _temporal_features(df)
    df = _lag_features(df)
    return df.sort_values(["date", "cell_i", "cell_j"]).reset_index(drop=True)

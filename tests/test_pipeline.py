"""Модульные тесты конвейера на синтетическом стенде.

Синтетические данные используются только для проверки корректности
преобразований: истинная структура набора известна, поэтому ошибки
очистки и построения признаков проявляются сразу.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import nir_pipeline as pipeline  # noqa: E402


def synthetic_frame() -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", "2024-12-31", freq="D")
    cells = pipeline.generate_city_grid(grid=4)
    frame = cells.merge(pd.DataFrame({"date": dates}), how="cross")
    weather = pipeline.generate_weather(dates)
    frame = frame.merge(weather, on="date", how="left")
    rng = np.random.default_rng(0)
    frame["accident_count"] = rng.poisson(0.05, len(frame))
    frame["target"] = (frame["accident_count"] > 0).astype(int)
    return frame


class CleaningTests(unittest.TestCase):
    def test_invalid_values_become_median(self):
        frame = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01", "2023-02-01", "2023-02-02"]),
            "temperature": [np.nan, -100.0, -5.0],
            "precipitation": [-1.0, 0.2, 0.0],
            "visibility": [0.0, 5.0, 3.0],
            "wind_speed": [-1.0, 2.0, 3.0],
            "target": [0, 1, 0],
        })
        cleaned = pipeline.clean_dataset(frame)
        self.assertEqual(int(cleaned.isna().sum().sum()), 0)
        self.assertTrue((cleaned["precipitation"] >= 0).all())

    def test_missing_target_is_dropped(self):
        frame = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "temperature": [1.0, 2.0],
            "precipitation": [0.0, 0.0],
            "visibility": [5.0, 5.0],
            "wind_speed": [1.0, 1.0],
            "target": [1, np.nan],
        })
        self.assertEqual(len(pipeline.clean_dataset(frame)), 1)


class FeatureTests(unittest.TestCase):
    def test_lag_features_use_only_past(self):
        frame = synthetic_frame()
        built = pipeline.build_features(frame)
        self.assertFalse(built[pipeline.FEATURES].isna().any().any())

        cell = built[(built.cell_i == 0) & (built.cell_j == 0)].sort_values("date")
        first = cell.iloc[0]
        self.assertEqual(first["lag_acc_1"], 0.0)
        second = cell.iloc[1]
        self.assertEqual(second["lag_acc_1"], first["accident_count"])

    def test_temporal_split_is_chronological(self):
        frame = pipeline.build_features(synthetic_frame())
        tr, va, te = pipeline.load_splits(frame)
        self.assertLess(tr["date"].max(), va["date"].min())
        self.assertLess(va["date"].max(), te["date"].min())
        self.assertEqual(len(tr) + len(va) + len(te), len(frame))


class RealDataContractTests(unittest.TestCase):
    def test_cell_index_stays_inside_grid(self):
        from real_data import GRID, _cell_index
        for lat, lon in [(55.57, 37.35), (55.92, 37.90), (60.0, 30.0)]:
            i, j = _cell_index(lat, lon)
            self.assertTrue(0 <= i < GRID and 0 <= j < GRID)


if __name__ == "__main__":
    unittest.main()

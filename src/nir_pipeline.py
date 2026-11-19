"""НИР: прогнозирование рисков ДТП в мегаполисе.
Синтетический набор данных + конвейер машинного обучения.

Запуск:  python nir_pipeline.py
Результаты: fig/*.pdf, fig/*.png, results.json
"""
from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
GRID = 30
START = "2023-01-01"
END = "2025-12-31"
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fig")
os.makedirs(OUT_DIR, exist_ok=True)


# --------------------------------------------------------------------------- #
# 1. Настройка единого стиля графиков
# --------------------------------------------------------------------------- #
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter
import matplotlib.font_manager as font_manager
import shutil
import subprocess


def _register_document_font() -> str:
    """Регистрирует в matplotlib тот же шрифт, что используется в тексте
    документа (Tempora из пакета LaTeX). Шрифт ищется средствами TeX Live
    (kpsewhich). Если он недоступен, берётся Times New Roman, затем DejaVu."""
    names = ("Tempora-Regular.otf", "Tempora-Bold.otf",
             "Tempora-Italic.otf", "Tempora-BoldItalic.otf")
    kpsewhich = shutil.which("kpsewhich")
    added = False
    for fn in names:
        path = None
        if kpsewhich:
            try:
                out = subprocess.run([kpsewhich, fn], capture_output=True,
                                     text=True, timeout=20)
                cand = out.stdout.strip()
                if cand and os.path.exists(cand):
                    path = cand
            except Exception:
                path = None
        if path:
            font_manager.fontManager.addfont(path)
            added = True
    if added:
        return "Tempora"
    return "Times New Roman"


# Гарнитура графиков совпадает с основным текстом документа
DOC_FONT = _register_document_font()

# Ширина текстового поля документа: 17 см = 483,7 pt = 6,69 дюйма.
# Рисунки строятся ровно под эту ширину, чтобы при вставке через
# \includegraphics[width=\linewidth] кегль шрифта не масштабировался.
LINE_WIDTH_IN = 6.69

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": [DOC_FONT, "Times New Roman", "DejaVu Serif"],
    # Математика в подписях набирается тем же шрифтом, что и текст
    "mathtext.fontset": "custom",
    "mathtext.rm": DOC_FONT,
    "mathtext.it": "{0}:italic".format(DOC_FONT),
    "mathtext.bf": "{0}:bold".format(DOC_FONT),
    # Кегль совпадает с основным текстом документа (14 pt)
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "text.color": "black",
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "black",
    "figure.dpi": 200,
    # Шрифты в PDF/PS как TrueType, чтобы совпадали с текстом документа
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# Отображаемые имена моделей, согласованные с таблицами в тексте работы
DISPLAY_NAMES = {
    "Градиентный бустинг LightGBM": "LightGBM",
    "Градиентный бустинг XGBoost": "XGBoost",
    "Частотная статистическая модель": "Статистическая модель",
}


def _disp_name(name: str) -> str:
    return DISPLAY_NAMES.get(name, name)


def _thousands(value, _pos=None) -> str:
    """Разряды через пробел без экспоненциальной записи."""
    return f"{value:,.0f}".replace(",", " ")


def _black_axes(ax) -> None:
    """Все координатные оси (рамка) — чёрные."""
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)


def _legend_below(fig, ax, ncol: int = 1, gap: float = 0.10):
    """Легенда под полем графика (в координатах рисунка).

    Размещается после tight_layout, поэтому не перекрывает ни кривые,
    ни подписи осей. Вызывается до _save (bbox_inches='tight').
    """
    handles, labels = ax.get_legend_handles_labels()
    y0 = ax.get_position().y0
    return fig.legend(handles, labels, loc="upper center",
                      bbox_to_anchor=(0.5, y0 - gap), ncol=ncol,
                      bbox_transform=fig.transFigure)


def _save(fig, stem: str) -> None:
    """Сохранение рисунка в PDF и PNG без обрезки подписей."""
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"{stem}.{ext}"),
                    bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 2. Генерация синтетического набора данных
# --------------------------------------------------------------------------- #
def generate_city_grid(grid: int = GRID, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = np.arange(grid)
    ii, jj = np.meshgrid(idx, idx, indexing="ij")
    ii = ii.ravel()
    jj = jj.ravel()
    c = (grid - 1) / 2.0
    dist = np.sqrt((ii - c) ** 2 + (jj - c) ** 2) / (np.sqrt(2.0) * c)

    centrality = np.exp(-2.2 * dist) + rng.normal(0, 0.03, dist.size)
    road_density = np.clip(0.92 * np.exp(-1.8 * dist) + rng.normal(0, 0.05, dist.size), 0.02, 1.0)
    intersection = np.clip(np.round(rng.poisson(3.0 * road_density + 0.2)), 0, 12).astype(int)
    lanes = np.clip(np.round(1 + 3.0 * centrality + rng.normal(0, 0.4, dist.size)), 1, 5).astype(int)
    speed = np.clip((40 + 40 * centrality) / 10, 4, 8).round() * 10
    poi = np.clip(np.exp(-2.0 * dist) + rng.normal(0, 0.05, dist.size), 0.0, 1.0)
    school = (rng.random(dist.size) < (0.06 + 0.20 * np.exp(-1.5 * dist))).astype(int)
    pavement = rng.choice([0, 1, 2], size=dist.size, p=[0.72, 0.22, 0.06])

    df = pd.DataFrame({
        "cell_i": ii, "cell_j": jj,
        "road_density": road_density.round(4),
        "intersection_count": intersection,
        "avg_lanes": lanes,
        "speed_limit": speed.astype(int),
        "dist_to_center": dist.round(4),
        "poi_density": poi.round(4),
        "is_school_zone": school,
        "centrality": centrality.round(4),
        "pavement_code": pavement,
    })
    base = (1.55 * df.road_density + 0.42 * df.intersection_count / 3.0
            + 1.00 * df.poi_density + 0.55 * df.speed_limit / 80.0
            + 0.55 * df.is_school_zone + 0.60 * df.centrality
            - 1.55 * df.dist_to_center - 2.40 * (df.centrality - 0.55) ** 2)
    df["risk_base"] = base
    return df


def generate_weather(dates: pd.DatetimeIndex, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    n = len(dates)
    doy = dates.dayofyear.to_numpy()
    seasonal = 8.0 * np.sin(2 * np.pi * (doy - 110) / 365.25)
    # AR(1)-процесс задаёт многодневные погодные периоды
    ar = np.zeros(n)
    for t in range(1, n):
        ar[t] = 0.72 * ar[t - 1] + rng.normal(0, 0.9)
    temperature = seasonal + 4.0 * ar + rng.normal(0, 2.0, n)
    prec_state = np.zeros(n)
    for t in range(1, n):
        prec_state[t] = 0.55 * prec_state[t - 1] + rng.normal(0, 1.0)
    precip = np.clip(rng.gamma(1.4, 1.1, n) + np.maximum(prec_state, 0) - 0.7, 0.0, None)
    visibility = np.clip(rng.normal(7.0, 3.0, n) - 0.6 * precip, 0.2, 10.0)
    wind = np.clip(rng.normal(3.5, 1.6, n), 0.0, None)
    is_snow = ((temperature < 1.0) & (precip > 0.6)).astype(int)
    is_fog = ((visibility < 2.0) & (precip < 0.5)).astype(int)
    holiday = dates.isin(pd.to_datetime([
        "2023-01-01", "2023-01-07", "2023-03-08", "2023-05-01", "2023-05-09",
        "2023-06-12", "2023-11-04", "2024-01-01", "2024-01-07", "2024-03-08",
        "2024-05-01", "2024-05-09", "2024-06-12", "2024-11-04", "2025-01-01",
        "2025-01-07", "2025-03-08", "2025-05-01", "2025-05-09", "2025-06-12",
        "2025-11-04",
    ])).astype(int)
    vac = dates.isin(pd.to_datetime(
        [d.strftime("%Y-%m-%d") for d in pd.date_range("2023-07-01", "2023-08-31")]
        + [d.strftime("%Y-%m-%d") for d in pd.date_range("2024-07-01", "2024-08-31")]
        + [d.strftime("%Y-%m-%d") for d in pd.date_range("2025-07-01", "2025-08-31")]
    )).astype(int)
    return pd.DataFrame({
        "date": dates,
        "temperature": temperature.round(2),
        "precipitation": precip.round(2),
        "visibility": visibility.round(2),
        "wind_speed": wind.round(2),
        "is_snow": is_snow,
        "is_fog": is_fog,
        "is_holiday": holiday,
        "is_vacation": vac,
    })


def generate_dataset() -> pd.DataFrame:
    cells = generate_city_grid()
    dates = pd.date_range(START, END, freq="D")
    weather = generate_weather(dates)

    df = cells.merge(pd.DataFrame({"date": dates}), how="cross")
    df = df.merge(weather, on="date", how="left")

    month = df["date"].dt.month.to_numpy()
    dow = df["date"].dt.dayofweek.to_numpy()
    month_eff = np.where(np.isin(month, [12, 1, 2]), 0.50,
                 np.where(np.isin(month, [3, 4, 11]), 0.12,
                 np.where(np.isin(month, [6, 7, 8]), -0.22, 0.0)))
    dow_eff = np.choose(dow, [-0.05, 0.0, 0.0, 0.05, 0.38, 0.50, 0.22])
    workday = (dow < 5).astype(float)

    logit = (df["risk_base"]
             + month_eff + dow_eff
             + 0.55 * df["is_holiday"] + 0.20 * df["is_vacation"]
             + 0.95 * df["is_snow"] + 0.80 * df["is_fog"]
             + 0.35 * np.clip(df["precipitation"], 0, 5) / 3.0
             - 0.45 * (df["temperature"] < -5.0)
             # нелинейности и взаимодействия признаков
             + 1.30 * (df["speed_limit"] / 80.0) * (df["intersection_count"] / 3.0)
             + 1.10 * ((df["road_density"] > 0.55).astype(int) ^ df["is_snow"])
             + 1.00 * df["poi_density"] * df["road_density"]
             + 0.90 * df["is_school_zone"] * (df["speed_limit"] >= 60)
             + 0.90 * (df["road_density"] * df["intersection_count"] / 3.0)
             + 0.70 * df["is_school_zone"] * workday
             + 0.65 * (df["speed_limit"] >= 70)
             + 1.05 * df["centrality"] * df["is_snow"]
             + 0.85 * df["centrality"] * df["is_fog"]
             + 0.90 * ((df["precipitation"] > 2.0) & (df["temperature"] < 0.0))
             + 0.80 * (df["visibility"] < 1.5))
    lam = np.exp(logit - 7.90)
    df["accident_count"] = np.random.default_rng(RANDOM_STATE + 7).poisson(lam)
    df["target"] = (df["accident_count"] > 0).astype(int)
    return df


# --------------------------------------------------------------------------- #
# 3. Конструирование признаков
# --------------------------------------------------------------------------- #
def temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    d = df["date"].dt
    df["month"] = d.month
    df["day_of_week"] = d.dayofweek
    df["is_weekend"] = (d.dayofweek >= 5).astype(int)
    df["doy_sin"] = np.sin(2 * np.pi * d.dayofyear / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * d.dayofyear / 365.25)
    df["is_winter"] = df["month"].isin([12, 1, 2]).astype(int)
    df["traffic_density"] = (0.6 * df["centrality"] + 0.3 * df["is_weekend"]
                             + 0.2 * (1 - df["is_winter"])).round(4)
    return df


def lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["cell_i", "cell_j", "date"]).copy()
    keys = ["cell_i", "cell_j"]
    grp = df.groupby(keys)["accident_count"]
    df["lag_acc_1"] = grp.shift(1)
    df["roll_acc_7"] = grp.transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    df["lag_acc_1"] = df["lag_acc_1"].fillna(0.0)
    df["roll_acc_7"] = df["roll_acc_7"].fillna(0.0)
    # пространственный лаг: среднее по 8 соседям за предыдущий день
    grid = GRID
    prev = df.pivot_table(index="date", columns=["cell_i", "cell_j"],
                          values="lag_acc_1", aggfunc="mean")
    neigh = []
    for i in range(grid):
        for j in range(grid):
            try:
                vals = []
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < grid and 0 <= nj < grid:
                            vals.append(prev[(ni, nj)])
                neigh.append(pd.concat(vals, axis=1).mean(axis=1))
            except KeyError:
                neigh.append(pd.Series(0.0, index=prev.index))
    n_mat = pd.concat(neigh, axis=1)
    n_mat.columns = pd.MultiIndex.from_tuples(
        [(i, j) for i in range(grid) for j in range(grid)],
        names=["cell_i", "cell_j"])
    n_long = n_mat.stack(["cell_i", "cell_j"], future_stack=True).rename("neigh_acc_1").reset_index()
    df = df.merge(n_long, on=["date", "cell_i", "cell_j"], how="left")
    df["neigh_acc_1"] = df["neigh_acc_1"].fillna(0.0)
    return df


FEATURES = [
    "road_density", "intersection_count", "avg_lanes", "speed_limit",
    "dist_to_center", "poi_density", "is_school_zone", "centrality", "pavement_code",
    "month", "day_of_week", "is_weekend", "is_holiday", "is_vacation",
    "doy_sin", "doy_cos", "is_winter", "traffic_density",
    "temperature", "precipitation", "visibility", "wind_speed", "is_snow", "is_fog",
    "lag_acc_1", "roll_acc_7", "neigh_acc_1",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = temporal_features(df)
    df = lag_features(df)
    return df


# --------------------------------------------------------------------------- #
# 4. Обучение моделей
# --------------------------------------------------------------------------- #
def load_splits(df: pd.DataFrame):
    tr = df[df["date"] < "2024-10-01"]
    va = df[(df["date"] >= "2024-10-01") & (df["date"] < "2025-01-01")]
    te = df[df["date"] >= "2025-01-01"]
    return tr, va, te


def train_models(tr, va, te, tune: bool = True):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    Xtr, ytr = tr[FEATURES].to_numpy(), tr["target"].to_numpy()
    Xva, yva = va[FEATURES].to_numpy(), va["target"].to_numpy()
    Xte, yte = te[FEATURES].to_numpy(), te["target"].to_numpy()
    spw = float((ytr == 0).sum()) / float((ytr == 1).sum())

    models = {}
    models["Логистическая регрессия"] = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000,
                           random_state=RANDOM_STATE))
    models["Случайный лес"] = RandomForestClassifier(
        n_estimators=200, max_depth=16, min_samples_leaf=3,
        class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE)

    best_params = {}
    if tune:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sub = tr.sample(n=min(200_000, len(tr)), random_state=RANDOM_STATE)
        Xs, ys = sub[FEATURES].to_numpy(), sub["target"].to_numpy()
        Xv, yv = Xva, yva

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 150, 500, step=50),
                "num_leaves": trial.suggest_int("num_leaves", 15, 120),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            }
            m = LGBMClassifier(scale_pos_weight=spw, random_state=RANDOM_STATE,
                               n_jobs=-1, verbose=-1, **params)
            m.fit(Xs, ys)
            from sklearn.metrics import roc_auc_score
            p = m.predict_proba(Xv)[:, 1]
            return roc_auc_score(yv, p)

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        study.optimize(objective, n_trials=25, show_progress_bar=False)
        best_params = study.best_params
        with open(os.path.join(OUT_DIR, "..", "optuna_history.json"), "w", encoding="utf-8") as f:
            json.dump([t.value for t in study.trials], f)
        models["Градиентный бустинг LightGBM"] = LGBMClassifier(
            scale_pos_weight=spw, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
            **best_params)
    else:
        models["Градиентный бустинг LightGBM"] = LGBMClassifier(
            n_estimators=300, num_leaves=48, learning_rate=0.06,
            scale_pos_weight=spw, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)

    models["Градиентный бустинг XGBoost"] = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.06,
        scale_pos_weight=spw, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, eval_metric="auc",
        tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE)

    for name, mdl in models.items():
        mdl.fit(Xtr, ytr)

    return models, (Xtr, ytr, Xva, yva, Xte, yte), best_params, spw


def statistical_baseline(tr, val, te):
    """Классический статистический подход: частота ДТП по ячейкам (эмпирическая
    оценка вероятности) с поправкой на месяц. Порог выбирается на валидации."""
    prior = float(tr["target"].mean())
    mc = tr.groupby(["cell_i", "cell_j", "month"])["target"].mean().rename("mc")
    cf = tr.groupby(["cell_i", "cell_j"])["target"].mean().rename("cf")

    def score_of(frame):
        f = frame[["cell_i", "cell_j", "month"]].merge(
            mc, on=["cell_i", "cell_j", "month"], how="left")
        f = f.merge(cf, on=["cell_i", "cell_j"], how="left")
        return f["mc"].fillna(f["cf"]).fillna(prior).to_numpy()

    s_val = score_of(val)
    thr = best_f1_threshold(val["target"].to_numpy(), s_val)
    s_te = score_of(te)
    return te["target"].to_numpy(), s_te, (s_te >= thr).astype(int)


def best_f1_threshold(y_true, y_score) -> float:
    from sklearn.metrics import f1_score
    grid = np.linspace(0.02, 0.90, 89)
    scores = [f1_score(y_true, (y_score >= t).astype(int), zero_division=0)
              for t in grid]
    return float(grid[int(np.argmax(scores))])


# --------------------------------------------------------------------------- #
# 5. Оценка качества
# --------------------------------------------------------------------------- #
def evaluate(y_true, y_score, y_pred) -> dict:
    from sklearn.metrics import (precision_score, recall_score, f1_score,
                                 roc_auc_score, average_precision_score,
                                 accuracy_score, confusion_matrix)
    return {
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_score)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "cm": confusion_matrix(y_true, y_pred).tolist(),
    }


# --------------------------------------------------------------------------- #
# 6. Графики
# --------------------------------------------------------------------------- #
def fig_temporal(df):
    fmt = FuncFormatter(_thousands)
    fig, ax = plt.subplots(1, 2, figsize=(LINE_WIDTH_IN, 3.3))
    m = df.groupby("month")["accident_count"].sum()
    ax[0].bar(m.index, m.values, color="#4C72B0", edgecolor="black", linewidth=0.6)
    ax[0].set_xlabel("Месяц")
    ax[0].set_ylabel("Число ДТП")
    ax[0].set_xticks(range(1, 13))
    ax[0].yaxis.set_major_formatter(fmt)
    d = df.groupby("day_of_week")["accident_count"].sum()
    ax[1].bar(range(1, 8), d.values, color="#DD8452", edgecolor="black", linewidth=0.6)
    ax[1].set_xlabel("День недели")
    ax[1].set_ylabel("Число ДТП")
    ax[1].set_xticks(range(1, 8))
    ax[1].set_xticklabels(["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"])
    ax[1].yaxis.set_major_formatter(fmt)
    for a in ax:
        _black_axes(a)
    fig.tight_layout()
    _save(fig, "temporal_profile")


def fig_roc(results, curves):
    fig, ax = plt.subplots(figsize=(LINE_WIDTH_IN, 4.6))
    for name in results:
        fpr, tpr, _ = curves[name]
        ax.plot(fpr, tpr, linewidth=1.6,
                label=f"{_disp_name(name)} (AUC = {results[name]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.0,
            label="Случайный прогноз")
    ax.set_xlabel("Доля ложноположительных срабатываний")
    ax.set_ylabel("Доля истинно положительных срабатываний")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    _black_axes(ax)
    fig.tight_layout()
    # Легенда вынесена под поле графика, чтобы не перекрывать кривые
    _legend_below(fig, ax, ncol=1)
    _save(fig, "roc_curves")


def fig_pr(results, prc):
    fig, ax = plt.subplots(figsize=(LINE_WIDTH_IN, 4.6))
    for name in results:
        prec, rec, _ = prc[name]
        ax.plot(rec, prec, linewidth=1.6,
                label=f"{_disp_name(name)} (PR-AUC = {results[name]['pr_auc']:.3f})")
    ax.set_xlabel("Полнота")
    ax.set_ylabel("Точность")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    _black_axes(ax)
    fig.tight_layout()
    # Легенда вынесена под поле графика, чтобы не перекрывать кривые
    _legend_below(fig, ax, ncol=1)
    _save(fig, "pr_curves")


def fig_confusion(results):
    import textwrap
    names = list(results)
    ncols = 3
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(LINE_WIDTH_IN, 2.95 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, name in zip(axes, names):
        cm = np.array(results[name]["cm"])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}".replace(",", " "),
                        ha="center", va="center", fontsize=15,
                        fontweight="bold",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_title(textwrap.fill(_disp_name(name), 14))
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["нет", "да"])
        ax.set_yticklabels(["нет", "да"])
        ax.set_xlabel("Прогноз")
        ax.set_ylabel("Факт")
        _black_axes(ax)
    for ax in axes[len(names):]:
        ax.axis("off")
    fig.tight_layout()
    _save(fig, "confusion_matrices")


def fig_metrics(results):
    metrics = ["precision", "recall", "f1", "roc_auc"]
    labels = ["Точность", "Полнота", "F1-мера", "ROC-AUC"]
    names = list(results)
    x = np.arange(len(metrics))
    w = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(LINE_WIDTH_IN, 4.3))
    for k, name in enumerate(names):
        vals = [results[name][m] for m in metrics]
        ax.bar(x + k * w - 0.4 + w / 2, vals, w, edgecolor="black",
               linewidth=0.6, label=_disp_name(name))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Значение метрики")
    ax.set_ylim(0, 1.15)
    _black_axes(ax)
    fig.tight_layout()
    # Легенда под полем графика, в две колонки
    _legend_below(fig, ax, ncol=2)
    _save(fig, "metrics_comparison")


def fig_importance(model, feat_names):
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return
    order = np.argsort(imp)[-15:]
    fig, ax = plt.subplots(figsize=(LINE_WIDTH_IN, 6.0))
    ax.barh(np.array(feat_names)[order], imp[order], color="#55A868",
            edgecolor="black", linewidth=0.6)
    ax.set_xlabel("Важность признака")
    _black_axes(ax)
    fig.tight_layout()
    _save(fig, "feature_importance")


def fig_risk_map(te, best_scores):
    grid = GRID
    risk = np.full((grid, grid), np.nan)
    tmp = te.assign(risk=best_scores).groupby(["cell_i", "cell_j"])["risk"].mean()
    for (i, j), v in tmp.items():
        risk[i, j] = v
    fig, ax = plt.subplots(figsize=(LINE_WIDTH_IN, 5.8))
    im = ax.imshow(risk.T, origin="lower", cmap="YlOrRd")
    # Уровень изолинии выбирается по данным, чтобы контур всегда был замкнутым
    if np.isfinite(risk).any() and np.nanmax(risk) > np.nanmin(risk):
        level = float(np.nanpercentile(risk, 90))
        ax.contour(risk.T, levels=[level], colors="black", linewidths=1.1)
    for k in range(grid + 1):
        ax.axhline(k - 0.5, color="black", linewidth=0.15, alpha=0.2)
        ax.axvline(k - 0.5, color="black", linewidth=0.15, alpha=0.2)
    ax.set_xlabel("Индекс ячейки по оси X")
    ax.set_ylabel("Индекс ячейки по оси Y")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Средний прогнозный риск")
    cbar.ax.tick_params(colors="black")
    cbar.outline.set_edgecolor("black")
    _black_axes(ax)
    fig.tight_layout()
    _save(fig, "risk_map")


def fig_optuna(path):
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        hist = [h for h in json.load(f) if h is not None]
    best = np.maximum.accumulate(hist)
    fig, ax = plt.subplots(figsize=(LINE_WIDTH_IN, 4.2))
    ax.plot(range(1, len(hist) + 1), hist, "o", color="#4C72B0",
            markersize=5, label="Значение на испытании")
    ax.plot(range(1, len(best) + 1), best, "-", color="#C44E52",
            linewidth=1.8, label="Лучшее значение")
    ax.set_xlabel("Номер испытания")
    ax.set_ylabel("ROC-AUC на валидации")
    _black_axes(ax)
    fig.tight_layout()
    # Легенда под полем графика, в две колонки
    _legend_below(fig, ax, ncol=2)
    _save(fig, "optuna_history")


# --------------------------------------------------------------------------- #
# 7. Полный конвейер
# --------------------------------------------------------------------------- #
def main():
    from sklearn.metrics import roc_curve, precision_recall_curve

    print("Генерация набора данных ...")
    df = generate_dataset()
    df = build_features(df)
    pos = float(df["target"].mean())
    print(f"Строк: {len(df):,}; доля положительных: {pos:.4f}; "
          f"всего ДТП: {int(df['accident_count'].sum()):,}")

    tr, va, te = load_splits(df)
    print(f"Обучающая: {len(tr):,}; валидационная: {len(va):,}; тестовая: {len(te):,}")

    fig_temporal(df)

    print("Обучение моделей и подбор гиперпараметров ...")
    models, splits, best_params, spw = train_models(tr, va, te, tune=True)
    Xtr, ytr, Xva, yva, Xte, yte = splits

    results, curves, prc = {}, {}, {}
    for name, mdl in models.items():
        score_va = mdl.predict_proba(Xva)[:, 1]
        thr = best_f1_threshold(yva, score_va)
        score = mdl.predict_proba(Xte)[:, 1]
        pred = (score >= thr).astype(int)
        results[name] = evaluate(yte, score, pred)
        results[name]["threshold"] = round(thr, 3)
        results[name]["_score"] = score
        curves[name] = roc_curve(yte, score)
        prc[name] = precision_recall_curve(yte, score)

    yb, sb, pb = statistical_baseline(tr, va, te)
    results["Частотная статистическая модель"] = evaluate(yb, sb, pb)
    results["Частотная статистическая модель"]["threshold"] = "-"
    # Кривые статистической модели также выводятся на рисунки ROC и PR,
    # поскольку они сравниваются с моделями машинного обучения в тексте.
    key_stat = "Частотная статистическая модель"
    curves[key_stat] = roc_curve(yb, sb)
    prc[key_stat] = precision_recall_curve(yb, sb)

    fig_roc({k: v for k, v in results.items() if k in curves}, curves)
    fig_pr({k: v for k, v in results.items() if k in prc}, prc)
    fig_confusion({k: v for k, v in results.items()})
    fig_metrics({k: v for k, v in results.items()})
    fig_importance(models["Градиентный бустинг LightGBM"], FEATURES)
    fig_risk_map(te, results["Градиентный бустинг LightGBM"]["_score"])
    fig_optuna(os.path.join(OUT_DIR, "..", "optuna_history.json"))

    summary = {
        "n_rows": int(len(df)),
        "positive_rate": round(pos, 4),
        "n_accidents": int(df["accident_count"].sum()),
        "n_cells": int(df.groupby(["cell_i", "cell_j"]).ngroups),
        "period": [START, END],
        "splits": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))},
        "scale_pos_weight": round(spw, 2),
        "best_params": best_params,
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "_score"}
                    for k, v in results.items()},
    }
    with open(os.path.join(OUT_DIR, "..", "results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Итоговые метрики ===")
    for name, m in results.items():
        print(f"{name:34s} P={m['precision']:.3f} R={m['recall']:.3f} "
              f"F1={m['f1']:.3f} ROC-AUC={m['roc_auc']:.3f} PR-AUC={m['pr_auc']:.3f}")
    print("\nЛучшие параметры LightGBM:", best_params)


if __name__ == "__main__":
    main()

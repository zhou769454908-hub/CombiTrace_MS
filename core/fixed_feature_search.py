"""Fixed-model random descriptor search for ESI response correction.

The user selects a candidate descriptor pool and one fixed model.  Only the
feature subset changes across trials.  Every trial is evaluated by grouped or
repeated out-of-fold prediction of log response factor, and all trial records
are retained.  Two independent leaderboards are produced:

* absolute/range: median and P80 fold error;
* trend: Spearman, pairwise concordance and top-20% recovery.

This is an exploratory model-selection layer.  Selecting the best trial from
many CV trials can still introduce selection optimism, so the output records
trial count, group-CV status and a clear guardrail statement.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from .concentration_levels import add_decile_columns

from .esi_descriptor_meta import descriptor_meta, display_name, mechanistic_profile
from .esi_model_benchmark import (
    _average_ranks,
    _key,
    _metrics,
    _num,
    _text,
    _trend_metrics,
    choose_features,
)

SKLEARN_AVAILABLE = False
SKLEARN_ERROR = ""
try:
    from sklearn.base import clone
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import BayesianRidge
    from sklearn.model_selection import LeaveOneGroupOut, RepeatedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    SKLEARN_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    SKLEARN_ERROR = str(exc)


FIXED_MODEL_OPTIONS: Tuple[Tuple[str, str], ...] = (
    ("Auto: previous best model", "auto_previous"),
    ("HistGradientBoosting", "HistGradientBoosting"),
    ("ExtraTrees", "ExtraTrees"),
    ("SVR-RBF", "SVR_RBF"),
    ("Bayesian Ridge", "BayesianRidge"),
    ("PLS", "PLS"),
    ("Random Forest", "RandomForest"),
    ("Physical Pairwise Ranker", "PhysicalPairwiseRanker"),
    ("Response-Factor ExtraTrees", "ResponseFactorExtraTrees"),
)
FIXED_MODEL_LABELS = [x[0] for x in FIXED_MODEL_OPTIONS]
FIXED_MODEL_MAP = {x[0]: x[1] for x in FIXED_MODEL_OPTIONS}

_MODEL_ALIASES = {
    "PairwiseRankSVM": "PhysicalPairwiseRanker",
    "PhysicalPairwiseRanker": "PhysicalPairwiseRanker",
    "ResponseFactorExtraTrees": "ResponseFactorExtraTrees",
    "ExtraTrees": "ExtraTrees",
    "HistGradientBoosting": "HistGradientBoosting",
    "SVR_RBF": "SVR_RBF",
    "SVR-RBF": "SVR_RBF",
    "BayesianRidge": "BayesianRidge",
    "Bayesian Ridge": "BayesianRidge",
    "PLS": "PLS",
    "RandomForest": "RandomForest",
    "Random Forest": "RandomForest",
}


@dataclass
class FixedFeatureSearchResult:
    enabled: bool
    model: str = ""
    candidate_features: List[str] = field(default_factory=list)
    all_trials: List[Dict[str, object]] = field(default_factory=list)
    top5_absolute: List[Dict[str, object]] = field(default_factory=list)
    top5_trend: List[Dict[str, object]] = field(default_factory=list)
    top5_absolute_cv: List[Dict[str, object]] = field(default_factory=list)
    top5_trend_cv: List[Dict[str, object]] = field(default_factory=list)
    top5_targets: List[Dict[str, object]] = field(default_factory=list)
    feature_effects: List[Dict[str, object]] = field(default_factory=list)
    manual_selection: List[Dict[str, object]] = field(default_factory=list)
    settings: List[Dict[str, object]] = field(default_factory=list)
    plot_paths: List[Path] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def resolve_fixed_model(requested: str, previous_best: str = "") -> str:
    requested = str(requested or "auto_previous")
    if requested == "auto_previous":
        candidate = _MODEL_ALIASES.get(str(previous_best or "").strip(), "")
        return candidate or "HistGradientBoosting"
    return _MODEL_ALIASES.get(requested, requested)


def _valid_records(records: Sequence[Dict[str, object]], require_concentration: bool) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in records:
        ratio = _num(row.get("Measured_ratio"))
        conc = _num(row.get("Actual_concentration"))
        if ratio is None or ratio <= 0:
            continue
        if require_concentration and (conc is None or conc <= 0):
            continue
        if not _key(row):
            continue
        out.append(dict(row))
    return out


def _matrix(records: Sequence[Dict[str, object]], features: Sequence[str]) -> np.ndarray:
    return np.asarray([
        [np.nan if _num(row.get(name)) is None else float(_num(row.get(name))) for name in features]
        for row in records
    ], dtype=float)


def _make_estimator(model_name: str, n_features: int, random_state: int):
    if model_name == "HistGradientBoosting":
        model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=7,
            min_samples_leaf=8, l2_regularization=1.0,
            random_state=int(random_state),
        )
        return Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)), ("model", model)])
    if model_name in {"ExtraTrees", "ResponseFactorExtraTrees"}:
        model = ExtraTreesRegressor(
            n_estimators=500, max_depth=7, min_samples_leaf=2,
            max_features=0.7, random_state=int(random_state), n_jobs=-1,
        )
        return Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)), ("model", model)])
    if model_name == "RandomForest":
        model = RandomForestRegressor(
            n_estimators=500, max_depth=6, min_samples_leaf=3,
            max_features=0.5, bootstrap=True,
            random_state=int(random_state), n_jobs=-1,
        )
        return Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)), ("model", model)])
    if model_name == "SVR_RBF":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", SVR(C=10.0, epsilon=0.10, gamma="scale")),
        ])
    if model_name == "BayesianRidge":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", BayesianRidge()),
        ])
    if model_name == "PLS":
        n_comp = max(1, min(5, int(n_features), 3))
        return Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", PLSRegression(n_components=n_comp, scale=False, max_iter=500)),
        ])
    raise ValueError(f"Unsupported fixed model: {model_name}")


def _predict(estimator, X: np.ndarray) -> np.ndarray:
    return np.asarray(estimator.predict(X), dtype=float).reshape(-1)


def _split_list(calibration: Sequence[Dict[str, object]], X: np.ndarray, y: np.ndarray, cv_splits: int, cv_repeats: int, random_state: int):
    groups = np.asarray([_text(r.get("Injection_Group")) for r in calibration], dtype=object)
    nonblank = sorted({str(x) for x in groups if str(x).strip()})
    if len(nonblank) >= 3 and all(str(x).strip() for x in groups):
        cv = LeaveOneGroupOut()
        return list(cv.split(X, y, groups)), f"Leave-one-injection-group-out ({len(nonblank)} groups)", groups
    n_splits = max(2, min(int(cv_splits), len(calibration) // 8 if len(calibration) >= 16 else 2))
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=max(1, int(cv_repeats)), random_state=int(random_state))
    return list(cv.split(X, y)), f"Repeated {n_splits}-fold x {max(1, int(cv_repeats))}", None


def _physical_pairwise_fold(
    X_train: np.ndarray, y_train: np.ndarray, log_ratio_train: np.ndarray,
    X_test: np.ndarray, log_ratio_test: np.ndarray, random_state: int,
) -> np.ndarray:
    # Reuse the physically constrained implementation and its OOF monotonic
    # mapping.  The configuration is intentionally fixed; only features vary.
    from .trend_rank_optimizer import _fit_predict_config
    config = {
        "subset": list(range(X_train.shape[1])),
        "alpha": 0.01,
        "pair_min_fold": 1.5,
        "margin": 0.10,
    }
    _base, pred, _cal, _pairs = _fit_predict_config(
        "PhysicalPairwiseRanker", X_train, y_train, log_ratio_train,
        X_test, log_ratio_test, config, random_state=int(random_state), groups=None,
    )
    return np.asarray(pred, dtype=float)


def _evaluate_subset(
    calibration: Sequence[Dict[str, object]],
    features: Sequence[str],
    model_name: str,
    split_list: Sequence[Tuple[np.ndarray, np.ndarray]],
    random_state: int,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    X = _matrix(calibration, features)
    actual = np.asarray([float(r["Actual_concentration"]) for r in calibration], dtype=float)
    y_log = np.log10(actual)
    ratios = np.asarray([float(r["Measured_ratio"]) for r in calibration], dtype=float)
    log_ratio = np.log10(ratios)
    predictions: Dict[int, List[float]] = {i: [] for i in range(len(calibration))}
    fold_rows: List[Dict[str, object]] = []
    failures = 0
    for fold_no, (tr, te) in enumerate(split_list, start=1):
        try:
            if model_name == "PhysicalPairwiseRanker":
                pred_log_c = _physical_pairwise_fold(X[tr], y_log[tr], log_ratio[tr], X[te], log_ratio[te], random_state + fold_no * 101)
            else:
                est = _make_estimator(model_name, len(features), random_state + fold_no * 101)
                y_log_rrf = log_ratio[tr] - y_log[tr]
                est.fit(X[tr], y_log_rrf)
                pred_log_rrf = _predict(est, X[te])
                pred_log_c = log_ratio[te] - pred_log_rrf
            for local, idx in enumerate(te):
                predictions[int(idx)].append(float(pred_log_c[local]))
        except Exception as exc:
            failures += 1
            fold_rows.append({"Fold": fold_no, "Status": "FAILED", "Error": str(exc)})
    cv_rows: List[Dict[str, object]] = []
    pred_values: List[float] = []
    actual_values: List[float] = []
    for idx, row in enumerate(calibration):
        vals = predictions.get(idx, [])
        if not vals:
            continue
        pred_log = float(statistics.median(vals))
        pred = float(10.0 ** pred_log)
        act = float(actual[idx])
        ratio = float(ratios[idx])
        cv_rows.append({
            "Calibration_Index": idx,
            "Source_file": row.get("Source_file", ""),
            "Source_sheet": row.get("Source_sheet", ""),
            "Source_row": row.get("Source_row", ""),
            "Injection_Group": row.get("Injection_Group", ""),
            "Combo": row.get("Combo", ""),
            "ABC_Formula_Key": _key(row),
            "Measured_ratio": ratio,
            "Actual_concentration": act,
            "Predicted_concentration": pred,
            "Predicted_log10_RRF": math.log10(ratio) - pred_log,
            "Fold_error": max(pred / act, act / pred),
            "Prediction_count": len(vals),
        })
        pred_values.append(pred)
        actual_values.append(act)
    if len(pred_values) < max(10, len(calibration) // 2):
        raise RuntimeError(f"Insufficient CV predictions ({len(pred_values)}/{len(calibration)}), failed folds={failures}")
    range_metrics = _metrics(actual_values, pred_values)
    trend_metrics = _trend_metrics(actual_values, pred_values)
    result: Dict[str, object] = {
        **range_metrics,
        **trend_metrics,
        "CV_prediction_rows": len(pred_values),
        "Failed_folds": failures,
    }
    return result, cv_rows


def _fit_full_predict(
    calibration: Sequence[Dict[str, object]], targets: Sequence[Dict[str, object]],
    features: Sequence[str], model_name: str, random_state: int,
) -> List[Dict[str, object]]:
    Xtr = _matrix(calibration, features)
    Xte = _matrix(targets, features)
    actual = np.asarray([float(r["Actual_concentration"]) for r in calibration], dtype=float)
    ratio_train = np.asarray([float(r["Measured_ratio"]) for r in calibration], dtype=float)
    ratio_target = np.asarray([float(r["Measured_ratio"]) for r in targets], dtype=float)
    y_log = np.log10(actual)
    log_ratio_train = np.log10(ratio_train)
    log_ratio_target = np.log10(ratio_target)
    if model_name == "PhysicalPairwiseRanker":
        from .trend_rank_optimizer import _fit_predict_config
        config = {"subset": list(range(len(features))), "alpha": 0.01, "pair_min_fold": 1.5, "margin": 0.10}
        _base, pred_log_c, _cal, _pairs = _fit_predict_config(
            "PhysicalPairwiseRanker", Xtr, y_log, log_ratio_train,
            Xte, log_ratio_target, config, random_state=int(random_state), groups=None,
        )
    else:
        est = _make_estimator(model_name, len(features), random_state)
        est.fit(Xtr, log_ratio_train - y_log)
        pred_log_rrf = _predict(est, Xte)
        pred_log_c = log_ratio_target - pred_log_rrf
    pred_c = 10.0 ** np.asarray(pred_log_c, dtype=float)
    ranks = _average_ranks(pred_c)
    rows: List[Dict[str, object]] = []
    for i, record in enumerate(targets):
        rows.append({
            "Source_file": record.get("Source_file", ""),
            "Source_sheet": record.get("Source_sheet", ""),
            "Source_row": record.get("Source_row", ""),
            "Combo": record.get("Combo", ""),
            "Formula": record.get("Formula", ""),
            "ABC_Formula_Key": _key(record),
            "Measured_ratio": record.get("Measured_ratio", ""),
            "Estimated_concentration": float(pred_c[i]),
            "Predicted_log10_RRF": math.log10(float(record["Measured_ratio"])) - float(pred_log_c[i]),
            "Rank": float(ranks[i]),
            "Percentile": float(ranks[i] / max(1, len(targets)) * 100.0),
        })
    add_decile_columns(rows, "Estimated_concentration", prefix="Concentration")
    return rows


def _random_subset(features: Sequence[str], rng: np.random.Generator, min_features: int, max_features: int) -> List[str]:
    upper = max(1, min(int(max_features), len(features)))
    lower = max(1, min(int(min_features), upper))
    k = int(rng.integers(lower, upper + 1))
    # Preserve mechanistic coverage by selecting at least one feature from each
    # of the main negative-ESI domains when possible, then fill randomly.
    buckets: Dict[str, List[str]] = {}
    for name in features:
        group = str(mechanistic_profile(name).get("Mechanistic_group", "Other"))
        buckets.setdefault(group, []).append(name)
    selected: List[str] = []
    priority_groups = [
        "Negative-ESI mechanism", "Ionization/charge", "Mobile-phase physics",
        "Physicochemical structure", "Experimental LC condition", "3D geometry",
    ]
    for group in priority_groups:
        choices = buckets.get(group, [])
        if choices and len(selected) < k and rng.random() < 0.70:
            selected.append(str(rng.choice(choices)))
    pool = [x for x in features if x not in selected]
    need = k - len(selected)
    if need > 0 and pool:
        selected.extend(rng.choice(np.asarray(pool, dtype=object), size=min(need, len(pool)), replace=False).tolist())
    return sorted(set(selected))


def _trial_row(trial: int, features: Sequence[str], metrics: Dict[str, object], status: str, error: str) -> Dict[str, object]:
    return {
        "Trial": int(trial),
        "Status": status,
        "Feature_count": len(features),
        "Features": ";".join(features),
        "Median_Fold_Error": metrics.get("Median_Fold_Error", ""),
        "P80_Fold_Error": metrics.get("P80_Fold_Error", ""),
        "P90_Fold_Error": metrics.get("P90_Fold_Error", ""),
        "Within_2x_pct": metrics.get("Within_2x_pct", ""),
        "Within_5x_pct": metrics.get("Within_5x_pct", ""),
        "RMSE_log10": metrics.get("RMSE_log10", ""),
        "R2_log_concentration": metrics.get("R2_log_concentration", ""),
        "Trend_Spearman_r": metrics.get("Trend_Spearman_r", ""),
        "Trend_Kendall_tau": metrics.get("Trend_Kendall_tau", ""),
        "Trend_Pairwise_Concordance_pct": metrics.get("Trend_Pairwise_Concordance_pct", ""),
        "Trend_Top20_Overlap_pct": metrics.get("Trend_Top20_Overlap_pct", ""),
        "CV_prediction_rows": metrics.get("CV_prediction_rows", ""),
        "Failed_folds": metrics.get("Failed_folds", ""),
        "Error": error,
    }


def _absolute_sort_key(row: Dict[str, object]):
    return (
        float(_num(row.get("Median_Fold_Error")) or 1e9),
        float(_num(row.get("P80_Fold_Error")) or 1e9),
        float(_num(row.get("RMSE_log10")) or 1e9),
        -float(_num(row.get("Within_2x_pct")) or 0.0),
    )


def _trend_sort_key(row: Dict[str, object]):
    spearman = _num(row.get("Trend_Spearman_r"))
    pairwise = _num(row.get("Trend_Pairwise_Concordance_pct"))
    top20 = _num(row.get("Trend_Top20_Overlap_pct"))
    median_fold = _num(row.get("Median_Fold_Error"))
    return (
        -float(spearman if spearman is not None else -1e9),
        -float(pairwise if pairwise is not None else 0.0),
        -float(top20 if top20 is not None else 0.0),
        float(median_fold if median_fold is not None else 1e9),
    )


def _feature_effects(
    trials: Sequence[Dict[str, object]], features: Sequence[str],
    top_abs: Sequence[Dict[str, object]], top_trend: Sequence[Dict[str, object]], language: str,
) -> List[Dict[str, object]]:
    valid = [r for r in trials if str(r.get("Status")) == "OK"]
    abs_sets = [set(str(r.get("Features", "")).split(";")) for r in top_abs]
    trend_sets = [set(str(r.get("Features", "")).split(";")) for r in top_trend]
    rows: List[Dict[str, object]] = []
    for name in features:
        included = [r for r in valid if name in set(str(r.get("Features", "")).split(";"))]
        excluded = [r for r in valid if name not in set(str(r.get("Features", "")).split(";"))]
        inc_s = [float(r["Trend_Spearman_r"]) for r in included if _num(r.get("Trend_Spearman_r")) is not None]
        exc_s = [float(r["Trend_Spearman_r"]) for r in excluded if _num(r.get("Trend_Spearman_r")) is not None]
        inc_f = [float(r["Median_Fold_Error"]) for r in included if _num(r.get("Median_Fold_Error")) is not None]
        exc_f = [float(r["Median_Fold_Error"]) for r in excluded if _num(r.get("Median_Fold_Error")) is not None]
        delta_s = (float(np.mean(inc_s)) - float(np.mean(exc_s))) if inc_s and exc_s else float("nan")
        # Positive means inclusion lowers median fold error.
        improve_f = (float(np.mean(exc_f)) - float(np.mean(inc_f))) if inc_f and exc_f else float("nan")
        if math.isfinite(delta_s) and math.isfinite(improve_f):
            if delta_s > 0.02 and improve_f > 0.05:
                effect = "Positive for both"
            elif delta_s > 0.02 and improve_f <= 0:
                effect = "Trend positive / absolute adverse"
            elif delta_s <= 0 and improve_f > 0.05:
                effect = "Absolute positive / trend adverse"
            elif delta_s < -0.02 and improve_f < -0.05:
                effect = "Negative for both"
            else:
                effect = "Weak or mixed"
        else:
            effect = "Insufficient comparisons"
        meta = descriptor_meta(name)
        mech = mechanistic_profile(name)
        rows.append({
            "Feature": name,
            "Name_zh": meta.get("Name_zh", name),
            "Name_en": meta.get("Name_en", name),
            "Mechanistic_group": mech.get("Mechanistic_group", ""),
            "Mechanistic_level": mech.get("Mechanistic_level", ""),
            "Trial_inclusion_pct": 100.0 * len(included) / max(1, len(valid)),
            "Trend_Spearman_mean_when_included": float(np.mean(inc_s)) if inc_s else "",
            "Trend_Spearman_mean_when_excluded": float(np.mean(exc_s)) if exc_s else "",
            "Trend_Spearman_delta_included_minus_excluded": delta_s if math.isfinite(delta_s) else "",
            "Absolute_MedianFold_mean_when_included": float(np.mean(inc_f)) if inc_f else "",
            "Absolute_MedianFold_mean_when_excluded": float(np.mean(exc_f)) if exc_f else "",
            "Absolute_MedianFold_improvement_included": improve_f if math.isfinite(improve_f) else "",
            "Top5_Absolute_frequency_pct": 100.0 * sum(name in s for s in abs_sets) / max(1, len(abs_sets)),
            "Top5_Trend_frequency_pct": 100.0 * sum(name in s for s in trend_sets) / max(1, len(trend_sets)),
            "Effect_class": effect,
            "Theory_zh": mech.get("Theory_zh", ""),
            "Theory_en": mech.get("Theory_en", ""),
            "Expected_direction_zh": mech.get("Expected_direction_zh", ""),
            "Expected_direction_en": mech.get("Expected_direction_en", ""),
        })
    def _effect_sort_key(r):
        freq = _num(r.get("Top5_Trend_frequency_pct"))
        delta = _num(r.get("Trend_Spearman_delta_included_minus_excluded"))
        return (
            -float(freq if freq is not None else 0.0),
            -float(delta if delta is not None else -999.0),
        )
    rows.sort(key=_effect_sort_key)
    return rows


def _manual_selection_rows(all_available: Sequence[str], selected: Sequence[str]) -> List[Dict[str, object]]:
    chosen = set(selected)
    rows: List[Dict[str, object]] = []
    for name in all_available:
        meta = descriptor_meta(name)
        mech = mechanistic_profile(name)
        rows.append({
            "Selected_for_candidate_pool": "Yes" if name in chosen else "No",
            "Feature": name,
            "Name_zh": meta.get("Name_zh", name),
            "Name_en": meta.get("Name_en", name),
            "Mechanistic_group": mech.get("Mechanistic_group", ""),
            "Mechanistic_level": mech.get("Mechanistic_level", ""),
            "Theory_zh": mech.get("Theory_zh", ""),
            "Theory_en": mech.get("Theory_en", ""),
            "Expected_direction_zh": mech.get("Expected_direction_zh", ""),
            "Expected_direction_en": mech.get("Expected_direction_en", ""),
        })
    return rows


def _make_plots(output_xlsx: Path, top_abs: Sequence[Dict[str, object]], top_trend: Sequence[Dict[str, object]], effects: Sequence[Dict[str, object]], language: str) -> Tuple[List[Path], List[str]]:
    warnings: List[str] = []
    paths: List[Path] = []
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return paths, [f"Fixed feature-search plots unavailable: {exc}"]
    plot_dir = Path(output_xlsx).parent / ("_fixedplots_" + str(abs(hash(str(output_xlsx))) % 10**8).zfill(8))
    plot_dir.mkdir(parents=True, exist_ok=True)
    zh = str(language).lower().startswith("zh")
    try:
        names = [f"T{int(r['Trial'])}" for r in top_abs]
        med = [float(r["Median_Fold_Error"]) for r in top_abs]
        p80 = [float(r["P80_Fold_Error"]) for r in top_abs]
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(names))
        ax.bar(x - 0.18, med, width=0.36, label="Median")
        ax.bar(x + 0.18, p80, width=0.36, label="P80")
        ax.set_xticks(x, names)
        ax.set_ylabel("Fold error")
        ax.set_title('Fixed model: top-5 absolute trials' if zh else "Fixed model: top-5 absolute trials")
        ax.legend(); fig.tight_layout()
        p = plot_dir / "27_fixed_top5_absolute.png"; fig.savefig(p, dpi=180); plt.close(fig); paths.append(p)
    except Exception as exc:
        warnings.append(f"Top-5 absolute plot failed: {exc}")
    try:
        names = [f"T{int(r['Trial'])}" for r in top_trend]
        rho = [float(r["Trend_Spearman_r"]) for r in top_trend]
        pair = [float(r["Trend_Pairwise_Concordance_pct"]) / 100.0 for r in top_trend]
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(names))
        ax.bar(x - 0.18, rho, width=0.36, label="Spearman")
        ax.bar(x + 0.18, pair, width=0.36, label="Pairwise")
        ax.axhline(0, linewidth=1)
        ax.set_xticks(x, names)
        ax.set_title('Fixed model: top-5 trend trials' if zh else "Fixed model: top-5 trend trials")
        ax.legend(); fig.tight_layout()
        p = plot_dir / "28_fixed_top5_trend.png"; fig.savefig(p, dpi=180); plt.close(fig); paths.append(p)
    except Exception as exc:
        warnings.append(f"Top-5 trend plot failed: {exc}")
    try:
        top = sorted(
            [r for r in effects if _num(r.get("Trend_Spearman_delta_included_minus_excluded")) is not None],
            key=lambda r: abs(float(r["Trend_Spearman_delta_included_minus_excluded"])), reverse=True,
        )[:20][::-1]
        fig, ax = plt.subplots(figsize=(10, 7))
        labels = [display_name(str(r["Feature"]), language) for r in top]
        vals = [float(r["Trend_Spearman_delta_included_minus_excluded"]) for r in top]
        ax.barh(labels, vals)
        ax.axvline(0, linewidth=1)
        ax.set_xlabel("Included-minus-excluded Spearman" if not zh else 'Included-minus-excluded Spearman')
        ax.set_title('Feature effect across random trials' if zh else "Feature effect across random trials")
        fig.tight_layout()
        p = plot_dir / "29_fixed_feature_effect.png"; fig.savefig(p, dpi=180); plt.close(fig); paths.append(p)
    except Exception as exc:
        warnings.append(f"Feature-effect plot failed: {exc}")
    return paths, warnings


def run_fixed_feature_search(
    calibration_records: Sequence[Dict[str, object]],
    target_records: Sequence[Dict[str, object]],
    descriptor_names: Sequence[str],
    output_xlsx: Path,
    *,
    fixed_model: str = "HistGradientBoosting",
    previous_best_model: str = "",
    manual_feature_pool: Sequence[str] = (),
    trial_count: int = 100,
    min_features: int = 4,
    max_features: int = 20,
    cv_splits: int = 5,
    cv_repeats: int = 3,
    random_state: int = 42,
    correlation_threshold: float = 0.97,
    output_language: str = "en",
    concentration_unit: str = "",
) -> FixedFeatureSearchResult:
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("Fixed feature search needs scikit-learn: " + SKLEARN_ERROR)
    calibration = _valid_records(calibration_records, True)
    targets = _valid_records(target_records, False)
    if len(calibration) < 30:
        raise ValueError(f"Fixed feature search requires at least 30 calibration rows; got {len(calibration)}")
    model_name = resolve_fixed_model(fixed_model, previous_best_model)
    numeric, _cat, manifest = choose_features(
        calibration, descriptor_names, expected_descriptor_names=(), three_d_enabled=True,
        correlation_threshold=float(correlation_threshold), include_categorical=False,
    )
    blocked = {"Measured_ratio", "Log_Measured_Ratio", "Actual_concentration"}

    # A manual choice must not be silently removed merely because it is highly
    # correlated with another descriptor.  For the manual pool, availability is
    # checked directly from the calibration matrix; correlation control is left
    # to the user's choice and the fixed model.  Without a manual pool, retain
    # the compact automatically de-redundant list returned by choose_features().
    direct_available: List[str] = []
    min_valid = max(10, int(math.ceil(0.50 * len(calibration))))
    direct_names = [
        "Exact_mass", "DBE", "Apex_RT_min", "Effective_gradient_time_min",
        "Mobile_phase_A_pct", "Mobile_phase_B_pct", "B_slope_pct_per_min",
    ] + list(descriptor_names)
    for name in dict.fromkeys(direct_names):
        if name in blocked or str(name).lower().startswith("measured_ratio"):
            continue
        vals = [_num(row.get(name)) for row in calibration]
        finite = [float(x) for x in vals if x is not None]
        if len(finite) < min_valid:
            continue
        if float(np.nanvar(np.asarray(finite, dtype=float))) <= 1e-14:
            continue
        direct_available.append(str(name))
    available = list(dict.fromkeys([x for x in numeric if x not in blocked and not x.lower().startswith("measured_ratio")]))
    manual = [x for x in manual_feature_pool if x in direct_available]
    candidate = list(dict.fromkeys(manual or available))
    if len(candidate) < 2:
        raise ValueError("Too few selected/available descriptors for fixed-model feature search")
    min_features = max(1, min(int(min_features), len(candidate)))
    max_features = max(min_features, min(int(max_features), len(candidate)))
    Xall = _matrix(calibration, candidate)
    yall = np.log10(np.asarray([float(r["Actual_concentration"]) for r in calibration], dtype=float))
    split_list, cv_label, _groups = _split_list(calibration, Xall, yall, cv_splits, cv_repeats, random_state)
    rng = np.random.default_rng(int(random_state) + 377)
    trial_rows: List[Dict[str, object]] = []
    predictions_by_trial: Dict[int, List[Dict[str, object]]] = {}
    used_subsets = set()
    for trial in range(1, max(1, int(trial_count)) + 1):
        subset = _random_subset(candidate, rng, min_features, max_features)
        signature = tuple(subset)
        # Avoid excessive duplicates while keeping the requested number of rows.
        attempts = 0
        while signature in used_subsets and attempts < 20 and len(used_subsets) < 2 ** min(len(candidate), 20):
            subset = _random_subset(candidate, rng, min_features, max_features)
            signature = tuple(subset); attempts += 1
        used_subsets.add(signature)
        try:
            metrics, cv_rows = _evaluate_subset(calibration, subset, model_name, split_list, random_state + trial * 1009)
            row = _trial_row(trial, subset, metrics, "OK", "")
            predictions_by_trial[trial] = cv_rows
        except Exception as exc:
            row = _trial_row(trial, subset, {}, "FAILED", str(exc))
        trial_rows.append(row)
    valid = [r for r in trial_rows if str(r.get("Status")) == "OK"]
    if not valid:
        raise RuntimeError("All fixed-model feature trials failed")
    abs_sorted = sorted(valid, key=_absolute_sort_key)
    trend_sorted = sorted(valid, key=_trend_sort_key)
    top_abs = [dict(r, Absolute_rank=i + 1) for i, r in enumerate(abs_sorted[:5])]
    top_trend = [dict(r, Trend_rank=i + 1) for i, r in enumerate(trend_sorted[:5])]
    top_abs_cv: List[Dict[str, object]] = []
    top_trend_cv: List[Dict[str, object]] = []
    for objective, top, destination in (("absolute", top_abs, top_abs_cv), ("trend", top_trend, top_trend_cv)):
        for rank, row in enumerate(top, start=1):
            for cv_row in predictions_by_trial.get(int(row["Trial"]), []):
                destination.append({
                    "Objective": objective, "Top_rank": rank, "Trial": row["Trial"],
                    "Model": model_name, "Features": row["Features"], **cv_row,
                })
    target_rows: List[Dict[str, object]] = []
    seen_configs = set()
    for objective, top in (("absolute", top_abs), ("trend", top_trend)):
        for rank, row in enumerate(top, start=1):
            trial = int(row["Trial"])
            features = [x for x in str(row["Features"]).split(";") if x]
            key = (objective, trial)
            if key in seen_configs:
                continue
            seen_configs.add(key)
            try:
                preds = _fit_full_predict(calibration, targets, features, model_name, random_state + 900000 + trial)
                for pred in preds:
                    target_rows.append({
                        "Objective": objective, "Top_rank": rank, "Trial": trial,
                        "Model": model_name, "Features": ";".join(features),
                        "Concentration_unit": concentration_unit, **pred,
                    })
            except Exception as exc:
                target_rows.append({
                    "Objective": objective, "Top_rank": rank, "Trial": trial,
                    "Model": model_name, "Features": ";".join(features), "Error": str(exc),
                })
    effects = _feature_effects(trial_rows, candidate, top_abs, top_trend, output_language)
    manual_rows = _manual_selection_rows(direct_available, candidate)
    plots, plot_warnings = _make_plots(output_xlsx, top_abs, top_trend, effects, output_language)
    settings = [
        {"Setting": "Fixed_model", "Value": model_name},
        {"Setting": "Previous_best_model", "Value": previous_best_model},
        {"Setting": "Random_feature_trials", "Value": int(trial_count)},
        {"Setting": "Candidate_feature_count", "Value": len(candidate)},
        {"Setting": "Feature_count_range", "Value": f"{min_features}-{max_features}"},
        {"Setting": "CV", "Value": cv_label},
        {"Setting": "Concentration_unit", "Value": concentration_unit},
        {"Setting": "Unknown_target_used_for_selection", "Value": False},
        {"Setting": "Guardrail", "Value": "Every trial uses out-of-fold predictions, but selecting the best of many trials remains exploratory and requires independent validation."},
    ]
    warnings = list(plot_warnings)
    if manual_feature_pool and not manual:
        warnings.append("None of the manually selected descriptors were available; the full available descriptor pool was used.")
    return FixedFeatureSearchResult(
        enabled=True, model=model_name, candidate_features=candidate,
        all_trials=trial_rows, top5_absolute=top_abs, top5_trend=top_trend,
        top5_absolute_cv=top_abs_cv, top5_trend_cv=top_trend_cv,
        top5_targets=target_rows, feature_effects=effects,
        manual_selection=manual_rows, settings=settings,
        plot_paths=plots, warnings=warnings,
    )


def _append_dict_sheet(wb, title: str, rows: Sequence[Dict[str, object]]) -> None:
    if title in wb.sheetnames:
        del wb[title]
    ws = wb.create_sheet(title)
    headers: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key); headers.append(str(key))
    if not headers:
        headers = ["Message"]
    ws.append(headers)
    for row in rows:
        ws.append([row.get(h, "") for h in headers])
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions


def append_fixed_feature_search_to_workbook(wb, result: FixedFeatureSearchResult) -> None:
    best_abs = result.top5_absolute[0] if result.top5_absolute else {}
    best_trend = result.top5_trend[0] if result.top5_trend else {}
    decision = [
        {"Metric": "Fixed_model", "Value": result.model},
        {"Metric": "Candidate_feature_count", "Value": len(result.candidate_features)},
        {"Metric": "Best_absolute_trial", "Value": best_abs.get("Trial", "")},
        {"Metric": "Best_absolute_features", "Value": best_abs.get("Features", "")},
        {"Metric": "Best_absolute_median_fold", "Value": best_abs.get("Median_Fold_Error", "")},
        {"Metric": "Best_absolute_P80_fold", "Value": best_abs.get("P80_Fold_Error", "")},
        {"Metric": "Best_absolute_R2_log", "Value": best_abs.get("R2_log_concentration", "")},
        {"Metric": "Best_trend_trial", "Value": best_trend.get("Trial", "")},
        {"Metric": "Best_trend_features", "Value": best_trend.get("Features", "")},
        {"Metric": "Best_trend_Spearman", "Value": best_trend.get("Trend_Spearman_r", "")},
        {"Metric": "Best_trend_pairwise_pct", "Value": best_trend.get("Trend_Pairwise_Concordance_pct", "")},
        {"Metric": "Best_trend_Top20_pct", "Value": best_trend.get("Trend_Top20_Overlap_pct", "")},
        {"Metric": "Best_trend_median_fold", "Value": best_trend.get("Median_Fold_Error", "")},
        {"Metric": "Interpretation", "Value": "Every trial uses out-of-fold predictions. The top-five tables are exploratory leaderboards and still require independent/grouped validation."},
    ] + [{"Metric": "Warning", "Value": x} for x in result.warnings]
    _append_dict_sheet(wb, "Fixed_Search_Decision", decision)
    _append_dict_sheet(wb, "Fixed_Trial_All", result.all_trials)
    _append_dict_sheet(wb, "Fixed_Top5_Absolute", result.top5_absolute)
    _append_dict_sheet(wb, "Fixed_Top5_Trend", result.top5_trend)
    _append_dict_sheet(wb, "Fixed_Top5_Absolute_CV", result.top5_absolute_cv)
    _append_dict_sheet(wb, "Fixed_Top5_Trend_CV", result.top5_trend_cv)
    _append_dict_sheet(wb, "Fixed_Top5_Targets", result.top5_targets)
    _append_dict_sheet(wb, "Fixed_Feature_Effects", result.feature_effects)
    _append_dict_sheet(wb, "Manual_Feature_Selection", result.manual_selection)
    _append_dict_sheet(wb, "Fixed_Search_Settings", result.settings)
    if result.plot_paths:
        try:
            from openpyxl.drawing.image import Image as XLImage
            if "Fixed_Search_Plots" in wb.sheetnames:
                del wb["Fixed_Search_Plots"]
            ws = wb.create_sheet("Fixed_Search_Plots")
            row = 1
            for path in result.plot_paths:
                if not Path(path).exists():
                    continue
                ws.cell(row=row, column=1, value=Path(path).name)
                img = XLImage(str(path)); img.width = min(img.width, 900); img.height = min(img.height, 650)
                ws.add_image(img, f"A{row + 1}")
                row += 35
        except Exception:
            pass

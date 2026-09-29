"""Independent continuous RRF audit. No concentration recalibration or clipping.

Compatible with the application's scikit-learn 1.3.2 API. All learned numeric
preprocessing and feature selection is local to each fit, including inner CV.
The unknown-sample records are never used for training or model selection.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler

from .training_limits import validate_training_count

VERSION = '18.45-audit-branch'
META = {
    'dataset', 'name', 'compound', 'formula', 'combo', 'injectiongroup',
    'injectiongroupcolumn', 'warnings', 'structurehash', 'structurestatus',
    'structuremethod', 'formulamatch', 'descriptorcount', '3dstatus',
    'productformulacalc', 'productmastermatch', 'productmasterformulakey',
    'productsmiles', 'afomula', 'aformula', 'bformula', 'cformula',
    'ratio', 'areais', 'arearatio', 'logratio', 'logarearatio', 'isarea',
    'peakarea', 'area', 'additionalsummedarea', 'rank', 'order', 'id', 'concentration', 'conc',
    'rrf', 'logrrf', 'log10rrf', 'correctionfactor', 'logcorrectionfactor',
    'responsefactor', 'responsecorrectionfactor', 'responsecorrectionindex',
}
CONDITION_FEATURES = [
    'Exact_mass', 'DBE', 'Apex_RT_min', 'Mobile_phase_A_pct',
    'Mobile_phase_B_pct', 'B_slope_pct_per_min',
]


def text(value):
    return '' if value is None else str(value).strip()


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else float('nan')
    except (ValueError, TypeError, OverflowError):
        return float('nan')


def norm(value):
    return ''.join(c.lower() for c in text(value) if c.isalnum())


def allowed_feature(name):
    key = norm(name)
    if not key or key in META or text(name).startswith('__'):
        return False
    if key.startswith(('source', 'actual', 'predicted', 'estimated', 'measured',
                       'known', 'target', 'oof', 'cv', 'fold', 'percentile')):
        return False
    return not any(token in key for token in
                   ('concentration', 'correctionfactor', 'logrrf', 'responsefactor'))


def feature_names(records, proposed):
    """Schema only; no global variance, correlation, median or target inspection."""
    present = set().union(*(r.keys() for r in records)) if records else set()
    return list(dict.fromkeys(n for n in list(proposed) + CONDITION_FEATURES
                              if n in present and allowed_feature(n)))


def matrix(records, names):
    return np.asarray([[number(r.get(n)) for n in names] for r in records],
                      dtype=float).reshape(len(records), len(names))


def compound_tokens(record):
    """Conservatively group duplicate identities; never use Formula alone."""
    out = []
    for key in ('Structure_hash', 'Product_SMILES', 'Combo'):
        value = text(record.get(key))
        if value:
            out.append(key + ':' + ''.join(value.split()))
    name, formula = text(record.get('Name')), text(record.get('Formula'))
    if name and formula:
        out.append('NameFormula:' + name + '|' + formula)
    return out


def independent_groups(records):
    """Union injection groups AND duplicate compounds, preventing either leakage."""
    n = len(records)
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    found_groups = 0
    for i, record in enumerate(records):
        tokens = compound_tokens(record)
        group = text(record.get('Injection_Group'))
        if group:
            found_groups += 1
            tokens.append('Injection:' + group.casefold())
        for token in tokens:
            if token in seen:
                parent[root(i)] = root(seen[token])
            else:
                seen[token] = i
    groups = np.asarray([root(i) for i in range(n)])
    notes = []
    if not found_groups:
        notes.append('No injection groups: this CV does NOT validate transfer to a new injection batch.')
    elif found_groups < n:
        notes.append('Injection group metadata are incomplete; grouped known rows are kept together, '
                     'but unknown batches cannot be protected. Supply complete group metadata.')
    if len(set(groups)) < n:
        notes.append('CV keeps connected injection/compound identity groups together.')
    return groups, notes


def splits(n, groups, count, seed):
    if n < 8:
        raise ValueError('A CV training partition has fewer than 8 usable standards.')
    distinct = len(set(groups))
    if distinct < 2:
        raise ValueError('Fewer than two independent groups in a CV partition; '
                         'cannot estimate held-out performance without group leakage.')
    count = min(int(count), distinct)
    if count < 2:
        raise ValueError('At least two CV folds are required.')
    if distinct == n:
        return list(KFold(count, shuffle=True, random_state=seed).split(np.arange(n)))
    return list(GroupKFold(count).split(np.arange(n), groups=groups))


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    scaler: str = 'standard'
    alpha: float = 1.0
    features: int = 8
    neighbors: int = 5
    leaf: int = 2


def default_specs():
    return [
        ModelSpec('RRF_median_baseline', 'median', features=0),
        ModelSpec('RRF_mean_baseline', 'mean', features=0),
        ModelSpec('Ridge_standard_a0.1_k8', 'ridge', alpha=.1),
        ModelSpec('Ridge_standard_a1_k8', 'ridge'),
        ModelSpec('Ridge_standard_a10_k8', 'ridge', alpha=10),
        ModelSpec('Ridge_standard_a100_k8', 'ridge', alpha=100),
        ModelSpec('Ridge_standard_a1_k16', 'ridge', features=16),
        ModelSpec('Ridge_robust_a1_k8', 'ridge', scaler='robust'),
        ModelSpec('Ridge_unscaled_a1_k8', 'ridge', scaler='none'),
        ModelSpec('LocalRRF_KNN3_standard', 'knn', neighbors=3),
        ModelSpec('LocalRRF_KNN7_standard', 'knn', neighbors=7),
        ModelSpec('RRF_ExtraTrees_leaf2', 'trees', scaler='none', features=12),
    ]


class FoldModel:
    """Numeric imputation, selection and scaling fitted ONLY to supplied X, y."""
    def __init__(self, spec, seed=42):
        self.spec, self.seed = spec, seed
        self.columns = np.asarray([], dtype=int)
        self.scaler = None

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        self.center = float(np.median(y) if self.spec.kind == 'median' else np.mean(y))
        if self.spec.kind in ('median', 'mean'):
            return self
        candidates, medians, values = [], [], []
        # Coverage and constant checks are fold-local; no all-training prefilter.
        for j in range(X.shape[1]):
            v = X[:, j]
            valid = np.isfinite(v)
            if np.count_nonzero(valid) < max(3, math.ceil(len(y) * .5)):
                continue
            median = float(np.median(v[valid]))
            filled = np.where(valid, v, median)
            # Relative variance tolerance avoids arbitrary dependence on units.
            if np.ptp(filled) <= np.finfo(float).eps * max(1e-300, np.max(np.abs(filled))):
                continue
            candidates.append(j)
            medians.append(median)
            values.append(filled)
        if not candidates:
            raise ValueError('No usable varying descriptor in this training fold.')
        values = np.column_stack(values)
        with np.errstate(invalid='ignore', divide='ignore'):
            # Unit-invariant univariate screen, followed by redundant-feature removal.
            z = values - np.mean(values, axis=0)
            sd = np.sqrt(np.sum(z * z, axis=0))
            yn = y - np.mean(y)
            ynorm = max(float(np.linalg.norm(yn)), 1e-300)
            standardized = z / sd
            scores = np.abs(standardized.T.dot(yn) / ynorm)
        scores = np.nan_to_num(scores)
        order = np.argsort(-scores, kind='stable')
        selected = []
        limit = max(1, min(self.spec.features, len(y) // 3))
        for k in order:
            if selected and np.any(np.abs(standardized[:, selected].T.dot(standardized[:, k])) > .98):
                continue
            selected.append(int(k))
            if len(selected) >= limit:
                break
        self.columns = np.asarray(candidates, dtype=int)[selected]
        self.medians = np.asarray(medians)[selected]
        Z = values[:, selected]
        self.minimum, self.maximum = np.min(Z, axis=0), np.max(Z, axis=0)
        if self.spec.scaler == 'standard':
            self.scaler = StandardScaler().fit(Z)
        elif self.spec.scaler == 'robust':
            self.scaler = RobustScaler().fit(Z)
        if self.scaler is not None:
            Z = self.scaler.transform(Z)
        if self.spec.kind == 'ridge':
            self.estimator = Ridge(alpha=self.spec.alpha)
        elif self.spec.kind == 'knn':
            self.estimator = KNeighborsRegressor(n_neighbors=min(self.spec.neighbors, len(y)),
                                                  weights='distance', algorithm='brute')
        elif self.spec.kind == 'trees':
            self.estimator = ExtraTreesRegressor(n_estimators=80, min_samples_leaf=self.spec.leaf,
                                                 random_state=self.seed, n_jobs=1)
        else:
            raise ValueError('Unknown model kind: ' + self.spec.kind)
        self.estimator.fit(Z, y)
        return self

    def transform(self, X):
        Z = np.asarray(X, float)[:, self.columns]
        Z = np.where(np.isfinite(Z), Z, self.medians)
        return self.scaler.transform(Z) if self.scaler is not None else Z

    def predict_rrf(self, X):
        if self.spec.kind in ('median', 'mean'):
            return np.full(len(X), self.center)
        return np.asarray(self.estimator.predict(self.transform(X))).reshape(-1)

    def predict_logc(self, X, log_ratio):
        # +1 log-response coefficient is NEVER recalibrated by Huber/isotonic.
        return np.asarray(log_ratio, float) - self.predict_rrf(X)

    def domain(self, X):
        if not len(self.columns):
            return [{'Selected_missing': 0, 'Out_of_range_features': 0,
                     'Descriptor_domain': 'Not_assessed_response_baseline'} for _ in X]
        Z = np.asarray(X, float)[:, self.columns]
        result = []
        for row in Z:
            valid = np.isfinite(row)
            missing = int(np.count_nonzero(~valid))
            outside = int(np.count_nonzero(valid & ((row < self.minimum) | (row > self.maximum))))
            result.append({'Selected_missing': missing, 'Out_of_range_features': outside,
                           'Descriptor_domain': ('All_selected_missing' if missing == len(row)
                              else 'Univariate_range_warning' if outside else 'Inside_univariate_ranges')})
        return result


def metrics(actual_log, predicted_log):
    y, p = np.asarray(actual_log, float), np.asarray(predicted_log, float)
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], p[valid]
    if not len(y):
        return {'N': 0}
    error = p - y
    denom = float(np.sum((y - np.mean(y)) ** 2))
    slope = float(np.dot(y - np.mean(y), p - np.mean(p)) / denom) if denom > 1e-20 else None
    ry, rp = rankdata(y), rankdata(p)
    rho = float(np.corrcoef(ry, rp)[0, 1]) if np.std(ry) > 0 and np.std(rp) > 0 else 0.0
    n20 = max(1, int(math.ceil(len(y) * .2)))
    order = np.argsort(y, kind='stable')
    return {
        'N': len(y), 'Log_RMSE': float(np.sqrt(np.mean(error ** 2))),
        'Log_MAE': float(np.mean(np.abs(error))),
        'Log_R2': float(1 - np.sum(error ** 2) / denom) if denom > 1e-20 else None,
        'Spearman': rho, 'Pred_on_actual_slope': slope,
        'Log_spread_ratio': float(np.std(p) / np.std(y)) if np.std(y) > 1e-10 else None,
        'Median_fold_error': pow10(float(np.median(np.abs(error)))),
        'P80_fold_error': pow10(float(np.quantile(np.abs(error), .8))),
        'Within_2x_pct': float(np.mean(np.abs(error) <= math.log10(2)) * 100),
        'Low20_log_bias': float(np.mean(error[order[:n20]])),
        'High20_log_bias': float(np.mean(error[order[-n20:]])),
        'Unique_predictions_12sig': len(set('%.12g' % v for v in p)),
    }


def pow10(value):
    if value is None or not math.isfinite(value) or value > 308 or value < -307:
        return None
    return float(10.0 ** value)


def selection_search(X, y, log_ratio, groups, specs, seed):
    partitions = splits(len(y), groups, 3, seed)
    results = []
    for spec in specs:
        pred = np.full(len(y), np.nan)
        failed = ''
        for ti, vi in partitions:
            try:
                model = FoldModel(spec, seed).fit(X[ti], log_ratio[ti] - y[ti])
                pred[vi] = model.predict_logc(X[vi], log_ratio[vi])
            except Exception as exc:
                failed = type(exc).__name__ + ': ' + str(exc)
                break
        if failed or not np.all(np.isfinite(pred)):
            results.append({'Model': spec.name, 'Inner_log_RMSE': None, 'Error': failed or 'Nonfinite predictions'})
        else:
            results.append({'Model': spec.name, 'Inner_log_RMSE': metrics(y, pred)['Log_RMSE'], 'Error': ''})
    available = [r for r in results if r['Inner_log_RMSE'] is not None]
    if not available:
        raise ValueError('All model candidates failed inside training-only cross-validation.')
    chosen = min(available, key=lambda r: (r['Inner_log_RMSE'], r['Model']))['Model']
    return next(s for s in specs if s.name == chosen), results


def row_info(record, i, dataset):
    return {'Dataset': dataset, 'Input_order': i + 1,
            'Source_file': record.get('Source_file', ''),
            'Source_sheet': record.get('Source_sheet', ''),
            'Source_row': record.get('Source_row', i + 2),
            'Compound': record.get('Name', ''), 'Formula': record.get('Formula', ''),
            'Combo': record.get('Combo', ''), 'Injection_Group': record.get('Injection_Group', '')}


def run_audit(training_records, target_records, descriptor_names, *, seed=42, folds=5,
              specs=None, progress=None):
    """Return plain tables, settings and arrays for a NEW, separate output folder."""
    emit = progress or (lambda message: None)
    validate_training_count(training_records)
    specs = list(specs or default_specs())
    if len({s.name for s in specs}) != len(specs):
        raise ValueError('Model names must be unique.')
    names = feature_names(training_records, descriptor_names)
    audit, valid_records, valid_indices = [], [], []
    for i, r in enumerate(training_records):
        ratio, conc = number(r.get('Measured_ratio')), number(r.get('Actual_concentration'))
        reason = ''
        if not np.isfinite(ratio) or ratio <= 0:
            reason = 'Missing_or_nonpositive_measured_ratio'
        elif not np.isfinite(conc) or conc <= 0:
            reason = 'Missing_or_nonpositive_known_concentration'
        elif norm(r.get('Formula_match')) in ('false', 'mismatch', 'no', 'fail'):
            reason = 'Explicit_formula_mismatch'
        if not reason:
            valid_records.append(r)
            valid_indices.append(i)
        audit.append(dict(row_info(r, i, 'Training'), Status='Skipped' if reason else 'Used', Reason=reason))
    if len(valid_records) < 24:
        raise ValueError('Need at least 24 valid standards for this nested diagnostic; '
                         '%d valid / %d input. This is a minimum for this workflow, not a guarantee of accuracy.'
                         % (len(valid_records), len(training_records)))
    X = matrix(valid_records, names)
    y = np.log10([number(r['Actual_concentration']) for r in valid_records])
    logr = np.log10([number(r['Measured_ratio']) for r in valid_records])
    if np.std(y) < 1e-10:
        raise ValueError('Known concentrations are constant; concentration trend cannot be validated.')
    groups, notes = independent_groups(valid_records)
    if len(set(groups)) < 3:
        raise ValueError('Need at least three independent injection/compound groups for nested validation. '
                         'Repeated compounds can connect multiple injection groups; do not break these groups to improve scores.')
    outer = splits(len(y), groups, folds, seed)
    pred = np.full(len(y), np.nan)
    constants = {'Constant_logC_mean': np.full(len(y), np.nan),
                 'Constant_logC_median': np.full(len(y), np.nan)}
    candidate_predictions = {s.name: np.full(len(y), np.nan) for s in specs}
    cv_rows, inner_rows, feature_rows = [], [], []
    row_models, row_folds = {}, {}
    for fold, (ti, vi) in enumerate(outer, start=1):
        emit('Outer fold %d/%d: model and preprocessing selection use this training fold only.' % (fold, len(outer)))
        chosen, search = selection_search(X[ti], y[ti], logr[ti], groups[ti], specs, seed + fold)
        inner_rows.extend(dict(row, Stage='Outer_%d' % fold, Selected=row['Model'] == chosen.name) for row in search)
        constants['Constant_logC_mean'][vi] = np.mean(y[ti])
        constants['Constant_logC_median'][vi] = np.median(y[ti])
        for spec in specs:
            try:
                model = FoldModel(spec, seed).fit(X[ti], logr[ti] - y[ti])
                prediction = model.predict_logc(X[vi], logr[vi])
                if not np.all(np.isfinite(prediction)):
                    raise ValueError('Nonfinite held-out predictions')
                candidate_predictions[spec.name][vi] = prediction
                if spec.name == chosen.name:
                    pred[vi] = prediction
                    for j in model.columns:
                        feature_rows.append({'Stage': 'Outer_%d' % fold, 'Model': chosen.name, 'Feature': names[j]})
            except Exception as exc:
                notes.append('Outer %d / %s failed: %s' % (fold, spec.name, str(exc)))
                if spec.name == chosen.name:
                    raise ValueError('The inner-selected model failed its outer fit: ' + str(exc)) from exc
        if set(groups[ti]) & set(groups[vi]):
            raise AssertionError('Training-validation group leakage')
        for partition, indices in [('Train', ti), ('Validation', vi)]:
            for j in indices:
                cv_rows.append({'Fold': fold, 'Partition': partition, 'Input_order': valid_indices[j] + 1,
                                'Independent_group': int(groups[j]), 'Selected_model': chosen.name})
        for j in vi:
            row_models[j], row_folds[j] = chosen.name, fold
    if not np.all(np.isfinite(pred)):
        raise ValueError('Incomplete outer predictions; no metric was silently computed on a selected subset.')
    emit('Outer validation complete. Selecting the deployment model within the full calibration set; unknown samples are not used for selection.')
    final_spec, search = selection_search(X, y, logr, groups, specs, seed + 999)
    inner_rows.extend(dict(row, Stage='Final_selection', Selected=row['Model'] == final_spec.name) for row in search)
    final_model = FoldModel(final_spec, seed).fit(X, logr - y)
    feature_rows.extend({'Stage': 'Final_fit', 'Model': final_spec.name, 'Feature': names[j]}
                        for j in final_model.columns)
    report = metrics(y, pred)
    comparison = [dict(report, Model='Nested_selected_strategy', Interpretation='Primary_nested_OOF')]
    for name, p in constants.items():
        comparison.append(dict(metrics(y, p), Model=name, Interpretation='Concentration_null_baseline'))
    for name, p in candidate_predictions.items():
        complete = np.all(np.isfinite(p))
        comparison.append(dict(metrics(y, p) if complete else {'N': int(np.count_nonzero(np.isfinite(p)))},
                               Model=name, Interpretation='Exploratory_fixed_candidate_OOF' if complete else 'Failed_in_some_folds'))
    null_rmse = min(metrics(y, p)['Log_RMSE'] for p in constants.values())
    flags = []
    if report['Log_spread_ratio'] < .35 or report['Pred_on_actual_slope'] < .25:
        flags.append('Concentration_trend_compression_warning')
    if report['Spearman'] <= 0:
        flags.append('No_positive_rank_relationship')
    if report['Log_R2'] <= 0 or report['Log_RMSE'] >= null_rmse or report['Spearman'] < .3:
        flags.append('Does_not_pass_basic_predictive_screen')
    if final_spec.kind in ('mean', 'median'):
        flags.append('Final_model_is_global_response_baseline_not_structure_learning')
    status = 'NOT_VALIDATED' if flags else 'PRELIMINARY_ONLY_external_validation_needed'
    notes.extend(['Status thresholds (spread .35, slope .25, rho .3) are diagnostic heuristics, not analytical validation criteria.',
                  'Fixed-candidate OOF scores are exploratory; selecting the best of them and quoting that score is optimistic.',
                  'P80 intervals use nested-strategy residuals, not compound-specific calibrated confidence intervals.',
                  'No residual-based outlier removal, target distribution matching, Huber concentration scaling or concentration clipping.',
                  'All numeric labels and measured responses must use consistent concentration units and Area/IS scales.',
                  'Descriptor range check is univariate only; it is not a complete applicability-domain test.'])
    known, by_input = [], {}
    for j, source_i in enumerate(valid_indices):
        out = dict(row_info(valid_records[j], source_i, 'Training'),
                   Actual_concentration=pow10(y[j]), Predicted_concentration=pow10(pred[j]),
                   Actual_log10_concentration=float(y[j]), Predicted_log10_concentration=float(pred[j]),
                   Fold_error=pow10(abs(pred[j] - y[j])), OOF_fold=row_folds[j],
                   Prediction_model=row_models[j], Prediction_status='Nested_OOF',
                   Constant_baseline_log10=float(constants['Constant_logC_mean'][j]))
        by_input[source_i] = out
    for i, r in enumerate(training_records):
        known.append(by_input.get(i, dict(row_info(r, i, 'Training'),
                                         Actual_concentration=r.get('Actual_concentration'),
                                         Prediction_status=audit[i]['Reason'])))
    unknown = []
    Xt = matrix(target_records, names)
    domains = final_model.domain(Xt)
    p80_log = float(np.quantile(np.abs(pred - y), .8))
    for i, r in enumerate(target_records):
        out = dict(row_info(r, i, 'Target'), Measured_ratio=r.get('Measured_ratio'),
                   Prediction_model=final_spec.name, Model_validation_status=status, **domains[i])
        ratio = number(r.get('Measured_ratio'))
        reason = ''
        if not np.isfinite(ratio) or ratio <= 0:
            reason = 'Missing_or_nonpositive_measured_ratio'
        elif norm(r.get('Formula_match')) in ('false', 'mismatch', 'no', 'fail'):
            reason = 'Explicit_formula_mismatch'
        elif domains[i]['Descriptor_domain'] == 'All_selected_missing':
            reason = 'All_selected_descriptors_missing'
        if not reason:
            logc = float(final_model.predict_logc(Xt[i:i+1], [math.log10(ratio)])[0])
            concentration = pow10(logc)
            if concentration is None:
                reason = 'Concentration_outside_numeric_float_range'
            else:
                out.update(Predicted_concentration=concentration, Predicted_log10_concentration=logc,
                           Predicted_log10_RRF=math.log10(ratio) - logc,
                           Empirical_P80_lower=pow10(logc - p80_log), Empirical_P80_upper=pow10(logc + p80_log),
                           Concentration_range_flag=('Outside_training_range_no_clipping' if logc < np.min(y) or logc > np.max(y) else 'Within_training_range'))
        out['Prediction_status'] = 'Skipped: ' + reason if reason else 'Predicted_exploratory'
        unknown.append(out)
        audit.append(dict(row_info(r, i, 'Target'), Status='Skipped' if reason else 'Predicted', Reason=reason))
    summary = {'Version': VERSION, 'Training_nonblank_input': len(training_records),
               'Training_maximum': 100, 'Training_used': len(y),
               'Training_skipped': len(training_records) - len(y),
               'Target_input': len(target_records), 'Target_predicted': sum('Predicted_concentration' in r for r in unknown),
               'Outer_folds': len(outer), 'Independent_groups': len(set(groups)),
               'Seed': seed, 'Final_model': final_spec.name, 'Numeric_scaling': final_spec.scaler,
               'Validation_status': status, 'Diagnostic_flags': '; '.join(flags),
               'Nested_OOF_metrics': report, 'Candidate_feature_count': len(names),
               'Selected_features': [names[j] for j in final_model.columns],
               'Final_config': asdict(final_spec), 'Notes': notes}
    return {'summary': summary, 'known': known, 'unknown': unknown, 'comparison': comparison,
            'row_audit': audit, 'cv_folds': cv_rows, 'inner_search': inner_rows,
            'feature_audit': feature_rows, 'actual_log': y, 'predicted_log': pred,
            'baseline_log': constants['Constant_logC_mean'], 'model': final_model,
            'candidate_features': names}

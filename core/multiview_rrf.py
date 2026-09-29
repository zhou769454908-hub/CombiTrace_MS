"""v18.46: existing-data-only, nested multiview effective response-factor models.

This module does NOT replace core.continuous_rrf (v18.45). It preserves the
physical offset log(C) = log(Area/IS) - log(RRF). It never rescales concentrations,
learns from unknown labels, deletes outliers by residual, or claims linearity was
experimentally verified. All ablations use the same outer/inner partitions.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, asdict

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge

from . import continuous_rrf as legacy
from .continuous_rrf import (number, norm, row_info, metrics, pow10, independent_groups,
                             splits, matrix, text)
from .training_limits import validate_training_count, is_nonblank_record
from .multiview_features import (NumericDesign, ComponentDesign, schema, coverage_rows,
                                 component_keys, block_of)

VERSION = '18.46-multiview-existing-data'
POLICY = {
    'Name': 'Balanced_error_rank_tail_v1',
    'RMSE_gate_relative_to_best': 1.20,
    'Weights': {'Normalized_log_RMSE': .45, 'Spearman_loss': .25,
                'Normalized_tail_RMSE': .15, 'Pairwise_discordance': .10,
                'Bounded_slope_deviation': .05},
    'Pairwise_min_log10_gap': math.log10(1.5),
    'Pairwise_gap_interpretation': 'Fixed diagnostic convention, not a detection limit',
    'Selection_data': 'Inner held-out predictions from training rows only',
    'Prediction_range_reward': 'None',
}
VIEW_LABELS = {
    'baseline': 'B0_Response_only',
    'structure': 'B1_Structure_LC',
    'observed': 'B2_Structure_LC_Observed',
    'abc': 'B3_Structure_LC_ABC',
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    view: str = 'structure'
    selector: str = 'all'
    scaler: str = 'standard'
    alpha: float = 10.0
    features: int = 16
    leaf: int = 3
    neighbors: int = 5
    interactions: bool = False
    component_pairs: bool = False
    rank_weight: float = .1


def default_specs():
    specs = [ModelSpec('Global_RRF_mean', 'mean', 'baseline'),
             ModelSpec('Global_RRF_median', 'median', 'baseline')]
    for view, prefix in [('structure', 'StructLC'), ('observed', 'Observed')]:
        specs.extend([
            ModelSpec(prefix + '_Ridge_all_a1', 'ridge', view, alpha=1),
            ModelSpec(prefix + '_Ridge_all_a10', 'ridge', view, alpha=10),
            ModelSpec(prefix + '_Ridge_all_a100', 'ridge', view, alpha=100),
            ModelSpec(prefix + '_Ridge_robust_a10', 'ridge', view, scaler='robust'),
            ModelSpec(prefix + '_Ridge_protected', 'ridge', view, selector='screen_protected'),
            ModelSpec(prefix + '_Interactions_a10', 'ridge', view, interactions=True),
            ModelSpec(prefix + '_Trees_all', 'trees', view, scaler='none'),
            ModelSpec(prefix + '_Local_RRF', 'local', view, selector='screen_protected'),
            ModelSpec(prefix + '_Pairwise_RRF', 'pairwise', view, selector='screen_protected'),
        ])
    specs.extend([
        ModelSpec('ABC_shrunken_main_a1', 'component', 'abc', alpha=1),
        ModelSpec('ABC_shrunken_main_a10', 'component', 'abc', alpha=10),
        ModelSpec('ABC_shrunken_pairs_a1', 'component', 'abc', alpha=1, component_pairs=True),
        ModelSpec('ABC_local_RRF_transfer', 'local_abc', 'abc', selector='screen_protected'),
    ])
    return specs


def more_metrics(y, p):
    m = metrics(y, p)
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) < 2 or not np.all(np.isfinite(p)):
        return m
    count = max(1, int(math.ceil(len(y) * .2)))
    order = np.argsort(y, kind='stable')
    low, high = order[:count], order[-count:]
    m['Low20_log_RMSE'] = float(np.sqrt(np.mean((p[low] - y[low]) ** 2)))
    m['High20_log_RMSE'] = float(np.sqrt(np.mean((p[high] - y[high]) ** 2)))
    m['Tail_log_RMSE'] = .5 * (m['Low20_log_RMSE'] + m['High20_log_RMSE'])
    a, b = np.triu_indices(len(y), 1)
    keep = np.abs(y[a] - y[b]) >= POLICY['Pairwise_min_log10_gap']
    a, b = a[keep], b[keep]
    if len(a):
        true_sign, pred_sign = np.sign(y[a] - y[b]), np.sign(p[a] - p[b])
        concordance = np.mean((true_sign == pred_sign) + .5 * (pred_sign == 0))
        m['Pairwise_concordance_pct'] = float(100 * concordance)
    else:
        m['Pairwise_concordance_pct'] = None
    m['Pairwise_pairs_evaluated'] = len(a)
    m['Top20_overlap_pct'] = float(100 * len(set(high) & set(np.argsort(p)[-count:])) / count)
    m['Within_5x_pct'] = float(100 * np.mean(np.abs(p - y) <= math.log10(5)))
    return m


def selection_score(m, null_rmse):
    scale = max(float(null_rmse), 1e-12)
    pair = m.get('Pairwise_concordance_pct')
    pair_loss = .5 if pair is None else 1 - pair / 100.0
    slope = m.get('Pred_on_actual_slope')
    slope_loss = 1.0 if slope is None else min(abs(slope - 1), 2) / 2
    return (.45 * m['Log_RMSE'] / scale + .25 * (1 - m['Spearman']) / 2 +
            .15 * m['Tail_log_RMSE'] / scale + .10 * pair_loss + .05 * slope_loss)


def choose_rows(rows, specs):
    allowed = {s.name for s in specs}
    usable = [r for r in rows if r['Model'] in allowed and not r.get('Error') and
              r.get('Log_RMSE') is not None and np.isfinite(r['Log_RMSE'])]
    if not usable:
        return None, {}
    best_error = min(r['Log_RMSE'] for r in usable)
    gate = POLICY['RMSE_gate_relative_to_best'] * best_error + 1e-12
    eligible = [r for r in usable if r['Log_RMSE'] <= gate]
    choice = min(eligible, key=lambda r: (r['Selection_score'], r['Log_RMSE'], r['Model']))
    spec = next(s for s in specs if s.name == choice['Model'])
    return spec, {'Best_inner_log_RMSE': best_error, 'Gate_log_RMSE': gate,
                  'Chosen_inner_log_RMSE': choice['Log_RMSE'], 'Chosen_score': choice['Selection_score']}


class PairwiseResponseRidge:
    """Response MSE + smooth concentration ordering + L2; no concentration map.

The ranking term uses only train-fold pairs with a >=1.5x true concentration
separation. The concentration offset log_ratio is fixed, not fitted. This is a
candidate objective, not a claim that ranking universally improves regression.
"""
    def fit(self, Z, rrf, logr, logc, alpha=10, rank_weight=.1):
        Z, rrf = np.asarray(Z, float), np.asarray(rrf, float)
        initial = Ridge(alpha=alpha).fit(Z, rrf)
        self.coef_, self.intercept_ = initial.coef_.copy(), float(initial.intercept_)
        a, b = np.triu_indices(len(Z), 1)
        keep = np.abs(logc[a] - logc[b]) >= POLICY['Pairwise_min_log10_gap']
        a, b = a[keep], b[keep]
        if not len(a):
            self.optimization_status = 'No_separated_pairs_Ridge_objective'
            return self
        delta = Z[a] - Z[b]
        offset = logr[a] - logr[b]
        sign = np.sign(logc[a] - logc[b])
        temperature = .2
        regularization = float(alpha) / len(Z)

        def objective(theta):
            w, intercept = theta[:-1], theta[-1]
            residual = Z.dot(w) + intercept - rrf
            margin = sign * (offset - delta.dot(w)) / temperature
            loss = (np.mean(residual ** 2) + regularization * np.dot(w, w) +
                    rank_weight * np.mean(np.logaddexp(0, -margin)))
            grad_w = 2 * Z.T.dot(residual) / len(Z) + 2 * regularization * w
            grad_w += rank_weight * delta.T.dot(expit(-margin) * sign) / (len(a) * temperature)
            return float(loss), np.r_[grad_w, 2 * residual.mean()]
        fit = minimize(objective, np.r_[self.coef_, self.intercept_], jac=True,
                       method='L-BFGS-B', options={'maxiter': 220, 'ftol': 1e-9})
        if not fit.success or not np.all(np.isfinite(fit.x)):
            raise ValueError('Pairwise optimizer did not converge: ' + str(fit.message))
        self.coef_, self.intercept_ = fit.x[:-1], float(fit.x[-1])
        self.optimization_status = 'Converged'
        return self

    def predict(self, Z):
        return Z.dot(self.coef_) + self.intercept_


class ResponseModel:
    def __init__(self, spec, names, seed=42):
        self.spec, self.names, self.seed = spec, list(names), seed
        self.design = None
        self.components = None
        self.selected_names = []

    def fit(self, records, logc, logr):
        rrf = np.asarray(logr) - np.asarray(logc)
        self.center = float(np.median(rrf) if self.spec.kind == 'median' else np.mean(rrf))
        self.fit_n = len(records)
        if self.spec.kind in ('mean', 'median'):
            return self
        view = 'observed' if self.spec.view == 'observed' else 'structure'
        self.design = NumericDesign(self.names, view, self.spec.selector, self.spec.scaler,
                                    self.spec.features, self.spec.interactions).fit(records, rrf)
        Z = self.design.transform(records)
        self.selected_names = list(self.design.output_names)
        if self.spec.view == 'observed' and not any(block_of(n) == 'Observed_ion_behavior'
                                                    for n in self.design.base_names):
            raise ValueError('No varying observed-ion descriptor in this train fold; observed view unavailable.')
        if self.spec.kind in ('component', 'local_abc'):
            self.components = ComponentDesign(self.spec.component_pairs).fit(records)
            if not self.components.output_names:
                raise ValueError('No varying A/B/C levels supported by >=2 distinct training combinations.')
            if self.spec.kind == 'component':
                Z = np.column_stack([Z, self.components.transform(records)])
                self.selected_names += self.components.output_names
            else:
                self.selected_names += ['ABC_role_similarity']
        if not Z.shape[1] and self.spec.kind != 'local_abc':
            raise ValueError('No usable varying feature in this train fold.')
        if not np.all(np.isfinite(Z)):
            raise ValueError('Nonfinite design (check descriptor magnitude and units).')
        if self.spec.kind in ('ridge', 'component'):
            self.estimator = Ridge(alpha=self.spec.alpha).fit(Z, rrf)
        elif self.spec.kind == 'trees':
            self.estimator = ExtraTreesRegressor(n_estimators=80, min_samples_leaf=self.spec.leaf,
                        max_features=.8, random_state=self.seed, n_jobs=1).fit(Z, rrf)
        elif self.spec.kind == 'pairwise':
            self.estimator = PairwiseResponseRidge().fit(Z, rrf, np.asarray(logr), np.asarray(logc),
                                                         self.spec.alpha, self.spec.rank_weight)
        elif self.spec.kind in ('local', 'local_abc'):
            self.train_Z, self.train_rrf = Z.copy(), rrf.copy()
            self.train_keys = [component_keys(r) for r in records]
        else:
            raise ValueError('Unknown model type: ' + self.spec.kind)
        return self

    def _transform(self, records):
        Z = self.design.transform(records)
        if self.spec.kind == 'component':
            Z = np.column_stack([Z, self.components.transform(records)])
        return Z

    def predict_rrf(self, records):
        if self.spec.kind in ('mean', 'median'):
            return np.full(len(records), self.center)
        Z = self._transform(records)
        if self.spec.kind not in ('local', 'local_abc'):
            return np.asarray(self.estimator.predict(Z)).reshape(-1)
        values = []
        for i, record in enumerate(records):
            distance = (np.mean((self.train_Z - Z[i]) ** 2, axis=1) if Z.shape[1]
                        else np.zeros(len(self.train_Z)))
            if self.spec.kind == 'local_abc':
                key = component_keys(record)
                shared = np.array([sum(bool(a) and a == b for a, b in zip(key, k))
                                   for k in self.train_keys], dtype=float)
                distance = distance + .75 * (3 - shared) / 3
            idx = np.argsort(distance, kind='stable')[:min(self.spec.neighbors, len(distance))]
            weights = np.exp(-np.minimum(distance[idx], 700) / 2.0)
            # Shrunken LOCAL RRF, never local average concentration. Distant
            # neighbors revert toward a response prior; no target-based tuning.
            values.append(float((weights.dot(self.train_rrf[idx]) + .5 * self.center) /
                                (weights.sum() + .5)))
        return np.asarray(values)

    def predict_logc(self, records, log_ratio=None):
        logr = (np.asarray(log_ratio, float) if log_ratio is not None else
                np.log10([number(r.get('Measured_ratio')) for r in records]))
        return logr - self.predict_rrf(records)

    def domain(self, records):
        if self.design is None:
            return [{'Descriptor_domain': 'Not_assessed_global_response_baseline',
                     'Selected_missing': 0, 'Out_of_range_features': 0} for _ in records]
        result = self.design.domain(records)
        if self.components is not None:
            for row, c in zip(result, self.components.domain(records)):
                row.update(c)
        return result


def search(records, logc, logr, groups, names, specs, seed):
    partitions = splits(len(records), groups, 3, seed)
    null = np.full(len(records), np.nan)
    for ti, vi in partitions:
        null[vi] = np.mean(logc[ti])
    null_rmse = more_metrics(logc, null)['Log_RMSE']
    rows = []
    for spec in specs:
        predictions = np.full(len(records), np.nan)
        error = ''
        for ti, vi in partitions:
            try:
                model = ResponseModel(spec, names, seed).fit([records[i] for i in ti], logc[ti], logr[ti])
                p = model.predict_logc([records[i] for i in vi], logr[vi])
                if not np.all(np.isfinite(p)):
                    raise ValueError('Nonfinite inner predictions')
                predictions[vi] = p
            except Exception as exc:
                error = type(exc).__name__ + ': ' + str(exc)
                break
        row = {'Model': spec.name, 'View': VIEW_LABELS.get(spec.view, spec.view), 'Error': error}
        if not error and np.all(np.isfinite(predictions)):
            m = more_metrics(logc, predictions)
            row.update(m, Selection_score=selection_score(m, null_rmse))
        else:
            row['Log_RMSE'] = None
        rows.append(row)
    chosen, rule = choose_rows(rows, specs)
    if chosen is None:
        raise ValueError('All candidates failed inside training-only CV. See feature/group input checks.')
    for row in rows:
        row['Within_global_error_gate'] = bool(row.get('Log_RMSE') is not None and
                                               row['Log_RMSE'] <= rule['Gate_log_RMSE'])
        row['Primary_selected'] = row['Model'] == chosen.name
        row['Gate_log_RMSE'] = rule['Gate_log_RMSE']
    return chosen, rows, partitions


def _feature_audit(model, stage):
    rows = []
    if model.design is not None:
        rows.extend(dict(r, Stage=stage, Model=model.spec.name) for r in model.design.audit)
    base = set(model.design.base_names) if model.design else set()
    for name in model.selected_names:
        if name not in base:
            rows.append({'Stage': stage, 'Model': model.spec.name, 'Feature': name,
                         'Block': 'Engineered_interaction_or_ABC', 'Selected_numeric': True,
                         'Reason': 'Fitted_inside_training_fold'})
    return rows


def _view_selections(rows, specs):
    out = {}
    for view in VIEW_LABELS:
        subset = [s for s in specs if s.view == view]
        chosen, _ = choose_rows(rows, subset)
        if chosen is not None:
            out[VIEW_LABELS[view]] = chosen
    return out


def _validation_status(report, null_rmse, final_spec):
    # Keep v18.45 basic criteria, never loosen in order to turn the status green.
    flags = []
    if report['Log_spread_ratio'] < .35 or report['Pred_on_actual_slope'] < .25:
        flags.append('Concentration_trend_compression_warning')
    if report['Spearman'] <= 0:
        flags.append('No_positive_rank_relationship')
    if report['Log_R2'] <= 0 or report['Log_RMSE'] >= null_rmse or report['Spearman'] < .3:
        flags.append('Does_not_pass_basic_predictive_screen')
    if final_spec.kind in ('mean', 'median'):
        flags.append('Final_model_is_global_response_baseline_not_structure_learning')
    return ('NOT_VALIDATED' if flags else 'PRELIMINARY_ONLY_external_validation_needed'), flags


def run_multiview(training_records, target_records, descriptor_names, *, seed=42, folds=5,
                  specs=None, compare_legacy=True, progress=None):
    emit = progress or (lambda s: None)
    count = validate_training_count(training_records)
    specs = list(default_specs() if specs is None else specs)
    if not specs or len({s.name for s in specs}) != len(specs):
        raise ValueError('Provide unique nonempty model candidate names.')
    if seed < 0 or seed > 2 ** 32 - 10000:
        raise ValueError('Seed must be a nonnegative integer below 2**32 - 10000.')
    names = schema(training_records, descriptor_names)
    records, indices, audit = [], [], []
    for i, record in enumerate(training_records):
        ratio, conc = number(record.get('Measured_ratio')), number(record.get('Actual_concentration'))
        reason = ''
        if not is_nonblank_record(record):
            reason = 'Blank_row'
        elif not np.isfinite(ratio) or ratio <= 0:
            reason = 'Missing_or_nonpositive_measured_ratio'
        elif not np.isfinite(conc) or conc <= 0:
            reason = 'Missing_or_nonpositive_known_concentration'
        elif norm(record.get('Formula_match')) in ('false', 'mismatch', 'no', 'fail'):
            reason = 'Explicit_formula_mismatch'
        if not reason:
            records.append(record); indices.append(i)
        audit.append(dict(row_info(record, i, 'Training'), Status='Skipped' if reason else 'Used', Reason=reason))
    if len(records) < 24:
        raise ValueError('This nested workflow needs at least 24 usable standards (maximum 100 input); found %d.' % len(records))
    y = np.log10([number(r['Actual_concentration']) for r in records])
    logr = np.log10([number(r['Measured_ratio']) for r in records])
    if np.std(y) < 1e-10:
        raise ValueError('Known concentrations are constant; cannot validate concentration ordering.')
    groups, notes = independent_groups(records)
    if len(set(groups)) < 3:
        raise ValueError('Need at least three independent injection/compound groups for nested CV; no leakage fallback.')
    outer = splits(len(records), groups, folds, seed)
    pred = np.full(len(y), np.nan)
    null_mean = np.full(len(y), np.nan)
    null_median = np.full(len(y), np.nan)
    view_pred = {v: np.full(len(y), np.nan) for v in VIEW_LABELS.values()}
    fixed = {s.name: np.full(len(y), np.nan) for s in specs}
    legacy_pred = np.full(len(y), np.nan)
    inner_rows, cv_rows, feature_audit, fold_metrics, legacy_rows = [], [], [], [], []
    fold_models, row_folds, primary_domain = {}, {}, {}
    model_by_view = {}
    # Original v18.45 schema / candidates, evaluated with the exact SAME rows
    # and splits. It is a comparator only and cannot override the new strategy.
    legacy_names = legacy.feature_names(records, descriptor_names)
    legacy_X = matrix(records, legacy_names)
    legacy_specs = legacy.default_specs()
    for fold, (ti, vi) in enumerate(outer, 1):
        emit('Outer fold %d/%d: joint error/ranking selection and same-fold feature-view comparisons.' % (fold, len(outer)))
        train, validation = [records[i] for i in ti], [records[i] for i in vi]
        chosen, rows, inner_partitions = search(train, y[ti], logr[ti], groups[ti], names, specs, seed + fold)
        inner_rows.extend(dict(r, Stage='Outer_%d' % fold) for r in rows)
        choices = _view_selections(rows, specs)
        null_mean[vi], null_median[vi] = np.mean(y[ti]), np.median(y[ti])
        fitted = {}
        for spec in specs:
            try:
                model = ResponseModel(spec, names, seed).fit(train, y[ti], logr[ti])
                p = model.predict_logc(validation, logr[vi])
                if not np.all(np.isfinite(p)):
                    raise ValueError('Nonfinite held-out predictions')
                fixed[spec.name][vi] = p
                fitted[spec.name] = model
            except Exception as exc:
                notes.append('Outer_%d/%s: %s' % (fold, spec.name, str(exc)))
                if spec.name == chosen.name:
                    raise ValueError('Inner-selected model failed outer fitting: ' + str(exc)) from exc
        model = fitted[chosen.name]
        pred[vi] = fixed[chosen.name][vi]
        feature_audit.extend(_feature_audit(model, 'Outer_%d' % fold))
        for j, d in zip(vi, model.domain(validation)):
            primary_domain[j] = d
            fold_models[j], row_folds[j] = chosen.name, fold
        for view, choice in choices.items():
            if choice.name in fitted:
                p = fixed[choice.name][vi]
                view_pred[view][vi] = p
                model_by_view[(fold, view)] = choice.name
                fold_metrics.append(dict(more_metrics(y[vi], p), Fold=fold, Strategy=view, Model=choice.name))
        fold_metrics.append(dict(more_metrics(y[vi], pred[vi]), Fold=fold,
                                 Strategy='Primary_nested_multiview', Model=chosen.name))
        if compare_legacy:
            try:
                old_spec, old_rows = legacy.selection_search(legacy_X[ti], y[ti], logr[ti], groups[ti],
                                                             legacy_specs, seed + fold)
                legacy_rows.extend(dict(r, Stage='Outer_%d' % fold, Selected=r['Model'] == old_spec.name)
                                   for r in old_rows)
                old_model = legacy.FoldModel(old_spec, seed).fit(legacy_X[ti], logr[ti] - y[ti])
                legacy_pred[vi] = old_model.predict_logc(legacy_X[vi], logr[vi])
                fold_metrics.append(dict(more_metrics(y[vi], legacy_pred[vi]), Fold=fold,
                                         Strategy='Legacy_v18_45_nested', Model=old_spec.name))
            except Exception as exc:
                notes.append('Legacy comparator fold %d unavailable: %s' % (fold, str(exc)))
        if set(groups[ti]) & set(groups[vi]):
            raise AssertionError('Independent group leakage')
        for partition, ids in [('Train', ti), ('Validation', vi)]:
            for j in ids:
                cv_rows.append({'Stage': 'Outer', 'Outer_fold': fold, 'Inner_fold': '',
                                'Partition': partition, 'Input_order': indices[j] + 1,
                                'Independent_group': int(groups[j]), 'Selected_model': chosen.name})
        for inner_fold, (ii, jj) in enumerate(inner_partitions, 1):
            for partition, ids in [('Train', ti[ii]), ('Validation', ti[jj])]:
                for j in ids:
                    cv_rows.append({'Stage': 'Inner', 'Outer_fold': fold, 'Inner_fold': inner_fold,
                                    'Partition': partition, 'Input_order': indices[j] + 1,
                                    'Independent_group': int(groups[j]), 'Selected_model': chosen.name})
    if not np.all(np.isfinite(pred)):
        raise ValueError('Incomplete primary nested predictions; do not report subset metrics.')
    emit('Final model selection uses inner validation on known standards only; target samples are not used.')
    final_spec, final_rows, final_partitions = search(records, y, logr, groups, names, specs, seed + 999)
    inner_rows.extend(dict(r, Stage='Final_selection') for r in final_rows)
    final_model = ResponseModel(final_spec, names, seed).fit(records, y, logr)
    feature_audit.extend(_feature_audit(final_model, 'Final_fit'))
    for f, (ti, vi) in enumerate(final_partitions, 1):
        for partition, ids in [('Train', ti), ('Validation', vi)]:
            for j in ids:
                cv_rows.append({'Stage': 'Final_inner', 'Outer_fold': '', 'Inner_fold': f,
                                'Partition': partition, 'Input_order': indices[j] + 1,
                                'Independent_group': int(groups[j]), 'Selected_model': final_spec.name})
    report = more_metrics(y, pred)
    comparison = [dict(report, Model='Primary_nested_multiview', Interpretation='Primary_nested_OOF')]
    for name, p in [('Constant_logC_mean', null_mean), ('Constant_logC_median', null_median)]:
        comparison.append(dict(more_metrics(y, p), Model=name, Interpretation='Concentration_null_baseline'))
    views = []
    for view, p in view_pred.items():
        complete = np.all(np.isfinite(p))
        row = dict(more_metrics(y, p) if complete else {'N': int(np.isfinite(p).sum())},
                   Model=view, Interpretation='Within_view_nested_OOF' if complete else 'View_unavailable_in_some_folds')
        views.append(row); comparison.append(row)
    if compare_legacy:
        complete = np.all(np.isfinite(legacy_pred))
        comparison.append(dict(more_metrics(y, legacy_pred) if complete else {'N': int(np.isfinite(legacy_pred).sum())},
                               Model='Legacy_v18_45_nested', Interpretation='Same_rows_splits_old_strategy' if complete else 'Legacy_comparator_incomplete'))
    for name, p in fixed.items():
        complete = np.all(np.isfinite(p))
        comparison.append(dict(more_metrics(y, p) if complete else {'N': int(np.isfinite(p).sum())},
                          Model=name, Interpretation='Exploratory_fixed_candidate_OOF' if complete else 'Candidate_unavailable_in_some_folds'))
    # Incremental information comparisons are diagnostic, not causal estimates.
    for view in views:
        if view['Model'] == 'B0_Response_only':
            continue
        ref_name = 'B0_Response_only' if view['Model'] == 'B1_Structure_LC' else 'B1_Structure_LC'
        ref = next((r for r in views if r['Model'] == ref_name), None)
        if ref and 'Log_RMSE' in ref and 'Log_RMSE' in view and view['Model'] != ref_name:
            view.update(Compared_to=ref_name, Delta_log_RMSE=view['Log_RMSE'] - ref['Log_RMSE'],
                        Delta_Spearman=view['Spearman'] - ref['Spearman'],
                        Interpretation_note='Same rows/splits; added features+within-view model choice, not proof of causality')
    null_rmse = min(more_metrics(y, p)['Log_RMSE'] for p in (null_mean, null_median))
    status, flags = _validation_status(report, null_rmse, final_spec)
    p80 = float(np.quantile(np.abs(pred - y), .8))
    by_input = {}
    for j, i in enumerate(indices):
        row = dict(row_info(records[j], i, 'Training'),
                   Actual_concentration=pow10(y[j]), Predicted_concentration=pow10(pred[j]),
                   Actual_log10_concentration=float(y[j]), Predicted_log10_concentration=float(pred[j]),
                   Fold_error=pow10(abs(pred[j] - y[j])), OOF_fold=row_folds[j],
                   Prediction_model=fold_models[j], Prediction_status='Nested_OOF',
                   Model_validation_status=status, Constant_baseline_log10=float(null_mean[j]),
                   Legacy_v18_45_log10=legacy_pred[j] if compare_legacy else None,
                   Measured_ratio=number(records[j]['Measured_ratio']), **primary_domain[j])
        for view, p in view_pred.items():
            row[view + '_log10_prediction'] = float(p[j]) if np.isfinite(p[j]) else None
        by_input[i] = row
    known = [by_input.get(i, dict(row_info(r, i, 'Training'),
               Actual_concentration=r.get('Actual_concentration'), Prediction_status=audit[i]['Reason']))
             for i, r in enumerate(training_records)]
    # Only after every model decision is complete may target values be inspected.
    domains = final_model.domain(target_records)
    unknown = []
    good_target_indices = []
    for i, record in enumerate(target_records):
        ratio = number(record.get('Measured_ratio'))
        domain = domains[i]
        out = dict(row_info(record, i, 'Target'), Measured_ratio=ratio,
                   Prediction_model=final_spec.name, Model_validation_status=status, **domain)
        reason = ''
        if not np.isfinite(ratio) or ratio <= 0:
            reason = 'Missing_or_nonpositive_measured_ratio'
        elif norm(record.get('Formula_match')) in ('false', 'mismatch', 'no', 'fail'):
            reason = 'Explicit_formula_mismatch'
        elif domain['Descriptor_domain'] == 'All_selected_missing' and not domain.get('ABC_supported_roles', 0):
            reason = 'All_selected_descriptors_missing'
        if not reason:
            logc = float(final_model.predict_logc([record], [math.log10(ratio)])[0])
            c = pow10(logc)
            if c is None:
                reason = 'Concentration_outside_numeric_float_range'
            else:
                good_target_indices.append(i)
                out.update(Predicted_concentration=c, Predicted_log10_concentration=logc,
                           Predicted_log10_RRF=math.log10(ratio) - logc,
                           Empirical_P80_lower=pow10(logc - p80), Empirical_P80_upper=pow10(logc + p80),
                           Concentration_range_flag='Outside_training_range_no_clipping' if logc < y.min() or logc > y.max() else 'Within_training_range')
        out['Prediction_status'] = 'Skipped: ' + reason if reason else 'Predicted_exploratory'
        unknown.append(out)
        audit.append(dict(row_info(record, i, 'Target'), Status='Skipped' if reason else 'Predicted', Reason=reason))
    # Deploy the independently inner-selected view models as side-by-side local
    # diagnostics only, never select by the spread of their unknown predictions.
    final_views = _view_selections(final_rows, specs)
    for view, spec in final_views.items():
        try:
            model = final_model if spec.name == final_spec.name else ResponseModel(spec, names, seed).fit(records, y, logr)
            subset = [target_records[i] for i in good_target_indices]
            p = model.predict_logc(subset) if subset else []
            view_domains = model.domain(subset)
            for i, value, d in zip(good_target_indices, p, view_domains):
                if d['Descriptor_domain'] == 'All_selected_missing' and not d.get('ABC_supported_roles', 0):
                    continue
                unknown[i][view + '_log10_prediction'] = float(value) if np.isfinite(value) else None
        except Exception as exc:
            notes.append('Final diagnostic view %s unavailable: %s' % (view, str(exc)))
    for row in unknown:
        values = [number(row.get(view + '_log10_prediction')) for view in VIEW_LABELS.values()]
        values = [v for v in values if np.isfinite(v)]
        row['Diagnostic_views_available'] = len(values)
        row['Between_view_log10_span'] = max(values) - min(values) if len(values) >= 2 else None
        row['Between_view_disagreement'] = ('Over_1_log10_review' if len(values) >= 2 and max(values) - min(values) > 1 else '')
    coverage = coverage_rows(records, target_records, names)
    observed_present = [r for r in coverage if r['Block'] == 'Observed_ion_behavior' and r['Availability'] == 'Available']
    if not observed_present:
        notes.append('Observed-ion fields are absent/constant/insufficient in training; no matrix or ion-behavior evidence was invented.')
    stability = []
    counts = Counter(fold_models[j] for j in range(len(y)))
    for name in sorted(counts):
        fs = sorted({row_folds[j] for j in range(len(y)) if fold_models[j] == name})
        stability.append({'Model': name, 'Outer_folds_selected': len(fs),
                          'Outer_folds_total': len(outer), 'Held_out_rows': counts[name]})
    group_rows = []
    for group in sorted(set(groups)):
        idx = np.flatnonzero(groups == group)
        group_rows.append(dict(more_metrics(y[idx], pred[idx]), Independent_group=int(group),
                              Description='Posthoc group residual diagnostic; not a model-selection criterion'))
    notes.extend([
        'Existing-data-only workflow. Approximate linear response is an assumption, not tested or guaranteed by internal standardization.',
        'No new experiments, uploads, online inference or external descriptor downloads are required by this module.',
        'Status thresholds are unchanged from v18.45 and remain heuristic screens, not analytical method validation.',
        'P80 is the nested-strategy residual range; not per-compound confidence and no external coverage guarantee.',
        'All feature screening, interactions, scaling and categorical support are fitted inside the relevant training fold.',
        'Feature coverage, group residuals, fixed candidate OOF and view deltas are diagnostics; do not select the best posthoc score and call it unbiased.',
        'Unknown concentrations and unknown distributions do not influence model selection. No automatic residual-based deletion or range stretching.',
        'Component categories retain index+label+formula. Unseen/rare roles shrink toward prior; no product formula-only merging.',
        'Univariate descriptor ranges and component support are incomplete applicability checks. Passing them does not prove transfer to new matrices.',
        'Repeatedly revising a workflow after seeing its OOF plots can still create dataset-level optimism; internal CV is not an independent external test.',
    ])
    summary = {'Version': VERSION, 'Training_maximum': 100, 'Training_nonblank_input': count,
               'Training_used': len(y), 'Training_skipped': len(training_records) - len(y),
               'Target_input': len(target_records), 'Target_predicted': len(good_target_indices),
               'Outer_folds': len(outer), 'Independent_groups': len(set(groups)), 'Seed': seed,
               'Final_model': final_spec.name, 'Final_config': asdict(final_spec),
               'Final_view': VIEW_LABELS[final_spec.view], 'Numeric_scaling': final_spec.scaler,
               'Validation_status': status, 'Diagnostic_flags': '; '.join(flags),
               'Nested_OOF_metrics': report, 'Selection_policy': POLICY,
               'Candidate_feature_count': len(names), 'Selected_feature_count': len(final_model.selected_names),
               'Selected_features': final_model.selected_names, 'Observed_features_available': len(observed_present),
               'Compare_v18_45_same_splits': bool(compare_legacy),
               'Approximate_linearity': 'Assumed_not_experimentally_verified_in_this_workflow',
               'Notes': notes}
    return {'summary': summary, 'known': known, 'unknown': unknown, 'comparison': comparison,
            'row_audit': audit, 'cv_folds': cv_rows, 'inner_search': inner_rows,
            'feature_audit': feature_audit, 'feature_coverage': coverage,
            'view_comparison': views, 'fold_metrics': fold_metrics, 'model_stability': stability,
            'group_residuals': group_rows, 'legacy_inner_search': legacy_rows,
            'actual_log': y, 'predicted_log': pred, 'baseline_log': null_mean,
            'legacy_log': legacy_pred if compare_legacy else None, 'view_predictions': view_pred,
            'model': final_model, 'candidate_features': names}

"""v18.46 fold-local multiview descriptors; no raw response/label as an input.

Feature schema is read from training keys only. All numerical decisions (coverage,
scaling, selection, interactions and categorical support) are fitted per fold.
This module does not calculate missing experimental quantities or read RAW files.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler

from .continuous_rrf import allowed_feature, text, norm, number, matrix
from .response_predictor import parse_combo

# These measured, dimensionless features are not absolute target response.
OBSERVED = (
    'Observed_Fragility_Index', 'Adduct_Cluster_Proneness_Index',
    'Primary_Ion_Fraction', 'Ion_Form_Diversity_Count',
    'Primary_Adduct_Fraction', 'Accepted_Channel_Count', 'Summed_Channel_Count',
    'Br_Fragment_Fraction',
    'Fragment_Fraction', 'Adduct_Fraction', 'Cluster_Fraction',
)
CONDITIONS = (
    'Apex_RT_min', 'Gradient_delay_min', 'Effective_gradient_time_min',
    'Mobile_phase_A_pct', 'Mobile_phase_B_pct', 'B_slope_pct_per_min',
    'Gradient_segment_start_min', 'Gradient_segment_end_min',
)
PROTECTED = (
    'MolLogP', 'TPSA', 'AcidicSiteCount_proxy', 'BasicSiteCount_proxy',
    'NegESI_DeprotonatableSiteCount_proxy', 'NegESI_AnionChargeDelocalization_proxy',
    'NegESI_AnionStability_proxy', 'GasteigerChargeRange',
    'GasteigerNegativeChargeAbsSum', 'HBD', 'HBA', 'FormalCharge',
    'Mobile_phase_B_pct', 'Apex_RT_min', 'Published_Apex_pH',
    'NegESI_pH_minus_pKa_proxy', 'Primary_Ion_Fraction',
    'Observed_Fragility_Index', 'Adduct_Cluster_Proneness_Index',
)
_OBSERVED_KEYS = {norm(v) for v in OBSERVED}
_CONDITION_KEYS = {norm(v) for v in CONDITIONS}
_METADATA = {
    'sourcerow', 'sourceindex', 'index', 'row', 'rowid', 'compoundid',
    'masterindex', 'productid', 'concentrationunit', 'dilutionfactor',
    'injectionorder', 'batchid', 'sampleid', 'peakindex', 'peaknumber',
    'concentrationlevel', 'level', 'class', 'label', 'logc', 'log10c',
    'yield', 'purity', 'response', 'rrf', 'log10rf', 'responsecorrectionindex',
    'isarearsd', 'ratiois', 'arearatiomean', 'arearatiorsd', 'arearatiostd',
}


def safe_feature(name):
    key = norm(name)
    if key in _OBSERVED_KEYS:
        return True
    if not allowed_feature(name) or key in _METADATA:
        return False
    if key.endswith(('smiles', 'formula', 'hash', 'status', 'warnings', 'comment')):
        return False
    if any(t in key for t in ('concentration', 'yield', 'predicted', 'actual',
                              'estimated', 'logrrf', 'responsefactor', 'logratio',
                              'arearatio', 'measuredratio', 'knownconc', 'log10conc',
                              'Concentration', 'Yield', 'prediction', 'Measured_concentration', 'Out_of_fold', 'Rank')):
        return False
    # Raw analyte/IS intensities, TIC and ranks are diagnostic only. Molecular
    # surface AREA descriptors are legitimate structural descriptors.
    if any(t in key for t in ('peakarea', 'isarea', 'summedarea', 'totalarea',
                              'peakheight', 'tic', 'intensity', 'areais')):
        if not any(t in key for t in ('aromatic', 'aliphatic', 'surface', 'topological')):
            return False
    if (key.endswith('area') or key.startswith(('quantarea', 'areamean', 'arearsd'))) and not any(
            t in key for t in ('surface', 'molecular', 'polar', 'vsa')):
        return False
    return True


def block_of(name):
    key = norm(name)
    if key in _OBSERVED_KEYS or key.startswith(('observed', 'ionform', 'primaryion',
                                                'adductcluster', 'fragmentfraction')):
        return 'Observed_ion_behavior'
    if (key in _CONDITION_KEYS or key.startswith(('publishedapex', 'publishedeluent',
            'mobilephase', 'gradient', 'effectivegradient', 'bslope'))):
        return 'LC_conditions'
    if key.startswith('negesi') and any(t in key for t in (
            'phminus', 'solutionionized', 'ionizationdelocal', 'organicfraction',
            'surfacetension', 'viscosity', 'polarityionization', 'hydrophobicion')):
        return 'Structure_condition_proxy'
    return 'Structure'


def schema(records, proposed):
    present = set().union(*(r.keys() for r in records)) if records else set()
    # Include cached observed descriptors even when they occur before the old
    # Product_Master_Formula_Key / Warnings descriptor window.
    ordered = list(dict.fromkeys(list(proposed) + ['Exact_mass', 'DBE'] + list(CONDITIONS) + list(OBSERVED)))
    return [n for n in ordered if n in present and safe_feature(n)]


def component_keys(record):
    """A/B/C index + label + formula; never use product formula as identity."""
    parts = parse_combo(text(record.get('Combo')))
    out = []
    for role in ('A', 'B', 'C'):
        p = parts.get(role, {})
        if p and p.get('index') is not None:
            out.append('%s#%s|%s|%s' % (role, p['index'], text(p.get('label')),
                                         text(p.get('formula'))))
        else:
            out.append('')
    return tuple(out)


class NumericDesign:
    """Fold-only median, selection, scale and a bounded interaction expansion."""
    def __init__(self, names, view='structure', selector='all', scaler='standard',
                 max_features=16, interactions=False):
        self.names = list(names)
        self.view, self.selector, self.scaler_kind = view, selector, scaler
        self.max_features, self.interactions = max_features, interactions
        self.indices = np.array([], dtype=int)
        self.audit = []

    def fit(self, records, log_rrf):
        X = matrix(records, self.names)
        cols, meds, filled = [], [], []
        for j, name in enumerate(self.names):
            reason = ''
            v = X[:, j]
            ok = np.isfinite(v)
            if self.view == 'structure' and block_of(name) == 'Observed_ion_behavior':
                reason = 'Not_in_this_view'
            elif ok.sum() < max(3, int(math.ceil(len(records) * .5))):
                reason = 'Insufficient_finite_training_values'
            if not reason:
                median = float(np.median(v[ok]))
                values = np.where(ok, v, median)
                if np.ptp(values) <= np.finfo(float).eps * max(1e-300, np.max(np.abs(values))):
                    reason = 'Constant_in_training_fold'
            self.audit.append({'Feature': name, 'Block': block_of(name),
                               'Train_finite_count': int(ok.sum()),
                               'Reason': reason or 'Eligible'})
            if not reason:
                cols.append(j); meds.append(median); filled.append(values)
        self.medians = np.asarray(meds, float)
        if not cols:
            self.scaler = None
            self.minimum = self.maximum = np.empty(0)
            self.base_names = self.output_names = []
            self.pairs = []
            return self
        Z = np.column_stack(filled)
        if self.selector == 'screen_protected':
            center = Z - Z.mean(axis=0)
            center /= np.maximum(np.linalg.norm(center, axis=0), 1e-300)
            target = np.asarray(log_rrf) - np.mean(log_rrf)
            scores = np.abs(center.T.dot(target) / max(np.linalg.norm(target), 1e-300))
            score_order = np.argsort(-scores, kind='stable').tolist()
            priorities = {norm(name): i for i, name in enumerate(PROTECTED)}
            protected = [k for k, j in enumerate(cols) if norm(self.names[j]) in priorities]
            protected.sort(key=lambda k: priorities[norm(self.names[cols[k]])])
            limit = min(max(4, len(records) // 3), self.max_features)
            selected = protected[:max(1, limit // 2)]
            selected += [k for k in score_order if k not in selected][:limit - len(selected)]
        else:
            selected = list(range(len(cols)))
        self.indices = np.asarray(cols, int)[selected]
        self.medians = self.medians[selected]
        self.base_names = [self.names[j] for j in self.indices]
        selected_set = set(self.base_names)
        for r in self.audit:
            r['Selected_numeric'] = r['Feature'] in selected_set
            if r['Reason'] == 'Eligible' and not r['Selected_numeric']:
                r['Reason'] = 'Not_selected_by_fold_screen'
        Z = Z[:, selected]
        self.minimum, self.maximum = Z.min(axis=0), Z.max(axis=0)
        self.scaler = (RobustScaler().fit(Z) if self.scaler_kind == 'robust' else
                       StandardScaler().fit(Z) if self.scaler_kind == 'standard' else None)
        Z = self.scaler.transform(Z) if self.scaler is not None else Z
        self.pairs = []
        self.output_names = list(self.base_names)
        if self.interactions:
            # Protect plausible chemistry/condition seeds from a univariate y
            # filter. Max 10 seeds / 45 interactions; no degree-3 explosion.
            priority = {norm(name): i for i, name in enumerate(PROTECTED)}
            seeds = [j for j, name in enumerate(self.base_names) if norm(name) in priority]
            seeds.sort(key=lambda j: priority[norm(self.base_names[j])])
            # Generic fallback permits testing with user-specific descriptors;
            # these products are statistical features, not measured chemistry.
            if len(seeds) < 2:
                seeds = sorted(range(len(self.base_names)), key=lambda j: self.base_names[j])[:8]
            seeds = seeds[:10]
            values, pairs = [], []
            for ii, a in enumerate(seeds):
                for b in seeds[ii + 1:]:
                    v = Z[:, a] * Z[:, b]
                    if np.all(np.isfinite(v)) and np.std(v) > 1e-12:
                        pairs.append((a, b)); values.append(v)
            if pairs:
                products = np.column_stack(values)
                self.pair_scaler = StandardScaler().fit(products)
                self.pairs = pairs
                self.output_names += ['Interaction[%s * %s]' % (self.base_names[a], self.base_names[b])
                                      for a, b in pairs]
        return self

    def transform(self, records):
        if not len(self.indices):
            return np.empty((len(records), 0))
        Z = matrix(records, self.names)[:, self.indices]
        Z = np.where(np.isfinite(Z), Z, self.medians)
        if self.scaler is not None:
            Z = self.scaler.transform(Z)
        if self.pairs:
            products = np.column_stack([Z[:, a] * Z[:, b] for a, b in self.pairs])
            Z = np.column_stack([Z, self.pair_scaler.transform(products)])
        return Z

    def domain(self, records):
        X = matrix(records, self.names)[:, self.indices]
        result = []
        for row in X:
            ok = np.isfinite(row)
            missing = int(np.count_nonzero(~ok))
            outside = int(np.count_nonzero(ok & ((row < self.minimum) | (row > self.maximum))))
            label = ('No_numeric_features' if not len(row) else
                     'All_selected_missing' if missing == len(row) else
                     'Univariate_range_warning' if outside else 'Inside_univariate_ranges')
            result.append({'Selected_missing': missing, 'Out_of_range_features': outside,
                           'Descriptor_domain': label})
        return result


class ComponentDesign:
    """Sparse category effects with role/pair shrinkage and unseen-level audit."""
    def __init__(self, pairs=False):
        self.use_pairs = pairs

    def fit(self, records):
        keys = [component_keys(r) for r in records]
        # Only independent full combinations count toward component support.
        unique = set(keys)
        self.support = [Counter(k[j] for k in unique if k[j]) for j in range(3)]
        self.levels = [(j, k) for j, counts in enumerate(self.support)
                       for k in sorted(counts) if 2 <= counts[k] < len(unique)]
        self.pair_support = {}
        self.pairs = []
        for a, b in ((0, 1), (0, 2), (1, 2)):
            counts = Counter((k[a], k[b]) for k in unique if k[a] and k[b])
            self.pair_support[(a, b)] = counts
            if self.use_pairs:
                self.pairs += [(a, b, v) for v in sorted(counts) if 3 <= counts[v] < len(unique)]
        self.output_names = ['Component[%s]' % k for _, k in self.levels]
        self.output_names += ['ComponentPair[%s + %s]' % v for _, _, v in self.pairs]
        return self

    def transform(self, records):
        keys = [component_keys(r) for r in records]
        cols = [np.array([float(k[j] == value) for k in keys]) / math.sqrt(5.0)
                for j, value in self.levels]
        cols += [np.array([float((k[a], k[b]) == value) for k in keys]) / math.sqrt(20.0)
                 for a, b, value in self.pairs]
        return np.column_stack(cols) if cols else np.empty((len(keys), 0))

    def domain(self, records):
        rows = []
        for record in records:
            keys = component_keys(record)
            support = [self.support[j].get(k, 0) if k else 0 for j, k in enumerate(keys)]
            supported = sum(v >= 2 for v in support)
            missing = sum(not k for k in keys)
            rows.append({'ABC_supported_roles': supported, 'ABC_missing_roles': missing,
                         'ABC_min_training_support': min(support),
                         'ABC_domain': 'All_roles_supported' if supported == 3 else
                            'Missing_or_unseen_roles_use_shrunken_prior'})
        return rows


def coverage_rows(training, targets, names):
    """Post-selection audit only; target coverage NEVER changes feature selection."""
    rows = []
    # Expected optional ion fields are listed even when absent from the cache.
    # This does not add any column or fabricated value to the fitted design.
    audit_names = list(dict.fromkeys(list(names) + list(OBSERVED)))
    for name in audit_names:
        in_cache = any(name in r for r in training)
        x = np.array([number(r.get(name)) for r in training])
        t = np.array([number(r.get(name)) for r in targets])
        finite = x[np.isfinite(x)]
        status = ('Not_in_cache' if not in_cache else 'All_missing' if not len(finite) else 'Constant' if np.ptp(finite) == 0 else
                  'Low_coverage' if len(finite) < len(training) * .5 else 'Available')
        rows.append({'Feature': name, 'Block': block_of(name),
                     'Training_finite': len(finite), 'Training_total': len(training),
                     'Target_finite': int(np.isfinite(t).sum()), 'Target_total': len(targets),
                     'Training_unique_values': len(np.unique(finite)), 'Availability': status,
                     'Meaning_warning': ('Structural/condition proxy, not a measured pKa or matrix suppression'
                                         if 'proxy' in name.casefold() else ''),
                     'Used_for_selection': 'No; descriptive audit only'})
    return rows

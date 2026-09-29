"""Synthetic software tests only, NOT evidence of accuracy on the user's samples."""
from __future__ import annotations
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.multiview_features import (NumericDesign, ComponentDesign, component_keys,
                                    schema, safe_feature, coverage_rows)
from core.multiview_rrf import (ModelSpec, ResponseModel, more_metrics, choose_rows,
                                selection_score, run_multiview, POLICY, default_specs)
from core.multiview_io import execute, aggregate_summary
from core.continuous_io import sha256, clean
from test_continuous_audit import fixture, write_cache

FAST = [ModelSpec('Global_RRF_mean', 'mean', 'baseline'),
        ModelSpec('Global_RRF_median', 'median', 'baseline'),
        ModelSpec('Small_Ridge_all', 'ridge', alpha=.1),
        ModelSpec('Small_Ridge_robust', 'ridge', scaler='robust', alpha=1)]


def logs(records):
    return (np.log10([r['Actual_concentration'] for r in records]),
            np.log10([r['Measured_ratio'] for r in records]))


class FeatureTests(unittest.TestCase):
    def test_schema_blocks_labels_raw_response_and_identity(self):
        blocked = ['Actual_concentration', 'Measured_ratio', 'Predicted_concentration',
                   'yield', 'Concentration', 'Product_SMILES', 'Source_row', 'Injection_Order',
                   'PeakArea', 'IS_area', 'Adduct_Cluster_Area', 'Total_TIC', 'Actual_log10_concentration',
                   'log_ratio', 'ResponseCorrectionIndex', 'log10_RRF']
        for n in blocked:
            self.assertFalse(safe_feature(n), n)
        for n in ['MolLogP', 'TPSA', 'LabuteASA', 'Primary_Ion_Fraction', 'AromaticRingCount']:
            self.assertTrue(safe_feature(n), n)
        row = {n: 1 for n in blocked + ['MolLogP', 'Primary_Ion_Fraction']}
        # Observed cache feature is recovered even when outside the old name span.
        self.assertEqual(schema([row], ['MolLogP']), ['MolLogP', 'Primary_Ion_Fraction'])

    def test_actual_panel_features_are_separated_from_structural_view(self):
        from core.multiview_features import block_of
        for name in ['Summed_Channel_Count', 'Br_Fragment_Fraction']:
            self.assertEqual(block_of(name), 'Observed_ion_behavior')
        self.assertEqual(schema([{'Exact_mass': 320.2, 'DBE': 5}], []), ['Exact_mass', 'DBE'])

    def test_no_supervised_prefilter_in_all_route(self):
        # Pure interaction with exactly zero individual correlations.
        rows = [{'d0': a, 'd1': b, 'd2': 1.0} for _ in range(10)
                for a, b in [(-1, -1), (-1, 1), (1, -1), (1, 1)]]
        y = np.array([r['d0'] * r['d1'] for r in rows], float)
        d = NumericDesign(['d0', 'd1', 'd2'], interactions=True).fit(rows, y)
        self.assertEqual(set(d.base_names), {'d0', 'd1'})
        self.assertEqual(len(d.pairs), 1)
        np.testing.assert_allclose(np.abs(np.corrcoef(d.transform(rows)[:, -1], y)[0, 1]), 1)

    def test_scaling_and_interaction_parameters_do_not_learn_test_values(self):
        tr, ta, names = fixture(98)
        y, lr = logs(tr)
        model = ResponseModel(ModelSpec('r', 'ridge', interactions=True), names).fit(tr, y, lr)
        means = model.design.scaler.mean_.copy()
        medians = model.design.medians.copy()
        pairs_mean = model.design.pair_scaler.mean_.copy()
        model.predict_logc([dict(ta[0], d0=1e6, d1=-1e6)])
        np.testing.assert_array_equal(means, model.design.scaler.mean_)
        np.testing.assert_array_equal(medians, model.design.medians)
        np.testing.assert_array_equal(pairs_mean, model.design.pair_scaler.mean_)

    def test_standardized_model_is_invariant_to_feature_units(self):
        tr, ta, names = fixture(98)
        y, lr = logs(tr)
        spec = ModelSpec('r', 'ridge', interactions=True)
        a = ResponseModel(spec, names).fit(tr, y, lr)
        trb = [dict(r, d0=r['d0'] * 1e6) for r in tr]
        tab = [dict(r, d0=r['d0'] * 1e6) for r in ta]
        b = ResponseModel(spec, names).fit(trb, y, lr)
        np.testing.assert_allclose(a.predict_logc(ta), b.predict_logc(tab), rtol=1e-8, atol=1e-8)

    def test_absent_observed_features_are_not_invented(self):
        tr, ta, names = fixture(98)
        for r in tr + ta:
            r['Primary_Ion_Fraction'] = None
        names += ['Primary_Ion_Fraction']
        y, lr = logs(tr)
        with self.assertRaisesRegex(ValueError, 'observed'):
            ResponseModel(ModelSpec('o', 'ridge', 'observed'), names).fit(tr, y, lr)
        audit = coverage_rows(tr, ta, names)
        self.assertEqual(next(r for r in audit if r['Feature'] == 'Primary_Ion_Fraction')['Availability'], 'All_missing')
        self.assertTrue(all(r['Primary_Ion_Fraction'] is None for r in tr))

    def test_observed_view_can_use_existing_measurements(self):
        tr, _, names = fixture(98)
        for i, r in enumerate(tr):
            r['Primary_Ion_Fraction'] = .1 + .8 * i / 98
        names += ['Primary_Ion_Fraction']
        y, lr = logs(tr)
        a = ResponseModel(ModelSpec('s', 'ridge'), names).fit(tr, y, lr)
        b = ResponseModel(ModelSpec('o', 'ridge', 'observed'), names).fit(tr, y, lr)
        self.assertNotIn('Primary_Ion_Fraction', a.selected_names)
        self.assertIn('Primary_Ion_Fraction', b.selected_names)

    def test_component_support_uses_distinct_combinations_and_no_test_vocab(self):
        def row(a, b, c):
            return {'Combo': 'A#%d:a:C2H4 | B#%d:b:C3H6 | C#%d:c:C5H10O2' % (a, b, c)}
        rows = [row(a, b, c) for a in range(1, 3) for b in range(1, 3) for c in range(1, 4)]
        d = ComponentDesign(pairs=True).fit(rows)
        before = list(d.levels)
        query = row(99, 1, 1)
        d.transform([query])
        self.assertEqual(before, d.levels)
        self.assertEqual(d.domain([query])[0]['ABC_supported_roles'], 2)
        self.assertEqual(len(ComponentDesign().fit([rows[0]] * 8).levels), 0)
        # Changing a formula for a same numeric index changes identity.
        self.assertNotEqual(component_keys(rows[0]), component_keys({'Combo': rows[0]['Combo'].replace('C2H4', 'C2H6')}))


class StrategyTests(unittest.TestCase):
    def test_trend_is_part_of_selection_but_error_gate_blocks_worse_models(self):
        specs = [ModelSpec('accurate', 'ridge'), ModelSpec('trend', 'ridge'), ModelSpec('bad', 'ridge')]
        def row(name, rmse, rho, slope):
            m = {'Log_RMSE': rmse, 'Spearman': rho, 'Tail_log_RMSE': rmse,
                 'Pairwise_concordance_pct': 50 + 50 * rho, 'Pred_on_actual_slope': slope}
            return dict(m, Model=name, Error='', Selection_score=selection_score(m, 1))
        rows = [row('accurate', 1, 0, 0), row('trend', 1.1, .95, .9), row('bad', 1.3, 1, 1)]
        chosen, rule = choose_rows(rows, specs)
        self.assertEqual(chosen.name, 'trend')
        self.assertAlmostEqual(rule['Gate_log_RMSE'], 1.2)
        rows[1] = row('trend', 1.21, .95, .9)
        self.assertEqual(choose_rows(rows, specs)[0].name, 'accurate')

    def test_pairwise_candidate_optimizes_and_keeps_offset(self):
        tr, ta, names = fixture(98)
        y, lr = logs(tr)
        m = ResponseModel(ModelSpec('pair', 'pairwise', alpha=1), names).fit(tr, y, lr)
        self.assertEqual(m.estimator.optimization_status, 'Converged')
        base = m.predict_logc(ta, np.zeros(len(ta)))
        np.testing.assert_allclose(m.predict_logc(ta, np.full(len(ta), 4)) - base, 4, atol=1e-12)

    def test_all_model_families_keep_response_offset_and_no_clipping(self):
        tr, ta, names = fixture(98)
        combos = [(a, b, c) for a in range(1, 8) for b in range(1, 6) for c in range(1, 5)]
        for i, r in enumerate(tr + ta):
            r['Primary_Ion_Fraction'] = .2 + .7 * (i % 10) / 10
            a, b, c = combos[i]
            r['Combo'] = 'A#%d:a:C2H4 | B#%d:b:C3H6 | C#%d:c:C5H10O2' % (a, b, c)
        names += ['Primary_Ion_Fraction']
        y, lr = logs(tr)
        for spec in default_specs():
            model = ResponseModel(spec, names).fit(tr, y, lr)
            a = model.predict_logc(ta[:2], [0, 0])
            b = model.predict_logc(ta[:2], [5, 5])
            np.testing.assert_allclose(b - a, 5, atol=1e-10, err_msg=spec.name)

    def test_nested_row_target_independence_and_no_residual_deletion(self):
        tr, ta, names = fixture(100)
        tr[5]['Measured_ratio'] = 0
        tr[71]['Actual_concentration'] = None
        before = copy.deepcopy(tr)
        result = run_multiview(tr, ta, names, specs=FAST, compare_legacy=False, folds=3)
        self.assertEqual(result['summary']['Training_used'], 98)
        self.assertEqual(len(result['known']), 100)
        self.assertEqual(tr, before)
        altered = [dict(r, Actual_concentration=1e70, d0=1e4, Combo='secret-combo') for r in ta]
        other = run_multiview(tr, altered, names, specs=FAST, compare_legacy=False, folds=3)
        np.testing.assert_array_equal(result['predicted_log'], other['predicted_log'])
        self.assertEqual(result['summary']['Final_model'], other['summary']['Final_model'])
        self.assertEqual(result['inner_search'], other['inner_search'])

    def test_inner_and_outer_groups_are_disjoint(self):
        tr, ta, names = fixture(98)
        r = run_multiview(tr, ta, names, specs=FAST, compare_legacy=False, folds=3)
        rows = r['cv_folds']
        keys = {(x['Stage'], x['Outer_fold'], x['Inner_fold']) for x in rows}
        for key in keys:
            subset = [x for x in rows if (x['Stage'], x['Outer_fold'], x['Inner_fold']) == key]
            a = {x['Independent_group'] for x in subset if x['Partition'] == 'Train'}
            b = {x['Independent_group'] for x in subset if x['Partition'] == 'Validation'}
            self.assertFalse(a & b)
        for f in range(1, 4):
            outer_test = {x['Input_order'] for x in rows if x['Stage'] == 'Outer' and x['Outer_fold'] == f and x['Partition'] == 'Validation'}
            inner_all = {x['Input_order'] for x in rows if x['Stage'] == 'Inner' and x['Outer_fold'] == f}
            self.assertFalse(outer_test & inner_all)

    def test_legacy_comparison_reproduces_original_strategy_exactly(self):
        from core.continuous_rrf import run_audit
        tr, ta, names = fixture(98)
        original = run_audit(tr, ta, names, folds=3)
        new = run_multiview(tr, ta, names, specs=FAST, compare_legacy=True, folds=3)
        np.testing.assert_allclose(original['predicted_log'], new['legacy_log'], atol=1e-12)

    def test_invalid_targets_retained_with_reasons(self):
        tr, ta, names = fixture(98)
        ta[0]['Measured_ratio'] = 0
        ta[1]['Formula_match'] = 'mismatch'
        ta[2].update({n: None for n in names})
        r = run_multiview(tr, ta, names, specs=FAST, compare_legacy=False, folds=3)
        self.assertEqual(len(r['unknown']), len(ta))
        for row in r['unknown'][:3]:
            self.assertIn('Skipped', row['Prediction_status'])

    def test_no_signal_does_not_pass(self):
        tr, ta, names = fixture(98)
        rng = np.random.RandomState(3)
        for record in tr:
            record['Measured_ratio'] = float(10 ** rng.normal(0, 4))
        r = run_multiview(tr, ta, names, specs=FAST, compare_legacy=False, folds=3)
        self.assertEqual(r['summary']['Validation_status'], 'NOT_VALIDATED')
        self.assertEqual(r['summary']['Training_used'], 98)

    def test_constant_labels_and_101_rows_rejected(self):
        tr, ta, names = fixture(101)
        with self.assertRaises(ValueError):
            run_multiview(tr, ta, names, specs=FAST, compare_legacy=False)
        tr = tr[:98]
        for row in tr:
            row['Actual_concentration'] = 1
        with self.assertRaisesRegex(ValueError, 'constant'):
            run_multiview(tr, ta, names, specs=FAST, compare_legacy=False)

    def test_empty_unknown_list_and_global_baseline_only(self):
        tr, _, names = fixture(98)
        r = run_multiview(tr, [], names, specs=FAST[:2], compare_legacy=False, folds=3)
        self.assertEqual(r['unknown'], [])
        self.assertEqual(r['summary']['Validation_status'], 'NOT_VALIDATED')

    def test_aggregate_does_not_contain_raw_identity_values_or_paths(self):
        tr, ta, names = fixture(98)
        secret = 'SECRET_LAB_VALUE_61731'
        for row in tr + ta:
            row.update(Source_file=secret, Product_SMILES='')
        r = run_multiview(tr, ta, names, specs=FAST, compare_legacy=False, folds=3)
        output = json.dumps(clean(aggregate_summary(r)), ensure_ascii=False)
        for value in [secret, tr[0]['Name'], tr[0]['Combo'], tr[0]['Formula'], 'Selected_features', 'Input_order']:
            self.assertNotIn(value, output)
        self.assertIn('Review_before_sharing', output)


class ExportTests(unittest.TestCase):
    def test_new_folder_roundtrip_and_source_unchanged(self):
        tr, ta, names = fixture(98)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache = root / 'private_cache.xlsx'
            write_cache(cache, tr, ta, names)
            before = sha256(cache)
            folder, result = execute(cache, root / 'out', specs=FAST, compare_legacy=False, folds=3)
            self.assertEqual(sha256(cache), before)
            self.assertIn('v18_46', folder.name)
            for name in ['START_HERE.html', 'completion.json', 'Feature_Coverage.csv',
                         'View_Comparison.csv', 'aggregate_summary_review_before_share.json',
                         'continuous_concentration_audit.xlsx', '02_known_sorted.png']:
                self.assertTrue((folder / name).is_file(), name)
            import joblib
            model = joblib.load(folder / 'fitted_model.joblib')['model']
            np.testing.assert_allclose(model.predict_logc(ta),
                  [row['Predicted_log10_concentration'] for row in result['unknown']], atol=1e-12)
            self.assertNotIn('<script', (folder / 'START_HERE.html').read_text().casefold())


if __name__ == '__main__':
    unittest.main(verbosity=2)

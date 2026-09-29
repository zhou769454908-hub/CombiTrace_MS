"""Read-only cache reuse / new-folder exports for the independent v18.46 path.

Workbook export delegates to the application's existing exporter. No dependency
is added, and the user's source tables or previous outputs are never saved over.
"""
from __future__ import annotations

import html
import math
import platform
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .continuous_io import (sha256, refresh_records, write_json, export_tables,
                            make_plots, METRIC_GUIDE)
from .model_rerun import load_descriptor_records
from .multiview_rrf import run_multiview, VERSION, VIEW_LABELS
from .training_limits import validate_training_count

NEW_GUIDE = [
    {'Metric': 'Selection_score', 'Meaning': 'Predefined inner-selection score: 45% error, 25% Spearman loss, 15% tail error, 10% pairwise loss and 5% bounded slope deviation. Lower is better; this is not an accuracy percentage.'},
    {'Metric': 'Within_global_error_gate', 'Meaning': 'Only candidates with inner log RMSE within 1.20 times the best candidate can enter primary selection. This permits a limited error/ranking trade-off, not arbitrary range expansion.'},
    {'Metric': 'View_Comparison', 'Meaning': 'Each feature view selects a model within the same training folds. Differences compare information sets; they do not establish causality.'},
    {'Metric': 'Feature_Coverage', 'Meaning': 'Cache availability, missingness and constant columns. Global coverage is a post-run audit and is not used for fold-specific selection. Missing observations are not fabricated.'},
    {'Metric': 'ABC_supported_roles', 'Meaning': 'Number of A/B/C components supported by at least two distinct training combinations. Unsupported effects shrink toward the prior and receive applicability flags.'},
    {'Metric': 'Legacy_v18_45_nested', 'Meaning': 'The original continuous-audit algorithm rerun on identical rows and outer splits. This is a same-fold control, not an import of previous scores.'},
    {'Metric': 'Between_view_log10_span', 'Meaning': 'Logarithmic range across feature-view predictions. A span above 1 means predictions differ by more than tenfold; this is a review flag, not a confidence interval.'},
    {'Metric': 'Approximate_linearity', 'Meaning': 'Approximately linear response is a modelling assumption. Single-concentration records do not independently establish linearity.'},
    {'Metric': 'Group_Residuals', 'Meaning': 'OOF residuals summarized by connected injection/compound groups. This is a post-run diagnostic; correlations and R2 are unstable in small groups.'},
]


def aggregate_summary(result):
    """Deliberate whitelist. No source paths, row data, structures or identifiers.

This file is local too. Aggregated metrics can still be commercially sensitive;
its filename explicitly asks the user to review before sharing.
"""
    s = result['summary']
    settings = {key: s[key] for key in (
        'Version', 'Training_nonblank_input', 'Training_used', 'Training_skipped',
        'Target_input', 'Target_predicted', 'Outer_folds', 'Independent_groups',
        'Seed', 'Final_model', 'Final_view', 'Validation_status',
        'Candidate_feature_count', 'Selected_feature_count', 'Observed_features_available',
        'Compare_v18_45_same_splits', 'Approximate_linearity')}
    settings['Primary_nested_metrics'] = s['Nested_OOF_metrics']
    allowed_models = {'Primary_nested_multiview', 'Constant_logC_mean', 'Constant_logC_median',
                      'Legacy_v18_45_nested'} | set(VIEW_LABELS.values())
    metric_keys = {'Model', 'Interpretation', 'N', 'Log_RMSE', 'Log_R2', 'Spearman',
                   'Pred_on_actual_slope', 'Log_spread_ratio', 'Median_fold_error',
                   'P80_fold_error', 'Within_2x_pct', 'Within_5x_pct',
                   'Low20_log_bias', 'High20_log_bias', 'Tail_log_RMSE',
                   'Pairwise_concordance_pct', 'Top20_overlap_pct'}
    settings['Aggregate_comparison'] = [{k: v for k, v in row.items() if k in metric_keys}
             for row in result['comparison'] if row['Model'] in allowed_models]
    settings['Review_before_sharing'] = ('Local aggregate metrics only; not an anonymization guarantee. '
        'No automatic upload. All other output files may contain confidential data.')
    return settings


def _extra_plots(folder, result):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    files = []

    def save(fig, name):
        FigureCanvasAgg(fig)
        fig.savefig(str(folder / name), dpi=180, bbox_inches='tight')
        files.append(name)
        fig.clear()

    rows = [r for r in result['view_comparison'] if 'Log_RMSE' in r]
    fig = Figure(figsize=(9, 4))
    ax = fig.add_subplot(111)
    if rows:
        ax.barh([r['Model'] for r in rows], [r['Log_RMSE'] for r in rows])
        ax.invert_yaxis()
        ax.set(xlabel='Nested held-out log10 RMSE (lower is better)',
               title='Information views: identical rows and validation splits')
    else:
        ax.text(.5, .5, 'No complete information-view comparison', ha='center')
    save(fig, '05_information_views.png')

    old = result.get('legacy_log')
    if old is not None and np.all(np.isfinite(old)):
        y, p = result['actual_log'], result['predicted_log']
        order = np.argsort(y, kind='stable')
        x = np.arange(1, len(y) + 1)
        fig = Figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        ax.plot(x, y[order], '.-', label='Actual')
        ax.plot(x, old[order], 'x', alpha=.65, label='v18.45 nested OOF (same splits)')
        ax.plot(x, p[order], '.', label='v18.46 nested OOF')
        ax.set(xlabel='Standards sorted by actual concentration (same row alignment)',
               ylabel='Concentration (log10)', title='New vs previous strategy: no range stretching')
        ax.legend(fontsize=8)
        save(fig, '06_v18_45_vs_v18_46.png')
    return files


def write_dashboard(folder, result, images):
    s = result['summary']
    whitelist = {'Primary_nested_multiview', 'Legacy_v18_45_nested',
                 'Constant_logC_mean', 'Constant_logC_median'} | set(VIEW_LABELS.values())
    columns = ['Model', 'N', 'Log_RMSE', 'Spearman', 'Pred_on_actual_slope', 'Low20_log_bias', 'High20_log_bias']
    table = []
    for row in result['comparison']:
        if row['Model'] not in whitelist:
            continue
        cells = []
        for key in columns:
            value = row.get(key, '')
            if isinstance(value, (int, float, np.number)):
                value = '%.4g' % value
            cells.append('<td>' + html.escape(str(value)) + '</td>')
        table.append('<tr>' + ''.join(cells) + '</tr>')
    text_content = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Multi-view response models</title>\n<style>body{font:16px/1.6 Arial,sans-serif;max-width:1160px;margin:32px auto;padding:0 24px}table{border-collapse:collapse;width:100%%}td,th{border:1px solid #ccc;padding:8px;text-align:left}img{max-width:100%%}code{background:#eee;padding:2px}</style>\n<h1>Multi-view response models</h1><p><strong>Validation status: %s</strong>. Calibration records: %d. Final configuration: %s.</p>\n<p>Reported predictions are held out, not full-training fitted values. Internal checks do not constitute analytical-method validation. Approximate linearity remains an assumption.</p>\n<h2>Same-fold comparison</h2><table><tr>%s</tr>%s</table>\n<p>Lower log RMSE indicates smaller errors; higher Spearman indicates stronger rank agreement. The identity slope is 1. Positive low/high-20%% bias indicates overestimation. Missing view scores indicate unavailable features or a failed fold, not a selected successful subset.</p>\n<h2>Files</h2><p><a href="continuous_concentration_audit.xlsx">Result workbook</a>: predictions, features, comparisons and exclusion reasons. It may contain confidential information.</p>\n<p><a href="aggregate_summary_review_before_share.json">Aggregate summary</a>: no row-level names, structures, paths or concentrations. Review before sharing. No data are uploaded automatically.</p>\n<p>The original descriptor workbook remains the ESI_Features cache; this result workbook is not a replacement cache.</p><h2>Plots</h2>%s</html>' % (html.escape(s['Validation_status']), s['Training_used'], html.escape(s['Final_model']),
        ''.join('<th>' + html.escape(k) + '</th>' for k in columns), ''.join(table),
        ''.join('<p><img src="%s" alt="%s"></p>' % (html.escape(n), html.escape(n)) for n in images
                if n in ('02_known_sorted.png', '05_information_views.png', '06_v18_45_vs_v18_46.png')))
    (folder / 'START_HERE.html').write_text(text_content, encoding='utf-8')


def execute(cache_path, output_parent, *, training_path='', target_path='', training_sheet='',
            target_sheet='', training_columns=None, target_columns=None,
            concentration_unit='same as known training table', seed=42, folds=5,
            compare_legacy=True, progress=None, specs=None):
    emit = progress or (lambda s: None)
    paths = [Path(cache_path).resolve()] + [Path(p).resolve() for p in (training_path, target_path) if p]
    inputs = [{'path': str(p), 'sha256': sha256(p)} for p in paths]
    parent = Path(output_parent).expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    folder = parent / ('v18_46_multiview_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6])
    folder.mkdir(exist_ok=False)
    try:
        emit('Reusing ESI_Features; no RAW access or additional experiments.')
        _, training, targets, features, messages = load_descriptor_records(paths[0])
        cache_n = len(training)
        matching, used_columns = [], {}
        if training_path:
            training, rows, used_columns['training'] = refresh_records(training, training_path, training_sheet,
                         is_training=True, column_overrides=training_columns)
            matching.extend(rows)
            emit('Current calibration table takes priority: %d cached rows -> %d current rows.' % (cache_n, len(training)))
        else:
            messages.append('Training row list/labels/ratios are from cached ESI_Features; no current training table supplied.')
        if target_path:
            targets, rows, used_columns['target'] = refresh_records(targets, target_path, target_sheet,
                         is_training=False, column_overrides=target_columns)
            matching.extend(rows)
        validate_training_count(training)
        # Small n models otherwise suffer severe BLAS/OpenMP oversubscription.
        # threadpoolctl is already an indirect dependency of scikit-learn.
        with threadpool_limits(limits=1):
            result = run_multiview(training, targets, features, seed=seed, folds=folds,
                         compare_legacy=compare_legacy, specs=specs, progress=emit)
        for row in result['known'] + result['unknown']:
            row['Concentration_unit'] = concentration_unit
        summary = result['summary']
        summary.update(Concentration_unit=concentration_unit, Training_records_in_cache=cache_n,
                       Input_column_mapping=used_columns)
        summary['Notes'].extend(messages)
        import sklearn
        import scipy
        summary['Runtime'] = {'python': platform.python_version(), 'numpy': np.__version__,
                              'scikit_learn': sklearn.__version__, 'scipy': scipy.__version__}
        write_json(folder / 'input_manifest.json', {'Version': VERSION, 'inputs': inputs,
                   'training_sheet': training_sheet, 'target_sheet': target_sheet, 'columns': used_columns,
                   'concentration_unit': concentration_unit, 'seed': seed, 'folds': folds,
                   'compare_legacy': compare_legacy, 'selection_policy': summary['Selection_policy'],
                   'response_equation': 'log10(C) = log10(Area/IS) - predicted_log10_RRF',
                   'linearity': 'Assumed, not experimentally verified here'})
        guide = [row for row in METRIC_GUIDE if row['Metric'] != 'Log_RMSE']
        guide.insert(0, {'Metric': 'Log_RMSE', 'Meaning': 'Root mean squared log10 concentration error. Multi-view inner selection also considers ranking and low/high concentration errors.'})
        tables = {'Run_Summary': [{'Setting': k, 'Value': v} for k, v in summary.items()],
                  'Model_Comparison': result['comparison'], 'View_Comparison': result['view_comparison'],
                  'Known_Standards_Predictions': result['known'], 'Unknown_Sample_Predictions': result['unknown'],
                  'Feature_Coverage': result['feature_coverage'], 'Feature_Audit': result['feature_audit'],
                  'Model_Stability': result['model_stability'], 'Fold_Metrics': result['fold_metrics'],
                  'Group_Residuals': result['group_residuals'], 'Row_Audit': result['row_audit'],
                  'OOF_Folds': result['cv_folds'], 'Inner_Search': result['inner_search'],
                  'Legacy_Inner_Search': result['legacy_inner_search'], 'Input_Matching': matching,
                  'Metric_Guide': guide + NEW_GUIDE}
        emit('Exporting row-level predictions, feature coverage, same-fold comparisons and summary files.')
        workbook = export_tables(folder, tables)
        images = make_plots(folder, result) + _extra_plots(folder, result)
        write_json(folder / 'run_summary.json', summary)
        write_json(folder / 'aggregate_summary_review_before_share.json', aggregate_summary(result))
        write_dashboard(folder, result, images)
        import joblib
        joblib.dump({'model': result['model'], 'feature_names': result['candidate_features'],
                    'unit': concentration_unit, 'version': VERSION,
                    'prediction_api': 'model.predict_logc(list_of_records)',
                    'privacy': 'This model may contain training-derived / local-neighbor data; keep private.'},
                   folder / 'fitted_model.joblib')
        for item in inputs:
            if sha256(item['path']) != item['sha256']:
                raise RuntimeError('An input changed during execution: ' + item['path'])
        write_json(folder / 'completion.json', {'status': 'completed', 'input_hashes_unchanged': True,
                   'workbook': workbook.name, 'plots': images, 'version': VERSION,
                   'real_or_synthetic': 'depends on the local input supplied by the user'})
        emit('Complete. Validation status: ' + summary['Validation_status'] + '\n' + str(folder / 'START_HERE.html'))
        return folder, result
    except Exception as exc:
        write_json(folder / 'FAILED.json', {'status': 'failed', 'error': type(exc).__name__ + ': ' + str(exc),
                                          'inputs': inputs})
        raise

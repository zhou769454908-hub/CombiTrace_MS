"""Small synthetic smoke checks, used before and AFTER freezing.

Only temporary synthetic objects are constructed. No user's RAW/cache/model is
opened. Import checks cannot establish successful real-instrument operation.
"""
import importlib
import json
import math
import platform
import sys
import time
import traceback
from pathlib import Path
from .catalog import BUILD_REVISION, dependencies, project_modules


def package_root():
    return Path(__file__).resolve().parent.parent


def read_bundle_config():
    path = package_root() / 'bundle_config.json'
    if path.is_file():
        return json.loads(path.read_text(encoding='utf-8'))
    return {'with_raw': True, 'modules': project_modules(package_root())}


def run_checks(*, with_raw=True, gui=False, output=None):
    result = {'revision': BUILD_REVISION, 'python': sys.version,
              'executable': sys.executable, 'platform': platform.platform(),
              'frozen': bool(getattr(sys, 'frozen', False)), 'with_raw': bool(with_raw),
              'checks': [], 'scope': 'imports + synthetic smoke; no real RAW/data validation'}
    def check(name, function):
        t = time.monotonic()
        try:
            detail = function()
            row = {'name': name, 'status': 'PASS', 'detail': str(detail or '')}
        except Exception as exc:
            row = {'name': name, 'status': 'FAIL', 'error': repr(exc),
                   'traceback': traceback.format_exc()}
        row['seconds'] = round(time.monotonic() - t, 3)
        result['checks'].append(row)
        print('[%s] %s: %s' % (row['status'], name, row.get('error', row.get('detail', ''))), flush=True)
    def require(condition, msg):
        if not condition:
            raise RuntimeError(msg)
    def import_one(name):
        obj = importlib.import_module(name)
        return '%s | %s' % (getattr(obj, '__version__', ''), getattr(obj, '__file__', 'builtin'))
    for name, dist in dependencies(with_raw):
        check('dependency:' + name, lambda n=name: import_one(n))
    def imports():
        errors = []
        modules = read_bundle_config()['modules']
        for name in modules:
            try:
                importlib.import_module(name)
            except Exception as exc:
                errors.append(name + ': ' + repr(exc))
        require(not errors, '\n'.join(errors))
        return str(len(modules)) + ' local module imports'
    check('project_modules', imports)
    def flags():
        from core import structure_features as sf, esi_model_benchmark as eb
        require(sf.RDKIT_AVAILABLE, 'RDKit disabled: ' + str(sf.RDKIT_ERROR))
        require(eb.SKLEARN_AVAILABLE, 'sklearn disabled: ' + str(eb.SKLEARN_ERROR))
        return 'RDKit and legacy ESI model backend enabled'
    check('backend_flags_not_silently_disabled', flags)
    def scientific():
        import numpy as np
        from scipy.optimize import minimize
        from scipy.special import expit
        from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
        from sklearn.linear_model import Ridge, BayesianRidge
        from sklearn.svm import SVR
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.feature_selection import SelectKBest, f_regression
        from sklearn.pipeline import make_pipeline
        import joblib
        from threadpoolctl import threadpool_limits
        from io import BytesIO
        rng = np.random.RandomState(42)
        x = rng.normal(size=(24, 4)); y = x[:, 0] - .2 * x[:, 1]
        estimators = [ExtraTreesRegressor(n_estimators=8, n_jobs=1, random_state=42),
                      RandomForestRegressor(n_estimators=8, n_jobs=1, random_state=42),
                      HistGradientBoostingRegressor(max_iter=5, min_samples_leaf=3),
                      Ridge(), BayesianRidge(), SVR(), KNeighborsRegressor(3),
                      PLSRegression(n_components=2), GaussianProcessRegressor(kernel=RBF(), optimizer=None)]
        with threadpool_limits(limits=1):
            for est in estimators:
                pipe = make_pipeline(SimpleImputer(), StandardScaler(), SelectKBest(f_regression, k=3), est)
                pipe.fit(x, y)
                pred = pipe.predict(x[:3]); require(np.isfinite(pred).all(), str(est))
            buf = BytesIO(); joblib.dump(pipe, buf); buf.seek(0)
            require(np.allclose(pred, joblib.load(buf).predict(x[:3])), 'model serialization failed')
        fit = minimize(lambda v: float((v[0] - 1) ** 2), np.array([0.0]))
        require(abs(fit.x[0] - 1) < .01 and np.isfinite(expit(1)), 'scipy optimizer failed')
        return '9 estimators, scipy optimization, joblib roundtrip'
    check('scientific_models', scientific)
    def chemistry():
        from rdkit import Chem
        from core.structure_features import _desc2d, _desc3d, _acid_base_proxy_descriptors, _extended_vsa_descriptors
        mol = Chem.MolFromSmiles('CC(=O)Oc1ccccc1C(=O)O')
        d2 = _desc2d(mol)
        require(all(k in d2 and math.isfinite(d2[k]) for k in ('MolLogP', 'TPSA', 'HBA')), 'RDKit descriptor error')
        d3, status = _desc3d(mol)
        require(bool(d3) and 'PMI1' in d3, '3D descriptor generation failed: ' + str(status))
        require(bool(_extended_vsa_descriptors(mol)), 'RDKit extended descriptor imports missing')
        require('IonizableSiteCount_proxy' in _acid_base_proxy_descriptors(mol), 'SMARTS proxy error')
        from rdkit.Chem.Draw import MolToImage
        require(MolToImage(mol).size[0] > 0, 'RDKit drawing error')
        return '2D/3D/VSA/SMARTS/drawing synthetic checks'
    check('rdkit_descriptors', chemistry)
    def plotting():
        from io import BytesIO
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from PIL import Image
        fig = Figure(); FigureCanvasAgg(fig)
        ax = fig.add_subplot(111); ax.plot([1, 2], [2, 3])
        b = BytesIO(); fig.savefig(b, format='png'); b.seek(0)
        im = Image.open(b); require(im.size[0] > 0, 'plot buffer empty')
        return 'Matplotlib Agg / PIL PNG in memory'
    check('plotting', plotting)
    def assets():
        root = package_root()
        required = ['VERSION.txt', 'templates/LC_gradient_default_v18_26.csv',
                    'templates/negative_ion_channel_registry.csv',
                    'templates/ABC_components_with_SMILES_template.xlsx',
                    'resources/README.md', 'r_scripts/ms2quant_bridge.R']
        missing = [x for x in required if not (root / x).is_file()]
        require(not missing, 'Missing resources: ' + ', '.join(missing))
        return 'resource/template/R-script paths exist (R itself not tested)'
    check('bundled_assets', assets)
    def final_tables():
        from core.final_table_selftest import run_synthetic_selftest
        return run_synthetic_selftest()
    check('final_table_postprocess', final_tables)
    def triplicate_tables():
        from core.final_table_selftest import run_triplicate_selftest
        return run_triplicate_selftest()
    check('final_table_triplicates', triplicate_tables)
    def display_choice_tables():
        from core.final_table_selftest import run_display_choice_selftest
        return run_display_choice_selftest()
    check('final_table_image_display_choice', display_choice_tables)
    def optional_captions():
        from core.final_table_captions import run_caption_selftest
        return run_caption_selftest()
    check('final_table_optional_captions', optional_captions)
    def existing_xic():
        from core.existing_xic_edit import run_existing_xic_selftest
        return run_existing_xic_selftest()
    check('existing_result_xic_edit', existing_xic)
    def numeric_ppm_tables():
        from core.final_table_selftest import run_numeric_ppm_selftest
        return run_numeric_ppm_selftest()
    check('final_table_numeric_ppm', numeric_ppm_tables)

    def observed_mass_tables():
        from core.observed_mass_selftest import run_observed_mass_selftest
        return run_observed_mass_selftest()
    check('observed_mass_synthetic_provenance', observed_mass_tables)
    if with_raw:
        def raw_api():
            from fisher_py import RawFile
            from fisher_py.data.business import TraceType
            require(RawFile is not None and hasattr(TraceType, 'TIC'), 'RAW bridge API not found')
            return 'RAW API imports; real RAW reading was NOT tested'
        check('raw_reader_bridge', raw_api)
    if gui:
        def gui_test():
            import tkinter as tk
            from app import ThermoBatchReportApp
            from core.multiview_lab import MultiviewLab
            from core.continuous_lab import ContinuousLab
            root = ThermoBatchReportApp(); root.withdraw()
            try:
                root.update_idletasks()
                require(root.notebook.tab(root.tab_final_tables, 'text') == 'Final tables', 'Final-table tab missing')
                panel = root.tab_final_tables
                require(not panel.caption_settings().enabled, 'Caption defaults must be off')
                panel.caption_replicate.set(True); panel.caption_changed()
                require('Rep 1' in panel.caption_preview.get() and 'Sample-1' not in panel.caption_preview.get(), 'Caption choice not reflected in preview')
                panel.clear_captions()
                require(not panel.caption_settings().enabled, 'Clear captions failed')
                editor, edit_panel = panel.open_existing_xic()
                editor.withdraw(); editor.update_idletasks()
                require(edit_panel.mode.get() == 'keep' and not edit_panel.captions().enabled, 'Existing-XIC editor defaults incorrect')
                edit_panel.destroy(); editor.destroy()
                a = MultiviewLab(root); a.withdraw(); a.update_idletasks(); a.destroy()
                b = ContinuousLab(root); b.withdraw(); b.update_idletasks(); b.destroy()
                root.update_idletasks()
            finally:
                root.destroy()
            return 'main UI + both independent modelling windows constructed'
        check('tk_gui_windows', gui_test)
    result['status'] = 'PASS' if all(x['status'] == 'PASS' for x in result['checks']) else 'FAIL'
    result['gui_tested'] = bool(gui)
    if output:
        p = Path(output); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result

"""Build-logic regression tests. Mocked freezer tests are NOT Windows EXE tests."""
import argparse
import ast
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import build_exe
from packaging_tools import catalog


class PackagingTests(unittest.TestCase):
    def test_catalog_all_core_source_names(self):
        names = catalog.project_modules(ROOT)
        for source in (ROOT / 'core').glob('*.py'):
            name = 'core' if source.stem == '__init__' else 'core.' + source.stem
            self.assertIn(name, names)
        self.assertIn('core.multiview_lab', names)
        self.assertIn('core.continuous_rrf', names)
        self.assertIn('app', names)
        self.assertNotIn('tests.test_multiview_v46', names)

    def test_import_distribution_names_not_confused(self):
        mapping = dict(catalog.dependencies())
        self.assertEqual(mapping['sklearn'], 'scikit-learn')
        self.assertEqual(mapping['docx'], 'python-docx')
        self.assertEqual(mapping['PIL'], 'Pillow')
        self.assertEqual(mapping['fisher_py'], 'fisher-py')

    def test_full_and_explicit_no_raw_profiles(self):
        self.assertIn('fisher_py', dict(catalog.dependencies(True)))
        self.assertNotIn('fisher_py', dict(catalog.dependencies(False)))
        self.assertIn('rdkit', dict(catalog.dependencies(False)))

    def test_submodule_filter(self):
        self.assertFalse(catalog.safe_submodule('sklearn.tests.test_pipeline'))
        self.assertFalse(catalog.safe_submodule('rdkit.Chem.Draw.IPythonConsole'))
        self.assertTrue(catalog.safe_submodule('sklearn.utils._typedefs'))
        self.assertTrue(catalog.safe_submodule('rdkit.Chem.rdMolDescriptors'))

    def test_only_explicit_project_asset_directories(self):
        self.assertEqual(set(catalog.BUNDLE_DIRS), {'resources', 'templates', 'r_scripts', 'docs'})
        self.assertNotIn('build_records', catalog.BUNDLE_DIRS)

    def test_python38_syntax(self):
        paths = list((ROOT / 'packaging_tools').glob('*.py')) + [ROOT / 'launch_reporter.py', ROOT / 'build_exe.py']
        for path in paths:
            ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 8))

    def test_bats_crlf_and_same_interpreter(self):
        for name in ('build_exe.bat', 'check_build_env.bat', 'build_exe_windowed.bat', 'install_build_tools.bat'):
            b = (ROOT / name).read_bytes()
            self.assertIn(b'\r\n', b)
            self.assertIn(b'THERMO_BUILD_PYTHON', b)
            self.assertNotIn(b'pip install -r requirements', b)

    def test_missing_dependency_stops_without_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            with mock.patch.object(build_exe, 'dependencies', return_value=[('missing_xyz', 'example-dist')]), \
                 mock.patch.object(build_exe.importlib.util, 'find_spec', return_value=None), \
                 mock.patch.object(build_exe, 'stream_command') as cmd:
                self.assertFalse(build_exe.source_preflight(run, True))
                cmd.assert_not_called()
            self.assertIn('missing_xyz', (run / 'missing_dependencies.txt').read_text())
            self.assertTrue((run / 'environment.json').exists())

    def test_source_preflight_requires_report_and_success_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            with mock.patch.object(build_exe, 'dependencies', return_value=[]), \
                 mock.patch.object(build_exe, 'stream_command', return_value=0):
                self.assertFalse(build_exe.source_preflight(run, False))
                (run / 'source_selftest.json').write_text('{"status":"PASS"}')
                self.assertTrue(build_exe.source_preflight(run, False))
            with mock.patch.object(build_exe, 'dependencies', return_value=[]), \
                 mock.patch.object(build_exe, 'stream_command', return_value=1):
                self.assertFalse(build_exe.source_preflight(run, False))

    def test_exe_exists_is_not_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp); exe = run / 'test.exe'; exe.write_bytes(b'not an executable')
            with mock.patch.object(build_exe, 'stream_command', return_value=0):
                self.assertFalse(build_exe.validate_frozen(run, exe))

    def test_frozen_check_rejects_failure_or_nonfrozen_or_no_gui(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp); exe = run / 'test.exe'; exe.write_bytes(b'placeholder')
            for report in ({'status':'FAIL','frozen':True,'gui_tested':True},
                           {'status':'PASS','frozen':False,'gui_tested':True},
                           {'status':'PASS','frozen':True,'gui_tested':False}):
                (run / 'frozen_selftest.json').write_text(json.dumps(report))
                with mock.patch.object(build_exe, 'stream_command', return_value=0):
                    self.assertFalse(build_exe.validate_frozen(run, exe))

    def test_frozen_success_requires_all_conditions(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp); exe = run / 'test.exe'; exe.write_bytes(b'placeholder')
            (run / 'frozen_selftest.json').write_text('{"status":"PASS","frozen":true,"gui_tested":true}')
            with mock.patch.object(build_exe, 'stream_command', return_value=0) as cmd:
                self.assertTrue(build_exe.validate_frozen(run, exe))
                self.assertEqual(cmd.call_args.kwargs['cwd'], run / 'empty_smoke_cwd')
            with mock.patch.object(build_exe, 'stream_command', return_value=2):
                self.assertFalse(build_exe.validate_frozen(run, exe))

    def test_development_import_paths_removed(self):
        with mock.patch.dict(os.environ, {'PYTHONPATH':'private-path','PYTHONHOME':'private-home'}):
            env = build_exe.isolated_runtime_env()
            self.assertNotIn('PYTHONPATH', env)
            self.assertNotIn('PYTHONHOME', env)
            self.assertIn('PYTHONPATH', os.environ)

    @unittest.skipIf(os.name == 'nt', 'This guard test applies to non-Windows hosts')
    def test_non_windows_does_not_claim_windows_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(check_only=False, without_raw=False)
            with self.assertRaisesRegex(RuntimeError, 'must run on Windows'):
                build_exe.build(args, Path(tmp))

    def test_spec_wiring_with_mocked_freezer(self):
        # This verifies Python spec logic, not collection by a real PyInstaller.
        hooks = types.ModuleType('PyInstaller.utils.hooks')
        hooks.collect_submodules = lambda pkg, **kw: [pkg + '.sample']
        hooks.collect_data_files = lambda pkg: [('fake-' + pkg, pkg)]
        hooks.collect_dynamic_libs = lambda pkg: []
        hooks.collect_delvewheel_libs_directory = lambda pkg, datas, binaries: (datas, binaries)
        hooks.copy_metadata = lambda dist: []
        hooks.conda_support = types.SimpleNamespace(collect_dynamic_libs=lambda *a, **k: [])
        for raw in (True, False):
            with tempfile.TemporaryDirectory() as tmp:
                cfg = Path(tmp) / 'bundle_config.json'
                cfg.write_text(json.dumps({'with_raw':raw,'console':True,'exe_name':'test'}))
                analysis = mock.Mock(return_value=types.SimpleNamespace(pure=[],scripts=[],binaries=[],datas=[]))
                scope = {'SPECPATH':str(ROOT), 'Analysis':analysis,
                         'PYZ':mock.Mock(), 'EXE':mock.Mock(), 'COLLECT':mock.Mock()}
                with mock.patch.dict(sys.modules, {'PyInstaller.utils.hooks':hooks}), \
                     mock.patch.dict(os.environ, {'THERMO_BUILD_ROOT':str(ROOT),'THERMO_BUILD_CONFIG':str(cfg)}):
                    exec(compile((ROOT/'ThermoRawReporter.spec').read_text(), 'test.spec', 'exec'), scope)
                kw = analysis.call_args.kwargs
                self.assertIn('core.multiview_lab', kw['hiddenimports'])
                self.assertIn('rdkit.Chem.AllChem', kw['hiddenimports'])
                self.assertEqual(('fisher_py' in kw['hiddenimports']), raw)
                self.assertEqual(scope['EXE'].call_args.kwargs['contents_directory'], '_internal')
                self.assertTrue(scope['EXE'].call_args.kwargs['exclude_binaries'])
                self.assertEqual(kw['hooksconfig']['matplotlib']['backends'], ['Agg','TkAgg'])


if __name__ == '__main__':
    unittest.main()

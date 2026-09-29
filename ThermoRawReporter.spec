# -*- mode: python ; coding: utf-8 -*-
# Invoke through build_exe.py. No experiment folders are recursively collected.
import importlib.metadata
import json
import os
from pathlib import Path
import sys

from PyInstaller.utils.hooks import (
    collect_submodules, collect_data_files, collect_dynamic_libs, copy_metadata,
    collect_delvewheel_libs_directory,
)
root = Path(os.environ.get('THERMO_BUILD_ROOT', SPECPATH)).resolve()
sys.path.insert(0, str(root))
from packaging_tools.catalog import (HIDDEN_ROOTS, BUNDLE_DIRS, BUNDLE_FILES,
                                      project_modules, safe_submodule)
config_path = Path(os.environ['THERMO_BUILD_CONFIG'])
config = json.loads(config_path.read_text(encoding='utf-8'))
name = config['exe_name']
with_raw = bool(config['with_raw'])
datas = [(str(root / d), d) for d in BUNDLE_DIRS]
datas += [(str(root / f), '.') for f in BUNDLE_FILES]
datas += [(str(config_path), '.')]
binaries = []
hiddenimports = project_modules(root) + list(HIDDEN_ROOTS)
# Scientific packages have binary extensions and dynamically discovered modules.
# Keep standard PyInstaller hooks active rather than replacing their DLL handling.
for pkg in ('sklearn', 'rdkit'):
    hiddenimports += collect_submodules(pkg, filter=safe_submodule, on_error='warn once')
for pkg in ('rdkit', 'docx', 'reportlab'):
    datas += collect_data_files(pkg)
for pkg in ('numpy', 'scipy', 'sklearn', 'rdkit', 'PIL', 'lxml'):
    binaries += collect_dynamic_libs(pkg)
    datas, binaries = collect_delvewheel_libs_directory(pkg, datas=datas, binaries=binaries)
if with_raw:
    hiddenimports += ['fisher_py', 'fisher_py.data.business', 'clr', 'pythonnet', 'clr_loader', '_cffi_backend']
    for pkg in ('fisher_py', 'pythonnet', 'clr_loader'):
        hiddenimports += collect_submodules(pkg, filter=safe_submodule, on_error='warn once')
        # .NET assemblies are copied verbatim, not treated as ELF/Win32 libraries.
        datas += collect_data_files(pkg)
        if pkg != 'fisher_py':
            binaries += collect_dynamic_libs(pkg)
# Conda stores some RDKit/shared dependencies under Library/bin, outside rdkit/.
if (Path(sys.prefix) / 'conda-meta').is_dir():
    from PyInstaller.utils.hooks import conda_support
    for pkg in ('rdkit',):
        try:
            datas += conda_support.collect_dynamic_libs(pkg, dependencies=True)
        except Exception as exc:
            print('Conda supplementary DLL collection warning for %s: %s' % (pkg, exc))
# Package metadata can be consulted through importlib.metadata at runtime.
for dist in ('numpy', 'scipy', 'scikit-learn', 'joblib', 'threadpoolctl', 'matplotlib',
             'openpyxl', 'python-docx', 'pypdf', 'reportlab', 'Pillow', 'lxml',
             'rdkit', 'rdkit-pypi') + (('fisher-py', 'pythonnet', 'clr-loader', 'cffi') if with_raw else ()):
    try:
        importlib.metadata.distribution(dist)
    except importlib.metadata.PackageNotFoundError:
        continue
    datas += copy_metadata(dist)
excludes = ['pytest', 'IPython', 'notebook', 'jupyter', 'torch', 'tensorflow',
            'PyQt5', 'PyQt6', 'PySide2', 'PySide6']
if not with_raw:
    excludes += ['fisher_py', 'pythonnet', 'clr_loader', 'clr']
a = Analysis(
    [str(root / 'launch_reporter.py')], pathex=[str(root)],
    binaries=list(dict.fromkeys(binaries)), datas=list(dict.fromkeys(datas)),
    hiddenimports=sorted(set(hiddenimports)), hookspath=[], runtime_hooks=[],
    hooksconfig={'matplotlib': {'backends': ['Agg', 'TkAgg']}},
    excludes=excludes, noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name=name,
          debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
          console=bool(config['console']), disable_windowed_traceback=False,
          contents_directory='_internal')
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name=name)

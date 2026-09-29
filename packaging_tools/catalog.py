"""Dependency contract for Windows packaging. Does not import optional backends."""
from pathlib import Path

BUILD_REVISION = '18.48.2-en.1'
APP_NAME = 'CombiTraceMS_v18_48_2_en1'
# Import names are deliberately distinct from pip distribution names.
BASE_DEPENDENCIES = (
    ('numpy', 'numpy'), ('scipy', 'scipy'), ('sklearn', 'scikit-learn'),
    ('joblib', 'joblib'), ('threadpoolctl', 'threadpoolctl'),
    ('matplotlib', 'matplotlib'), ('openpyxl', 'openpyxl'),
    ('PIL', 'Pillow'), ('docx', 'python-docx'), ('lxml.etree', 'lxml'),
    ('reportlab', 'reportlab'), ('pypdf', 'pypdf'), ('tkinter', 'Python/Tcl-Tk'),
    ('rdkit', 'rdkit'),
)
RAW_DEPENDENCIES = (
    ('fisher_py', 'fisher-py'), ('pythonnet', 'pythonnet'),
    ('clr_loader', 'clr-loader'), ('clr', 'pythonnet'), ('_cffi_backend', 'cffi'),
)
BUNDLE_DIRS = ('resources', 'templates', 'r_scripts', 'docs')
BUNDLE_FILES = ('VERSION.txt', 'INSTALLATION.md', 'BUILD_EXE.md', 'THIRD_PARTY_NOTICES.md')
# Included even when imported only on a button click or inside an optional try block.
HIDDEN_ROOTS = (
    'numpy', 'scipy', 'scipy.optimize', 'scipy.special', 'scipy.stats',
    'sklearn', 'sklearn.ensemble', 'sklearn.linear_model', 'sklearn.svm',
    'sklearn.cross_decomposition', 'sklearn.gaussian_process', 'sklearn.inspection',
    'sklearn.impute', 'sklearn.neighbors', 'sklearn.isotonic',
    'joblib', 'threadpoolctl', 'matplotlib.backends.backend_agg',
    'matplotlib.backends.backend_tkagg', 'tkinter', '_tkinter',
    'tkinter.filedialog', 'tkinter.messagebox', 'tkinter.ttk',
    'PIL.ImageTk', 'PIL._tkinter_finder', 'openpyxl', 'docx', 'reportlab', 'pypdf',
    'rdkit.Chem.AllChem', 'rdkit.Chem.Crippen', 'rdkit.Chem.Descriptors',
    'rdkit.Chem.Lipinski', 'rdkit.Chem.rdMolDescriptors',
    'rdkit.Chem.rdChemReactions', 'rdkit.Chem.Draw',
)


def project_modules(root):
    """Enumerate source names without importing them in the build interpreter."""
    root = Path(root)
    result = ['app', 'packaging_tools', 'packaging_tools.catalog',
              'packaging_tools.runtime_check', 'packaging_tools.runtime_logging']
    for path in sorted((root / 'core').rglob('*.py')):
        if '__pycache__' in path.parts:
            continue
        rel = path.relative_to(root).with_suffix('')
        parts = list(rel.parts)
        if parts[-1] == '__init__':
            parts.pop()
        result.append('.'.join(parts))
    return sorted(set(result))


def dependencies(with_raw=True):
    return BASE_DEPENDENCIES + (RAW_DEPENDENCIES if with_raw else ())


def safe_submodule(name):
    parts = name.split('.')
    bad = {'tests', 'test', 'testing', 'conftest', 'test_data', 'Contrib',
           'IPythonConsole', 'Demo', 'Demos', 'examples'}
    return not any(p in bad or p.startswith('test_') for p in parts)

# Installation

## Existing installation

Use the same 64-bit Python environment and RAW backend as the working
18.48.2 installation. Extract this edition to a separate directory. Do not
copy it over a newer experimental branch, and do not copy an old `dist`
directory into this source tree.

```bat
cd /d D:\CombiTrace-MS
python -c "import sys; print(sys.executable)"
python app.py
```

The path above is an example. The interpreter printed by the second command
should be the interpreter previously used for the project.

## Fresh source installation

The preserved dependency constraints include NumPy below 2, scikit-learn
1.3.2, SciPy below 1.11 and fisher-py 1.0.22. Python 3.10 x64 is a reasonable
starting point for resolving these historical constraints. A fresh Windows
installation of that combination was not tested for this English edition;
retain the installed versions of a known-working environment when available.

```bash
python -m pip install -r requirements.txt
python -m pip install rdkit
python app.py
```

RDKit is required for structure generation and structure-derived descriptors.
The original requirements file is retained without changing its scientific
version constraints. Record the RDKit version used in a study rather than
silently upgrading it between the two chemical series.

The native window uses Tkinter. Test it with `python -m tkinter`. Tkinter is
usually provided by the Python distribution; it is not installed by a package
named `tkinter` from PyPI. On Linux, the corresponding Tcl/Tk system package
and a graphical display are needed.

## RAW backend

The application uses `fisher-py==1.0.22` and its RawFileReader/.NET bridge.
Install and configure that chain according to its own documentation, then
verify it in the same interpreter:

```bash
python -c "from fisher_py import RawFile; print('RAW bridge import succeeded')"
```

An import check is not a test of reading an experimental RAW file. Test a
small local file before processing a whole study. A .NET load failure differs
from a missing Python module. Do not repeatedly reinstall NumPy or the entire
scientific environment to fix a missing .NET runtime or vendor assembly.

For model-only or table-only use on a system without the RAW backend,
`requirements-analysis.txt` lists the non-RAW dependencies. This intentionally
does not provide RAW extraction. The optional ms2quant comparison also
requires a separate R installation and its packages; Python installation does
not install R automatically.

## Dependency errors

| Error | Check |
|---|---|
| `No module named 'sklearn'` | Install the `scikit-learn` distribution in the selected interpreter. |
| `No module named 'PIL'` | The distribution is `Pillow`. |
| `No module named 'docx'` | The distribution is `python-docx`. |
| `No module named 'rdkit'` | Install RDKit into the selected environment. |
| `No module named 'core'` | Extract the complete source tree and run from its root. |
| `clr`, `System`, or vendor assembly errors | Check Python.NET, .NET and RawFileReader together. |
| `DLL load failed` | Check native runtime dependencies and matching architecture. |

The source checker records module import errors and the interpreter path. It
does not install or remove packages.

References: [fisher_py](https://github.com/ethz-institute-of-microbiology/fisher_py),
[RawFileReader](https://github.com/thermofisherlsms/RawFileReader).

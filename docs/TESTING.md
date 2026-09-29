# Release testing

## Observed checks

All numerical examples and workbooks used for this English-release check were
synthetic. No study RAW file, concentration table or private structure library
was used.

| Check | Observed result |
|---|---|
| Unit suite | 254 tests passed: 235 retained tests plus 19 English/legacy-schema tests. |
| Source smoke checks | 28 passed with `--without-raw --self-test-gui`. |
| GUI | Ten retained tabs constructed; standard control text inspected for non-English labels. Three unrelated tabs absent. |
| Legacy tables | Chinese and earlier English header aliases, worksheet preference and preserved compound names tested. |
| New triplicate table | English headers and 3/3 mean values verified; generated table read back through the application reader. |
| Model regression | Same synthetic calibration/target data and settings run against pristine 18.48.2 and this edition. 3,357 numeric fields compared, zero mismatches at relative tolerance 1e-11 / absolute tolerance 1e-12; selected model and descriptor names identical. |
| Existing-XIC editor | Original values/formulas/caches and image recovery tested by the retained suite. |
| Public example | Three known rows, four targets and 21 synthetic PNGs exported; signed ppm and unchanged input verified. |
| Input templates | Four English .xlsx schemas recreated and checked; generic or empty fields, no private structures. |
| Python syntax | Project Python sources parsed with Python 3.8 grammar. This is not execution under Python 3.8. |

The deterministic numerical comparison uses 30 calibration rows, eight targets,
three validation folds and one repeat, with feature selection enabled and
outlier screening/deep tuning disabled. It covers one retained main-model
configuration, not every possible setting. The comparison is a regression
check for translation changes, not a benchmark of analytical accuracy.

## Reproduce

```bash
python -m unittest discover -s tests -p "test*.py" -v
python launch_reporter.py --self-test --without-raw --self-test-gui --self-test-output source_check.json
python examples/create_demo.py
```

Use a graphical desktop or `xvfb-run -a` for the Tk checks on Linux. The build
suite intentionally tests a missing module called `missing_xyz`; its diagnostic
message is expected when that unit test passes. Check the test runner's final
status rather than interpreting the fixture as a missing application dependency.

## Environment and untested scope

The recorded environment is available in `release_test_summary.json`. This
release was checked on Linux/Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0,
scikit-learn 1.8.0, RDKit 2025.09.4 and openpyxl 3.1.5.

The original pinned scientific requirements are retained. This is not evidence
that the original Windows dependency combination or a fresh installation of it
was tested. Real RAW reading through fisher_py/.NET, a Windows frozen build,
Microsoft Excel/WPS behaviour and real experimental results were not tested
here. The source self-check excludes RAW access explicitly. Optional R/ms2quant
execution was not verified.

## Change boundaries

UI layout, program text, standard schemas, compatibility readers and help/build
resources were updated. Three unused interfaces and their unreferenced report/
deuteration modules were removed. The old institutional report logo and
historical debug/change documents are not distributed. Core model formulas,
ion mass constants, integration methods and numeric scientific defaults were
not intentionally changed. Numeric and input/output regressions above are the
measured evidence for that scope, not an assertion that every core file is
byte-identical.

Input workbooks and user values are not translated in place. The new English
schemas are for generated output; compatibility code can read old schemas.
The executable must still pass its own Windows build/self-check sequence.

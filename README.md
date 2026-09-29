# CombiTrace-MS

**Version 18.48.2-en.1**

CombiTrace-MS is a desktop workflow for combinatorial product enumeration,
Thermo RAW XIC processing, triplicate internal-standard normalization, and
relative-response-factor (RRF) modelling. Results are exported as Excel
workbooks, CSV tables and figures. Processing is local; experimental data are
not uploaded by the application.

This English edition retains the analysis methods and workbook layout of
18.48.2. The unrelated batch spectrum-report, deuteration and report-signoff
tabs have been removed. It does not incorporate the later experimental
fine-resolution or local-structure modelling branches.

![Triplicate XIC interface](docs/images/triplicate_xic.png)

## Start

Use the Python environment that already runs the original application:

```bash
python app.py
```

Windows users can also run `START_COMBITRACE.bat`. For a new installation,
read [Installation](INSTALLATION.md) before installing dependencies. RAW access
requires the existing `fisher-py` / RawFileReader / .NET chain. Table editing
and model-only analysis do not require a RAW file to be opened.

The complete [User guide](docs/USER_GUIDE.md) includes input layouts, button
names, output tables and a reproducible synthetic demonstration. A local HTML
copy is accessible from **Help > User guide**. A [PDF edition](docs/USER_GUIDE.pdf)
is included for supplementary-material use.

## Workflow

| Tab | Purpose |
|---|---|
| Enumeration | Enumerate A/B/C component CSVs and retain duplicate formulas. |
| Triplicate XIC | Enumerate component worksheets; extract and integrate three replicate RAW files; review negative-ion channels. |
| SMILES | Construct and check product structures using mapped attachment sites. |
| ESI models | Generate structural/gradient descriptors or reuse a descriptor workbook; compare response models. |
| Final tables | Add theoretical and observed masses, ppm differences, XIC images and cross-table names. |
| Table tools | Update component labels and translate legacy summary schemas. |
| CSV filter | Remove formulas listed in a reference CSV. |
| Response | Run the original response-factor transfer model. |
| Local RRF | Run the original structure-guided local response transfer model. |
| XIC | Extract a target CSV from a RAW directory. |

A typical study uses **Triplicate XIC > SMILES > ESI models > Final tables**.
The other tabs are utilities or alternative analysis routes; they do not have
to be run in sequence. Negative-ion channel review is retained inside
Triplicate XIC. It is not the removed report-signoff feature.

## Data and interpretation

Known standards supply concentration labels. The larger unknown mixture is
used for prediction, not for assigning training concentrations to matching
names. The calibration table accepts up to 100 nonempty records; 94, 98 and 99
records are valid table sizes. Different series can be fitted separately with
the same descriptor definitions and selection settings.

The main response model learns `log10(RRF)`, where
`RRF = (target area / internal-standard area) / concentration`. Concentration
is recovered from the observed ratio and predicted RRF. Ratio values expressed
as percentages are also accepted, provided the same scale is used in the
calibration and target tables. No automatic division by 100 is inferred from
a column name.

Model completion and prediction coverage are not measures of quantitative
accuracy. Review fold errors, controls and trend diagnostics together. The
retained main workflow has global availability/redundancy prescreening and
optional data-dependent outlier filtering; its results should not be described
as fully independent external validation. The implementation and limits are
specified in [Methods](docs/METHODS.md).

`Exact_mass` is calculated neutral mass. It is not observed m/z. Instrument ppm
requires an observed ion m/z and the corresponding theoretical ion m/z.
Missing measurements remain missing. Source plots are not evidence of a
successful internal-standard QC decision by themselves.

## Compatibility

New interface text, generated schemas and standard messages are English.
Legacy Chinese and earlier English summary headers are recognized when
reading tables. Compound names, worksheet identities and original plot
content are not translated. The existing-XIC editor preserves previous
workbook values and formulas; it does not translate or regenerate an old
experimental workbook merely to change an image caption.

## Tests and Windows executable

```bash
python -m unittest discover -s tests -p "test*.py" -v
python launch_reporter.py --self-test --without-raw --self-test-gui --self-test-output source_check.json
```

Use a desktop display, or `xvfb-run` on a headless Linux system. The second
command excludes the RAW backend intentionally. See [Testing](docs/TESTING.md)
for the actual release checks and their limits.

Windows build scripts are included. Run `check_build_env.bat`, install the
build tools only when required, then run `build_exe.bat`. A build is accepted
only after the generated executable passes the included smoke check. Copy the
whole executable directory, including `_internal`. See
[Building on Windows](BUILD_EXE.md).

## Publication and licensing

This source archive contains no experimental RAW files, fitted study models,
private concentration tables, font files or third-party DLLs. The examples
are synthetic software demonstrations, not supplementary experimental data.

Before publishing a repository or supplementary release, complete
[Release checklist](docs/RELEASE_CHECKLIST.md), select a project license, and
add the actual authors, repository URL and associated publication to
[Citation](CITATION.md). No license or authorship has been assigned on the
maintainers' behalf. See [License status](LICENSE_STATUS.md) and
[Third-party notices](THIRD_PARTY_NOTICES.md).

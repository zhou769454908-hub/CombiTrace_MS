# Retained analysis methods

This note describes the main ESI benchmark in 18.48.2-en.1. It is an
implementation record, not a revised validation claim. The separate continuous
and multiview windows retain their own strategies. Their results are labelled
separately and should not be mixed with those of the main benchmark.

## Measurements and response model

For valid replicate `r`, the percentage response is
`R_ir = 100 × Area_ir / IS_Area_r`. A ratio replicate requires a found target,
a found internal standard, positive finite IS area and passing IS QC.
Means exclude invalid or missing replicates. The strict triplicate mean is
reported only for 3/3 valid replicates. Target area and ratio validity counts
are separate; missing values are not zero. Source Found and QC flags remain
available for review.

The benchmark fits `q_i = log10(R_i / C_i)` and predicts
`C_hat_j = R_j / 10**q_hat_j`. `R` must be on the same scale for standards and
targets. The formulation does not force different compounds to have equal
response factors. Theoretical exact mass, extraction tolerance and observed
mass deviation are distinct quantities.

## Descriptor generation

The library combines formula-derived values, standard RDKit 2D functions,
optional conformer-dependent 3D functions, pattern counts and transparent
computed interactions. Method gradients are interpolated at effective apex
RT. The original empirical pKa classes, charge/stability proxies and related
heuristics remain in this version. They are not measured pKa, quantum
energies, COSMO-WAPS or a verified simulation of electrospray.

Generation switches and reused-cache contents are different: reusing a cache
does not regenerate or erase its columns. Legacy privacy exports can retain
numeric descriptors while omitting complete Product_SMILES. Formula-only
identity links do not independently confirm isomers.

## Main feature-selection sequence

1. Combine fixed base fields, expected descriptor names and available descriptor
   names; remove duplicate names. Field order is retained.
2. In the current calibration set, require at least
   `max(5, ceil(0.5*N))` finite values for a numeric field. Reject variance
   at or below `1e-14` on available raw values.
3. Temporarily median-impute for pairwise correlation screening. At the default
   absolute Pearson threshold `0.97`, remove a later field redundant with an
   earlier retained field. This decision is not a mechanistic ranking.
4. Each fold-level pipeline fits numeric median imputation and, where applicable,
   StandardScaler. Optional categorical fields are one-hot encoded.
   `SafeSelectKBest` uses `f_regression` against log RRF to select up to `k`
   processed columns. This is top-k linear association, not a p<0.05 test,
   SHAP selection or feature-importance selection by the final ExtraTrees.
5. With automatic selection enabled, inner validation uses BayesianRidge and
   RBF-SVR to compare candidate counts. The default 6–40 range yields
   `6, 8, 12, 16, 24, 32, 40`. Choose the smallest count with error no greater
   than `best + max(0.005, 0.02*best)` for mean absolute log RRF error.
6. Count selections over recorded repeated outer folds. The default stability
   threshold is 50%. Retain by frequency, supplement to the configured minimum
   if needed, and truncate to the maximum. Frequency ties follow candidate
   order. If no events exist, default 100/0 flags are not stability evidence.
7. Form the retained raw descriptor set, then refit the final pipeline with
   `k` based on the median chosen fold count. A further SelectKBest can change
   the actual model-input columns.

The main GUI initially requests five folds, three repeats and seed 42; the
minimum and maximum descriptor settings are 6 and 40. The default final-model
objective is response correction/trend. Optional deep tuning and outlier modes
have GUI defaults that differ from some direct function defaults. Report
actual run settings, not this list as a substitute for the run record.

`Selected_Descriptors` describes retained raw fields. `Descriptor_Selection`
contains frequencies and reasons; `Descriptor_Tuning` contains count/error
comparisons. Final input masks should be taken from the fitted selector when
exact encoded columns are required. `Feature_Importance` is an exploratory
post-fit diagnostic, not independent causal evidence.

## Validation boundary

Initial availability, variance and redundancy screening use the current full
calibration set, not only each validation fold. The optional categorical
vocabulary is also constructed from that calibration set. Fold-level
estimators fit their own numeric imputation, scaling and score selectors.
The complete retained workflow therefore must not be described as having all
preprocessing strictly nested inside every fold.

Optional automatic outlier handling uses cross-model OOF residual consensus
and then reruns evaluation after exclusion. Post-screening metrics are
conditional on that data-dependent step. Preserve pre/post-QC counts,
reasons and results. A report is not external validation merely because its
predictions are labelled OOF. Repeated tuning against the same results can
also create dataset-level selection bias.

OOF fold error is `max(C_hat/C, C/C_hat)`. P80 is the empirical 80th percentile
of that error distribution; `[C_hat/P80, C_hat*P80]` is a shared empirical
range, not a compound-specific confidence interval. A high fraction within
2-fold error can coexist with poor resolution of nearby concentrations.
Report controls and trend limitations as well as typical error. Prediction
coverage is the proportion with numerical output, not predictive accuracy.

## Data and version integrity

The English release changes interface strings, standard export schemas, header
compatibility and documentation. It removes three unrelated interfaces and
unreferenced report/deuteration modules. Model feature rules, numerical
formulas and default scientific settings are not intentionally changed.
Schema aliases are applied to headers, not compound names or raw values.
The image editor preserves original mass, concentration and formula caches.
See TESTING.md for measured regression coverage and untested platforms.

## Sources

- Main implementation: `core/esi_model_benchmark.py`, `core/esi_descriptors.py`.
- Triplicates and channels: `core/excel_sheet_xic.py`, `core/xic_quant.py`.
- Structure: `core/structure_features.py`, `core/negative_esi_literature.py`.
- [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html).
- [SelectKBest](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html).
- [f_regression](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html).
- [RDKit descriptors](https://www.rdkit.org/docs/GettingStartedInPython.html).
- Liigand et al. Quantification for non-targeted LC/MS screening without
  standard substances. Scientific Reports 10, 5808 (2020).
  DOI: [10.1038/s41598-020-62573-z](https://doi.org/10.1038/s41598-020-62573-z).

These sources support API definitions and methodological context, not the
accuracy of a particular study or the project's empirical proxy coefficients.

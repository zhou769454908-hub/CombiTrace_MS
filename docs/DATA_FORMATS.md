# Data formats and table conventions

## Core inputs

| File | Required content |
|---|---|
| Component CSVs | Formula columns for A, B and C; identity fields retained in Combo. |
| Component Excel | A/B/C column blocks with formula/mass, optionally ID and mapped SMILES. |
| Calibration table | Compound identity, formula, positive finite known concentration and measured ratio; per-compound RT where needed. |
| Target table | Compound identity, formula and measured ratio; no fabricated concentration labels. |
| Structure master | Verified full product SMILES plus stable Combo/structure/formula evidence. |
| ESI descriptor cache | `ESI_Features` sheet from an ESI export. Privacy export may omit full structures. |
| Original XIC indices | Per-RAW `__XIC_quant.csv`, enumeration CSV and triplicate summary; keep alongside plot folders. |
| Final illustrated table | Existing postprocessed Excel with its audit and drawing metadata. |

CSV readers accept UTF-8/UTF-8 BOM and legacy supported Chinese encodings.
Use `.xlsx` or unencrypted `.xlsm` for Excel. Legacy `.xls` must be converted
before use. Protected or signed workbooks are not bypassed by the image editor.

## English triplicate fields

New triplicate summaries use these standard headers:

| Field | Meaning |
|---|---|
| `Rep1_Found`, `Rep2_Found`, `Rep3_Found` | Target detection status in each replicate. |
| `RepN_Area`, `RepN_Apex_RT` | Target peak area and apex RT for N=1,2,3. |
| `RepN_Ratio_%` | Target/IS peak-area ratio multiplied by 100. |
| `RepN_IS_Found`, `RepN_IS_QC` | Internal-standard detection and QC state. |
| `RepN_IS_Area`, `RepN_IS_Apex_RT` | Internal-standard area and apex RT. |
| `Valid_Area_Replicate_Count` | Number of valid target-area replicates. |
| `Valid_Ratio_Replicate_Count` | Number of valid target/IS ratio replicates. |
| `Triplicates_Complete` | Completeness flag from the retained averaging rules. |
| `Mean_Peak_Area`, `Peak_Area_SD`, `Peak_Area_RSD_%` | Valid-replicate area summary. |
| `Mean_Area_to_IS_Ratio_%`, `Ratio_SD`, `Ratio_RSD_%` | Valid-replicate percentage-ratio summary. |
| `Mean_Apex_RT` | Averaged apex RT under the retained valid-data rules. |
| `Final_Triplicate_Mean_Result` | Selected reported mean. |
| `Strict_Triplicate_Mean_Result_3of3` | Mean only when all three applicable replicates are valid. |
| `Result_Type`, `Valid_Result_Replicate_Count` | Result definition and usable replicate count. |
| `Averaging_Rule`, `IS_QC_Notes` | Explicit averaging and internal-standard QC notes. |

`Observed_Fragility_Index`, `Adduct_Cluster_Proneness_Index`,
`Primary_Ion_Fraction`, `Ion_Form_Diversity_Count`, `Accepted_Channel_Count`,
`Summed_Channel_Count`, `Br_Fragment_Fraction`, `Negative_Ion_Panel` and
`Accepted_Channels` retain their names. They refer to observed/processed channel
behaviour, not directly measured fundamental molecular energies.

Legacy Chinese and earlier English spellings are accepted through
`core/legacy_schema.py`. Header aliases are added to in-memory records so
explicit column selections remain valid. If two equivalent columns contain
conflicting values, the reader requests resolution instead of silently
choosing one. Compound names, Combo values and user paths are not translated.

New generated standard sheets include `Run Summary`, `Triplicate Average`,
`Triplicate XIC Summary`, `XIC Long Table` and `Internal Standard`. Input sheets
retain their actual identities; existing-table editing does not rename every
old sheet or translate original content.

## Mass fields

| Field | Meaning |
|---|---|
| `Exact_mass` / `Theoretical_mass` | Formula-derived neutral monoisotopic mass. |
| `Observed_adduct` | Single-ion identity used for mass comparison. |
| `Theoretical_mz` | Formula/adduct-derived ion m/z. |
| `Observed_mz` | Supplied observation or actual centroid read from the selected RAW scan. |
| `Observed_neutral_mass` | Neutral mass inferred from the observed ion and specified adduct. |
| `Mass_difference_ppm` | Signed observed-versus-theoretical ion error in m/z mode. |
| `Mass_QC` | Source definition, missingness, ambiguity and threshold checks. |

Do not confuse `XIC_mz` (often extraction centre) or a `PPM` tolerance with an
observed centroid or a measured ppm deviation. Source scans, RTs and replicates
are stored separately. A theoretical-to-theoretical comparison may be zero.

## Audit and privacy

Generated logs, workbook audits, model files, drawing alternative descriptions
and retained embedded image blocks may contain compound identities, local paths
or training information. Removing a visible image caption does not anonymize
the workbook. Only share the aggregate summary after reviewing its contents;
never assume a fitted model or full workbook is de-identified.

# Release checklist

## Repository metadata

- Select the project license with the rights holder; add the actual license text.
- Enter the real authors, repository URL and associated article in `CITATION.md`.
- Tag the release and record its commit. Archive the software version cited by the manuscript.

## Files to publish

Publish source, documentation, schema templates and explicitly labelled
synthetic examples. Do not publish `build_records`, `dist`, local environments,
RAW files, private structure workbooks, fitted models or unreviewed audit logs.
The `.gitignore` excludes typical run outputs but is not a confidentiality
review. Inspect the staged files before uploading.

Audits, hidden workbook sheets, embedded pictures and model files can retain
compound identities and local paths even when visible names have been removed.
Do not describe a caption-free workbook as de-identified without inspecting it.

## Scientific supplementary material

For each chemical series, report the actual standard count, exclusions and
reasons, concentration unit, ratio scale, internal-standard definition, ion
panel, gradient, descriptor options, model-selection objective, random seed,
validation settings and final predictor. Preserve the fold-level predictions
and selection records. Do not combine metrics from one predictor with a target
table exported by another predictor.

The source package does not include study data or establish that the method
achieves a particular accuracy. Retain unsuccessful diagnostics and the
limitations of post-QC internal validation.

## Local release check

Run the tests and source checker, then check one representative real RAW file
on the intended Windows system. Verify the exported workbook in the spreadsheet
application used for submission. A Windows EXE requires a separate successful
frozen-application check and appropriate third-party distribution permissions.

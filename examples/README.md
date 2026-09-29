# Synthetic demonstration

Run `python examples/create_demo.py` from the repository root. The script creates
three known-table rows, four target-table rows and 21 labelled synthetic XIC
images, then exercises the real final-table exporter. It checks the signed ppm
values, image counts and read-only input policy. No RAW file or network is used.

The example identities, formulas, chromatograms and m/z offsets are software
fixtures. They are not experimental standards or validation data for the RRF
model. Generated workbooks are excluded from version control. Use the unit tests
for model/legacy-schema regression, not the demo as a claim of model accuracy.

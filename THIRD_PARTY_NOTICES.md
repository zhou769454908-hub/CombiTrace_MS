# Third-party notices

CombiTrace-MS is independent software. It is not a Thermo Fisher Scientific
product and does not imply endorsement by an instrument or library vendor.

## Thermo RawFileReader

**RawFileReader reading tool. Copyright © 2016 by Thermo Fisher Scientific,
Inc. All rights reserved.**

RAW access is made through the locally installed fisher_py / RawFileReader
chain. The RawFileReader assemblies have separate vendor terms; the fisher_py
project specifically notes restrictions on redistribution by recipients of
its bundled assemblies. No vendor assemblies are included in this source
archive. Obtain any required distribution permission before releasing a
frozen application containing them.

Primary sources:

- [RawFileReader project](https://github.com/thermofisherlsms/RawFileReader)
- [fisher_py RawFileReader license](https://github.com/ethz-institute-of-microbiology/fisher_py/blob/main/RawFileReaderLicense.md)

## fisher_py

The fisher_py wrapper is Copyright 2021 ethz-institute-of-microbiology and is
published under the MIT license. Its wrapper license does not replace the
RawFileReader license. The wrapper source and binaries are installed as a
dependency rather than copied into this repository.

Source and license: [fisher_py](https://github.com/ethz-institute-of-microbiology/fisher_py).

## Other dependencies

The application imports RDKit, NumPy, SciPy, scikit-learn, Matplotlib,
openpyxl, Pillow, python-docx, ReportLab, pypdf, Python.NET and related runtime
packages. These are not vendored into this source archive. Preserve the
license notices supplied with each installed package when distributing any
permitted compiled bundle. Optional R/ms2quant components require their own
installation and terms.

No font files are distributed. Figure and caption rendering uses fonts
available on the local system.

# Building a Windows executable

Build on 64-bit Windows, from the Python environment that runs this source
release successfully. PyInstaller bundles the active interpreter and is not a
cross-platform compiler. This archive does not contain a precompiled EXE.

## Build sequence

```bat
check_build_env.bat
install_build_tools.bat
build_exe.bat
```

Run `install_build_tools.bat` only when the checker requests it. It installs
PyInstaller and its hook package, not the full scientific requirements. For a
non-default interpreter, set its actual path first:

```bat
set "THERMO_BUILD_PYTHON=C:\path\to\environment\python.exe"
build_exe.bat
```

The default build keeps a console for diagnosis. After it works, use
`build_exe_windowed.bat` for a windowed executable. Both routes run the same
source and frozen-application checks.

## Successful output

A successful attempt prints `BUILD + FROZEN SMOKE CHECK PASSED` and writes
`BUILD_SUCCESS.json` in that attempt's `build_records` directory. An EXE
created before a failed smoke check is not a completed build. Logs include
`environment.json`, `source_selftest.log`, `pyinstaller.log`, and
`frozen_selftest.log`; `BUILD_FAILED.json` identifies a failed stage.

The executable name is `CombiTraceMS_v18_48_2_en1.exe`. Keep the entire output
directory:

```text
CombiTraceMS_v18_48_2_en1/
    CombiTraceMS_v18_48_2_en1.exe
    _internal/
    RUN_WITH_LOG.bat
    OPEN_MULTIVIEW.bat
    OPEN_FINAL_TABLES.bat
    EDIT_EXISTING_XIC.bat
```

Use a desktop shortcut instead of moving the EXE away from `_internal`.
Package collection includes the project modules, templates and documentation;
it does not recursively collect experimental output directories.

`python build_exe.py --without-raw` is a deliberately reduced build without RAW
reading. Its name contains `_NoRAW`. It is not a workaround that restores
full RAW functionality.

## Distribution

Do not upload a built directory containing Thermo assemblies to GitHub merely
because it builds successfully. Check the applicable redistribution permissions
first. The public source release intentionally excludes those binaries.

References: [PyInstaller operating mode](https://pyinstaller.org/en/stable/operating-mode.html)
and [third-party notices](THIRD_PARTY_NOTICES.md).

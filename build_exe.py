"""Reproducible Windows onedir builder with fail-fast checks and persistent logs.

Uses the CURRENT interpreter; never upgrades or reinstalls scientific packages.
Each run writes a NEW build_records subfolder; no old build/dist folder is erased.
"""
import argparse
from datetime import datetime
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import traceback
import uuid

from packaging_tools.catalog import APP_NAME, BUILD_REVISION, dependencies, project_modules
ROOT = Path(__file__).resolve().parent


def version_of(dist):
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def stream_command(command, logfile, *, cwd, env=None, timeout=1800):
    """Run with output redirected to a file (bounded memory; timeout is enforced)."""
    print('RUN: ' + subprocess.list2cmdline([str(x) for x in command]), flush=True)
    print('Log: ' + str(logfile), flush=True)
    env = dict(os.environ if env is None else env, PYTHONUTF8='1')
    with open(str(logfile), 'w', encoding='utf-8') as stream:
        stream.write('Command: ' + subprocess.list2cmdline([str(x) for x in command]) + '\n')
        stream.flush()
        try:
            proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, cwd=str(cwd), env=env)
        except Exception as exc:
            stream.write(traceback.format_exc())
            raise RuntimeError('Cannot start process: ' + str(exc)) from exc
        started = time.monotonic()
        try:
            while True:
                left = timeout - (time.monotonic() - started)
                if left <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    return proc.wait(timeout=min(15, left))
                except subprocess.TimeoutExpired:
                    if time.monotonic() - started >= timeout:
                        raise
                    print('[working] %.0f s; details are being written to %s' %
                          (time.monotonic() - started, logfile.name), flush=True)
        except subprocess.TimeoutExpired:
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'],
                               stdout=stream, stderr=subprocess.STDOUT, check=False)
            else:
                proc.kill()
            proc.wait()
            stream.write('\nTIMEOUT: process terminated.\n')
            return 124


def source_preflight(run, with_raw):
    details = {'python': sys.version, 'executable': sys.executable,
               'platform': platform.platform(), 'prefix': sys.prefix,
               'packages': [], 'with_raw': with_raw}
    missing = []
    for module, dist in dependencies(with_raw):
        try:
            spec = importlib.util.find_spec(module)
            origin = spec.origin if spec else None
            error = '' if spec else 'module not found'
        except Exception as exc:
            origin, error = None, repr(exc)
        row = {'module': module, 'distribution': dist, 'version': version_of(dist),
               'origin': origin, 'error': error}
        details['packages'].append(row)
        if error:
            missing.append(row)
    details['pyinstaller'] = version_of('pyinstaller')
    details['hooks_contrib'] = version_of('pyinstaller-hooks-contrib')
    write_json(run / 'environment.json', details)
    if missing:
        lines = ['Missing dependencies in the selected Python interpreter:', sys.executable]
        lines += ['  import %-20s -> distribution %s (%s)' % (r['module'], r['distribution'], r['error']) for r in missing]
        lines += ['', 'Use the SAME environment that successfully runs app.py.',
                  'No package was installed/removed by this check.',
                  'Do not run "pip install docx" or "pip install sklearn"; these are import names, not the intended distributions.',
                  'See BUILD_EXE.md for dependency-specific guidance.']
        text = '\n'.join(lines)
        (run / 'missing_dependencies.txt').write_text(text, encoding='utf-8')
        print(text, flush=True)
        return False
    command = [sys.executable, str(ROOT / 'launch_reporter.py'), '--self-test', '--self-test-gui',
               '--self-test-output', str(run / 'source_selftest.json')]
    if not with_raw:
        command += ['--without-raw']
    code = stream_command(command, run / 'source_selftest.log', cwd=ROOT, timeout=420)
    try:
        valid = json.loads((run / 'source_selftest.json').read_text(encoding='utf-8'))['status'] == 'PASS'
    except Exception:
        valid = False
    return code == 0 and valid


def isolated_runtime_env():
    env = dict(os.environ)
    for key in list(env):
        if key.upper() in ('PYTHONHOME', 'PYTHONPATH'):
            env.pop(key, None)
    # Avoid accidentally resolving DLLs from the development interpreter.
    if os.name == 'nt':
        prefixes = [Path(sys.prefix).resolve(), ROOT.resolve()]
        path_key = next((k for k in env if k.upper() == 'PATH'), 'PATH')
        kept = []
        for item in env.get(path_key, '').split(os.pathsep):
            if not item:
                continue
            try:
                path = Path(item).resolve()
                below = any(path == p or p in path.parents for p in prefixes)
            except Exception:
                below = False
            if not below:
                kept.append(item)
        env[path_key] = os.pathsep.join(kept)
    return env


def validate_frozen(run, exe):
    """EXE existence alone is NOT success. Require the frozen smoke report."""
    if not exe.is_file():
        return False
    cwd = run / 'empty_smoke_cwd'; cwd.mkdir(exist_ok=True)
    out = run / 'frozen_selftest.json'
    command = [str(exe), '--self-test', '--self-test-gui', '--self-test-output', str(out)]
    rc = stream_command(command, run / 'frozen_selftest.log', cwd=cwd,
                        env=isolated_runtime_env(), timeout=420)
    try:
        report = json.loads(out.read_text(encoding='utf-8'))
    except Exception:
        return False
    return rc == 0 and report.get('status') == 'PASS' and report.get('frozen') is True and report.get('gui_tested') is True


def build(args, run):
    with_raw = not args.without_raw
    if not args.check_only:
        if os.name != 'nt':
            raise RuntimeError('Windows EXE build must run on Windows; this is not a cross-compiler.')
        if sys.maxsize <= 2 ** 32:
            raise RuntimeError('Use a 64-bit Python environment for this build.')
    if sys.version_info < (3, 8):
        raise RuntimeError('Python 3.8 or newer is required.')
    print('Selected interpreter: ' + sys.executable, flush=True)
    print('Logs/results: ' + str(run), flush=True)
    if not source_preflight(run, with_raw):
        raise RuntimeError('Source dependency/smoke check failed. See missing_dependencies.txt or source_selftest.log.')
    if args.check_only:
        write_json(run / 'check_result.json', {'status': 'SOURCE_CHECK_PASS', 'built_exe': False})
        print('Source dependency check passed. No executable was built.', flush=True)
        return
    pyiv = version_of('pyinstaller')
    if not pyiv or int(pyiv.split('.')[0]) < 6 or int(pyiv.split('.')[0]) >= 7:
        raise RuntimeError('PyInstaller 6.x is required. Run install_build_tools.bat in this same environment; then rebuild.')
    if not version_of('pyinstaller-hooks-contrib'):
        raise RuntimeError('pyinstaller-hooks-contrib is missing. Run install_build_tools.bat first.')
    exe_name = APP_NAME + ('_NoRAW' if not with_raw else '')
    config = {'revision': BUILD_REVISION, 'with_raw': with_raw, 'console': not args.windowed,
              'exe_name': exe_name, 'modules': project_modules(ROOT)}
    config_path = run / 'bundle_config.json'; write_json(config_path, config)
    env = dict(os.environ, THERMO_BUILD_ROOT=str(ROOT), THERMO_BUILD_CONFIG=str(config_path), PYTHONUTF8='1')
    command = [sys.executable, '-m', 'PyInstaller', '--clean', '--noconfirm',
               '--distpath', str(run / 'dist'), '--workpath', str(run / 'work'), str(ROOT / 'ThermoRawReporter.spec')]
    rc = stream_command(command, run / 'pyinstaller.log', cwd=ROOT, env=env, timeout=3600)
    if rc:
        raise RuntimeError('PyInstaller failed with exit code %s. See pyinstaller.log (no success declared).' % rc)
    exe = run / 'dist' / exe_name / (exe_name + '.exe')
    if not validate_frozen(run, exe):
        raise RuntimeError('EXE created but frozen self-test FAILED. See frozen_selftest.log/json. Do not deploy this build.')
    notes = ('Build smoke checks passed on the build PC. Keep this ENTIRE folder together.\n'
             'Do not move only the EXE or delete _internal.\n'
             'Real RAW reading and external R/Office tools were not tested by the build smoke test.\n'
             'For RAW support the target PC needs a compatible .NET runtime.\n'
             'Thermo RawFileReader redistribution requires appropriate vendor rights.\n')
    (exe.parent / 'KEEP_ENTIRE_FOLDER.txt').write_text(notes, encoding='utf-8')
    runner = '@echo off\r\ncd /d "%~dp0"\r\n"%~dp0' + exe.name + '" %*\r\npause\r\n'
    (exe.parent / 'RUN_WITH_LOG.bat').write_bytes(runner.encode('ascii'))
    multi = '@echo off\r\ncd /d "%~dp0"\r\n"%~dp0' + exe.name + '" --multiview\r\n'
    (exe.parent / 'OPEN_MULTIVIEW.bat').write_bytes(multi.encode('ascii'))
    final_tables = multi.replace('--multiview', '--postprocess')
    (exe.parent / 'OPEN_FINAL_TABLES.bat').write_bytes(final_tables.encode('ascii'))
    (exe.parent / 'EDIT_EXISTING_XIC.bat').write_bytes(multi.replace('--multiview', '--existing-xic').encode('ascii'))
    write_json(run / 'BUILD_SUCCESS.json', {'status': 'BUILD_AND_SMOKE_PASS', 'exe': str(exe),
                                          'whole_folder': str(exe.parent), 'real_raw_tested': False,
                                          'other_pc_tested': False, 'revision': BUILD_REVISION})
    (ROOT / 'LAST_BUILD_SUCCESS.txt').write_text(str(exe.parent) + '\n', encoding='utf-8')
    print('\nBUILD + FROZEN SMOKE CHECK PASSED\n' + str(exe) + '\nKeep the WHOLE folder including _internal.', flush=True)
    if not args.no_open:
        os.startfile(str(exe.parent))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--windowed', action='store_true', help='Build without console; crash log remains enabled')
    p.add_argument('--without-raw', action='store_true', help='Explicitly omit RAW reader; EXE is labeled NoRAW')
    p.add_argument('--no-open', action='store_true')
    args = p.parse_args(argv)
    run = ROOT / 'build_records' / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6])
    run.mkdir(parents=True, exist_ok=False)
    (ROOT / 'LAST_BUILD_ATTEMPT.txt').write_text(str(run) + '\n', encoding='utf-8')
    try:
        build(args, run)
    except Exception as exc:
        info = {'status': 'FAILED', 'error': str(exc), 'traceback': traceback.format_exc(),
                'python': sys.executable, 'revision': BUILD_REVISION}
        write_json(run / 'BUILD_FAILED.json', info)
        print('\nFAILED: %s\nLogs: %s' % (exc, run), flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

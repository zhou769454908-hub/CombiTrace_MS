"""Dependency-free crash reporting; never reads user experiment files."""
import logging
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback


def setup_logging():
    roots = [Path(os.environ.get('LOCALAPPDATA', str(Path.home()))) /
             'ThermoRawReporter' / 'logs', Path(tempfile.gettempdir()) / 'ThermoRawReporter_logs']
    for root in roots:
        try:
            root.mkdir(parents=True, exist_ok=True)
            path = root / ('startup_%s_%s.log' % (time.strftime('%Y%m%d_%H%M%S'), os.getpid()))
            stream = open(str(path), 'a', encoding='utf-8', buffering=1)
            break
        except OSError:
            continue
    else:
        return None
    # PyInstaller windowed executables may have no stdout/stderr.
    if sys.stdout is None:
        sys.stdout = stream
    if sys.stderr is None:
        sys.stderr = stream
    logging.basicConfig(filename=str(path), level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s', force=True)
    logging.info('Python=%s; executable=%s; frozen=%s', sys.version, sys.executable,
                 bool(getattr(sys, 'frozen', False)))
    return path


def notify_exception(exc_type, exc, tb, log_path, show_dialog=True):
    text = ''.join(traceback.format_exception(exc_type, exc, tb))
    logging.error('%s', text)
    try:
        sys.stderr.write(text + '\n')
    except Exception:
        pass
    if not show_dialog:
        return
    short = '%s: %s' % (exc_type.__name__, exc)
    message = ('The application encountered an error.\n\n' + short +
               '\n\nFull error log:\n' + str(log_path or 'console') +
               "\n\nKeep the complete module name from any 'No module named' error. Run the executable from the complete dist folder; do not move the EXE alone.")
    try:
        import tkinter as tk
        from tkinter import messagebox
        existing = getattr(tk, '_default_root', None)
        root = existing or tk.Tk()
        if existing is None:
            root.withdraw()
        messagebox.showerror('ThermoRawReporter startup error', message, parent=root)
        if existing is None:
            root.destroy()
    except Exception:
        if os.name == 'nt':
            try:
                import ctypes
                ctypes.windll.user32.MessageBoxW(None, message, 'ThermoRawReporter error', 0x10)
            except Exception:
                pass


def install_handlers(log_path):
    sys.excepthook = lambda t, e, tb: notify_exception(t, e, tb, log_path)
    try:
        import threading
        threading.excepthook = lambda a: notify_exception(a.exc_type, a.exc_value, a.exc_traceback, log_path)
    except Exception:
        pass
    import tkinter as tk
    def callback_error(self, exc_type, exc, tb):
        notify_exception(exc_type, exc, tb, log_path)
    tk.Tk.report_callback_exception = callback_error

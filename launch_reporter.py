"""Frozen-safe entry point; leaves app.py and every scientific core module unchanged."""
import multiprocessing
import sys


def main():
    # Required before heavy imports/argument parsing for Windows frozen workers.
    multiprocessing.freeze_support()
    from packaging_tools.runtime_logging import setup_logging, install_handlers, notify_exception
    log_path = setup_logging()
    try:
        import argparse
        parser = argparse.ArgumentParser(description='CombiTrace-MS desktop launcher')
        parser.add_argument('--self-test', action='store_true')
        parser.add_argument('--self-test-gui', action='store_true')
        parser.add_argument('--self-test-output')
        parser.add_argument('--without-raw', action='store_true', help='Source smoke check only; omit RAW API')
        parser.add_argument('--multiview', action='store_true')
        parser.add_argument('--continuous', action='store_true')
        parser.add_argument('--postprocess', action='store_true')
        parser.add_argument('--existing-xic', action='store_true')
        args = parser.parse_args()
        if args.self_test:
            from packaging_tools.runtime_check import run_checks, read_bundle_config
            config = read_bundle_config()
            # A full frozen build must always check RAW; it cannot be overridden by CLI.
            raw = config['with_raw'] if getattr(sys, 'frozen', False) else not args.without_raw
            return 0 if run_checks(with_raw=raw, gui=args.self_test_gui,
                                   output=args.self_test_output)['status'] == 'PASS' else 2
        install_handlers(log_path)
        if args.existing_xic:
            from core.existing_xic_lab import main as run
            run()
        elif args.postprocess:
            from core.final_table_lab import main as run
            run()
        elif args.multiview:
            from core.multiview_lab import main as run
            run()
        elif args.continuous:
            from core.continuous_lab import main as run
            run()
        else:
            from app import ThermoBatchReportApp
            app = ThermoBatchReportApp()
            from packaging_tools.runtime_check import read_bundle_config
            if not read_bundle_config().get('with_raw', True):
                app.title(app.title() + ' [NO RAW READER IN THIS BUILD]')
            app.mainloop()
        return 0
    except Exception:
        notify_exception(*sys.exc_info(), log_path, show_dialog='--self-test' not in sys.argv)
        return 1


if __name__ == '__main__':
    sys.exit(main())

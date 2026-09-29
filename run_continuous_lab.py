"""Launch the independent lab, or run it from the command line with --cache."""
from __future__ import annotations
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='CombiTrace v18.45 continuous RRF audit (read-only inputs)')
    parser.add_argument('--cache', help='Existing ESI descriptor workbook containing ESI_Features')
    parser.add_argument('--training', default='', help='Optional current authoritative training XLSX/CSV (<=100 rows)')
    parser.add_argument('--target', default='', help='Optional current target XLSX/CSV')
    parser.add_argument('--training-sheet', default='')
    parser.add_argument('--target-sheet', default='')
    parser.add_argument('--output', default='v18_45_runs', help='Parent folder; each run creates a new unique child')
    parser.add_argument('--unit', default='same as training table')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--folds', type=int, default=5)
    args = parser.parse_args()
    if not args.cache:
        from core.continuous_lab import main as gui_main
        gui_main()
        return
    if not 3 <= args.folds <= 10:
        parser.error('--folds must be between 3 and 10')
    from core.continuous_io import execute
    folder, _ = execute(args.cache, args.output, training_path=args.training, target_path=args.target,
                       training_sheet=args.training_sheet, target_sheet=args.target_sheet,
                       concentration_unit=args.unit, seed=args.seed, folds=args.folds, progress=print)
    print('Saved to:', folder)


if __name__ == '__main__':
    main()

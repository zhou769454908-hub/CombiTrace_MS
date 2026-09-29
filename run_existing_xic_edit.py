"""Edit XIC presentation in an existing result; no RAW backend is needed."""
if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    from core.existing_xic_lab import main
    main()

"""Independent entry, no RAW/model setup needed for final-table processing."""
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    from core.final_table_lab import main
    main()

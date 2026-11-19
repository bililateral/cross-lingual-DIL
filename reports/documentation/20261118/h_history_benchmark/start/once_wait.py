import time

if __name__ == "__main__":
    started = time.monotonic()
    from step28_benchmark_wait import main
    main(started)

from numpy_bench_helper import NumpyBenchSuite, np


def main():
    suite = NumpyBenchSuite(
        "NumPy Timsort & Argsort Comprehensive Benchmark Suite",
        "sort",
    )
    sizes = [1000, 10000, 50000]

    def register_track(label: str, gen_fn):
        suite.group(label)
        for size in sizes:
            template = gen_fn(size)
            target = np.empty_like(template)
            iters = 300 if size <= 10000 else 100

            suite.bench(
                f"Direct sort() [{size}]",
                lambda t=target: np.sort(t, kind="quicksort"),
                setup_fn=lambda dst=target, src=template: np.copyto(dst, src),
                iterations=iters,
            )
            suite.bench(
                f"Indirect argsort() [{size}]",
                lambda t=target: np.argsort(t, kind="quicksort"),
                setup_fn=lambda dst=target, src=template: np.copyto(dst, src),
                iterations=iters,
            )

    register_track(
        "Random Array",
        lambda sz: np.random.default_rng(42).random(sz, dtype=np.float64) * 1000.0,
    )
    register_track(
        "Already Sorted",
        lambda sz: np.arange(sz, dtype=np.float64),
    )
    register_track(
        "Reverse Sorted",
        lambda sz: np.arange(sz, 0, -1, dtype=np.float64),
    )

    suite.finish()


if __name__ == "__main__":
    main()

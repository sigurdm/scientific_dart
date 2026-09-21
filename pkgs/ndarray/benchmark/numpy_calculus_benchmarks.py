from numpy_bench_helper import NumpyBenchSuite, np


def main():
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    suite = NumpyBenchSuite(
        "NumPy Calculus Benchmarks",
        "calculus",
    )

    y_1d = np.arange(1000000, dtype=np.float64)
    f_1d = np.arange(1000000, dtype=np.float64) ** 2
    f_2d = np.arange(1000000, dtype=np.float64).reshape((1000, 1000))

    suite.bench(
        "Calculus | trapz 1D (Float64) [size=1,000,000]",
        lambda: trapz_fn(y_1d),
        iterations=100,
    )
    suite.bench(
        "Calculus | gradient 1D (Float64) [size=1,000,000]",
        lambda: np.gradient(f_1d),
        iterations=100,
    )
    suite.bench(
        "Calculus | gradient 2D (Float64) [size=1,000x1,000]",
        lambda: np.gradient(f_2d, axis=0),
        iterations=100,
    )

    suite.finish()


if __name__ == "__main__":
    main()

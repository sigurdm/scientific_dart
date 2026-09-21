from numpy_bench_helper import NumpyBenchSuite, np


def main():
    size = 100000
    grid_dim = 500
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))

    suite = NumpyBenchSuite(
        "NumPy Calculus & Numerical Integration Benchmark Suite",
        "calculus_integration",
    )

    vec1d = np.linspace(0.0, 100.0, size, dtype=np.float64)
    grid2d = np.linspace(0.0, 100.0, grid_dim * grid_dim, dtype=np.float64).reshape(
        (grid_dim, grid_dim)
    )

    suite.group("1. Numerical Differentiation")
    suite.bench(
        "diff(a, n=1) [100k points]",
        lambda: np.diff(vec1d, n=1),
        iterations=300,
    )
    suite.bench(
        "diff(a, n=2) [100k points]",
        lambda: np.diff(vec1d, n=2),
        iterations=300,
    )
    suite.bench(
        "gradientArray(grid2D) [500x500]",
        lambda: np.gradient(grid2d),
        iterations=150,
    )

    suite.group("2. Numerical Integration")
    suite.bench(
        "trapz(y, spacing=0.01) [100k points]",
        lambda: trapz_fn(vec1d, dx=0.01),
        iterations=300,
    )
    suite.bench(
        "trapz(grid2D, axis=0) [500x500]",
        lambda: trapz_fn(grid2d, dx=0.01, axis=0),
        iterations=200,
    )

    suite.finish()


if __name__ == "__main__":
    main()

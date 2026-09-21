from numpy_bench_helper import NumpyBenchSuite, np


def main():
    suite = NumpyBenchSuite(
        "NumPy Non-Contiguous Strided Math Benchmark",
        "strided_math",
    )

    mat = (np.arange(1500 * 1500, dtype=np.float64) / 100000.0).reshape(
        (1500, 1500)
    )
    mat_t = mat.T
    out = np.empty((1500, 1500), dtype=np.float64)

    suite.bench(
        "strided tan(matT) [shape=1500x1500 transposed]",
        lambda: np.tan(mat_t, out=out),
        iterations=30,
        warmup=5,
    )
    suite.bench(
        "strided exp(matT) [shape=1500x1500 transposed]",
        lambda: np.exp(mat_t, out=out),
        iterations=30,
        warmup=5,
    )

    suite.finish()


if __name__ == "__main__":
    main()

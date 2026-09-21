from numpy_bench_helper import NumpyBenchSuite, np


def main():
    suite = NumpyBenchSuite(
        "NumPy Non-Contiguous sin() Performance Benchmark",
        "strided_trig",
    )

    mat = (np.arange(2000 * 2000, dtype=np.float64) / 10000.0).reshape(
        (2000, 2000)
    )
    mat_t = mat.T
    out = np.empty((2000, 2000), dtype=np.float64)

    suite.bench(
        "strided sin(matT) [shape=2000x2000 transposed]",
        lambda: np.sin(mat_t, out=out),
        iterations=25,
        warmup=4,
    )

    suite.finish()


if __name__ == "__main__":
    main()

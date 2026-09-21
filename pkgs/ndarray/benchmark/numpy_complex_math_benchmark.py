from numpy_bench_helper import NumpyBenchSuite, np


def main():
    size = 100000

    suite = NumpyBenchSuite(
        "NumPy Complex Number Vectorized Math Benchmark Suite",
        "complex_math",
    )

    idx = np.arange(size, dtype=np.float64)
    c_a = (idx * 0.01) + 1j * ((size - idx) * 0.01)
    c_b = ((idx % 100.0) + 1.0) - 0.5j
    c_a = c_a.astype(np.complex128)
    c_b = c_b.astype(np.complex128)

    suite.group("1. Vectorized Complex Arithmetic (Complex128)")
    suite.bench(
        "cMul (cA * cB) [size=100,000 Complex128]",
        lambda: np.multiply(c_a, c_b),
        iterations=300,
    )
    suite.bench(
        "cDiv (cA / cB) [size=100,000 Complex128]",
        lambda: np.divide(c_a, c_b),
        iterations=300,
    )
    suite.bench(
        "cAdd (cA + cB) [size=100,000 Complex128]",
        lambda: np.add(c_a, c_b),
        iterations=300,
    )

    suite.group("2. Complex Transformations & Projections")
    suite.bench(
        "conj(cA) [size=100,000 Complex128]",
        lambda: np.conj(c_a),
        iterations=300,
    )
    suite.bench(
        "abs(cA) (Magnitude) [size=100,000 Complex128 -> Float64]",
        lambda: np.abs(c_a),
        iterations=300,
    )
    suite.bench(
        "angle(cA) (Phase) [size=100,000 Complex128 -> Float64]",
        lambda: np.angle(c_a),
        iterations=200,
    )
    suite.bench(
        "exp(cA) (Complex Exponential) [size=100,000 Complex128]",
        lambda: np.exp(c_a),
        iterations=200,
    )

    suite.finish()


if __name__ == "__main__":
    main()

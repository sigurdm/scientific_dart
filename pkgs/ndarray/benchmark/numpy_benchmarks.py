import time
import numpy as np

# ============================================================================
# BENCHMARK HELPER RUNNER
# ============================================================================
def run_benchmark(name, fn_run, fn_setup=None, reps=200, warmup=15):
    """
    A custom Python benchmarking runner mimicking Dart's benchmark_harness.
    Performs warmup iterations and averages repetition counts to compute precise microseconds.
    """
    # 1. Setup if needed
    context = fn_setup() if fn_setup else None
    
    # 2. Warmup cycles to heat cache and memory pages
    for _ in range(warmup):
        fn_run(context)
        
    # 3. Timed execution repetitions loop
    start_time = time.perf_counter()
    for _ in range(reps):
        fn_run(context)
    end_time = time.perf_counter()
    
    total_duration_seconds = end_time - start_time
    # Convert total duration to average microseconds per single run!
    avg_microseconds = (total_duration_seconds / reps) * 1_000_000
    
    print(f"{name}: {avg_microseconds:.2f} us per run")

# ============================================================================
# 1. PROBABILITY DISTRIBUTIONS & RNG TRACK (Section 2)
# ============================================================================
def bench_normal(ctx):
    rng = np.random.default_rng(42)
    rng.normal(loc=0.0, scale=1.0, size=50000)

def bench_poisson(ctx):
    rng = np.random.default_rng(42)
    rng.poisson(lam=35.0, size=20000)

def bench_binomial(ctx):
    rng = np.random.default_rng(42)
    rng.binomial(n=60, p=0.4, size=20000)

# ============================================================================
# 2. SORTING & SEARCHING TRACK (Section 7)
# ============================================================================
def setup_qsort():
    template = np.arange(30000, 0, -1, dtype=np.float64)
    target = np.zeros(30000, dtype=np.float64)
    return template, target

def bench_qsort(ctx):
    template, target = ctx
    # High-speed C-level bit copy, bypassing object allocations inside timed loops!
    np.copyto(target, template)
    np.sort(target)

def setup_where():
    cond = np.zeros((100, 100), dtype=np.int32)
    x = np.ones(100, dtype=np.float64) # 1D vector stretching broadcast
    y = np.ones((100, 100), dtype=np.float64)
    return cond, x, y

def bench_where(ctx):
    cond, x, y = ctx
    np.where(cond, x, y)

# ============================================================================
# 3. ADVANCED LINEAR ALGEBRA & SIGNALS TRACK (Section 6)
# ============================================================================
def setup_inv():
    return np.eye(100, dtype=np.float64)

def bench_inv(a):
    np.linalg.inv(a)

def setup_fft():
    return np.zeros(2048, dtype=np.float64)

def bench_fft(signal):
    np.fft.fft(signal)

# ============================================================================
# 4. UFUNCS, REDUCTIONS & MEMORY TRACK (Section 9 Target)
# ============================================================================
def setup_add():
    x = np.ones(300000, dtype=np.float64)
    y = np.ones(300000, dtype=np.float64)
    return x, y

def bench_add(ctx):
    x, y = ctx
    np.add(x, y)

def setup_scalar():
    return np.ones(300000, dtype=np.float64)

def bench_scalar(x):
    # Scalar array broadcasting
    x + 5.0

def setup_sin():
    return np.ones(100000, dtype=np.float64)

def bench_sin(x):
    np.sin(x)

def setup_sum():
    return np.ones(300000, dtype=np.float64)

def bench_sum(x):
    np.sum(x)

# ============================================================================
# MAIN EXECUTION ENTRYPOINT
# ============================================================================
if __name__ == "__main__":
    print("============================================================================")
    print("         PYTHON NUMPY ALL-INCLUSIVE PERFORMANCE BENCHMARK SUITE MASTER       ")
    print("============================================================================")
    print("Establishing high-precision NumPy baseline metrics for twin comparison...\n")

    print("--- TRACK A: RANDOM DISTRIBUTIONS & RNG SOLVERS ---")
    run_benchmark("RNG Track  | Seeded normal() (Gaussian samples)       [size=50,000]", bench_normal, reps=150)
    run_benchmark("RNG Track  | Seeded poisson() (Knuth vs Gaussian)      [size=20,000]", bench_poisson, reps=150)
    run_benchmark("RNG Track  | Seeded binomial() (Bernoulli vs Normal)   [size=20,000]", bench_binomial, reps=150)

    print("\n--- TRACK B: NATIVE C HEAP SORTING & SEARCHING BROADCASTS ---")
    run_benchmark("SORT Track | Native C Heap sort() (Contiguous vector)   [size=30,000]", bench_qsort, setup_qsort, reps=100)
    run_benchmark("SORT Track | Ternary where() 3-Way Broadcasting       [shape=100x100]", bench_where, setup_where, reps=200)

    print("\n--- TRACK C: OPENBLAS LINEAR ALGEBRA & NATIVE POCKETFFT SIGNALS ---")
    run_benchmark("LINALG Track| OpenBLAS LU Matrix Inversion (inv)       [shape=100x100]", bench_inv, setup_inv, reps=300)
    run_benchmark("LINALG Track| Native Mixed-Radix C FFI pocketfft (fft) [length=2048]", bench_fft, setup_fft, reps=300)

    print("\n--- TRACK D: UNIVERSAL UFUNCS, REDUCTIONS & MEMORY STRIDES (SECTION 9 TARGET) ---")
    run_benchmark("MEMORY Track| Element-wise Same-Shape add(x, y)       [size=300,000]", bench_add, setup_add, reps=200)
    run_benchmark("MEMORY Track| Scalar Array Broadcast add(x, scalar)   [size=300,000]", bench_scalar, setup_scalar, reps=200)
    run_benchmark("MEMORY Track| Universal math function sin(x)          [size=100,000]", bench_sin, setup_sin, reps=300)
    run_benchmark("MEMORY Track| Flat Memory Reduction walk sum(x)       [size=300,000]", bench_sum, setup_sum, reps=200)

    print("\n============================================================================")
    print("Exhaustive Python NumPy Baseline Performance Benchmarks completed.")
    print("============================================================================")

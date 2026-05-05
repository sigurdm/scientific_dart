import 'dart:math';
import 'package:num_dart/num_dart.dart';

void main() {
  print('=== NDArray Random Probability Distributions Examples ===\n');
  runNormalDistributionExample();
  runExponentialDistributionExample();
  runPoissonDistributionExample();
  runBinomialDistributionExample();
  runUniformDistributionExample();
  runRandintDistributionExample();
  runSeededReproducibilityExample();
}

void runNormalDistributionExample() {
  print('--- Normal (Gaussian) Distribution ---');
  // Draw a 2x3 normal array with mean=0.0 and stddev=1.0
  final a = normal([2, 3], loc: 0.0, scale: 1.0, dtype: DType.float64);
  print('Normal shape: ${a.shape}');
  print('Data: ${a.data}');

  // Draw with a shifted center and scale
  final b = normal([5], loc: 100.0, scale: 15.0);
  print('Shifted normal (loc=100, scale=15): ${b.data}');
}

void runExponentialDistributionExample() {
  print('\n--- Exponential Distribution ---');
  // Draw exponential distribution with scale parameter (beta) = 2.0
  final a = exponential([5], scale: 2.0);
  print('Exponential data (scale=2.0): ${a.data}');
}

void runPoissonDistributionExample() {
  print('\n--- Poisson Distribution ---');
  // Small lambda using Knuth exact algorithm
  final smallLam = poisson([5], lam: 4.0, dtype: DType.int64);
  print('Poisson data (small lam=4.0, Knuth): ${smallLam.data}');

  // Large lambda using Gaussian approximation to avoid performance stalls
  final largeLam = poisson([5], lam: 50.0, dtype: DType.int64);
  print('Poisson data (large lam=50.0, Gaussian approx): ${largeLam.data}');
}

void runBinomialDistributionExample() {
  print('\n--- Binomial Distribution (Bernoulli Trials) ---');
  // Small n: 20 trials with success probability 0.5
  final smallN = binomial([5], 20, 0.5, dtype: DType.int32);
  print('Binomial data (n=20, p=0.5, Bernoulli trials): ${smallN.data}');

  // Large n: 1000 trials with p=0.3, triggers fast normal approximation
  final largeN = binomial([5], 1000, 0.3, dtype: DType.int64);
  print('Binomial data (n=1000, p=0.3, Normal approx): ${largeN.data}');
}

void runSeededReproducibilityExample() {
  print('\n--- Seeded Reproducibility (Seeding for Science) ---');
  // Passing the same seed object guarantees exactly identical random draws!
  final rand1 = Random(42);
  final rand2 = Random(42);

  final draw1 = normal([3], loc: 10.0, scale: 2.0, random: rand1);
  final draw2 = normal([3], loc: 10.0, scale: 2.0, random: rand2);

  print('Draw 1 (Seed 42): ${draw1.data}');
  print('Draw 2 (Seed 42): ${draw2.data}');
  print(
    'Do draws match perfectly? ${draw1.data[0] == draw2.data[0] && draw1.data[1] == draw2.data[1] ? "YES" : "NO"}',
  );
}

void runUniformDistributionExample() {
  print('\n--- Uniform Distribution ---');
  // Draw a 2x3 uniform array with values between 0.0 and 1.0
  final a = uniform([2, 3], dtype: DType.float64);
  print('Uniform shape: ${a.shape}');
  print('Data: ${a.data}');
}

void runRandintDistributionExample() {
  print('\n--- Uniform Integer (randint) Distribution ---');
  // Draw a 2x3 uniform integer array with values in [0, 10)
  final a = randint([2, 3], 0, 10, dtype: DType.int64);
  print('Randint shape: ${a.shape}');
  print('Data: ${a.data}');
}

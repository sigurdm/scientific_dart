import 'dart:typed_data';
import 'package:num_dart/num_dart.dart';

void main() {
  print('=== NDArray nan_to_num() Dataset Sanitation Example ===\n');

  // 1. Initialize a Float64 vector containing NaN and Infinities
  final a = NDArray.fromList(
    Float64List.fromList([
      1.0,
      double.nan,
      double.infinity,
      -2.0,
      double.negativeInfinity,
    ]),
    [5],
    DType.float64,
  );
  print('Raw array a: ${a.toList()}');

  // 2. Default cleaning: NaN -> 0.0, inf -> max_float, -inf -> min_float
  final cleanDefault = nan_to_num(a);
  print('\nCleaned (default): ${cleanDefault.toList()}');

  // 3. Custom cleaning: NaN -> 99.0, posinf -> 500.0, neginf -> -500.0
  final cleanCustom = nan_to_num(a, nan: 99.0, posinf: 500.0, neginf: -500.0);
  print('Cleaned (custom): ${cleanCustom.toList()}');

  // 4. Blazingly fast in-place recycling to completely bypass allocations!
  print('\n--- Allocation-Free In-Place Recycler ---');
  nan_to_num(a, nan: 0.0, out: a); // mutate a in-place!
  print('Raw array a (after in-place clean): ${a.toList()}');

  // Cleanup memory
  a.dispose();
  cleanDefault.dispose();
  cleanCustom.dispose();
}

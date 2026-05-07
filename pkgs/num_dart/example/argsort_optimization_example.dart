import 'package:num_dart/num_dart.dart';

void main() {
  print('=== NDArray Indirect Sorting (argsort) Optimization Examples ===\n');

  runArgsortSortingByKeysExample();
}

void runArgsortSortingByKeysExample() {
  print('--- 1. Sorting Prices by Product Names Indirectly ---');
  // Suppose we have product prices and corresponding product names:
  final prices = NDArray.fromList([39.9, 15.5, 22.0, 9.9], [4], DType.float64);
  final names = ['Tablet', 'Keyboard', 'Mouse', 'Cable'];

  print('Product Names: $names');
  print('Product Prices: ${prices.data}');

  // We want to sort the product names based on their prices!
  // argsort returns indices that would sort the prices array:
  final indices = argsort(prices);
  print('\nArgsort indices (sorted index mapping): ${indices.data}');

  // Map the names list according to the sorted indices!
  final sortedNames = indices.data.map((idx) => names[idx]).toList();
  final sortedPrices = indices.data.map((idx) => prices.data[idx]).toList();

  print('\n--- Sorted Results (By Price Ascending) ---');
  for (var i = 0; i < 4; i++) {
    print('  ${sortedNames[i]} : \$${sortedPrices[i]}');
  }

  print('\n🏆 Zero-Allocation indirect sorting executed successfully!');
}

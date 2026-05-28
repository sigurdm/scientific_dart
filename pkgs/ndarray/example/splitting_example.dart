import 'package:ndarray/ndarray.dart';

void main() {
  print('=== NDArray Splitting Examples ===\n');
  runSplitExamples();
  runArraySplitExamples();
  runHSplitExamples();
  runVSplitExamples();
}

// #docregion split
void runSplitExamples() {
  print('--- 1. Equal Splitting (split / split_at) ---');
  NDArray.scope(() {
    final a = NDArray.fromList([1, 2, 3, 4], [4], DType.int32);
    print('Original array: ${a.toList()}');

    // Split into 2 equal sections
    final splits = split(a, 2);
    print('Split into 2 equal sections:');
    print('Sub-array 0: ${splits[0].toList()}');
    print('Sub-array 1: ${splits[1].toList()}');

    // Split at custom index 2
    final splitsAt = split_at(a, [2]);
    print('Split at index [2]:');
    print('Sub-array 0: ${splitsAt[0].toList()}');
    print('Sub-array 1: ${splitsAt[1].toList()}\n');
  });
}
// #enddocregion vsplit
// #enddocregion hsplit
// #enddocregion array_split
// #enddocregion split

// #docregion array_split
void runArraySplitExamples() {
  print('--- 2. Unequal Splitting (array_split / array_split_at) ---');
  NDArray.scope(() {
    final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
    print('Original array: ${a.toList()}');

    // Split 3 elements into 2 sections (unequal)
    final splits = array_split(a, 2);
    print('Split into 2 sections (as equal as possible):');
    print('Sub-array 0: ${splits[0].toList()}');
    print('Sub-array 1: ${splits[1].toList()}');

    // Split at custom indices [1, 2]
    final splitsAt = array_split_at(a, [1, 2]);
    print('Split at indices [1, 2]:');
    for (var i = 0; i < splitsAt.length; i++) {
      print('Sub-array $i: ${splitsAt[i].toList()}');
    }
    print('');
  });
}

// #docregion hsplit
void runHSplitExamples() {
  print('--- 3. Horizontal Splitting (hsplit / hsplit_at) ---');
  NDArray.scope(() {
    final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
    print('Original 2D matrix:\n$a');

    final splits = hsplit(a, 2);
    print('hsplit columns into 2 sections:');
    print('Sub-array 0:\n${splits[0]}');
    print('Sub-array 1:\n${splits[1]}');

    final splitsAt = hsplit_at(a, [1]);
    print('hsplit_at column index [1]:');
    print('Sub-array 0:\n${splitsAt[0]}');
    print('Sub-array 1:\n${splitsAt[1]}\n');
  });
}

// #docregion vsplit
void runVSplitExamples() {
  print('--- 4. Vertical Splitting (vsplit / vsplit_at) ---');
  NDArray.scope(() {
    final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
    print('Original 2D matrix:\n$a');

    final splits = vsplit(a, 2);
    print('vsplit rows into 2 sections:');
    print('Sub-array 0:\n${splits[0]}');
    print('Sub-array 1:\n${splits[1]}');

    final splitsAt = vsplit_at(a, [1]);
    print('vsplit_at row index [1]:');
    print('Sub-array 0:\n${splitsAt[0]}');
    print('Sub-array 1:\n${splitsAt[1]}\n');
  });
}

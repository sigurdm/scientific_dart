import 'package:num_dart/num_dart.dart';
import 'package:test/test.dart';
import 'dart:math' as math;
import 'dart:typed_data';

void main() {
  group('NDArray Tests', () {
    test('Creation and Strides', () {
      final a = NDArray.fromList(Float64List.fromList([1, 2, 3, 4, 5, 6]), [
        2,
        3,
      ], DType.float64);
      addTearDown(a.dispose);
      expect(a.shape, [2, 3]);
      expect(a.strides, [3, 1]);
    });

    test('Broadcasting Shape', () {
      final a = NDArray.fromList(Float64List.fromList([1, 2]), [
        2,
        1,
      ], DType.float64);
      addTearDown(a.dispose);
      final b = NDArray.fromList(Float64List.fromList([10, 20, 30]), [
        1,
        3,
      ], DType.float64);
      addTearDown(b.dispose);
      final result = broadcast(a, b);
      expect(result.shape, [2, 3]);
      expect(result.stridesA, [1, 0]);
      expect(result.stridesB, [0, 1]);
    });

    test('Element-wise Addition with Broadcasting', () {
      final a = NDArray.fromList(Float64List.fromList([1, 2]), [
        2,
        1,
      ], DType.float64);
      addTearDown(a.dispose);
      final b = NDArray.fromList(Float64List.fromList([10, 20, 30]), [
        1,
        3,
      ], DType.float64);
      addTearDown(b.dispose);
      final c = add(a, b);
      addTearDown(c.dispose);
      expect(c.shape, [2, 3]);
      expect(c.data, [11.0, 21.0, 31.0, 12.0, 22.0, 32.0]);
    });

    test('Matrix Multiplication (OpenBLAS)', () {
      final a = NDArray<double>.fromList(Float64List.fromList([1, 2, 3, 4]), [
        2,
        2,
      ], DType.float64);
      addTearDown(a.dispose);
      final b = NDArray<double>.fromList(Float64List.fromList([5, 6, 7, 8]), [
        2,
        2,
      ], DType.float64);
      addTearDown(b.dispose);
      final c = matmul(a, b);
      addTearDown(c.dispose);
      expect(c.shape, [2, 2]);
      expect(c.data, [19.0, 22.0, 43.0, 50.0]);
    });

    test('View Shares Memory', () {
      final a = NDArray.fromList(Float64List.fromList([1, 2, 3, 4]), [
        2,
        2,
      ], DType.float64);
      addTearDown(a.dispose);
      final view = NDArray.view(
        a,
        shape: [2],
        strides: [1],
        offsetElements: 1,
      ); // View of [2, 3] if we flatten or just take from offset 1
      addTearDown(view.dispose);
      // Wait, strides for view of [2] with stride 1 from offset 1 will be elements at index 1 and 2.
      // data at index 1 is 2.0.
      // data at index 2 is 3.0.
      expect(view.data[0], 2.0);
      expect(view.data[1], 3.0);

      // Modify view
      view.data[0] = 99.0;
      // Check original
      expect(a.data[1], 99.0);
    });

    test('Element-wise Subtraction with Broadcasting', () {
      final a = NDArray.fromList(Float64List.fromList([1, 2]), [
        2,
        1,
      ], DType.float64);
      addTearDown(a.dispose);
      final b = NDArray.fromList(Float64List.fromList([10, 20, 30]), [
        1,
        3,
      ], DType.float64);
      addTearDown(b.dispose);
      final c = subtract(a, b);
      addTearDown(c.dispose);
      expect(c.shape, [2, 3]);
      expect(c.data, [-9.0, -19.0, -29.0, -8.0, -18.0, -28.0]);
    });

    test('Element-wise Multiplication with Broadcasting', () {
      final a = NDArray.fromList(Float64List.fromList([1, 2]), [
        2,
        1,
      ], DType.float64);
      addTearDown(a.dispose);
      final b = NDArray.fromList(Float64List.fromList([10, 20, 30]), [
        1,
        3,
      ], DType.float64);
      addTearDown(b.dispose);
      final c = multiply(a, b);
      addTearDown(c.dispose);
      expect(c.shape, [2, 3]);
      expect(c.data, [10.0, 20.0, 30.0, 20.0, 40.0, 60.0]);
    });

    test('Element-wise Division with Broadcasting', () {
      final a = NDArray.fromList(Float64List.fromList([10, 20]), [
        2,
        1,
      ], DType.float64);
      addTearDown(a.dispose);
      final b = NDArray.fromList(Float64List.fromList([2, 4, 5]), [
        1,
        3,
      ], DType.float64);
      addTearDown(b.dispose);
      final c = divide(a, b);
      addTearDown(c.dispose);
      expect(c.shape, [2, 3]);
      expect(c.data, [5.0, 2.5, 2.0, 10.0, 5.0, 4.0]);
    });

    test('Zeros Factory', () {
      final a = NDArray<double>.zeros([2, 3], DType.float64);
      addTearDown(a.dispose);
      expect(a.shape, [2, 3]);
      expect(a.data, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    });

    test('Ones Factory', () {
      final a = NDArray<double>.ones([2, 3], DType.float64);
      addTearDown(a.dispose);
      expect(a.shape, [2, 3]);
      expect(a.data, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    });

    test('Arange Factory', () {
      final a = NDArray<double>.arange(
        0.0,
        5.0,
        step: 1.0,
        dtype: DType.float64,
      );
      addTearDown(a.dispose);
      expect(a.shape, [5]);
      expect(a.data, [0.0, 1.0, 2.0, 3.0, 4.0]);
    });

    test('Linspace Factory', () {
      final a = NDArray<double>.linspace(0.0, 1.0, 5, dtype: DType.float64);
      addTearDown(a.dispose);
      expect(a.shape, [5]);
      expect(a.data, [0.0, 0.25, 0.5, 0.75, 1.0]);
    });

    test('Global Sum', () {
      final a = NDArray.fromList(Float64List.fromList([1, 2, 3, 4]), [
        2,
        2,
      ], DType.float64);
      addTearDown(a.dispose);
      final s = sum(a);
      expect(s, 10.0);
    });

    test('Manual Dispose', () {
      final a = NDArray<double>.create([2, 2], DType.float64);
      addTearDown(a.dispose);
      // Should not throw
      a.dispose();
    });

    test('Eye Factory', () {
      final a = NDArray<double>.eye(3, DType.float64);
      addTearDown(a.dispose);
      expect(a.shape, [3, 3]);
      expect(a.data, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    });

    test('Uniform Factory', () {
      final a = uniform([2, 3], dtype: DType.float64);
      addTearDown(a.dispose);
      expect(a.shape, [2, 3]);
      expect(a.data.length, 6);
      for (final value in a.data) {
        expect(value, greaterThanOrEqualTo(0.0));
        expect(value, lessThan(1.0));
      }
    });

    test('Reshape', () {
      final a = NDArray.fromList(Float64List.fromList([1, 2, 3, 4]), [
        4,
      ], DType.float64);
      addTearDown(a.dispose);
      final b = a.reshape([2, 2]);
      addTearDown(b.dispose);
      expect(b.shape, [2, 2]);
      expect(b.strides, [2, 1]);
      expect(b.data, [1.0, 2.0, 3.0, 4.0]);

      // Modify view
      b.data[0] = 99.0;
      // Check original
      expect(a.data[0], 99.0);
    });

    test('Reshape non-contiguous view disposal memory safety', () {
      final parent = NDArray.fromList(
        Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        [3, 2],
        DType.float64,
      );
      final viewT = parent.transposed;
      final reshaped = viewT.reshape([6]);

      expect(reshaped.isDisposed, false);
      reshaped.dispose();
      expect(reshaped.isDisposed, true);

      expect(parent.isDisposed, false);
      expect(viewT.isDisposed, false);

      parent.dispose();
      viewT.dispose();
    });

    test('Dart Addition (Fallback)', () {
      final a = NDArray.fromList(Float64List.fromList([1, 2, 3, 4]), [
        2,
        2,
      ], DType.float64);
      addTearDown(a.dispose);
      final b = NDArray.fromList(Float64List.fromList([10, 20, 30, 40]), [
        2,
        2,
      ], DType.float64);
      addTearDown(b.dispose);

      // Force Dart path
      ffiThresholds[Operation.add] = 10000;

      final c = add(a, b);
      addTearDown(c.dispose);
      expect(c.shape, [2, 2]);
      expect(c.data, [11.0, 22.0, 33.0, 44.0]);
    });

    test('SIMD Addition (Float32)', () {
      final a = NDArray.fromList(
        Float32List.fromList([1.0, 2.0, 3.0, 4.0, 5.0]),
        [5],
        DType.float32,
      );
      addTearDown(a.dispose);
      final b = NDArray.fromList(
        Float32List.fromList([10.0, 20.0, 30.0, 40.0, 50.0]),
        [5],
        DType.float32,
      );
      addTearDown(b.dispose);

      // This should trigger SIMD path because it is float32 and contiguous
      final c = add(a, b);
      addTearDown(c.dispose);
      expect(c.shape, [5]);
      expect(c.data, [11.0, 22.0, 33.0, 44.0, 55.0]);
    });

    test('Sum along Axis 0', () {
      final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
        2,
        2,
      ], DType.float64);
      addTearDown(a.dispose);
      final s = sum(a, axis: 0) as NDArray<double>;
      addTearDown(s.dispose);
      expect(s.shape, [2]);
      expect(s.data, [4.0, 6.0]);
    });

    test('Sum along Axis 1', () {
      final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
        2,
        2,
      ], DType.float64);
      addTearDown(a.dispose);
      final s = sum(a, axis: 1) as NDArray<double>;
      addTearDown(s.dispose);
      expect(s.shape, [2]);
      expect(s.data, [3.0, 7.0]);
    });

    test('NDArray.create supports Int32', () {
      final a = NDArray<int>.create([2, 2], DType.int32);
      addTearDown(a.dispose);
      expect(a.shape, [2, 2]);
      expect(a.dtype, DType.int32);
    });

    test('NDArray.create supports Int64', () {
      final a = NDArray<int>.create([2, 2], DType.int64);
      addTearDown(a.dispose);
      expect(a.shape, [2, 2]);
      expect(a.dtype, DType.int64);
    });

    test('Randint Factory', () {
      final a = randint([2, 3], low: 0, high: 10, dtype: DType.int64);
      addTearDown(a.dispose);
      expect(a.shape, [2, 3]);
      expect(a.data.length, 6);
      for (final value in a.data) {
        expect(value, greaterThanOrEqualTo(0));
        expect(value, lessThan(10));
      }
    });

    test('Sqrt Operation', () {
      final a = NDArray.fromList(Float64List.fromList([1.0, 4.0, 9.0]), [
        3,
      ], DType.float64);
      addTearDown(a.dispose);
      final b = sqrt(a);
      addTearDown(b.dispose);
      expect(b.shape, [3]);
      expect(b.data, [1.0, 2.0, 3.0]);
    });

    test('Sin Operation', () {
      final a = NDArray.fromList(Float64List.fromList([0.0, math.pi / 2]), [
        2,
      ], DType.float64);
      addTearDown(a.dispose);
      final b = sin(a);
      addTearDown(b.dispose);
      expect(b.shape, [2]);
      expect(b.data[0], closeTo(0.0, 1e-10));
      expect(b.data[1], closeTo(1.0, 1e-10));
    });

    test('Cos Operation', () {
      final a = NDArray.fromList(Float64List.fromList([0.0, math.pi]), [
        2,
      ], DType.float64);
      addTearDown(a.dispose);
      final b = cos(a);
      addTearDown(b.dispose);
      expect(b.shape, [2]);
      expect(b.data[0], closeTo(1.0, 1e-10));
      expect(b.data[1], closeTo(-1.0, 1e-10));
    });

    test('Exp Operation', () {
      final a = NDArray.fromList(Float64List.fromList([0.0, 1.0]), [
        2,
      ], DType.float64);
      addTearDown(a.dispose);
      final b = exp(a);
      addTearDown(b.dispose);
      expect(b.shape, [2]);
      expect(b.data[0], closeTo(1.0, 1e-10));
      expect(b.data[1], closeTo(math.e, 1e-10));
    });

    test('Log Operation', () {
      final a = NDArray.fromList(Float64List.fromList([1.0, math.e]), [
        2,
      ], DType.float64);
      addTearDown(a.dispose);
      final b = log(a);
      addTearDown(b.dispose);
      expect(b.shape, [2]);
      expect(b.data[0], closeTo(0.0, 1e-10));
      expect(b.data[1], closeTo(1.0, 1e-10));
    });

    test('Mean Operation', () {
      final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
        2,
        2,
      ], DType.float64);
      addTearDown(a.dispose);
      final m = mean(a);
      expect(m, 2.5);

      final m0 = mean(a, axis: 0) as NDArray<double>;
      addTearDown(m0.dispose);
      expect(m0.shape, [2]);
      expect(m0.data, [2.0, 3.0]);
    });

    test('Min Operation', () {
      final a = NDArray.fromList(Float64List.fromList([3.0, 1.0, 4.0, 2.0]), [
        2,
        2,
      ], DType.float64);
      final m = min(a);
      expect(m.scalar, 1.0);

      final m0 = min(a, axis: 0);
      expect(m0.shape, [2]);
      expect(m0.data, [3.0, 1.0]); // Min along rows: min(3,4)=3, min(1,2)=1
    });

    test('Max Operation', () {
      final a = NDArray.fromList(Float64List.fromList([3.0, 1.0, 4.0, 2.0]), [
        2,
        2,
      ], DType.float64);
      final m = max(a);
      expect(m.scalar, 4.0);

      final m0 = max(a, axis: 0);
      expect(m0.shape, [2]);
      expect(m0.data, [4.0, 2.0]); // Max along rows: max(3,4)=4, max(1,2)=2
    });

    test('Prod Operation', () {
      final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
        2,
        2,
      ], DType.float64);
      addTearDown(a.dispose);
      final p = prod(a);
      expect(p, 24.0);

      final p0 = prod(a, axis: 0) as NDArray<double>;
      addTearDown(p0.dispose);
      expect(p0.shape, [2]);
      expect(p0.toList(), [3.0, 8.0]); // Prod along rows: 1*3=3, 2*4=8
    });

    test('Variance Operation', () {
      final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
        4,
      ], DType.float64);
      addTearDown(a.dispose);
      final v = variance(a);
      expect(
        v,
        closeTo(1.25, 1e-10),
      ); // Mean=2.5, SqDiffs=[2.25, 0.25, 0.25, 2.25], Sum=5, Var=1.25

      final a2 = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
        2,
        2,
      ], DType.float64);
      addTearDown(a2.dispose);
      final v0 = variance(a2, axis: 0) as NDArray<double>;
      addTearDown(v0.dispose);
      expect(v0.shape, [2]);
      expect(v0.toList(), [
        1.0,
        1.0,
      ]); // Variance along rows: var([1,3])=1, var([2,4])=1
    });

    test('Std Operation', () {
      final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
        4,
      ], DType.float64);
      addTearDown(a.dispose);
      final s = std(a);
      expect(s, closeTo(math.sqrt(1.25), 1e-10));

      final a2 = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
        2,
        2,
      ], DType.float64);
      addTearDown(a2.dispose);
      final s0 = std(a2, axis: 0) as NDArray<double>;
      addTearDown(s0.dispose);
      expect(s0.shape, [2]);
      expect(s0.toList(), [1.0, 1.0]);
    });

    test('Determinant Operation', () {
      final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
        2,
        2,
      ], DType.float64);
      addTearDown(a.dispose);
      final d = det(a);
      expect(d, closeTo(-2.0, 1e-10)); // 1*4 - 2*3 = -2

      final a3 = NDArray.fromList(
        Float64List.fromList([1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0]),
        [3, 3],
        DType.float64,
      );
      addTearDown(a3.dispose);
      final d3 = det(a3);
      expect(
        d3,
        closeTo(1.0, 1e-10),
      ); // 1*(0-24) - 2*(0-20) + 3*(0-5) = -24 + 40 - 15 = 1
    });

    test('Determinant of Singular Matrix', () {
      final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 2.0, 4.0]), [
        2,
        2,
      ], DType.float64);
      addTearDown(a.dispose);
      final d = det(a);
      expect(d, closeTo(0.0, 1e-10));
    });

    test('Solve Linear System (1D b)', () {
      final a = NDArray.fromList(Float64List.fromList([3.0, 1.0, 1.0, 2.0]), [
        2,
        2,
      ], DType.float64);
      addTearDown(a.dispose);
      final b = NDArray.fromList(Float64List.fromList([9.0, 8.0]), [
        2,
      ], DType.float64);
      addTearDown(b.dispose);
      final x = solve(a, b);
      addTearDown(x.dispose);
      expect(x.shape, [2]);
      expect(x.data[0], closeTo(2.0, 1e-10));
      expect(x.data[1], closeTo(3.0, 1e-10));
    });

    test('Solve Linear System (2D b)', () {
      final a = NDArray.fromList(Float64List.fromList([3.0, 1.0, 1.0, 2.0]), [
        2,
        2,
      ], DType.float64);
      addTearDown(a.dispose);
      final b = NDArray.fromList(Float64List.fromList([9.0, 5.0, 8.0, 5.0]), [
        2,
        2,
      ], DType.float64);
      addTearDown(b.dispose);
      final x = solve(a, b);
      addTearDown(x.dispose);
      expect(x.shape, [2, 2]);
      expect(x.data[0], closeTo(2.0, 1e-10));
      expect(x.data[1], closeTo(1.0, 1e-10));
      expect(x.data[2], closeTo(3.0, 1e-10));
      expect(x.data[3], closeTo(2.0, 1e-10));
    });

    test('Solve Singular Matrix throws', () {
      final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 2.0, 4.0]), [
        2,
        2,
      ], DType.float64);
      addTearDown(a.dispose);
      final b = NDArray.fromList(Float64List.fromList([5.0, 10.0]), [
        2,
      ], DType.float64);
      addTearDown(b.dispose);
      expect(() => solve(a, b), throwsArgumentError);
    });

    test('Solve Float32', () {
      final a = NDArray.fromList(Float32List.fromList([3.0, 1.0, 1.0, 2.0]), [
        2,
        2,
      ], DType.float32);
      addTearDown(a.dispose);
      final b = NDArray.fromList(Float32List.fromList([9.0, 8.0]), [
        2,
      ], DType.float32);
      addTearDown(b.dispose);
      final x = solve(a, b);
      addTearDown(x.dispose);
      expect(x.shape, [2]);
      expect(x.dtype, DType.float32);
      expect(x.data[0], closeTo(2.0, 1e-5));
      expect(x.data[1], closeTo(3.0, 1e-5));
    });

    test('Solve Int32 (converts to Float64)', () {
      final a = NDArray.fromList([3, 1, 1, 2], [2, 2], DType.int32);
      addTearDown(a.dispose);
      final b = NDArray.fromList([9, 8], [2], DType.int32);
      addTearDown(b.dispose);
      final x = solve(a, b);
      addTearDown(x.dispose);
      expect(x.shape, [2]);
      expect(x.dtype, DType.float64);
      expect(x.data[0], closeTo(2.0, 1e-10));
      expect(x.data[1], closeTo(3.0, 1e-10));
    });

    test('Solve Complex128', () {
      final a = NDArray<Complex>.create([2, 2], DType.complex128);
      addTearDown(a.dispose);
      a.data[0] = Complex(3.0, 0.0);
      a.data[1] = Complex(1.0, 0.0);
      a.data[2] = Complex(1.0, 0.0);
      a.data[3] = Complex(2.0, 0.0);

      final b = NDArray<Complex>.create([2], DType.complex128);
      addTearDown(b.dispose);
      b.data[0] = Complex(9.0, 0.0);
      b.data[1] = Complex(8.0, 0.0);

      final x = solve(a, b);
      addTearDown(x.dispose);
      expect(x.shape, [2]);
      expect(x.dtype, DType.complex128);
      expect(x.data[0], Complex(2.0, 0.0));
      expect(x.data[1], Complex(3.0, 0.0));
    });

    test('Eigen Decompositions (Real Matrix)', () {
      final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 2.0, 1.0]), [
        2,
        2,
      ], DType.float64);
      addTearDown(a.dispose);
      final result = eig(a);
      final w = result['eigenvalues']!;
      addTearDown(w.dispose);
      final vr = result['eigenvectors']!;
      addTearDown(vr.dispose);

      expect(w.shape, [2]);
      expect(vr.shape, [2, 2]);

      final val0 = w.data[0];
      final val1 = w.data[1];

      expect(
        (val0.real - 3.0).abs() < 1e-5 || (val0.real - (-1.0)).abs() < 1e-5,
        true,
      );
      expect(
        (val1.real - 3.0).abs() < 1e-5 || (val1.real - (-1.0)).abs() < 1e-5,
        true,
      );
      expect(val0.imag, closeTo(0.0, 1e-5));
      expect(val1.imag, closeTo(0.0, 1e-5));
    });

    test('Eigen Decompositions (Complex Matrix)', () {
      final a = NDArray<Complex>.create([2, 2], DType.complex128);
      addTearDown(a.dispose);
      a.data[0] = Complex(0.0, 1.0);
      a.data[1] = Complex(0.0, 0.0);
      a.data[2] = Complex(0.0, 0.0);
      a.data[3] = Complex(0.0, 1.0);

      final result = eig(a);
      final w = result['eigenvalues']!;
      addTearDown(w.dispose);

      expect(w.shape, [2]);
      expect(w.data[0], Complex(0.0, 1.0));
      expect(w.data[1], Complex(0.0, 1.0));
    });

    test('Matrix Inversion', () {
      final a = NDArray.fromList(Float64List.fromList([4.0, 7.0, 2.0, 6.0]), [
        2,
        2,
      ], DType.float64);
      addTearDown(a.dispose);
      final b = inv(a);
      addTearDown(b.dispose);
      expect(b.shape, [2, 2]);
      expect(b.data[0], closeTo(0.6, 1e-10));
      expect(b.data[1], closeTo(-0.7, 1e-10));
      expect(b.data[2], closeTo(-0.2, 1e-10));
      expect(b.data[3], closeTo(0.4, 1e-10));
    });

    test('Multi-dimensional Indexing', () {
      final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
        2,
        2,
      ], DType.float64);
      addTearDown(a.dispose);
      expect(a[[0, 0]], 1.0);
      expect(a[[0, 1]], 2.0);
      expect(a[[1, 0]], 3.0);
      expect(a[[1, 1]], 4.0);

      a[[0, 0]] = 10.0;
      expect(a[[0, 0]], 10.0);
    });

    test('Basic Slicing', () {
      final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
        4,
      ], DType.float64);
      addTearDown(a.dispose);
      final b = a.slice([Slice(start: 1, stop: 3)]);
      addTearDown(b.dispose);
      expect(b.shape, [2]);
      expect(b.toList(), [2.0, 3.0]);
    });

    test('Slicing shares memory', () {
      final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
        4,
      ], DType.float64);
      addTearDown(a.dispose);
      final b = a.slice([Slice(start: 1, stop: 3)]);
      addTearDown(b.dispose);
      b.data[0] = 20.0;
      expect(a.data[1], 20.0);
    });

    test('Slicing with step', () {
      final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
        4,
      ], DType.float64);
      addTearDown(a.dispose);
      final b = a.slice([Slice(start: 0, stop: 4, step: 2)]);
      addTearDown(b.dispose);
      expect(b.shape, [2]);
      expect(b.toList(), [1.0, 3.0]);
    });

    test('Slicing reduces rank with int selector', () {
      final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
        2,
        2,
      ], DType.float64);
      addTearDown(a.dispose);
      final b = a.slice([Index(1)]); // Select row 1
      addTearDown(b.dispose);
      expect(b.shape, [2]);
      expect(b.toList(), [3.0, 4.0]);
    });

    test('Integer Array Indexing', () {
      final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
        4,
      ], DType.float64);
      addTearDown(a.dispose);
      final b = a.slice([
        Indices([0, 2, 3]),
      ]);
      addTearDown(b.dispose);
      expect(b.shape, [3]);
      expect(b.toList(), [1.0, 3.0, 4.0]);
    });

    test('Integer Array Indexing (Copy)', () {
      final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
        4,
      ], DType.float64);
      addTearDown(a.dispose);
      final b = a.slice([
        Indices([0, 2]),
      ]);
      addTearDown(b.dispose);
      b.data[0] = 10.0;
      expect(a.data[0], 1.0); // Original should not change!
    });

    test('Comparison Operators', () {
      final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
        4,
      ], DType.float64);
      addTearDown(a.dispose);
      final b = a > 2.0;
      addTearDown(b.dispose);
      expect(b.toList(), [false, false, true, true]);

      final c = a < 3.0;
      addTearDown(c.dispose);
      expect(c.toList(), [true, true, false, false]);

      final d = a.eq(2.0);
      addTearDown(d.dispose);
      expect(d.toList(), [false, true, false, false]);
    });

    test('Boolean Masking', () {
      final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
        4,
      ], DType.float64);
      addTearDown(a.dispose);
      final mask = a > 2.0;
      addTearDown(mask.dispose);
      final b = a.slice([Mask(BooleanMask(mask))]);
      addTearDown(b.dispose);
      expect(b.shape, [2]);
      expect(b.toList(), [3.0, 4.0]);
    });

    test('Boolean Masking (Copy)', () {
      final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
        4,
      ], DType.float64);
      addTearDown(a.dispose);
      final mask = a > 2.0;
      addTearDown(mask.dispose);
      final b = a.slice([Mask(BooleanMask(mask))]);
      addTearDown(b.dispose);
      b.data[0] = 10.0;
      expect(a.data[2], 3.0); // Original should not change!
    });

    test('Take Method', () {
      final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
        2,
        2,
      ], DType.float64);
      addTearDown(a.dispose);
      final b = a.take([0, 1], axis: 1); // Select columns 0 and 1
      addTearDown(b.dispose);
      expect(b.shape, [2, 2]);
      expect(b.toList(), [1.0, 2.0, 3.0, 4.0]);

      final c = a.take([1], axis: 0); // Select row 1
      addTearDown(c.dispose);
      expect(c.shape, [1, 2]);
      expect(c.toList(), [3.0, 4.0]);
    });

    test('ApplyMask Method', () {
      final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
        4,
      ], DType.float64);
      addTearDown(a.dispose);
      final mask = a > 2.0;
      addTearDown(mask.dispose);
      final b = a.applyMask(mask);
      addTearDown(b.dispose);
      expect(b.shape, [2]);
      expect(b.toList(), [3.0, 4.0]);
    });

    test('Flatten', () {
      final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
        2,
        2,
      ], DType.float64);
      addTearDown(a.dispose);
      final b = a.flatten();
      addTearDown(b.dispose);
      expect(b.shape, [4]);
      expect(b.toList(), [1.0, 2.0, 3.0, 4.0]);

      b.data[0] = 10.0;
      expect(a.data[0], 1.0);
    });

    test('Flatten contiguous arrays across all DTypes (FFI block copy)', () {
      for (final dtype in DType.values) {
        final NDArray a;
        final List expected;
        if (dtype == DType.complex128 || dtype == DType.complex64) {
          a = NDArray<Complex>.create([2, 2], dtype);
          a.data[0] = Complex(1.0, 1.0);
          a.data[1] = Complex(2.0, 2.0);
          a.data[2] = Complex(3.0, 3.0);
          a.data[3] = Complex(4.0, 4.0);
          expected = [
            Complex(1.0, 1.0),
            Complex(2.0, 2.0),
            Complex(3.0, 3.0),
            Complex(4.0, 4.0),
          ];
        } else if (dtype == DType.boolean) {
          a = NDArray<bool>.fromList([true, false, true, false], [2, 2], dtype);
          expected = [true, false, true, false];
        } else if (dtype == DType.int32 || dtype == DType.int64) {
          a = NDArray<int>.fromList([1, 2, 3, 4], [2, 2], dtype);
          expected = [1, 2, 3, 4];
        } else {
          a = NDArray<double>.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], dtype);
          expected = [1.0, 2.0, 3.0, 4.0];
        }
        addTearDown(a.dispose);

        final b = a.flatten();
        addTearDown(b.dispose);

        expect(b.shape, [4]);
        expect(b.dtype, dtype);
        expect(b.toList(), expected);
      }
    });

    test('Ravel (Contiguous)', () {
      final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
        2,
        2,
      ], DType.float64);
      addTearDown(a.dispose);
      final b = a.ravel();
      addTearDown(b.dispose);
      expect(b.shape, [4]);
      expect(b.toList(), [1.0, 2.0, 3.0, 4.0]);

      b.data[0] = 10.0;
      expect(a.data[0], 10.0);
    });

    test('Ravel (Non-Contiguous)', () {
      final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
        2,
        2,
      ], DType.float64);
      addTearDown(a.dispose);
      final b = a.slice([
        Slice(start: 0, stop: 2),
        Index(0),
      ]); // Select column 0
      addTearDown(b.dispose);
      final c = b.ravel();
      addTearDown(c.dispose);
      expect(c.shape, [2]);
      expect(c.toList(), [1.0, 3.0]);

      c.data[0] = 10.0;
      expect(b.data[0], 1.0);
    });

    test('Transpose (Default)', () {
      final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
        2,
        2,
      ], DType.float64);
      addTearDown(a.dispose);
      final b = a.transposed;
      addTearDown(b.dispose);
      expect(b.shape, [2, 2]);
      expect(b.toList(), [1.0, 3.0, 2.0, 4.0]);

      b.data[0] = 10.0;
      expect(a.data[0], 10.0);
    });

    test('Transpose with Axes', () {
      final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
        2,
        2,
      ], DType.float64);
      addTearDown(a.dispose);
      final b = a.transpose([1, 0]);
      addTearDown(b.dispose);
      expect(b.shape, [2, 2]);
      expect(b.toList(), [1.0, 3.0, 2.0, 4.0]);
    });

    test('Concatenate (Axis 0)', () {
      final a = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
        1,
        2,
      ], DType.float64);
      addTearDown(a.dispose);
      final b = NDArray.fromList(Float64List.fromList([3.0, 4.0]), [
        1,
        2,
      ], DType.float64);
      addTearDown(b.dispose);
      final c = concatenate([a, b], axis: 0);
      addTearDown(c.dispose);
      expect(c.shape, [2, 2]);
      expect(c.toList(), [1.0, 2.0, 3.0, 4.0]);
    });

    test('Concatenate (Axis 1)', () {
      final a = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
        1,
        2,
      ], DType.float64);
      addTearDown(a.dispose);
      final b = NDArray.fromList(Float64List.fromList([3.0, 4.0]), [
        1,
        2,
      ], DType.float64);
      addTearDown(b.dispose);
      final c = concatenate([a, b], axis: 1);
      addTearDown(c.dispose);
      expect(c.shape, [1, 4]);
      expect(c.toList(), [1.0, 2.0, 3.0, 4.0]);
    });

    test('vstack', () {
      final a = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
        1,
        2,
      ], DType.float64);
      addTearDown(a.dispose);
      final b = NDArray.fromList(Float64List.fromList([3.0, 4.0]), [
        1,
        2,
      ], DType.float64);
      addTearDown(b.dispose);
      final c = vstack([a, b]);
      addTearDown(c.dispose);
      expect(c.shape, [2, 2]);
      expect(c.toList(), [1.0, 2.0, 3.0, 4.0]);
    });

    test('hstack', () {
      final a = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
        1,
        2,
      ], DType.float64);
      addTearDown(a.dispose);
      final b = NDArray.fromList(Float64List.fromList([3.0, 4.0]), [
        1,
        2,
      ], DType.float64);
      addTearDown(b.dispose);
      final c = hstack([a, b]);
      addTearDown(c.dispose);
      expect(c.shape, [1, 4]);
      expect(c.toList(), [1.0, 2.0, 3.0, 4.0]);
    });
    test('Complex Array Creation', () {
      final a = NDArray<Complex>.create([2], DType.complex128);
      addTearDown(a.dispose);
      expect(a.shape, [2]);
      expect(a.dtype, DType.complex128);

      a.data[0] = Complex(1.0, 2.0);
      a.data[1] = Complex(3.0, 4.0);

      expect(a.data[0].real, 1.0);
      expect(a.data[0].imag, 2.0);
      expect(a.data[1].real, 3.0);
      expect(a.data[1].imag, 4.0);
    });

    test('Complex Array Addition', () {
      final a = NDArray<Complex>.create([2], DType.complex128);
      addTearDown(a.dispose);
      a.data[0] = Complex(1.0, 2.0);
      a.data[1] = Complex(3.0, 4.0);

      final b = NDArray<Complex>.create([2], DType.complex128);
      addTearDown(b.dispose);
      b.data[0] = Complex(10.0, 20.0);
      b.data[1] = Complex(30.0, 40.0);

      final c = add(a, b);
      addTearDown(c.dispose);
      expect(c.shape, [2]);
      expect(c.data[0], Complex(11.0, 22.0));
      expect(c.data[1], Complex(33.0, 44.0));
    });

    test('Complex and Real Array Interaction', () {
      final a = NDArray<Complex>.create([2], DType.complex128);
      addTearDown(a.dispose);
      a.data[0] = Complex(1.0, 2.0);
      a.data[1] = Complex(3.0, 4.0);

      final b = NDArray<double>.fromList([10.0, 20.0], [2], DType.float64);
      addTearDown(b.dispose);

      final c = add(a, b);
      addTearDown(c.dispose);
      expect(c.shape, [2]);
      expect(c.data[0], Complex(11.0, 2.0));
      expect(c.data[1], Complex(23.0, 4.0));
    });

    group('NDArray Bounds, Formats, and Error Exceptions Tests', () {
      test('Transpose axes validations', () {
        final a = NDArray.zeros([2, 3], DType.float64);
        addTearDown(a.dispose);
        expect(() => a.transpose([0]), throwsArgumentError);
        expect(() => a.transpose([0, -5]), throwsRangeError);
        expect(() => a.transpose([0, 0]), throwsArgumentError);
      });

      test('getCell and setCell coordinate checks', () {
        final a = NDArray.zeros([2, 3], DType.float64);
        addTearDown(a.dispose);
        expect(() => a.getCell([0]), throwsArgumentError);
        expect(() => a.getCell([0, 5]), throwsRangeError);
        expect(() => a.setCell([0], 1.0), throwsArgumentError);
        expect(() => a.setCell([0, 5], 1.0), throwsRangeError);
      });

      test('setByMask dimensions validation', () {
        final a = NDArray.zeros([2, 3], DType.float64);
        addTearDown(a.dispose);
        final mask1D = NDArray<bool>.zeros([2], DType.boolean);
        addTearDown(mask1D.dispose);
        expect(() => a.setByMaskScalar(mask1D, 5.0), throwsArgumentError);

        final mask2D = NDArray<bool>.zeros([2, 2], DType.boolean);
        addTearDown(mask2D.dispose);
        expect(() => a.setByMaskScalar(mask2D, 5.0), throwsArgumentError);
      });

      test('setIndices and setIndicesScalar bounds checks', () {
        final a = NDArray.zeros([2, 3], DType.float64);
        addTearDown(a.dispose);
        final indices = NDArray<int>.fromList([0], [1], DType.int32);
        addTearDown(indices.dispose);
        expect(
          () => a.setIndicesScalar(indices, 1.0, axis: 5),
          throwsRangeError,
        );

        final badIndices = NDArray<int>.fromList([5], [1], DType.int32);
        addTearDown(badIndices.dispose);
        expect(() => a.setIndicesScalar(badIndices, 1.0), throwsRangeError);

        final val = NDArray.zeros([1], DType.float64);
        addTearDown(val.dispose);
        expect(() => a.setIndices(badIndices, val), throwsRangeError);
      });

      test('setIndices multi-dimensional slice assignment', () {
        final a = NDArray.zeros([3, 3], DType.float64);
        addTearDown(a.dispose);

        final indices = NDArray.fromList([0, 2], [2], DType.int32);
        addTearDown(indices.dispose);

        final vals = NDArray.fromList(
          Float64List.fromList([1.0, 2.0, 3.0, 4.0, 0.0, 0.0]),
          [2, 3],
          DType.float64,
        );
        addTearDown(vals.dispose);

        a.setIndices(indices, vals, axis: 0);

        expect(a.toList(), [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0]);
      });

      test('operator [] fancy index parameter checks', () {
        final a = NDArray.zeros([2, 3], DType.float64);
        addTearDown(a.dispose);
        expect(() => a[[0]], throwsArgumentError);

        final badMask = NDArray<bool>.zeros([2, 2], DType.boolean);
        addTearDown(badMask.dispose);
        expect(() => a[badMask], throwsArgumentError);
      });
    });
  });
}

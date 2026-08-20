import 'package:test/test.dart';
import 'package:gpuarray/gpuarray.dart';
import 'package:gpuarray/random.dart' as rng;
import 'package:resource_scope/resource_scope.dart';

void main() {
  group('GpuArray Pseudo-Random Number Generation (gpuarray.random)', () {
    test('Philox4x32 uniform random sampling (rand, uniform)', () {
      ResourceScope.scope(() {
        rng.seed(42);
        final u = rng.rand([100]);
        expect(u.shape, equals([100]));
        expect(u.dtype, equals(DType.float64));

        final uList = u.toList().cast<double>();
        for (final val in uList) {
          expect(val, greaterThanOrEqualTo(0.0));
          expect(val, lessThan(1.0));
        }

        // Custom uniform range [10.0, 20.0)
        final customU = rng.uniform(low: 10.0, high: 20.0, shape: [50]);
        final cList = customU.toList().cast<double>();
        for (final val in cList) {
          expect(val, greaterThanOrEqualTo(10.0));
          expect(val, lessThan(20.0));
        }
      });
    });

    test('Gaussian normal sampling (randn, normal)', () {
      ResourceScope.scope(() {
        rng.seed(123);
        final g = rng.randn([1000]);
        expect(g.shape, equals([1000]));

        final gList = g.toList().cast<double>();
        var sum = 0.0;
        for (final val in gList) {
          sum += val;
        }
        final mean = sum / gList.length;
        expect(mean, closeTo(0.0, 0.15)); // Sample mean of N(0, 1) near 0
      });
    });

    test('Random integers and exponential (randint, exponential)', () {
      ResourceScope.scope(() {
        final ints = rng.randint(5, 15, [50]);
        expect(ints.shape, equals([50]));
        final intList = ints.toList().cast<int>();
        for (final v in intList) {
          expect(v, greaterThanOrEqualTo(5));
          expect(v, lessThan(15));
        }

        final exp = rng.exponential(scale: 2.0, shape: [100]);
        final expList = exp.toList().cast<double>();
        for (final v in expList) {
          expect(v, greaterThanOrEqualTo(0.0));
        }
      });
    });

    test('Random choice, permutation and shuffle', () {
      ResourceScope.scope(() {
        final arr = GpuArray.fromList([10.0, 20.0, 30.0, 40.0, 50.0], [5], DType.float64);

        final ch = rng.choice(arr, size: 3);
        expect(ch.shape, equals([3]));
        for (final item in ch.toList().cast<double>()) {
          expect(arr.toList().cast<double>(), contains(item));
        }

        final perm = rng.permutation(5) as GpuArray;
        expect(perm.shape, equals([5]));
        expect(perm.toList().cast<int>().toSet().length, equals(5));

        final orig = GpuArray.fromList([1.0, 2.0, 3.0, 4.0, 5.0], [5], DType.float64);
        rng.shuffle(orig);
        expect(orig.toList().cast<double>().toSet().length, equals(5));
      });
    });
  });
}

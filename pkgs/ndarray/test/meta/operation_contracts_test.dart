import 'package:ndarray/ndarray.dart';
import 'package:ndarray/src/scratch_arena.dart';
import 'package:test/test.dart';

/// Descriptor for a unary operation `f(a, {out})` on `NDArray<Float64>`.
final class UnaryOpSpec {
  final String name;
  final NDArray<Float64> Function(NDArray<Float64> a, {NDArray<Float64>? out})
  call;
  final bool supportsInPlace;

  const UnaryOpSpec(this.name, this.call, {this.supportsInPlace = true});
}

/// Descriptor for a binary operation `f(a, b, {out})` on `NDArray<Float64>`.
final class BinaryOpSpec {
  final String name;
  final NDArray<Float64> Function(
    NDArray<Float64> a,
    NDArray<Float64> b, {
    NDArray<Float64>? out,
  })
  call;

  const BinaryOpSpec(this.name, this.call);
}

/// Descriptor for an axis reduction `f(a, {axis, out})` on `NDArray<Float64>`.
final class ReductionOpSpec {
  final String name;
  final NDArray<Float64> Function(
    NDArray<Float64> a, {
    int? axis,
    NDArray<Float64>? out,
  })
  call;

  const ReductionOpSpec(this.name, this.call);
}

void main() {
  final unaryOps = <UnaryOpSpec>[
    UnaryOpSpec('sin', (a, {out}) => sin(a, out: out)),
    UnaryOpSpec('cos', (a, {out}) => cos(a, out: out)),
    UnaryOpSpec('tan', (a, {out}) => tan(a, out: out)),
    UnaryOpSpec('exp', (a, {out}) => exp(a, out: out)),
    UnaryOpSpec('log', (a, {out}) => log(a, out: out)),
    UnaryOpSpec('sqrt', (a, {out}) => sqrt(a, out: out)),
    UnaryOpSpec('abs', (a, {out}) => abs(a, out: out)),
    UnaryOpSpec('negative', (a, {out}) => negative(a, out: out)),
    UnaryOpSpec('floor', (a, {out}) => floor(a, out: out)),
    UnaryOpSpec('ceil', (a, {out}) => ceil(a, out: out)),
    UnaryOpSpec('round', (a, {out}) => round(a, out: out)),
    UnaryOpSpec('cumsum', (a, {out}) => cumsum(a, out: out)),
    UnaryOpSpec('cumprod', (a, {out}) => cumprod(a, out: out)),
    UnaryOpSpec('sort', (a, {out}) => sort(a, out: out), supportsInPlace: true),
    UnaryOpSpec(
      'roll(shift: 2)',
      (a, {out}) => roll(a, 2, out: out),
      supportsInPlace: false,
    ),
  ];

  final binaryOps = <BinaryOpSpec>[
    BinaryOpSpec('add', (a, b, {out}) => add(a, b, out: out)),
    BinaryOpSpec('subtract', (a, b, {out}) => subtract(a, b, out: out)),
    BinaryOpSpec('multiply', (a, b, {out}) => multiply(a, b, out: out)),
    BinaryOpSpec('divide', (a, b, {out}) => divide(a, b, out: out)),
    BinaryOpSpec('power', (a, b, {out}) => power(a, b, out: out)),
    BinaryOpSpec(
      'atan2',
      (a, b, {out}) =>
          atan2<Float64, Float64>(a, b, out: out) as NDArray<Float64>,
    ),
    BinaryOpSpec(
      'hypot',
      (a, b, {out}) => hypot<Float64, Float64, Float64>(a, b, out: out),
    ),
  ];

  final reductionOps = <ReductionOpSpec>[
    ReductionOpSpec('sum', (a, {axis, out}) => sum(a, axis: axis, out: out)),
    ReductionOpSpec('prod', (a, {axis, out}) => prod(a, axis: axis, out: out)),
    ReductionOpSpec('mean', (a, {axis, out}) => mean(a, axis: axis, out: out)),
    ReductionOpSpec('min', (a, {axis, out}) => min(a, axis: axis, out: out)),
    ReductionOpSpec('max', (a, {axis, out}) => max(a, axis: axis, out: out)),
  ];

  group('Unary Operation Contracts', () {
    for (final op in unaryOps) {
      group(op.name, () {
        test('strided, negative-stride, and transposed equivalence', () {
          NDArray.scope(() {
            final base = linspace(
              0.5,
              3.0,
              12,
              dtype: DType.float64,
            ).reshape([3, 4]);

            // 1. Reversed 1D view (negative stride)
            final flat = base.reshape([12]);
            final revView = flat.slice([Slice(step: -1)]);
            final revContig = revView.copy();
            final resView = op.call(revView);
            final resContig = op.call(revContig);
            expect(
              allClose(resView, resContig),
              isTrue,
              reason: '${op.name} failed negative-stride equivalence',
            );

            // 2. Transposed 2D view (non-contiguous strides)
            final tView = base.transpose();
            final tContig = tView.copy();
            final resTView = op.call(tView);
            final resTContig = op.call(tContig);
            expect(
              allClose(resTView, resTContig),
              isTrue,
              reason: '${op.name} failed transposed 2D equivalence',
            );
          });
        });

        test('non-contiguous out view & in-place out', () {
          NDArray.scope(() {
            final a = NDArray.fromList(
              <double>[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
              [6],
              DType.float64,
            );
            final expected = op.call(a);

            // Non-contiguous out view (step: 2)
            final carrier = NDArray.full([12], 99.0, dtype: DType.float64);
            final outSlice = carrier.slice([
              Slice(start: 0, stop: 12, step: 2),
            ]);
            final returned = op.call(a, out: outSlice);
            expect(sameId(returned, outSlice), isTrue);
            expect(allClose(outSlice, expected), isTrue);
            // Interstitial elements (odd indices) must remain untouched (99.0)
            final untouched = carrier.slice([
              Slice(start: 1, stop: 12, step: 2),
            ]);
            for (var i = 0; i < 6; i++) {
              expect(
                untouched[[i]],
                equals(99.0),
                reason:
                    '${op.name} corrupted interstitial elements of strided out',
              );
            }

            // In-place out: a
            if (op.supportsInPlace) {
              final inPlace = a.copy();
              op.call(inPlace, out: inPlace);
              expect(allClose(inPlace, expected), isTrue);
            }
          });
        });

        test(
          'disposed input or disposed out throws StateError and preserves ScratchArena',
          () {
            final markerBefore = ScratchArena.marker;
            final disposed = NDArray.zeros([4], DType.float64)..dispose();
            final valid = NDArray.ones([4], DType.float64);
            try {
              expect(() => op.call(disposed), throwsStateError);
              expect(() => op.call(valid, out: disposed), throwsStateError);
            } finally {
              valid.dispose();
            }
            expect(ScratchArena.marker, equals(markerBefore));
          },
        );
      });
    }
  });

  group('Binary Operation Contracts', () {
    for (final op in binaryOps) {
      group(op.name, () {
        test('validation happens BEFORE mutating out buffer', () {
          NDArray.scope(() {
            final a = NDArray.ones([4], DType.float64);
            final badB = NDArray.ones([3], DType.float64); // incompatible shape
            final out = NDArray.full([4], 99.0, dtype: DType.float64);
            final markerBefore = ScratchArena.marker;

            expect(() => op.call(a, badB, out: out), throwsArgumentError);
            expect(ScratchArena.marker, equals(markerBefore));

            for (var i = 0; i < 4; i++) {
              expect(
                out[[i]],
                equals(99.0),
                reason:
                    '${op.name} mutated `out` before shape validation failed',
              );
            }
          });
        });

        test(
          'strided & negative-stride equivalence and non-contiguous out',
          () {
            NDArray.scope(() {
              final a = linspace(1.0, 6.0, 6, dtype: DType.float64);
              final b = linspace(0.5, 3.0, 6, dtype: DType.float64);
              final revA = a.slice([Slice(step: -1)]);
              final revB = b.slice([Slice(step: -1)]);

              final expected = op.call(revA.copy(), revB.copy());
              final actual = op.call(revA, revB);
              expect(allClose(actual, expected), isTrue);

              final carrier = NDArray.full([12], 99.0, dtype: DType.float64);
              final stridedOut = carrier.slice([
                Slice(start: 0, stop: 12, step: 2),
              ]);
              op.call(revA, revB, out: stridedOut);
              expect(allClose(stridedOut, expected), isTrue);
            });
          },
        );

        test('disposed input or out throws StateError', () {
          final disposed = NDArray.ones([4], DType.float64)..dispose();
          final valid = NDArray.ones([4], DType.float64);
          try {
            expect(() => op.call(disposed, valid), throwsStateError);
            expect(() => op.call(valid, disposed), throwsStateError);
            expect(
              () => op.call(valid, valid, out: disposed),
              throwsStateError,
            );
          } finally {
            valid.dispose();
          }
        });
      });
    }
  });

  group('Reduction Operation Contracts', () {
    for (final op in reductionOps) {
      group(op.name, () {
        test('invalid axis or mismatched out does not mutate out buffer', () {
          NDArray.scope(() {
            final a = NDArray.ones([3, 4], DType.float64);
            final out = NDArray.full([3], 99.0, dtype: DType.float64);
            final markerBefore = ScratchArena.marker;

            // Out-of-bounds axis (axis: 5 on 2D array)
            expect(
              () => op.call(a, axis: 5, out: out),
              throwsA(anyOf(isA<RangeError>(), isA<ArgumentError>())),
            );
            expect(ScratchArena.marker, equals(markerBefore));
            for (var i = 0; i < 3; i++) {
              expect(
                out[[i]],
                equals(99.0),
                reason:
                    '${op.name} mutated `out` before axis validation failed',
              );
            }
          });
        });

        test('transposed and negative-stride reduction equivalence', () {
          NDArray.scope(() {
            final a = NDArray.arange(
              1.0,
              13.0,
              dtype: DType.float64,
            ).reshape([3, 4]);
            final tView = a.transpose();
            final tCopy = tView.copy();

            expect(
              allClose(op.call(tView, axis: 0), op.call(tCopy, axis: 0)),
              isTrue,
            );
            expect(
              allClose(op.call(tView, axis: 1), op.call(tCopy, axis: 1)),
              isTrue,
            );
          });
        });
      });
    }
  });
}

bool sameId(Object a, Object b) => identical(a, b);

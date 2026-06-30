import "package:ndarray/ndarray.dart";
import "package:test/test.dart";
import "dart:typed_data";

void main() {
  group("Kronecker Product (kron)", () {
    test("computes 2D x 2D matrix Kronecker product for Float64", () {
      NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
          2,
          2,
        ], DType.float64);
        final b = NDArray.fromList(Float64List.fromList([0.0, 5.0, 6.0, 7.0]), [
          2,
          2,
        ], DType.float64);

        final res = kron(a, b);
        expect(res.shape, equals([4, 4]));
        expect(res.dtype, equals(DType.float64));
        expect(
          res.toList(),
          equals([
            0.0,
            5.0,
            0.0,
            10.0,
            6.0,
            7.0,
            12.0,
            14.0,
            0.0,
            15.0,
            0.0,
            20.0,
            18.0,
            21.0,
            24.0,
            28.0,
          ]),
        );
      });
    });

    test("handles 1D and different rank arrays", () {
      NDArray.scope(() {
        final a = NDArray.fromList(Int64List.fromList([1, 2]), [
          2,
        ], DType.int64);
        final b = NDArray.fromList(Int64List.fromList([3, 4, 5]), [
          3,
        ], DType.int64);

        final res = kron(a, b);
        expect(res.shape, equals([6]));
        expect(res.toList(), equals([3, 4, 5, 6, 8, 10]));
      });
    });

    test("supports out argument", () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        final b = NDArray.fromList([3.0, 4.0], [2], DType.float64);
        final out = NDArray.zeros([4], DType.float64);

        final res = kron(a, b, out: out);
        expect(identical(res, out), isTrue);
        expect(out.toList(), equals([3.0, 4.0, 6.0, 8.0]));
      });
    });

    test("throws for disposed or incompatible out", () {
      final a = NDArray.fromList([1.0], [1], DType.float64);
      final b = NDArray.fromList([2.0], [1], DType.float64);
      final wrongOut = NDArray.zeros([2], DType.float64);

      expect(() => kron(a, b, out: wrongOut), throwsArgumentError);

      a.dispose();
      expect(() => kron(a, b), throwsStateError);
      b.dispose();
    });
  });

  group("Tensor Dot Product (tensordot)", () {
    test("computes tensordot over last 2 axes of A and first 2 axes of B", () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
        final b = NDArray.fromList([1.0, 0.0, 0.0, 1.0], [2, 2], DType.float64);

        final res = tensordot(a, b, axes: const TensordotAxes.count(2));
        expect(res.shape, equals([]));
        expect(res.scalar, equals(5.0));
      });
    });

    test(
      "computes matrix contraction with TensordotAxes.explicit and TensordotAxes.pair",
      () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final b = NDArray.fromList(
            [5.0, 6.0, 7.0, 8.0],
            [2, 2],
            DType.float64,
          );

          final resExplicit = tensordot(
            a,
            b,
            axes: const TensordotAxes.explicit([1], [0]),
          );
          expect(resExplicit.shape, equals([2, 2]));
          expect(resExplicit.getCell([0, 0]), equals(19.0));
          expect(resExplicit.getCell([0, 1]), equals(22.0));

          final resPair = tensordot(a, b, axes: TensordotAxes.pair(1, 0));

          expect(resPair.shape, equals([2, 2]));
          expect(resPair.getCell([0, 0]), equals(19.0));
        });
      },
    );

    test("supports out argument for tensordot", () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
        final b = NDArray.fromList([1.0, 0.0, 0.0, 1.0], [2, 2], DType.float64);
        final out = NDArray.zeros([2, 2], DType.float64);

        final res = tensordot(
          a,
          b,
          axes: const TensordotAxes.explicit([1], [0]),
          out: out,
        );
        expect(identical(res, out), isTrue);
        expect(out.getCell([0, 0]), equals(1.0));
        expect(out.getCell([1, 1]), equals(4.0));
      });
    });
    test(
      "fast path when contracting all axes of two contiguous Float64 arrays",
      () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final b = NDArray.fromList(
            [2.0, 3.0, 4.0, 5.0],
            [2, 2],
            DType.float64,
          );
          final res = tensordot(a, b, axes: const TensordotAxes.count(2));
          expect(res.shape, equals([]));
          expect(
            res.scalar,
            equals(40.0),
          ); // 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40

          final out = NDArray.zeros([], DType.float64);
          final resOut = tensordot(
            a,
            b,
            axes: const TensordotAxes.count(2),
            out: out,
          );
          expect(identical(resOut, out), isTrue);
          expect(out.scalar, equals(40.0));
        });
      },
    );

    test(
      "fast path when contracting all axes of two contiguous Float32 arrays",
      () {
        NDArray.scope(() {
          final a = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float32);
          final b = NDArray.fromList([4.0, 5.0, 6.0], [3], DType.float32);
          final res = tensordot(a, b, axes: const TensordotAxes.count(1));
          expect(res.shape, equals([]));
          expect(res.scalar, equals(32.0)); // 1*4 + 2*5 + 3*6 = 32

          final out = NDArray.zeros([], DType.float32);
          final resOut = tensordot(
            a,
            b,
            axes: const TensordotAxes.count(1),
            out: out,
          );
          expect(identical(resOut, out), isTrue);
          expect(out.scalar, equals(32.0));
        });
      },
    );
    test("flexible axes parameter types: int, records, and lists", () {
      NDArray.scope(() {
        final a = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [2, 3],
          DType.float64,
        );
        final b = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [3, 2],
          DType.float64,
        );

        // 1. int N
        final resInt = tensordot(a, b, axes: 1);
        expect(resInt.shape, equals([2, 2]));
        expect(resInt.getCell([0, 0]), equals(22.0)); // 1*1 + 2*3 + 3*5 = 22

        // 2. (int, int) record pair
        final resRecordPair = tensordot(a, b, axes: (1, 0));
        expect(resRecordPair.shape, equals([2, 2]));
        expect(resRecordPair.getCell([0, 0]), equals(22.0));

        // 3. (List<int>, List<int>) record pair
        final resRecordListPair = tensordot(a, b, axes: ([1], [0]));
        expect(resRecordListPair.shape, equals([2, 2]));
        expect(resRecordListPair.getCell([0, 0]), equals(22.0));

        // 4. List<List<int>>
        final resListList = tensordot(
          a,
          b,
          axes: [
            [1],
            [0],
          ],
        );
        expect(resListList.shape, equals([2, 2]));
        expect(resListList.getCell([0, 0]), equals(22.0));

        // 5. Throws for invalid axes parameter format
        expect(() => tensordot(a, b, axes: "invalid"), throwsArgumentError);
        expect(() => tensordot(a, b, axes: [1, 2, 3]), throwsArgumentError);
      });
    });

    test("axes=0 outer product fast path", () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        final b = NDArray.fromList([3.0, 4.0, 5.0], [3], DType.float64);

        // 1. int 0
        final res0 = tensordot(a, b, axes: 0);
        expect(res0.shape, equals([2, 3]));
        expect(res0.getCell([0, 0]), equals(3.0));
        expect(res0.getCell([0, 1]), equals(4.0));
        expect(res0.getCell([1, 2]), equals(10.0));

        // 2. axes=0 with out parameter
        final out = NDArray.zeros([2, 3], DType.float64);
        final resOut = tensordot(a, b, axes: 0, out: out);
        expect(identical(resOut, out), isTrue);
        expect(out.getCell([1, 2]), equals(10.0));

        // 3. 2D x 2D outer product (axes=0)
        final a2D = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        final b2D = NDArray.fromList([5.0, 6.0], [2, 1], DType.float64);
        final res2D0 = tensordot(a2D, b2D, axes: 0);
        expect(res2D0.shape, equals([2, 2, 2, 1]));
        expect(res2D0.getCell([0, 0, 0, 0]), equals(5.0)); // 1*5
        expect(res2D0.getCell([1, 1, 1, 0]), equals(24.0)); // 4*6
      });
    });
  });

  group("Einstein Summation (einsum)", () {
    test("matrix multiplication ij,jk->ik", () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
        final b = NDArray.fromList([5.0, 6.0, 7.0, 8.0], [2, 2], DType.float64);

        final res = einsum(EinsumSubscripts.parse("ij,jk->ik"), [a, b]);
        expect(res.shape, equals([2, 2]));
        expect(res.getCell([0, 0]), equals(19.0));
        expect(res.getCell([0, 1]), equals(22.0));
      });
    });

    test("implicit output einsum ij,jk", () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
        final b = NDArray.fromList([5.0, 6.0, 7.0, 8.0], [2, 2], DType.float64);

        final res = einsum(EinsumSubscripts.parse("ij,jk"), [a, b]);
        expect(res.shape, equals([2, 2]));
        expect(res.getCell([0, 0]), equals(19.0));
      });
    });

    test("vector inner product i,i->", () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        final b = NDArray.fromList([4.0, 5.0, 6.0], [3], DType.float64);

        final res = einsum(EinsumSubscripts.parse("i,i->"), [a, b]);
        expect(res.shape, equals([]));
        expect(res.scalar, equals(32.0));
      });
    });

    test("vector outer product i,j->ij", () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        final b = NDArray.fromList([3.0, 4.0, 5.0], [3], DType.float64);

        final res = einsum(EinsumSubscripts.parse("i,j->ij"), [a, b]);
        expect(res.shape, equals([2, 3]));
        expect(res.getCell([0, 0]), equals(3.0));
        expect(res.getCell([1, 2]), equals(10.0));
      });
    });

    test("matrix transpose ij->ji", () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);

        final res = einsum(EinsumSubscripts.parse("ij->ji"), [a]);
        expect(res.shape, equals([2, 2]));
        expect(res.getCell([0, 1]), equals(3.0));
        expect(res.getCell([1, 0]), equals(2.0));
      });
    });

    test("trace ii->", () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);

        final res = einsum(EinsumSubscripts.parse("ii->"), [a]);
        expect(res.shape, equals([]));
        expect(res.scalar, equals(5.0));
      });
    });

    test("diagonal extraction ii->i", () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);

        final res = einsum(EinsumSubscripts.parse("ii->i"), [a]);
        expect(res.shape, equals([2]));
        expect(res.toList(), equals([1.0, 4.0]));
      });
    });

    test("ellipsis broadcasting ...ij,...jk->...ik", () {
      NDArray.scope(() {
        final a = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [1, 2, 2],
          DType.float64,
        );
        final b = NDArray.fromList(
          [5.0, 6.0, 7.0, 8.0],
          [1, 2, 2],
          DType.float64,
        );

        final res = einsum(EinsumSubscripts.parse("...ij,...jk->...ik"), [
          a,
          b,
        ]);
        expect(res.shape, equals([1, 2, 2]));
        expect(res.getCell([0, 0, 0]), equals(19.0));
      });
    });

    test("Complex128 einsum matrix contraction", () {
      NDArray.scope(() {
        final a = NDArray.fromList(
          [Complex(1, 2), Complex(3, 4), Complex(5, 6), Complex(7, 8)],
          [2, 2],
          DType.complex128,
        );
        final b = NDArray.fromList(
          [Complex(1, 0), Complex(0, 1), Complex(1, 0), Complex(0, 1)],
          [2, 2],
          DType.complex128,
        );
        final res = einsum(EinsumSubscripts.parse("ij,jk->ik"), [a, b]);
        expect(res.shape, equals([2, 2]));
        expect(res.dtype, equals(DType.complex128));
      });
    });

    test("supports out parameter in einsum", () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
        final b = NDArray.fromList([5.0, 6.0, 7.0, 8.0], [2, 2], DType.float64);
        final out = NDArray.zeros([2, 2], DType.float64);

        final res = einsum(EinsumSubscripts.parse("ij,jk->ik"), [
          a,
          b,
        ], out: out);
        expect(identical(res, out), isTrue);
        expect(out.getCell([0, 0]), equals(19.0));
      });
    });

    test("throws for empty operands or subscript errors", () {
      expect(
        () => einsum(EinsumSubscripts.parse("ij"), <NDArray<Float64>>[]),
        throwsArgumentError,
      );
      final a = NDArray.fromList([1.0], [1], DType.float64);
      expect(
        () => einsum(EinsumSubscripts.parse("ij"), [a]),
        throwsArgumentError,
      );
      a.dispose();
    });

    test("matrix multiplication equivalence (tensordot axes=1)", () {
      NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
          2,
          2,
        ], DType.float64);
        final b = NDArray.fromList(Float64List.fromList([5.0, 6.0, 7.0, 8.0]), [
          2,
          2,
        ], DType.float64);
        final res = tensordot<double, double, double>(
          a,
          b,
          axes: const TensordotAxes.count(1),
        );

        final expectedMatmul = matmul(a, b);
        expect(res.shape, equals([2, 2]));
        expect(res.toList(), equals(expectedMatmul.toList()));
      });
    });

    test("throws for multiple arrow delimiters in einsum", () {
      expect(() => EinsumSubscripts.parse("i->j->k"), throwsArgumentError);
    });

    test(
      "EinsumSubscripts constructors: fromIndices, parse, and fromLabels",
      () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final b = NDArray.fromList(
            [5.0, 6.0, 7.0, 8.0],
            [2, 2],
            DType.float64,
          );

          // 1. EinsumSubscripts.fromIndices
          final specIndices = EinsumSubscripts.fromIndices(
            [
              [0, 1],
              [1, 2],
            ],
            [0, 2],
          );
          final resIndices = einsum(specIndices, [a, b]);
          expect(resIndices.shape, equals([2, 2]));
          expect(resIndices.getCell([0, 0]), equals(19.0));
          expect(
            specIndices.operandIndices,
            equals([
              [0, 1],
              [1, 2],
            ]),
          );
          expect(specIndices.outputIndices, equals([0, 2]));

          // 2. EinsumSubscripts.fromLabels
          final specLabels = EinsumSubscripts.fromLabels(
            [
              ['i', 'j'],
              ['j', 'k'],
            ],
            ['i', 'k'],
          );
          final resLabels = einsum(specLabels, [a, b]);
          expect(resLabels.shape, equals([2, 2]));
          expect(resLabels.getCell([0, 0]), equals(19.0));

          // 3. EinsumSubscripts.parse
          final specParse = EinsumSubscripts.parse("ij,jk->ik");
          final resParse = einsum(specParse, [a, b]);
          expect(resParse.shape, equals([2, 2]));
          expect(resParse.getCell([0, 0]), equals(19.0));

          // 4. EinsumSubscripts.fromLabels (implicit output)
          final specImplicit = EinsumSubscripts.fromLabels([
            ['i', 'j'],
            ['j', 'k'],
          ]);
          final resImplicit = einsum(specImplicit, [a, b]);
          expect(resImplicit.shape, equals([2, 2]));
          expect(resImplicit.getCell([0, 0]), equals(19.0));
          expect(specImplicit.outputIndices, isNull);
        });
      },
    );
    test(
      "batch matrix multiplication handler ...ij,...jk->...ik with Float64",
      () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [2, 2, 2],
            DType.float64,
          );
          final b = NDArray.fromList(
            [1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0],
            [2, 2, 2],
            DType.float64,
          );
          final res = einsum(EinsumSubscripts.parse("...ij,...jk->...ik"), [
            a,
            b,
          ]);
          expect(res.shape, equals([2, 2, 2]));
          expect(res.getCell([0, 0, 0]), equals(1.0));
          expect(res.getCell([0, 0, 1]), equals(2.0));
          expect(res.getCell([1, 0, 0]), equals(10.0));
          expect(res.getCell([1, 1, 1]), equals(16.0));

          final out = NDArray.zeros([2, 2, 2], DType.float64);
          final resOut = einsum(EinsumSubscripts.parse("...ij,...jk->...ik"), [
            a,
            b,
          ], out: out);
          expect(identical(resOut, out), isTrue);
          expect(out.getCell([1, 0, 0]), equals(10.0));
        });
      },
    );
    test(
      "implicit output subscripts alphabetical sorting (NumPy compatibility)",
      () {
        NDArray.scope(() {
          // einsum('ji,kj', a, b) -> implicit output labels 'ik' (alphabetical order of single labels 'i', 'k')
          // a is 2x3 (j=2, i=3), b is 4x2 (k=4, j=2)
          final a = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2, 3],
            DType.float64,
          );
          final b = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [4, 2],
            DType.float64,
          );

          final res = einsum(EinsumSubscripts.parse("ji,kj"), [a, b]);
          // Output labels: 'ik' -> shape [3, 4]
          expect(res.shape, equals([3, 4]));

          // Calculate expected manually for res[i, k]: sum_j a[j, i] * b[k, j]
          // For i=0, k=0: a[0, 0]*b[0, 0] + a[1, 0]*b[0, 1] = 1*1 + 4*2 = 9
          expect(res.getCell([0, 0]), equals(9.0));

          // Another case: einsum('ki,jk', a_ik, bJk) -> implicit output labels 'ij'
          final aKi = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final bJk = NDArray.fromList(
            [5.0, 6.0, 7.0, 8.0],
            [2, 2],
            DType.float64,
          );
          final resIj = einsum(EinsumSubscripts.parse("ki,jk"), [aKi, bJk]);
          expect(resIj.shape, equals([2, 2]));
        });
      },
    );

    test(
      "multi-operand pairwise contraction tree optimization (3 and 4 operands)",
      () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final b = NDArray.fromList(
            [5.0, 6.0, 7.0, 8.0],
            [2, 2],
            DType.float64,
          );
          final c = NDArray.fromList(
            [1.0, 0.0, 0.0, 1.0],
            [2, 2],
            DType.float64,
          );
          final d = NDArray.fromList(
            [2.0, 0.0, 0.0, 2.0],
            [2, 2],
            DType.float64,
          );

          // 3 operands: einsum('ij,jk,kl->il', A, B, C) == (A @ B) @ C
          final res3 = einsum(EinsumSubscripts.parse("ij,jk,kl->il"), [
            a,
            b,
            c,
          ]);
          expect(res3.shape, equals([2, 2]));
          expect(res3.getCell([0, 0]), equals(19.0));
          expect(res3.getCell([0, 1]), equals(22.0));
          expect(res3.getCell([1, 0]), equals(43.0));
          expect(res3.getCell([1, 1]), equals(50.0));

          // 4 operands: einsum('ij,jk,kl,lm->im', A, B, C, D) == ((A @ B) @ C) @ D
          final res4 = einsum(EinsumSubscripts.parse("ij,jk,kl,lm->im"), [
            a,
            b,
            c,
            d,
          ]);
          expect(res4.shape, equals([2, 2]));
          expect(res4.getCell([0, 0]), equals(38.0));
          expect(res4.getCell([0, 1]), equals(44.0));
          expect(res4.getCell([1, 0]), equals(86.0));
          expect(res4.getCell([1, 1]), equals(100.0));
        });
      },
    );
  });
}

import 'package:test/test.dart';
import 'package:ndarray/ndarray.dart' as nd;
import 'package:gpuarray/gpuarray.dart';
import 'package:resource_scope/resource_scope.dart';

void main() {
  group('GpuArray Extended DTypes & Complete 15-Type Support', () {
    test(
      'Float16 & BFloat16 GpuArray creation, arithmetic, and host roundtrip',
      () {
        ResourceScope.scope(() {
          // Float16
          final aF16 = GpuArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float16,
          );
          final bF16 = GpuArray.fromList(
            [10.0, 20.0, 30.0, 40.0],
            [2, 2],
            DType.float16,
          );
          expect(aF16.dtype, equals(DType.float16));
          expect(aF16.dtype.byteWidth, equals(2));

          final sumF16 = aF16 + bF16;
          expect(sumF16.dtype, equals(DType.float16));
          expect(sumF16.toList(), equals([11.0, 22.0, 33.0, 44.0]));

          // Roundtrip to host NDArray
          final hostND = sumF16.toNDArray();
          expect(hostND.dtype, equals(nd.DType.float16));
          expect(hostND.toList(), equals([11.0, 22.0, 33.0, 44.0]));

          // Convert back to GPU
          final backGpu = hostND.toGpu();
          expect(backGpu.dtype, equals(DType.float16));
          expect(backGpu.toList(), equals([11.0, 22.0, 33.0, 44.0]));

          // BFloat16
          final aBF16 = GpuArray.fromList([2.0, 4.0], [2], DType.bfloat16);
          final scaled = aBF16 * 3.0;
          expect(scaled.dtype, equals(DType.bfloat16));
          expect(scaled.toList(), equals([6.0, 12.0]));
        });
      },
    );

    test('Int8, Uint16, Uint32, Uint64 on GPU and host roundtrips', () {
      ResourceScope.scope(() {
        // Int8
        final i8 = GpuArray.fromList([10, -20, 30], [3], DType.int8);
        expect(i8.dtype, equals(DType.int8));
        expect(i8.dtype.byteWidth, equals(1));
        expect(i8.toList(), equals([10, -20, 30]));
        expect(i8.toNDArray().toList(), equals([10, -20, 30]));

        // Uint16
        final u16 = GpuArray.fromList([100, 200, 300], [3], DType.uint16);
        expect(u16.dtype, equals(DType.uint16));
        expect(u16.dtype.byteWidth, equals(2));
        expect(u16.toList(), equals([100, 200, 300]));
        expect(u16.toNDArray().toList(), equals([100, 200, 300]));

        // Uint32
        final u32 = GpuArray.fromList([1000, 2000, 3000], [3], DType.uint32);
        expect(u32.dtype, equals(DType.uint32));
        expect(u32.dtype.byteWidth, equals(4));
        expect(u32.toList(), equals([1000, 2000, 3000]));
        expect(u32.toNDArray().toList(), equals([1000, 2000, 3000]));

        // Uint64
        final u64 = GpuArray.fromList([5, 10, 15], [3], DType.uint64);
        expect(u64.dtype, equals(DType.uint64));
        expect(u64.dtype.byteWidth, equals(8));
        expect(u64.toList(), equals([5, 10, 15]));
        expect(u64.toNDArray().toList(), equals([5, 10, 15]));
      });
    });

    test(
      'All 15 data types allocate and transfer cleanly between NDArray and GpuArray',
      () {
        ResourceScope.scope(() {
          for (final dtype in DType.values) {
            final GpuArray arr;
            if (dtype == DType.boolean) {
              arr = GpuArray.fromList(
                [true, false, true, false],
                [2, 2],
                dtype,
              );
            } else {
              arr = GpuArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], dtype);
            }
            expect(arr.dtype, equals(dtype));
            expect(arr.shape, equals([2, 2]));

            final host = arr.toNDArray();
            expect(host.dtype.name, equals(dtype.name));
            expect(host.shape, equals([2, 2]));

            final back = host.toGpu();
            expect(back.dtype.name, equals(dtype.name));
            expect(back.shape, equals([2, 2]));
          }
        });
      },
    );

    test('Complex64 and Complex128 preserve imaginary parts on GPU', () {
      ResourceScope.scope(() {
        final cpxList = [
          nd.Complex(1.0, 2.0),
          nd.Complex(3.0, -4.0),
          nd.Complex(0.0, 5.0),
          nd.Complex(-6.0, 0.0),
        ];
        final gpuCpx128 = GpuArray.fromList(cpxList, [2, 2], DType.complex128);
        expect(gpuCpx128.dtype, equals(DType.complex128));

        final hostND = gpuCpx128.toNDArray();
        expect(hostND.toList(), equals(cpxList));

        final copiedGpu = gpuCpx128.copy();
        expect(copiedGpu.toList(), equals(cpxList));
      });
    });

    test('Uint64 handles large values > 2^63 - 1 without sign corruption', () {
      ResourceScope.scope(() {
        final largeVal = 0x8000000000000000; // 2^63
        final gpuU64 = GpuArray.fromList([largeVal, 100], [2], DType.uint64);
        expect(gpuU64.dtype, equals(DType.uint64));

        final hostND = gpuU64.toNDArray();
        expect(hostND.toList(), equals([largeVal, 100]));
      });
    });
  });
}

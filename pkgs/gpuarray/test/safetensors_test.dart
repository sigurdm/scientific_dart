import 'dart:io';
import 'package:test/test.dart';
import 'package:gpuarray/gpuarray.dart';
import 'package:gpuarray/safetensors.dart' as st;
import 'package:resource_scope/resource_scope.dart';

void main() {
  group('GpuArray SafeTensors Serialization (gpuarray.safetensors)', () {
    test('Roundtrip in-memory SafeTensors serialization', () {
      ResourceScope.scope(() {
        final w = GpuArray.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], DType.float32);
        final b = GpuArray.fromList([0.1, 0.2], [2], DType.float64);
        final mask = GpuArray.fromList([true, false, true, true], [4], DType.boolean);

        final tensorDict = {
          'model.weight': w,
          'model.bias': b,
          'model.mask': mask,
        };

        final bytes = st.saveSafetensors(tensorDict, metadata: {'format': 'pt'});
        expect(bytes.isNotEmpty, isTrue);

        final loadedDict = st.loadSafetensors(bytes);
        expect(loadedDict.keys, containsAll(['model.weight', 'model.bias', 'model.mask']));

        expect(loadedDict['model.weight']!.shape, equals([2, 3]));
        expect(loadedDict['model.weight']!.dtype, equals(DType.float32));
        expect(loadedDict['model.weight']!.toList().cast<double>(), equals([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));

        expect(loadedDict['model.bias']!.shape, equals([2]));
        expect(loadedDict['model.bias']!.dtype, equals(DType.float64));
        expect(loadedDict['model.bias']!.toList().cast<double>(), equals([0.1, 0.2]));

        expect(loadedDict['model.mask']!.shape, equals([4]));
        expect(loadedDict['model.mask']!.dtype, equals(DType.boolean));
        expect(loadedDict['model.mask']!.toList().cast<bool>(), equals([true, false, true, true]));
      });
    });

    test('Roundtrip file-based SafeTensors save and load', () {
      ResourceScope.scope(() {
        final tempDir = Directory.systemTemp.createTempSync('safetensors_test');
        final filePath = '${tempDir.path}/test_model.safetensors';

        try {
          final emb = GpuArray.fromList([10, 20, 30, 40], [2, 2], DType.int32);
          st.saveSafetensorsFile(filePath, {'embeddings': emb});

          final loaded = st.loadSafetensorsFile(filePath);
          expect(loaded.containsKey('embeddings'), isTrue);
          expect(loaded['embeddings']!.shape, equals([2, 2]));
          expect(loaded['embeddings']!.dtype, equals(DType.int32));
          expect(loaded['embeddings']!.toList().cast<int>(), equals([10, 20, 30, 40]));
        } finally {
          tempDir.deleteSync(recursive: true);
        }
      });
    });
  });
}

import 'package:test/test.dart';
import 'package:gpuarray/gpuarray.dart';
import 'package:gpuarray/nn.dart' as nn;
import 'package:resource_scope/resource_scope.dart';

void main() {
  group('GpuArray Neural Network Primitives (gpuarray.nn)', () {
    test('Linear layer forward and parameter registration', () {
      ResourceScope.scope(() {
        final fc = nn.Linear(4, 2);
        expect(fc.parameters().length, equals(2)); // weight and bias
        expect(fc.namedParameters().keys, containsAll(['weight', 'bias']));

        final x = GpuArray.fromList([
          1.0, 2.0, 3.0, 4.0,
          0.5, 1.5, 2.5, 3.5,
        ], [2, 4], DType.float64);

        final out = fc(x);
        expect(out.shape, equals([2, 2]));
        expect(out.requiresGrad, isTrue);
      });
    });

    test('Sequential MLP with ReLU and MSE Loss optimization (SGD / Adam)', () {
      ResourceScope.scope(() {
        // Target function: y = 2*x1 - 3*x2
        final model = nn.Sequential([
          nn.Linear(2, 4),
          nn.ReLU(),
          nn.Linear(4, 1),
        ]);

        final optimizer = nn.Adam(model.parameters(), lr: 0.05);

        final xTrain = GpuArray.fromList([
          1.0, 1.0,
          2.0, 0.0,
          0.0, 1.0,
          -1.0, 2.0,
        ], [4, 2], DType.float64);

        final yTrain = GpuArray.fromList([
          -1.0,
          4.0,
          -3.0,
          -8.0,
        ], [4, 1], DType.float64);

        // Train for 20 steps
        double? initialLoss;
        double? finalLoss;

        for (var epoch = 0; epoch < 20; epoch++) {
          optimizer.zeroGrad();
          final pred = model(xTrain);
          final loss = nn.mse_loss(pred, yTrain);

          final lossVal = loss.toList().cast<double>().first;
          if (epoch == 0) initialLoss = lossVal;
          if (epoch == 19) finalLoss = lossVal;

          loss.backward();
          optimizer.step();
        }

        expect(initialLoss, isNotNull);
        expect(finalLoss, isNotNull);
        expect(finalLoss!, lessThan(initialLoss!)); // Loss decreased
      });
    });

    test('LayerNorm and Dropout', () {
      ResourceScope.scope(() {
        final ln = nn.LayerNorm([4]);
        final x = GpuArray.fromList([
          1.0, 2.0, 3.0, 4.0,
        ], [1, 4], DType.float64);

        final out = ln(x);
        expect(out.shape, equals([1, 4]));

        final outList = out.toList().cast<double>();
        var sum = 0.0;
        for (final v in outList) {
          sum += v;
        }
        expect(sum / 4.0, closeTo(0.0, 1e-4)); // Normalized mean is 0

        final drop = nn.Dropout(p: 0.0);
        final dropOut = drop(x);
        expect(dropOut.toList(), equals(x.toList()));
      });
    });

    test('Activations & Softmax', () {
      ResourceScope.scope(() {
        final x = GpuArray.fromList([-2.0, 0.0, 2.0], [3], DType.float64);

        final reluOut = nn.relu(x);
        expect(reluOut.toList(), equals([0.0, 0.0, 2.0]));

        final sm = nn.softmax(x);
        final smList = sm.toList().cast<double>();
        var sum = 0.0;
        for (final v in smList) {
          sum += v;
        }
        expect(sum, closeTo(1.0, 1e-4)); // Probabilities sum to 1
      });
    });
  });
}

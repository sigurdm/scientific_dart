import 'dart:io';
import 'package:path/path.dart' as p;
import 'package:test/test.dart';
import 'package:notebook/notebook.dart';
import 'package:ndarray/ndarray.dart';
import 'dart:math' as math;

void main() {
  test('Audio generates valid 16-bit PCM WAV Data URL and HTML', () {
    final numSamples = 4410; // 0.1s audio @ 44100Hz
    final samples = NDArray<Float64>.zeros([numSamples], DType.float64);
    for (var i = 0; i < numSamples; i++) {
      samples[[i]] = Float64(math.sin(2 * math.pi * 440 * (i / 44100.0)));
    }

    final audio = Audio(samples, sampleRate: 44100);
    final html = audio.toHtml();

    expect(html, contains('<audio controls'));
    expect(html, contains('src="data:audio/wav;base64,'));
  });

  test('prettyFormat formats Displayable objects, Html, Audio, and Image', () {
    final htmlObj = Html('<h1>Hello Notebook</h1>');
    expect(prettyFormat(htmlObj), equals('<h1>Hello Notebook</h1>'));

    final numSamples = 441;
    final samples = NDArray<Float64>.zeros([numSamples], DType.float64);
    final audioObj = Audio(samples);
    expect(prettyFormat(audioObj), contains('<audio controls'));

    final imgArr = NDArray<Float64>.zeros([10, 10, 3], DType.float64);
    final imgObj = Image(imgArr);
    expect(prettyFormat(imgObj), contains('data:image/bmp;base64,'));
  });

  test(
    'kernel captures display(Audio) and display(Spectrogram) without JSON truncation',
    () async {
      final workspaceDir = Directory.current.path;
      final sdkPath =
          Platform.environment['DART_SDK'] ??
          p.dirname(p.dirname(Platform.resolvedExecutable));
      final kernel = NotebookKernel(
        workspaceDir: workspaceDir,
        dartSdkPath: sdkPath,
      );
      await kernel.start();
      try {
        final code = '''
var time = linspace<Float64>(Float64(0.0), Float64(0.1), 4410);
var samples = sin(time * 440 * 2 * math.pi) * 0.5;
display(Audio(samples, sampleRate: 44100));
display(Spectrogram(samples, sampleRate: 44100));
''';
        final result = await kernel.execute(code);
        expect(result, contains('data:audio/wav;base64,'));
        expect(result, contains('spectrogram-container'));
      } finally {
        await kernel.stop();
      }
    },
  );
}

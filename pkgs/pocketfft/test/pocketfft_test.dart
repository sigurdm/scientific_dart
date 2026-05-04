import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart';
import 'package:pocketfft/pocketfft.dart';
import 'package:test/test.dart';

void main() {
  group('package:pocketfft Raw FFI Bindings Tests', () {
    test(
      'Perform basic raw FFI kiss_fft transform on unmanaged heap structs',
      () {
        const nfft = 4;
        // Allocate forward transform plan (0 for forward)
        final cfg = kiss_fft_alloc(nfft, 0, ffi.nullptr, ffi.nullptr);
        expect(cfg.address != 0, true);

        final fin = malloc<kiss_fft_cpx>(nfft);
        final fout = malloc<kiss_fft_cpx>(nfft);

        try {
          // Initialize flat Kronecker delta spike input: [1.0 + 0i, 0i, 0i, 0i]
          fin[0].r = 1.0;
          fin[0].i = 0.0;
          fin[1].r = 0.0;
          fin[1].i = 0.0;
          fin[2].r = 0.0;
          fin[2].i = 0.0;
          fin[3].r = 0.0;
          fin[3].i = 0.0;

          // Execute native FFT
          kiss_fft(cfg, fin, fout);

          // Kronecker delta spike FFT is a perfectly flat spectrum: [1.0 + 0i, 1.0 + 0i, 1.0 + 0i, 1.0 + 0i]!
          for (var i = 0; i < nfft; i++) {
            expect(fout[i].r, closeTo(1.0, 1e-9));
            expect(fout[i].i, closeTo(0.0, 1e-9));
          }
        } finally {
          // Release heap configurations and buffers
          free(cfg.cast<ffi.Void>());
          malloc.free(fin);
          malloc.free(fout);
        }
      },
    );
  });
}

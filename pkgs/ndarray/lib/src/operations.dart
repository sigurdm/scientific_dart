// ignore_for_file: non_constant_identifier_names
@ffi.DefaultAsset('package:openblas/openblas')
library operations;

import 'dart:typed_data';
import 'dart:math' as math;
import 'ndarray.dart';
import 'broadcasting.dart';
import 'package:openblas/openblas.dart';
import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart';
import 'ndarray_bindings.dart';
import 'scratch_arena.dart';

part 'operations/math.dart';
part 'operations/stats.dart';
part 'operations/sorting.dart';
part 'operations/linalg.dart';
part 'operations/spacers.dart';
part 'operations/manipulation.dart';

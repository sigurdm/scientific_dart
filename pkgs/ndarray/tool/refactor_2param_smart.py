import re
import sys

def fix_linalg(content):
    print("Applying specific fixes to linalg.dart...")
    # --- eig fixes ---
    # 1. Make eig generic on MC
    content = content.replace(
        '({NDArray<Complex, Complex128Marker> eigenvalues, NDArray<Complex, Complex128Marker> eigenvectors}) eig<T, MT extends Marker>(\n  NDArray<T, MT> a, {\n  ({NDArray<Complex, Complex128Marker> eigenvalues, NDArray<Complex, Complex128Marker> eigenvectors})? out,\n}) {',
        '({NDArray<Complex, MC> eigenvalues, NDArray<Complex, MC> eigenvectors}) eig<T, MT extends Marker, MC extends Marker>(\n  NDArray<T, MT> a, {\n  ({NDArray<Complex, MC> eigenvalues, NDArray<Complex, MC> eigenvectors})? out,\n}) {'
    )
    # w and vr declarations in eig
    content = content.replace(
        '    final NDArray<Complex, Complex128Marker> w;\n    final NDArray<Complex, Complex128Marker> vr;',
        '    final NDArray<Complex, MC> w;\n    final NDArray<Complex, MC> vr;'
    )
    # Cast compDType definition in eig
    content = content.replace(
        '  final compDType = (a.dtype == DType.float32 || a.dtype == DType.complex64)\n      ? DType.complex64\n      : DType.complex128;',
        '  final compDType = ((a.dtype == DType.float32 || a.dtype == DType.complex64)\n      ? DType.complex64\n      : DType.complex128) as DType<Complex, MC>;'
    )
    
    # 2. complex128 case in eig (create with compDType, no casts in view)
    content = content.replace(
        '        case DType.complex128:\n          final w2D = NDArray.create([n], DType.complex128);\n          final vr2D = NDArray.create([n, n], DType.complex128);\n\n          final info = LAPACKE_zgeev(\n            101, // ROW_MAJOR\n            jobvl,\n            jobvr,\n            n,\n            sliceCopy.pointer.cast<ffi.Double>(),\n            n,\n            w2D.pointer.cast<ffi.Double>(),\n            ffi.nullptr.cast<ffi.Double>(),\n            n,\n            vr2D.pointer.cast<ffi.Double>(),\n            n,\n          );\n\n          if (info < 0) {\n            throw ArgumentError(\n              \'Illegal value in call to LAPACKE_zgeev: $info\',\n            );\n          }\n          if (info > 0) {\n            throw ArgumentError(\n              \'The LAPACK QR algorithm failed to converge; only eigenvalues from 1-based index ${info + 1} to $n successfully converged.\',\n            );\n          }\n\n          final wView = NDArray.view(\n            w,\n            shape: [n],\n            strides: w.strides.isEmpty ? [1] : [w.strides.last],\n            offsetElements: offsetW,\n          );\n          w2D.copy(out: wView);\n\n          final vrView = NDArray.view(\n            vr,\n            shape: [n, n],\n            strides: vr.strides.sublist(rank - 2),\n            offsetElements: offsetVR,\n          );\n          vr2D.copy(out: vrView);',
        '        case DType.complex128:\n          final w2D = NDArray.create([n], compDType);\n          final vr2D = NDArray.create([n, n], compDType);\n\n          final info = LAPACKE_zgeev(\n            101, // ROW_MAJOR\n            jobvl,\n            jobvr,\n            n,\n            sliceCopy.pointer.cast<ffi.Double>(),\n            n,\n            w2D.pointer.cast<ffi.Double>(),\n            ffi.nullptr.cast<ffi.Double>(),\n            n,\n            vr2D.pointer.cast<ffi.Double>(),\n            n,\n          );\n\n          if (info < 0) {\n            throw ArgumentError(\n              \'Illegal value in call to LAPACKE_zgeev: $info\',\n            );\n          }\n          if (info > 0) {\n            throw ArgumentError(\n              \'The LAPACK QR algorithm failed to converge; only eigenvalues from 1-based index ${info + 1} to $n successfully converged.\',\n            );\n          }\n\n          final wView = NDArray.view(\n            w,\n            shape: [n],\n            strides: w.strides.isEmpty ? [1] : [w.strides.last],\n            offsetElements: offsetW,\n          );\n          w2D.copy(out: wView);\n\n          final vrView = NDArray.view(\n            vr,\n            shape: [n, n],\n            strides: vr.strides.sublist(rank - 2),\n            offsetElements: offsetVR,\n          );\n          vr2D.copy(out: vrView);'
    )
    
    # 3. complex64 case in eig (create with compDType, no casts in view)
    content = content.replace(
        '        case DType.complex64:\n          final w2D = NDArray.create([n], DType.complex64);\n          final vr2D = NDArray.create([n, n], DType.complex64);\n\n          final info = LAPACKE_cgeev(\n            101, // ROW_MAJOR\n            jobvl,\n            jobvr,\n            n,\n            sliceCopy.pointer.cast<ffi.Float>(),\n            n,\n            w2D.pointer.cast<ffi.Float>(),\n            ffi.nullptr.cast<ffi.Float>(),\n            n,\n            vr2D.pointer.cast<ffi.Float>(),\n            n,\n          );\n\n          if (info < 0) {\n            throw ArgumentError(\n              \'Illegal value in call to LAPACKE_cgeev: $info\',\n            );\n          }\n          if (info > 0) {\n            throw ArgumentError(\n              \'The LAPACK QR algorithm failed to converge; only eigenvalues from 1-based index ${info + 1} to $n successfully converged.\',\n            );\n          }\n\n          final wView = NDArray.view(\n            w,\n            shape: [n],\n            strides: w.strides.isEmpty ? [1] : [w.strides.last],\n            offsetElements: offsetW,\n          );\n          w2D.copy(out: wView);\n\n          final vrView = NDArray.view(\n            vr,\n            shape: [n, n],\n            strides: vr.strides.sublist(rank - 2),\n            offsetElements: offsetVR,\n          );\n          vr2D.copy(out: vrView);',
        '        case DType.complex64:\n          final w2D = NDArray.create([n], compDType);\n          final vr2D = NDArray.create([n, n], compDType);\n\n          final info = LAPACKE_cgeev(\n            101, // ROW_MAJOR\n            jobvl,\n            jobvr,\n            n,\n            sliceCopy.pointer.cast<ffi.Float>(),\n            n,\n            w2D.pointer.cast<ffi.Float>(),\n            ffi.nullptr.cast<ffi.Float>(),\n            n,\n            vr2D.pointer.cast<ffi.Float>(),\n            n,\n          );\n\n          if (info < 0) {\n            throw ArgumentError(\n              \'Illegal value in call to LAPACKE_cgeev: $info\',\n            );\n          }\n          if (info > 0) {\n            throw ArgumentError(\n              \'The LAPACK QR algorithm failed to converge; only eigenvalues from 1-based index ${info + 1} to $n successfully converged.\',\n            );\n          }\n\n          final wView = NDArray.view(\n            w,\n            shape: [n],\n            strides: w.strides.isEmpty ? [1] : [w.strides.last],\n            offsetElements: offsetW,\n          );\n          w2D.copy(out: wView);\n\n          final vrView = NDArray.view(\n            vr,\n            shape: [n, n],\n            strides: vr.strides.sublist(rank - 2),\n            offsetElements: offsetVR,\n          );\n          vr2D.copy(out: vrView); '
    )

    # --- qr fixes ---
    # 1. targetDType and qMat/rMat declarations in qr
    content = content.replace(
        '  final DType<double, Float64Marker> targetDType = a.dtype == DType.float32\n      ? DType.float32 as DType<double, Float64Marker>\n      : DType.float64 as DType<double, Float64Marker>;\n\n  final qShape = [...stackShape, m, k];\n  final rShape = [...stackShape, k, n];\n\n  final NDArray<double, Float64Marker> qMat;\n  final NDArray<double, Float64Marker> rMat;\n  if (out != null) {\n    qMat = out.Q as NDArray<double, Float64Marker>;\n    rMat = out.R as NDArray<double, Float64Marker>;',
        '  final targetDType = (a.dtype == DType.float32\n      ? DType.float32\n      : DType.float64) as DType<T, MT>;\n\n  final qShape = [...stackShape, m, k];\n  final rShape = [...stackShape, k, n];\n\n  final NDArray<T, MT> qMat;\n  final NDArray<T, MT> rMat;\n  if (out != null) {\n    qMat = out.Q;\n    rMat = out.R;'
    )
    # 2. q2D/r2D creation in qr (use targetDType instead of DType.float32/64)
    content = content.replace(
        '      final r2D = targetDType == DType.float32\n          ? NDArray.zeros([k, n], DType.float32)\n          : NDArray.zeros([k, n], DType.float64);\n      final q2D = targetDType == DType.float32\n          ? NDArray.zeros([m, k], DType.float32)\n          : NDArray.zeros([m, k], DType.float64);',
        '      final r2D = NDArray.zeros([k, n], targetDType);\n      final q2D = NDArray.zeros([m, k], targetDType);'
    )

    # --- svd fixes ---
    # 1. Make svd generic on MS
    content = content.replace(
        '({NDArray<T, MT> U, NDArray<double, Float64Marker> S, NDArray<T, MT> Vh}) svd<T, MT extends Marker>(\n  NDArray<T, MT> a, {\n  ({NDArray<T, MT> U, NDArray<double, Float64Marker> S, NDArray<T, MT> Vh})? out,\n}) {',
        '({NDArray<T, MT> U, NDArray<double, MS> S, NDArray<T, MT> Vh}) svd<T, MT extends Marker, MS extends Marker>(\n  NDArray<T, MT> a, {\n  ({NDArray<T, MT> U, NDArray<double, MS> S, NDArray<T, MT> Vh})? out,\n}) {'
    )
    # dtypeS in svd
    content = content.replace(
        '  final dtypeS = a.dtype.isComplex\n      ? (a.dtype == DType.complex128 ? DType.float64 : DType.float32)\n      : a.dtype;',
        '  final dtypeS = (a.dtype.isComplex\n      ? (a.dtype == DType.complex128 ? DType.float64 : DType.float32)\n      : a.dtype) as DType<double, MS>;'
    )
    # 2. Make _svd generic on MS
    content = content.replace(
        '({NDArray<T, MT> U, NDArray<double, Float64Marker> S, NDArray<T, MT> Vh}) _svd<T, MT extends Marker>(\n  NDArray<T, MT> a, {\n  ({NDArray<T, MT> U, NDArray<double, Float64Marker> S, NDArray<T, MT> Vh})? out,\n}) {',
        '({NDArray<T, MT> U, NDArray<double, MS> S, NDArray<T, MT> Vh}) _svd<T, MT extends Marker, MS extends Marker>(\n  NDArray<T, MT> a, {\n  ({NDArray<T, MT> U, NDArray<double, MS> S, NDArray<T, MT> Vh})? out,\n}) {'
    )
    # dtypeS in _svd
    content = content.replace(
        '  final dtypeS = a.dtype.isComplex\n      ? (a.dtype == DType.complex128 ? DType.float64 : DType.float32)\n      : a.dtype;',
        '  final dtypeS = (a.dtype.isComplex\n      ? (a.dtype == DType.complex128 ? DType.float64 : DType.float32)\n      : a.dtype) as DType<double, MS>;'
    )
    # sMat creation in _svd
    content = content.replace(
        '  final sMat =\n      out?.S ?? NDArray.zeros(sShape, dtypeS as DType<double, Float64Marker>);',
        '  final sMat =\n      out?.S ?? NDArray.zeros(sShape, dtypeS);'
    )
    # s2D creation in _svd: create with dtypeS
    content = content.replace(
        '      final s2D =\n          (a.dtype == DType.float32 || a.dtype == DType.complex64)\n          ? NDArray.zeros([n], DType.float32)\n          : NDArray.zeros([n], DType.float64);',
        '      final s2D = NDArray.zeros([n], dtypeS);'
    )
    # _svd recursive call (line 2368)
    content = content.replace(
        '    final resT = _svd(aT);',
        '    final ({NDArray<T, MT> U, NDArray<double, MS> S, NDArray<T, MT> Vh}) resT = _svd(aT);'
    )
    return content

def fix_math(content):
    print("Applying specific fixes to math.dart...")
    # --- atan2 fixes ---
    # 1. Make atan2 generic on MR
    content = content.replace(
        'NDArray<double, Float64Marker> atan2<Ty, MTy extends Marker, Tx, MTx extends Marker>(\n  NDArray<Ty, MTy> y,\n  NDArray<Tx, MTx> x, {\n  NDArray<double, Float64Marker>? out,\n}) {',
        'NDArray<double, MR> atan2<Ty, MTy extends Marker, Tx, MTx extends Marker, MR extends FloatingMarker>(\n  NDArray<Ty, MTy> y,\n  NDArray<Tx, MTx> x, {\n  NDArray<double, MR>? out,\n}) {'
    )
    # targetDType in atan2
    content = content.replace(
        '  final DType<double, Float64Marker> targetDType =\n      (y.dtype == DType.float32 && x.dtype == DType.float32)\n      ? DType.float32\n      : DType.float64;',
        '  final targetDType = ((y.dtype == DType.float32 && x.dtype == DType.float32)\n      ? DType.float32\n      : DType.float64) as DType<double, MR>;'
    )
    # result declaration in atan2
    content = content.replace(
        '  final NDArray<double, Float64Marker> result;',
        '  final NDArray<double, MR> result;'
    )

    # --- hypot fixes ---
    # 1. Make hypot generic on MR
    content = content.replace(
        'NDArray<double, Float64Marker> hypot(NDArray x1, NDArray x2, {NDArray<double, Float64Marker>? out}) {',
        'NDArray<double, MR> hypot<MR extends FloatingMarker>(NDArray x1, NDArray x2, {NDArray<double, MR>? out}) {'
    )
    # targetDType in hypot
    content = content.replace(
        '  final DType<double, Float64Marker> targetDType =\n      (x1.dtype == DType.complex64 || x2.dtype == DType.complex64)\n      ? DType.float32 as DType<double, Float64Marker>\n      : DType.float64 as DType<double, Float64Marker>;',
        '  final targetDType = ((x1.dtype == DType.complex64 || x2.dtype == DType.complex64)\n      ? DType.float32\n      : DType.float64) as DType<double, MR>;'
    )
    # result declaration in hypot
    content = content.replace(
        '  final NDArray<double, Float64Marker> result;',
        '  final NDArray<double, MR> result;'
    )

    # --- norm fixes ---
    # 1. Make norm generic on MR (matching exact file whitespace & keepdims parameter name)
    content = content.replace(
        'NDArray<double, Float64Marker> norm<T, MT extends Marker>(\n  NDArray<T, MT> a, {\n  dynamic ord,\n  dynamic axis,\n  bool keepdims = false,\n  NDArray<double, Float64Marker>? out,\n}) {',
        'NDArray<double, MR> norm<T, MT extends Marker, MR extends FloatingMarker>(\n  NDArray<T, MT> a, {\n  dynamic ord,\n  dynamic axis,\n  bool keepdims = false,\n  NDArray<double, MR>? out,\n}) {'
    )
    # targetDType in norm
    content = content.replace(
        '  final targetDType = (a.dtype == DType.float32 || a.dtype == DType.complex64)\n      ? DType.float32\n      : DType.float64;',
        '  final targetDType = ((a.dtype == DType.float32 || a.dtype == DType.complex64)\n      ? DType.float32\n      : DType.float64) as DType<double, MR>;'
    )

    # --- lstsq fixes ---
    # Branching s creation with explicit type arguments & remove sDType
    content = content.replace(
        '  final sDType = (a.dtype == DType.complex64 || a.dtype == DType.float32)\n      ? DType.float32\n      : DType.float64;\n  final s = NDArray.zeros([minMN], sDType as dynamic);',
        '  final NDArray<double, Marker> s;\n  if (a.dtype == DType.complex64 || a.dtype == DType.float32) {\n    s = NDArray<double, Float32Marker>.zeros([minMN], DType.float32);\n  } else {\n    s = NDArray<double, Float64Marker>.zeros([minMN], DType.float64);\n  }'
    )
    # residuals declaration
    content = content.replace(
        '    final NDArray<double, Float64Marker> residuals;',
        '    final NDArray<double, Marker> residuals;'
    )
    # Branching residuals creation when computed with explicit type arguments
    content = content.replace(
        '      residuals = NDArray.zeros(resShape, sDType as dynamic);',
        '      if (a.dtype == DType.complex64 || a.dtype == DType.float32) {\n        residuals = NDArray<double, Float32Marker>.zeros(resShape, DType.float32);\n      } else {\n        residuals = NDArray<double, Float64Marker>.zeros(resShape, DType.float64);\n      }'
    )
    # Branching residuals creation when empty with explicit type arguments
    content = content.replace(
        '      residuals = NDArray.zeros([0], sDType as dynamic);',
        '      if (a.dtype == DType.complex64 || a.dtype == DType.float32) {\n        residuals = NDArray<double, Float32Marker>.zeros([0], DType.float32);\n      } else {\n        residuals = NDArray<double, Float64Marker>.zeros([0], DType.float64);\n      }'
    )
    # Promote and return
    content = content.replace(
        '    return LstsqResult<T, MT>(x: x, residuals: residuals, rank: rank, s: s);',
        '    final NDArray<double, Float64Marker> sPromoted;\n    if (s.dtype == DType.float32) {\n      sPromoted = castNDArray(s as NDArray<double, Float32Marker>, DType.float64);\n      s.dispose();\n    } else {\n      sPromoted = s as NDArray<double, Float64Marker>;\n    }\n\n    final NDArray<double, Float64Marker> resPromoted;\n    if (residuals.dtype == DType.float32) {\n      resPromoted = castNDArray(residuals as NDArray<double, Float32Marker>, DType.float64);\n      residuals.dispose();\n    } else {\n      resPromoted = residuals as NDArray<double, Float64Marker>;\n    }\n\n    return LstsqResult(x: x, residuals: resPromoted, rank: rank, s: sPromoted);'
    )

    # Fix factory type inference by casting DType instead of result
    content = re.sub(
        r'\bNDArray\.(create|zeros)\(([^,]+),\s*targetDType\)\s*as\s*NDArray<R,\s*MR>',
        r'NDArray.\1(\2, targetDType as DType<R, MR>)',
        content
    )
    return content

def fix_calculus(content):
    print("Applying specific fixes to calculus.dart...")
    # 1. trapz/gradient case complex64 spacingArray type
    content = content.replace(
        '            case DType.complex64:\n              final dxStruct = ScratchArena.allocate<cpx_f_t>(\n                ffi.sizeOf<cpx_f_t>(),\n              );\n              dxStruct.ref.r = 1.0;\n              dxStruct.ref.i = 0.0;\n              NDArray<Complex, Complex128Marker>? spacingArray;',
        '            case DType.complex64:\n              final dxStruct = ScratchArena.allocate<cpx_f_t>(\n                ffi.sizeOf<cpx_f_t>(),\n              );\n              dxStruct.ref.r = 1.0;\n              dxStruct.ref.i = 0.0;\n              NDArray<Complex, Complex64Marker>? spacingArray;'
    )
    # 2. trapz case real spacingArray type & cast
    content = content.replace(
        '          NDArray<double, Float64Marker>? spacingArray;\n          spacingArray = NDArray.fromList(\n            doubleValues,\n            [N],\n            y.dtype.isFloating ? y.dtype as DType<double, Float64Marker> : DType.float64,\n          );',
        '          NDArray<double, Marker>? spacingArray;\n          spacingArray = NDArray.fromList(\n            doubleValues,\n            [N],\n            y.dtype.isFloating ? y.dtype as DType<double, Marker> : DType.float64,\n          );',
    )
    # 3. gradient case real spacingArray type & cast
    content = content.replace(
        '          NDArray<double, Float64Marker>? spacingArray;\n          spacingArray = NDArray.fromList(\n            doubleValues,\n            [N],\n            f.dtype.isFloating ? f.dtype as DType<double, Float64Marker> : DType.float64,\n          );',
        '          NDArray<double, Marker>? spacingArray;\n          spacingArray = NDArray.fromList(\n            doubleValues,\n            [N],\n            f.dtype.isFloating ? f.dtype as DType<double, Marker> : DType.float64,\n          );',
    )
    return content

def fix_fft(content):
    print("Applying specific fixes to fft.dart...")
    content = content.replace(
        '      final transposedResult = fft(transposedInput, n: n);',
        '      final NDArray<R, MR> transposedResult = fft(transposedInput, n: n);'
    )
    content = content.replace(
        '      final transposedResult = ifft(transposedInput, n: n);',
        '      final NDArray<R, MR> transposedResult = ifft(transposedInput, n: n);'
    )
    return content

def refactor_file(filepath):
    print(f"Refactoring {filepath}...")
    with open(filepath, 'r') as f:
        content = f.read()

    # Pass 0: Remove explicit NDArray type annotations from local variables of CONCRETE types to let Dart infer the new 2-parameter types.
    # We only match concrete types that we want to infer (double, int, Complex, bool, num, Float32, Float64, Complex64, Complex128)
    concrete_types_pat = r'(?:double|int|Complex|bool|num|Float32|Float64|Complex64|Complex128)'
    # final NDArray<concrete_type> name = ... -> final name = ...
    content = re.sub(r'\bfinal\s+NDArray<' + concrete_types_pat + r'>\s+(\w+)\s*=', r'final \1 =', content)
    # NDArray<concrete_type> name = ... -> var name = ...
    content = re.sub(r'\bNDArray<' + concrete_types_pat + r'>\s+(\w+)\s*=', r'var \1 =', content)

    # Remove type arguments from NDArray factory calls: NDArray<T>.zeros(...) -> NDArray.zeros(...)
    # BUT ONLY if there are no commas inside the type arguments (i.e. keep 2-generic calls!).
    content = re.sub(
        r'\bNDArray<[^,>]+>\.(view|create|zeros|fromList|arange|ones|empty|eye|fromPointer)\(',
        r'NDArray.\1(',
        content
    )

    # 1. Refactor function definitions (column 0)
    func_pattern = re.compile(
        r'^(([^\s][\w\s({,})<>]*)\s+(\w+)<([^>]+)>\((.*?)\)\s*\{)',
        re.MULTILINE | re.DOTALL
    )

    new_content = ""
    last_idx = 0
    
    for match in func_pattern.finditer(content):
        start, end = match.span(1)
        new_content += content[last_idx:start]
        
        ret_type = match.group(2).strip()
        func_name = match.group(3)
        type_params_raw = match.group(4)
        params_raw = match.group(5)
        
        # Parse type parameters
        type_params = []
        for tp in type_params_raw.split(','):
            tp = tp.strip()
            if not tp: continue
            name = tp.split('extends')[0].strip()
            type_params.append((name, tp))

        # Generate marker parameters
        type_to_marker = {}
        for name, full_decl in type_params:
            type_to_marker[name] = f"M{name}"

        # Reconstruct type parameters list
        new_type_params_list = []
        for name, full_decl in type_params:
            new_type_params_list.append(full_decl)
            new_type_params_list.append(type_to_marker[name] + f" extends Marker")
            
        new_type_params_str = ", ".join(new_type_params_list)

        # Refactor return type and parameters signature
        new_ret_type = ret_type
        new_params = params_raw
        
        for name, marker in type_to_marker.items():
            new_ret_type = re.sub(
                r'\bNDArray<' + name + r'>(?!\s*,\s*\w+)',
                f'NDArray<{name}, {marker}>',
                new_ret_type
            )
            new_params = re.sub(
                r'\bNDArray<' + name + r'>(?!\s*,\s*\w+)',
                f'NDArray<{name}, {marker}>',
                new_params
            )

        # Handle concrete types
        concrete_map = {
            'double': 'Float64Marker',
            'int': 'Int64Marker',
            'Complex': 'Complex128Marker',
            'bool': 'BooleanMarker'
        }
        for c_type, c_marker in concrete_map.items():
            new_ret_type = re.sub(
                r'\bNDArray<' + c_type + r'>(?!\s*,\s*\w+)',
                f'NDArray<{c_type}, {c_marker}>',
                new_ret_type
            )
            new_params = re.sub(
                r'\bNDArray<' + c_type + r'>(?!\s*,\s*\w+)',
                f'NDArray<{c_type}, {c_marker}>',
                new_params
            )

        reconstructed = f"{new_ret_type} {func_name}<{new_type_params_str}>({new_params}) {{"
        new_content += reconstructed
        last_idx = end

    new_content += content[last_idx:]

    # 2. Second pass: global replacements
    concrete_map = {
        # Old extension types to new markers
        r'\bNDArray<Float32>': 'NDArray<double, Float32Marker>',
        r'\bNDArray<Float64>': 'NDArray<double, Float64Marker>',
        r'\bNDArray<Complex64>': 'NDArray<Complex, Complex64Marker>',
        r'\bNDArray<Complex128>': 'NDArray<Complex, Complex128Marker>',

        # Concrete types
        r'\bNDArray<double>': 'NDArray<double, Float64Marker>',
        r'\bNDArray<int>': 'NDArray<int, Int64Marker>',
        r'\bNDArray<Complex>': 'NDArray<Complex, Complex128Marker>',
        r'\bNDArray<bool>': 'NDArray<bool, BooleanMarker>',
        r'\bas DType<double>': 'as DType<double, Float64Marker>',
        r'\bas DType<Complex>': 'as DType<Complex, Complex128Marker>',
        r'\bDType<dynamic>': 'DType<dynamic, Marker>',
        r'\bNDArray<dynamic>': 'NDArray<dynamic, Marker>',
        r'\bNDArray<num>': 'NDArray<num, Marker>',
        r'\bNDArray<Object>': 'NDArray<Object, Marker>',
        
        # DType concrete replacements
        r'\bDType<double>': 'DType<double, Float64Marker>',
        r'\bDType<int>': 'DType<int, Int64Marker>',
        r'\bDType<Complex>': 'DType<Complex, Complex128Marker>',
        r'\bDType<bool>': 'DType<bool, BooleanMarker>',
        r'\bDType<num>': 'DType<num, Marker>',
        
        r'\bNDArray<R>': 'NDArray<R, MR>',
        r'\bDType<R>': 'DType<R, MR>',
        r'\bNDArray<T>': 'NDArray<T, MT>',
        r'\bDType<T>': 'DType<T, MT>',
        r'\bNDArray<Ta>': 'NDArray<Ta, MTa>',
        r'\bDType<Ta>': 'DType<Ta, MTa>',
        r'\bNDArray<Tb>': 'NDArray<Tb, MTb>',
        r'\bDType<Tb>': 'DType<Tb, MTb>',
        r'\bNDArray<Tr>': 'NDArray<Tr, MTr>',
        r'\bDType<Tr>': 'DType<Tr, MTr>',
        
        # NDEnumerate / LstsqResult specific replacements
        r'\bfinal class LstsqResult<T>': 'final class LstsqResult<T, MT extends Marker>',
        r'\bLstsqResult<T>': 'LstsqResult<T, MT>',
        r'\bLstsqResult<T>\(': 'LstsqResult(',
        r'\bfinal class NDEnumerate<T>': 'final class NDEnumerate<T, MT extends Marker>',
        r'\bNDEnumerate<T>': 'NDEnumerate<T, MT>',
        r'\bNDEnumerate\(': 'NDEnumerate(',

        # Primitive wrappers removal (replace Float64(99.0) -> 99.0)
        r'\bFloat64\(([^)]+)\)': r'\1',
        r'\bFloat32\(([^)]+)\)': r'\1',
        r'\bInt64\(([^)]+)\)': r'\1',
        r'\bInt32\(([^)]+)\)': r'\1',
        r'\bComplex128\(([^)]+)\)': r'Complex(\1)',
        r'\bComplex64\(([^)]+)\)': r'Complex(\1)',
        
        # .getCell().value removal
        r'\.getCell\((.*?)\)\.value\b': r'.getCell(\1)',
    }
    
    for pattern, replacement in concrete_map.items():
        new_content = re.sub(pattern, replacement, new_content)

    # 3. Third pass: Remove type arguments from calls to generic helper functions
    # and also ufuncs/reductions/spacers calls.
    helpers_to_clean = [
        "_vectorNorm", "_matrixNorm", "_prepareBinaryBitwise", "_prepareBinary", 
        "_prepareUnary", "unaryOp", "binaryOp", "flatUnaryOp", 
        "flatBinaryOp", "_resolveDType", "_prepareUnaryBitwise",
        "_prepareBinaryBitwiseInPlace",
        # ufuncs
        "multiply", "divide", "add", "subtract", "maximum", "minimum",
        # helpers
        "reduceRecursive", "nanReduceRecursive", "defaultDType", "toNDArray",
        # spacers
        "linspaceGrid", "_linspaceGridInternal", "linspace", "linspaceWithStep", "logspace", "geomspace",
        # sorting / reductions
        "sort", "count_nonzero", "_argminmaxFFI", "sum", "prod",
        # more helpers
        "elementWiseOp", "ternaryOp", "linspaceInternal",
        # recursive helpers in helpers.dart
        "whereOpRec", "countNonzeroRecursive", "argMinMaxRecursive",
        # calculus / fft helpers
        "gradient", "fft", "ifft",
        # det / ndenumerate
        "det", "NDEnumerate",
    ]
    
    for helper in helpers_to_clean:
        pattern = r'\b' + helper + r'<(?!.*?\bextends\b)[^>]+>\('
        new_content = re.sub(
            pattern,
            f'{helper}(',
            new_content
        )

    # 4. Post-processing for specific files
    if filepath.endswith('linalg.dart'):
        new_content = fix_linalg(new_content)
    elif filepath.endswith('math.dart'):
        new_content = fix_math(new_content)
    elif filepath.endswith('calculus.dart'):
        new_content = fix_calculus(new_content)
    elif filepath.endswith('fft.dart'):
        new_content = fix_fft(new_content)
    elif filepath.endswith('sorting_searching_test.dart'):
        print("Applying specific fixes to sorting_searching_test.dart...")
        new_content = new_content.replace(
            'as List<NDArray<int, Int64Marker>>',
            'as List<NDArray<int, Int32Marker>>'
        )

    with open(filepath, 'w') as f:
        f.write(new_content)
    print(f"Done refactoring {filepath}")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        refactor_file(sys.argv[1])
    else:
        refactor_file('/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/lib/src/operations/math.dart')

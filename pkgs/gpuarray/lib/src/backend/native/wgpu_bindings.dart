// ignore_for_file: constant_identifier_names, non_constant_identifier_names, camel_case_types
import 'dart:ffi' as ffi;
import 'dart:io' show Platform;
import 'package:ffi/ffi.dart';
import '../../exceptions.dart';

// =============================================================================
// WebGPU Standard Usage and Enumeration Constants
// =============================================================================

/// Buffer usage bitmask flags indicating allowed operations on GPU buffers.
abstract final class WGPUBufferUsage {
  static const int none = 0x00000000;
  static const int mapRead = 0x00000001;
  static const int mapWrite = 0x00000002;
  static const int copySrc = 0x00000004;
  static const int copyDst = 0x00000008;
  static const int index = 0x00000010;
  static const int vertex = 0x00000020;
  static const int uniform = 0x00000040;
  static const int storage = 0x00000080;
  static const int indirect = 0x00000100;
  static const int queryResolve = 0x00000200;
}

// Global alias constants matching C API macro names
const int WGPUBufferUsage_None = WGPUBufferUsage.none;
const int WGPUBufferUsage_MapRead = WGPUBufferUsage.mapRead;
const int WGPUBufferUsage_MapWrite = WGPUBufferUsage.mapWrite;
const int WGPUBufferUsage_CopySrc = WGPUBufferUsage.copySrc;
const int WGPUBufferUsage_CopyDst = WGPUBufferUsage.copyDst;
const int WGPUBufferUsage_Index = WGPUBufferUsage.index;
const int WGPUBufferUsage_Vertex = WGPUBufferUsage.vertex;
const int WGPUBufferUsage_Uniform = WGPUBufferUsage.uniform;
const int WGPUBufferUsage_Storage = WGPUBufferUsage.storage;
const int WGPUBufferUsage_Indirect = WGPUBufferUsage.indirect;
const int WGPUBufferUsage_QueryResolve = WGPUBufferUsage.queryResolve;

/// Map mode bitmask flags for buffer mapping.
abstract final class WGPUMapMode {
  static const int none = 0x00000000;
  static const int read = 0x00000001;
  static const int write = 0x00000002;
}

const int WGPUMapMode_None = WGPUMapMode.none;
const int WGPUMapMode_Read = WGPUMapMode.read;
const int WGPUMapMode_Write = WGPUMapMode.write;

/// WebGPU Structure Type identifiers for chained descriptor structs.
abstract final class WGPUSType {
  static const int invalid = 0x00000000;
  static const int surfaceDescriptorFromMetalLayer = 0x00000001;
  static const int surfaceDescriptorFromWindowsHWND = 0x00000002;
  static const int surfaceDescriptorFromXlibWindow = 0x00000003;
  static const int surfaceDescriptorFromCanvasHTMLSelector = 0x00000004;
  static const int shaderModuleWGSLDescriptor = 0x00000006;
  static const int primitiveDepthClipControl = 0x00000007;
}

const int WGPUSType_Invalid = WGPUSType.invalid;
const int WGPUSType_ShaderModuleWGSLDescriptor = WGPUSType.shaderModuleWGSLDescriptor;

/// Buffer map asynchronous operation status.
abstract final class WGPUBufferMapAsyncStatus {
  static const int success = 0x00000000;
  static const int validationError = 0x00000001;
  static const int unknown = 0x00000002;
  static const int deviceLost = 0x00000003;
  static const int destroyedBeforeCallback = 0x00000004;
  static const int unmappedBeforeCallback = 0x00000005;
  static const int offsetOutOfRange = 0x00000006;
  static const int sizeOutOfRange = 0x00000007;
}

const int WGPUBufferMapAsyncStatus_Success = WGPUBufferMapAsyncStatus.success;
const int WGPUBufferMapAsyncStatus_ValidationError = WGPUBufferMapAsyncStatus.validationError;
const int WGPUBufferMapAsyncStatus_Unknown = WGPUBufferMapAsyncStatus.unknown;
const int WGPUBufferMapAsyncStatus_DeviceLost = WGPUBufferMapAsyncStatus.deviceLost;
const int WGPUBufferMapAsyncStatus_DestroyedBeforeCallback = WGPUBufferMapAsyncStatus.destroyedBeforeCallback;
const int WGPUBufferMapAsyncStatus_UnmappedBeforeCallback = WGPUBufferMapAsyncStatus.unmappedBeforeCallback;
const int WGPUBufferMapAsyncStatus_OffsetOutOfRange = WGPUBufferMapAsyncStatus.offsetOutOfRange;
const int WGPUBufferMapAsyncStatus_SizeOutOfRange = WGPUBufferMapAsyncStatus.sizeOutOfRange;

/// Adapter request status.
abstract final class WGPURequestAdapterStatus {
  static const int success = 0x00000000;
  static const int unavailable = 0x00000001;
  static const int error = 0x00000002;
  static const int unknown = 0x00000003;
}

const int WGPURequestAdapterStatus_Success = WGPURequestAdapterStatus.success;

/// Device request status.
abstract final class WGPURequestDeviceStatus {
  static const int success = 0x00000000;
  static const int error = 0x00000001;
  static const int unknown = 0x00000002;
}

const int WGPURequestDeviceStatus_Success = WGPURequestDeviceStatus.success;

/// Power preference hint for GPU adapter selection.
abstract final class WGPUPowerPreference {
  static const int undefined = 0x00000000;
  static const int lowPower = 0x00000001;
  static const int highPerformance = 0x00000002;
}

const int WGPUPowerPreference_Undefined = WGPUPowerPreference.undefined;
const int WGPUPowerPreference_LowPower = WGPUPowerPreference.lowPower;
const int WGPUPowerPreference_HighPerformance = WGPUPowerPreference.highPerformance;

/// Backend graphics API type.
abstract final class WGPUBackendType {
  static const int undefined = 0x00000000;
  static const int nullBackend = 0x00000001;
  static const int webGpu = 0x00000002;
  static const int d3d11 = 0x00000003;
  static const int d3d12 = 0x00000004;
  static const int metal = 0x00000005;
  static const int vulkan = 0x00000006;
  static const int openGl = 0x00000007;
  static const int openGlEs = 0x00000008;
}

const int WGPUBackendType_Undefined = WGPUBackendType.undefined;
const int WGPUBackendType_Null = WGPUBackendType.nullBackend;
const int WGPUBackendType_WebGPU = WGPUBackendType.webGpu;
const int WGPUBackendType_D3D11 = WGPUBackendType.d3d11;
const int WGPUBackendType_D3D12 = WGPUBackendType.d3d12;
const int WGPUBackendType_Metal = WGPUBackendType.metal;
const int WGPUBackendType_Vulkan = WGPUBackendType.vulkan;
const int WGPUBackendType_OpenGL = WGPUBackendType.openGl;
const int WGPUBackendType_OpenGLES = WGPUBackendType.openGlEs;

// =============================================================================
// WebGPU C-FFI Native Struct Definitions
// =============================================================================

/// Base struct for extensible WebGPU descriptor chains.
final class WGPUChainedStruct extends ffi.Struct {
  external ffi.Pointer<WGPUChainedStruct> next;

  @ffi.Uint32()
  external int sType;
}

/// Descriptor for creating a `WGPUInstance`.
final class WGPUInstanceDescriptor extends ffi.Struct {
  external ffi.Pointer<WGPUChainedStruct> nextInChain;
}

/// Options for requesting a `WGPUAdapter`.
final class WGPURequestAdapterOptions extends ffi.Struct {
  external ffi.Pointer<WGPUChainedStruct> nextInChain;
  external ffi.Pointer<ffi.Void> compatibleSurface;

  @ffi.Uint32()
  external int powerPreference;

  @ffi.Uint32()
  external int backendType;

  @ffi.Uint32()
  external int forceFallbackAdapter;
}

/// Descriptor for requesting a `WGPUDevice`.
final class WGPUDeviceDescriptor extends ffi.Struct {
  external ffi.Pointer<WGPUChainedStruct> nextInChain;
  external ffi.Pointer<Utf8> label;

  @ffi.UintPtr()
  external int requiredFeatureCount;

  external ffi.Pointer<ffi.Uint32> requiredFeatures;
  external ffi.Pointer<ffi.Void> requiredLimits;
  external ffi.Pointer<WGPUChainedStruct> defaultQueueNextInChain;
  external ffi.Pointer<Utf8> defaultQueueLabel;
  external ffi.Pointer<ffi.Void> deviceLostCallback;
  external ffi.Pointer<ffi.Void> deviceLostUserdata;
}

/// Descriptor for creating a `WGPUBuffer`.
final class WGPUBufferDescriptor extends ffi.Struct {
  external ffi.Pointer<WGPUChainedStruct> nextInChain;
  external ffi.Pointer<Utf8> label;

  @ffi.Uint64()
  external int usage;

  @ffi.Uint64()
  external int size;

  @ffi.Uint32()
  external int mappedAtCreation;
}

/// WGSL shader source descriptor chained to `WGPUShaderModuleDescriptor`.
final class WGPUShaderModuleWGSLDescriptor extends ffi.Struct {
  external WGPUChainedStruct chain;
  external ffi.Pointer<Utf8> code;
}

/// Descriptor for creating a `WGPUShaderModule`.
final class WGPUShaderModuleDescriptor extends ffi.Struct {
  external ffi.Pointer<WGPUChainedStruct> nextInChain;
  external ffi.Pointer<Utf8> label;

  @ffi.UintPtr()
  external int hintCount;

  external ffi.Pointer<ffi.Void> hints;
}

/// Programmable stage descriptor for compute or graphics pipelines.
final class WGPUProgrammableStageDescriptor extends ffi.Struct {
  external ffi.Pointer<WGPUChainedStruct> nextInChain;
  external ffi.Pointer<ffi.Void> module;
  external ffi.Pointer<Utf8> entryPoint;

  @ffi.UintPtr()
  external int constantCount;

  external ffi.Pointer<ffi.Void> constants;
}

/// Descriptor for creating a `WGPUComputePipeline`.
final class WGPUComputePipelineDescriptor extends ffi.Struct {
  external ffi.Pointer<WGPUChainedStruct> nextInChain;
  external ffi.Pointer<Utf8> label;
  external ffi.Pointer<ffi.Void> layout;
  external WGPUProgrammableStageDescriptor compute;
}

/// Single binding entry within a `WGPUBindGroupDescriptor`.
final class WGPUBindGroupEntry extends ffi.Struct {
  external ffi.Pointer<WGPUChainedStruct> nextInChain;

  @ffi.Uint32()
  external int binding;

  external ffi.Pointer<ffi.Void> buffer;

  @ffi.Uint64()
  external int offset;

  @ffi.Uint64()
  external int size;

  external ffi.Pointer<ffi.Void> sampler;
  external ffi.Pointer<ffi.Void> textureView;
}

/// Descriptor for creating a `WGPUBindGroup`.
final class WGPUBindGroupDescriptor extends ffi.Struct {
  external ffi.Pointer<WGPUChainedStruct> nextInChain;
  external ffi.Pointer<Utf8> label;
  external ffi.Pointer<ffi.Void> layout;

  @ffi.UintPtr()
  external int entryCount;

  external ffi.Pointer<WGPUBindGroupEntry> entries;
}

/// Descriptor for creating a `WGPUCommandEncoder`.
final class WGPUCommandEncoderDescriptor extends ffi.Struct {
  external ffi.Pointer<WGPUChainedStruct> nextInChain;
  external ffi.Pointer<Utf8> label;
}

/// Descriptor for creating a `WGPUComputePassEncoder`.
final class WGPUComputePassDescriptor extends ffi.Struct {
  external ffi.Pointer<WGPUChainedStruct> nextInChain;
  external ffi.Pointer<Utf8> label;

  @ffi.UintPtr()
  external int timestampWritesCount;

  external ffi.Pointer<ffi.Void> timestampWrites;
}

/// Descriptor for creating a `WGPUCommandBuffer`.
final class WGPUCommandBufferDescriptor extends ffi.Struct {
  external ffi.Pointer<WGPUChainedStruct> nextInChain;
  external ffi.Pointer<Utf8> label;
}

// =============================================================================
// Internal C Callback Typedefs & Static State
// =============================================================================

typedef _wgpuRequestAdapterCallback_C = ffi.Void Function(
  ffi.Uint32 status,
  ffi.Pointer<ffi.Void> adapter,
  ffi.Pointer<Utf8> message,
  ffi.Pointer<ffi.Void> userdata,
);

typedef _wgpuRequestDeviceCallback_C = ffi.Void Function(
  ffi.Uint32 status,
  ffi.Pointer<ffi.Void> device,
  ffi.Pointer<Utf8> message,
  ffi.Pointer<ffi.Void> userdata,
);

typedef _wgpuBufferMapCallback_C = ffi.Void Function(
  ffi.Uint32 status,
  ffi.Pointer<ffi.Void> userdata,
);

abstract final class _WgpuCallbackState {
  static ffi.Pointer<ffi.Void> lastAdapter = ffi.nullptr;
  static int lastAdapterStatus = -1;
  static String? lastAdapterMessage;

  static ffi.Pointer<ffi.Void> lastDevice = ffi.nullptr;
  static int lastDeviceStatus = -1;
  static String? lastDeviceMessage;

  static int lastMapStatus = -1;
}

void _onAdapterReceived(
  int status,
  ffi.Pointer<ffi.Void> adapter,
  ffi.Pointer<Utf8> message,
  ffi.Pointer<ffi.Void> userdata,
) {
  _WgpuCallbackState.lastAdapterStatus = status;
  _WgpuCallbackState.lastAdapter = adapter;
  if (message != ffi.nullptr) {
    _WgpuCallbackState.lastAdapterMessage = message.toDartString();
  } else {
    _WgpuCallbackState.lastAdapterMessage = null;
  }
}

void _onDeviceReceived(
  int status,
  ffi.Pointer<ffi.Void> device,
  ffi.Pointer<Utf8> message,
  ffi.Pointer<ffi.Void> userdata,
) {
  _WgpuCallbackState.lastDeviceStatus = status;
  _WgpuCallbackState.lastDevice = device;
  if (message != ffi.nullptr) {
    _WgpuCallbackState.lastDeviceMessage = message.toDartString();
  } else {
    _WgpuCallbackState.lastDeviceMessage = null;
  }
}

void _onBufferMapped(
  int status,
  ffi.Pointer<ffi.Void> userdata,
) {
  _WgpuCallbackState.lastMapStatus = status;
}

// =============================================================================
// High-Level Data Helper for Bind Group Entries
// =============================================================================

/// Convenience representation of a buffer binding entry in a bind group.
final class WgpuBindGroupEntryData {
  final int binding;
  final ffi.Pointer<ffi.Void> buffer;
  final int offset;
  final int size;

  const WgpuBindGroupEntryData({
    required this.binding,
    required this.buffer,
    this.offset = 0,
    this.size = 0,
  });
}

// =============================================================================
// WebGPU Dynamic Native Library Loader & Driver Interface
// =============================================================================

/// Dynamic loader and FFI interface to native WebGPU libraries (`libwgpu_native`, Dawn).
final class WgpuNativeLib {
  final ffi.DynamicLibrary dylib;
  final String libraryPath;

  // C function pointers
  late final ffi.Pointer<ffi.Void> Function(ffi.Pointer<WGPUInstanceDescriptor>) _wgpuCreateInstance;
  late final void Function(
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<WGPURequestAdapterOptions>,
    ffi.Pointer<ffi.NativeFunction<_wgpuRequestAdapterCallback_C>>,
    ffi.Pointer<ffi.Void>,
  ) _wgpuInstanceRequestAdapter;
  late final void Function(
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<WGPUDeviceDescriptor>,
    ffi.Pointer<ffi.NativeFunction<_wgpuRequestDeviceCallback_C>>,
    ffi.Pointer<ffi.Void>,
  ) _wgpuAdapterRequestDevice;
  late final ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>) _wgpuDeviceGetQueue;
  late final ffi.Pointer<ffi.Void> Function(
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<WGPUBufferDescriptor>,
  ) _wgpuDeviceCreateBuffer;
  late final void Function(
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<ffi.Void>,
    int,
    ffi.Pointer<ffi.Void>,
    int,
  ) _wgpuQueueWriteBuffer;
  late final void Function(
    ffi.Pointer<ffi.Void>,
    int,
    int,
    int,
    ffi.Pointer<ffi.NativeFunction<_wgpuBufferMapCallback_C>>,
    ffi.Pointer<ffi.Void>,
  )? _wgpuBufferMapAsync;
  late final ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>, int, int)? _wgpuBufferGetMappedRange;
  late final void Function(ffi.Pointer<ffi.Void>)? _wgpuBufferUnmap;
  late final ffi.Pointer<ffi.Void> Function(
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<WGPUShaderModuleDescriptor>,
  ) _wgpuDeviceCreateShaderModule;
  late final ffi.Pointer<ffi.Void> Function(
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<WGPUComputePipelineDescriptor>,
  ) _wgpuDeviceCreateComputePipeline;
  late final ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>, int)? _wgpuComputePipelineGetBindGroupLayout;
  late final ffi.Pointer<ffi.Void> Function(
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<WGPUBindGroupDescriptor>,
  ) _wgpuDeviceCreateBindGroup;
  late final ffi.Pointer<ffi.Void> Function(
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<WGPUCommandEncoderDescriptor>,
  ) _wgpuDeviceCreateCommandEncoder;
  late final ffi.Pointer<ffi.Void> Function(
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<WGPUComputePassDescriptor>,
  ) _wgpuCommandEncoderBeginComputePass;
  late final void Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Void>) _wgpuComputePassEncoderSetPipeline;
  late final void Function(
    ffi.Pointer<ffi.Void>,
    int,
    ffi.Pointer<ffi.Void>,
    int,
    ffi.Pointer<ffi.Uint32>,
  ) _wgpuComputePassEncoderSetBindGroup;
  late final void Function(ffi.Pointer<ffi.Void>, int, int, int) _wgpuComputePassEncoderDispatchWorkgroups;
  late final void Function(ffi.Pointer<ffi.Void>) _wgpuComputePassEncoderEnd;
  late final void Function(
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<ffi.Void>,
    int,
    ffi.Pointer<ffi.Void>,
    int,
    int,
  )? _wgpuCommandEncoderCopyBufferToBuffer;
  late final ffi.Pointer<ffi.Void> Function(
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<WGPUCommandBufferDescriptor>,
  ) _wgpuCommandEncoderFinish;
  late final void Function(ffi.Pointer<ffi.Void>, int, ffi.Pointer<ffi.Pointer<ffi.Void>>) _wgpuQueueSubmit;
  late final int Function(ffi.Pointer<ffi.Void>, int, ffi.Pointer<ffi.Void>)? _wgpuDevicePoll;
  late final void Function(ffi.Pointer<ffi.Void>) _wgpuBufferDestroy;
  late final void Function(ffi.Pointer<ffi.Void>)? _wgpuDeviceDestroy;
  late final void Function(ffi.Pointer<ffi.Void>)? _wgpuInstanceRelease;
  late final void Function(ffi.Pointer<ffi.Void>)? _wgpuAdapterRelease;
  late final void Function(ffi.Pointer<ffi.Void>)? _wgpuDeviceRelease;
  late final void Function(ffi.Pointer<ffi.Void>)? _wgpuQueueRelease;
  late final void Function(ffi.Pointer<ffi.Void>)? _wgpuBufferRelease;
  late final void Function(ffi.Pointer<ffi.Void>)? _wgpuShaderModuleRelease;
  late final void Function(ffi.Pointer<ffi.Void>)? _wgpuComputePipelineRelease;
  late final void Function(ffi.Pointer<ffi.Void>)? _wgpuBindGroupRelease;
  late final void Function(ffi.Pointer<ffi.Void>)? _wgpuBindGroupLayoutRelease;
  late final void Function(ffi.Pointer<ffi.Void>)? _wgpuCommandEncoderRelease;
  late final void Function(ffi.Pointer<ffi.Void>)? _wgpuComputePassEncoderRelease;
  late final void Function(ffi.Pointer<ffi.Void>)? _wgpuCommandBufferRelease;

  WgpuNativeLib(this.dylib, {this.libraryPath = "dynamic"}) {
    _lookupSymbols();
  }

  static T? _tryLookup<T>(T Function() lookupFn) {
    try {
      return lookupFn();
    } catch (_) {
      return null;
    }
  }

  void _lookupSymbols() {
    _wgpuCreateInstance = dylib
        .lookup<ffi.NativeFunction<ffi.Pointer<ffi.Void> Function(ffi.Pointer<WGPUInstanceDescriptor>)>>(
          "wgpuCreateInstance",
        )
        .asFunction();

    _wgpuInstanceRequestAdapter = dylib
        .lookup<
          ffi.NativeFunction<
            ffi.Void Function(
              ffi.Pointer<ffi.Void>,
              ffi.Pointer<WGPURequestAdapterOptions>,
              ffi.Pointer<ffi.NativeFunction<_wgpuRequestAdapterCallback_C>>,
              ffi.Pointer<ffi.Void>,
            )
          >
        >("wgpuInstanceRequestAdapter")
        .asFunction();

    _wgpuAdapterRequestDevice = dylib
        .lookup<
          ffi.NativeFunction<
            ffi.Void Function(
              ffi.Pointer<ffi.Void>,
              ffi.Pointer<WGPUDeviceDescriptor>,
              ffi.Pointer<ffi.NativeFunction<_wgpuRequestDeviceCallback_C>>,
              ffi.Pointer<ffi.Void>,
            )
          >
        >("wgpuAdapterRequestDevice")
        .asFunction();

    _wgpuDeviceGetQueue = dylib
        .lookup<ffi.NativeFunction<ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>)>>("wgpuDeviceGetQueue")
        .asFunction();

    _wgpuDeviceCreateBuffer = dylib
        .lookup<
          ffi.NativeFunction<
            ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>, ffi.Pointer<WGPUBufferDescriptor>)
          >
        >("wgpuDeviceCreateBuffer")
        .asFunction();

    _wgpuQueueWriteBuffer = dylib
        .lookup<
          ffi.NativeFunction<
            ffi.Void Function(
              ffi.Pointer<ffi.Void>,
              ffi.Pointer<ffi.Void>,
              ffi.Uint64,
              ffi.Pointer<ffi.Void>,
              ffi.UintPtr,
            )
          >
        >("wgpuQueueWriteBuffer")
        .asFunction();

    _wgpuBufferMapAsync = _tryLookup(
      () => dylib
          .lookup<
            ffi.NativeFunction<
              ffi.Void Function(
                ffi.Pointer<ffi.Void>,
                ffi.Uint32,
                ffi.UintPtr,
                ffi.UintPtr,
                ffi.Pointer<ffi.NativeFunction<_wgpuBufferMapCallback_C>>,
                ffi.Pointer<ffi.Void>,
              )
            >
          >("wgpuBufferMapAsync")
          .asFunction(),
    );

    _wgpuBufferGetMappedRange = _tryLookup(
      () => dylib
          .lookup<
            ffi.NativeFunction<ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>, ffi.UintPtr, ffi.UintPtr)>
          >("wgpuBufferGetMappedRange")
          .asFunction(),
    );

    _wgpuBufferUnmap = _tryLookup(
      () => dylib
          .lookup<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>("wgpuBufferUnmap")
          .asFunction(),
    );

    _wgpuDeviceCreateShaderModule = dylib
        .lookup<
          ffi.NativeFunction<
            ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>, ffi.Pointer<WGPUShaderModuleDescriptor>)
          >
        >("wgpuDeviceCreateShaderModule")
        .asFunction();

    _wgpuDeviceCreateComputePipeline = dylib
        .lookup<
          ffi.NativeFunction<
            ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>, ffi.Pointer<WGPUComputePipelineDescriptor>)
          >
        >("wgpuDeviceCreateComputePipeline")
        .asFunction();

    _wgpuComputePipelineGetBindGroupLayout = _tryLookup(
      () => dylib
          .lookup<ffi.NativeFunction<ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>, ffi.Uint32)>>(
            "wgpuComputePipelineGetBindGroupLayout",
          )
          .asFunction(),
    );

    _wgpuDeviceCreateBindGroup = dylib
        .lookup<
          ffi.NativeFunction<
            ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>, ffi.Pointer<WGPUBindGroupDescriptor>)
          >
        >("wgpuDeviceCreateBindGroup")
        .asFunction();

    _wgpuDeviceCreateCommandEncoder = dylib
        .lookup<
          ffi.NativeFunction<
            ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>, ffi.Pointer<WGPUCommandEncoderDescriptor>)
          >
        >("wgpuDeviceCreateCommandEncoder")
        .asFunction();

    _wgpuCommandEncoderBeginComputePass = dylib
        .lookup<
          ffi.NativeFunction<
            ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>, ffi.Pointer<WGPUComputePassDescriptor>)
          >
        >("wgpuCommandEncoderBeginComputePass")
        .asFunction();

    _wgpuComputePassEncoderSetPipeline = dylib
        .lookup<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Void>)>>(
          "wgpuComputePassEncoderSetPipeline",
        )
        .asFunction();

    _wgpuComputePassEncoderSetBindGroup = dylib
        .lookup<
          ffi.NativeFunction<
            ffi.Void Function(
              ffi.Pointer<ffi.Void>,
              ffi.Uint32,
              ffi.Pointer<ffi.Void>,
              ffi.UintPtr,
              ffi.Pointer<ffi.Uint32>,
            )
          >
        >("wgpuComputePassEncoderSetBindGroup")
        .asFunction();

    _wgpuComputePassEncoderDispatchWorkgroups = dylib
        .lookup<
          ffi.NativeFunction<
            ffi.Void Function(ffi.Pointer<ffi.Void>, ffi.Uint32, ffi.Uint32, ffi.Uint32)
          >
        >("wgpuComputePassEncoderDispatchWorkgroups")
        .asFunction();

    _wgpuComputePassEncoderEnd = dylib
        .lookup<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>(
          "wgpuComputePassEncoderEnd",
        )
        .asFunction();

    _wgpuCommandEncoderCopyBufferToBuffer = _tryLookup(
      () => dylib
          .lookup<
            ffi.NativeFunction<
              ffi.Void Function(
                ffi.Pointer<ffi.Void>,
                ffi.Pointer<ffi.Void>,
                ffi.Uint64,
                ffi.Pointer<ffi.Void>,
                ffi.Uint64,
                ffi.Uint64,
              )
            >
          >("wgpuCommandEncoderCopyBufferToBuffer")
          .asFunction(),
    );

    _wgpuCommandEncoderFinish = dylib
        .lookup<
          ffi.NativeFunction<
            ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>, ffi.Pointer<WGPUCommandBufferDescriptor>)
          >
        >("wgpuCommandEncoderFinish")
        .asFunction();

    _wgpuQueueSubmit = dylib
        .lookup<
          ffi.NativeFunction<
            ffi.Void Function(ffi.Pointer<ffi.Void>, ffi.UintPtr, ffi.Pointer<ffi.Pointer<ffi.Void>>)
          >
        >("wgpuQueueSubmit")
        .asFunction();

    _wgpuDevicePoll = _tryLookup(
      () => dylib
          .lookup<ffi.NativeFunction<ffi.Uint32 Function(ffi.Pointer<ffi.Void>, ffi.Uint32, ffi.Pointer<ffi.Void>)>>(
            "wgpuDevicePoll",
          )
          .asFunction(),
    );

    _wgpuBufferDestroy = dylib
        .lookup<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>("wgpuBufferDestroy")
        .asFunction();

    _wgpuDeviceDestroy = _tryLookup(
      () => dylib
          .lookup<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>("wgpuDeviceDestroy")
          .asFunction(),
    );

    _wgpuInstanceRelease = _tryLookup(
      () => dylib
          .lookup<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>("wgpuInstanceRelease")
          .asFunction(),
    );

    _wgpuAdapterRelease = _tryLookup(
      () => dylib
          .lookup<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>("wgpuAdapterRelease")
          .asFunction(),
    );

    _wgpuDeviceRelease = _tryLookup(
      () => dylib
          .lookup<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>("wgpuDeviceRelease")
          .asFunction(),
    );

    _wgpuQueueRelease = _tryLookup(
      () => dylib
          .lookup<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>("wgpuQueueRelease")
          .asFunction(),
    );

    _wgpuBufferRelease = _tryLookup(
      () => dylib
          .lookup<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>("wgpuBufferRelease")
          .asFunction(),
    );

    _wgpuShaderModuleRelease = _tryLookup(
      () => dylib
          .lookup<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>("wgpuShaderModuleRelease")
          .asFunction(),
    );

    _wgpuComputePipelineRelease = _tryLookup(
      () => dylib
          .lookup<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>("wgpuComputePipelineRelease")
          .asFunction(),
    );

    _wgpuBindGroupRelease = _tryLookup(
      () => dylib
          .lookup<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>("wgpuBindGroupRelease")
          .asFunction(),
    );

    _wgpuBindGroupLayoutRelease = _tryLookup(
      () => dylib
          .lookup<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>("wgpuBindGroupLayoutRelease")
          .asFunction(),
    );

    _wgpuCommandEncoderRelease = _tryLookup(
      () => dylib
          .lookup<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>("wgpuCommandEncoderRelease")
          .asFunction(),
    );

    _wgpuComputePassEncoderRelease = _tryLookup(
      () => dylib
          .lookup<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>("wgpuComputePassEncoderRelease")
          .asFunction(),
    );

    _wgpuCommandBufferRelease = _tryLookup(
      () => dylib
          .lookup<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>("wgpuCommandBufferRelease")
          .asFunction(),
    );
  }

  /// Whether the library is actively loaded.
  bool get isAvailable => true;

  // ===========================================================================
  // High-Level FFI Invocation Methods
  // ===========================================================================

  /// Creates a standard `WGPUInstance`.
  ffi.Pointer<ffi.Void> createInstance() {
    return using((arena) {
      final desc = arena<WGPUInstanceDescriptor>();
      desc.ref.nextInChain = ffi.nullptr;
      return _wgpuCreateInstance(desc);
    });
  }

  /// Requests a `WGPUAdapter` with the specified power preference and backend.
  Future<ffi.Pointer<ffi.Void>> requestAdapter(
    ffi.Pointer<ffi.Void> instance, {
    int powerPreference = WGPUPowerPreference.highPerformance,
    int backendType = WGPUBackendType.undefined,
  }) async {
    _WgpuCallbackState.lastAdapter = ffi.nullptr;
    _WgpuCallbackState.lastAdapterStatus = -1;
    _WgpuCallbackState.lastAdapterMessage = null;

    using((arena) {
      final options = arena<WGPURequestAdapterOptions>();
      options.ref.nextInChain = ffi.nullptr;
      options.ref.compatibleSurface = ffi.nullptr;
      options.ref.powerPreference = powerPreference;
      options.ref.backendType = backendType;
      options.ref.forceFallbackAdapter = 0;

      final cb = ffi.Pointer.fromFunction<_wgpuRequestAdapterCallback_C>(_onAdapterReceived);
      _wgpuInstanceRequestAdapter(instance, options, cb, ffi.nullptr);
    });

    if (_WgpuCallbackState.lastAdapterStatus != WGPURequestAdapterStatus.success ||
        _WgpuCallbackState.lastAdapter == ffi.nullptr) {
      throw GpuDeviceException(
        "Failed to obtain WebGPU Adapter (status: ${_WgpuCallbackState.lastAdapterStatus}, message: ${_WgpuCallbackState.lastAdapterMessage}).",
      );
    }
    return _WgpuCallbackState.lastAdapter;
  }

  /// Requests a `WGPUDevice` from a `WGPUAdapter`.
  Future<ffi.Pointer<ffi.Void>> requestDevice(
    ffi.Pointer<ffi.Void> adapter, {
    String? label,
  }) async {
    _WgpuCallbackState.lastDevice = ffi.nullptr;
    _WgpuCallbackState.lastDeviceStatus = -1;
    _WgpuCallbackState.lastDeviceMessage = null;

    using((arena) {
      final desc = arena<WGPUDeviceDescriptor>();
      desc.ref.nextInChain = ffi.nullptr;
      desc.ref.label = label != null ? label.toNativeUtf8(allocator: arena) : ffi.nullptr;
      desc.ref.requiredFeatureCount = 0;
      desc.ref.requiredFeatures = ffi.nullptr;
      desc.ref.requiredLimits = ffi.nullptr;
      desc.ref.defaultQueueNextInChain = ffi.nullptr;
      desc.ref.defaultQueueLabel = ffi.nullptr;
      desc.ref.deviceLostCallback = ffi.nullptr;
      desc.ref.deviceLostUserdata = ffi.nullptr;

      final cb = ffi.Pointer.fromFunction<_wgpuRequestDeviceCallback_C>(_onDeviceReceived);
      _wgpuAdapterRequestDevice(adapter, desc, cb, ffi.nullptr);
    });

    if (_WgpuCallbackState.lastDeviceStatus != WGPURequestDeviceStatus.success ||
        _WgpuCallbackState.lastDevice == ffi.nullptr) {
      throw GpuDeviceException(
        "Failed to obtain WebGPU Device (status: ${_WgpuCallbackState.lastDeviceStatus}, message: ${_WgpuCallbackState.lastDeviceMessage}).",
      );
    }
    return _WgpuCallbackState.lastDevice;
  }

  /// Retrieves the default `WGPUQueue` for a device.
  ffi.Pointer<ffi.Void> deviceGetQueue(ffi.Pointer<ffi.Void> device) {
    final queue = _wgpuDeviceGetQueue(device);
    if (queue == ffi.nullptr) {
      throw GpuDeviceException("Failed to retrieve device queue.");
    }
    return queue;
  }

  /// Creates a `WGPUBuffer` with the requested [size] and [usage] bitmask.
  ffi.Pointer<ffi.Void> createBuffer(
    ffi.Pointer<ffi.Void> device, {
    required int size,
    required int usage,
    bool mappedAtCreation = false,
    String? label,
  }) {
    return using((arena) {
      final desc = arena<WGPUBufferDescriptor>();
      desc.ref.nextInChain = ffi.nullptr;
      desc.ref.label = label != null ? label.toNativeUtf8(allocator: arena) : ffi.nullptr;
      desc.ref.usage = usage;
      desc.ref.size = size;
      desc.ref.mappedAtCreation = mappedAtCreation ? 1 : 0;

      final buf = _wgpuDeviceCreateBuffer(device, desc);
      if (buf == ffi.nullptr) {
        throw GpuMemoryException("Failed to allocate GPU buffer of size $size bytes.");
      }
      return buf;
    });
  }

  /// Writes host data directly into a GPU buffer via device queue.
  void queueWriteBuffer(
    ffi.Pointer<ffi.Void> queue,
    ffi.Pointer<ffi.Void> buffer, {
    int bufferOffset = 0,
    required ffi.Pointer<ffi.Void> data,
    required int size,
  }) {
    if (size == 0) return;
    _wgpuQueueWriteBuffer(queue, buffer, bufferOffset, data, size);
  }

  /// Asynchronously maps a GPU buffer for host CPU reading or writing.
  Future<int> bufferMapAsync(
    ffi.Pointer<ffi.Void> buffer, {
    int mode = WGPUMapMode.read,
    int offset = 0,
    required int size,
  }) async {
    if (_wgpuBufferMapAsync == null) return WGPUBufferMapAsyncStatus.unknown;
    _WgpuCallbackState.lastMapStatus = -1;

    final cb = ffi.Pointer.fromFunction<_wgpuBufferMapCallback_C>(_onBufferMapped);
    _wgpuBufferMapAsync(buffer, mode, offset, size, cb, ffi.nullptr);
    return _WgpuCallbackState.lastMapStatus;
  }

  /// Gets a pointer to the mapped memory range of a buffer.
  ffi.Pointer<ffi.Void> bufferGetMappedRange(
    ffi.Pointer<ffi.Void> buffer, {
    int offset = 0,
    required int size,
  }) {
    if (_wgpuBufferGetMappedRange == null) return ffi.nullptr;
    return _wgpuBufferGetMappedRange(buffer, offset, size);
  }

  /// Unmaps a previously mapped buffer.
  void bufferUnmap(ffi.Pointer<ffi.Void> buffer) {
    _wgpuBufferUnmap?.call(buffer);
  }

  /// Compiles a WGSL shader source into a `WGPUShaderModule`.
  ffi.Pointer<ffi.Void> createShaderModule(
    ffi.Pointer<ffi.Void> device,
    String wgslSource, {
    String? label,
  }) {
    return using((arena) {
      final wgslDesc = arena<WGPUShaderModuleWGSLDescriptor>();
      wgslDesc.ref.chain.next = ffi.nullptr;
      wgslDesc.ref.chain.sType = WGPUSType.shaderModuleWGSLDescriptor;
      wgslDesc.ref.code = wgslSource.toNativeUtf8(allocator: arena);

      final desc = arena<WGPUShaderModuleDescriptor>();
      desc.ref.nextInChain = wgslDesc.cast<WGPUChainedStruct>();
      desc.ref.label = label != null ? label.toNativeUtf8(allocator: arena) : ffi.nullptr;
      desc.ref.hintCount = 0;
      desc.ref.hints = ffi.nullptr;

      final module = _wgpuDeviceCreateShaderModule(device, desc);
      if (module == ffi.nullptr) {
        throw GpuComputeException("Failed to compile WGSL shader module.");
      }
      return module;
    });
  }

  /// Creates a `WGPUComputePipeline` from a compiled shader module.
  ffi.Pointer<ffi.Void> createComputePipeline(
    ffi.Pointer<ffi.Void> device, {
    required ffi.Pointer<ffi.Void> shaderModule,
    String entryPoint = "main",
    ffi.Pointer<ffi.Void>? layout,
    String? label,
  }) {
    return using((arena) {
      final desc = arena<WGPUComputePipelineDescriptor>();
      desc.ref.nextInChain = ffi.nullptr;
      desc.ref.label = label != null ? label.toNativeUtf8(allocator: arena) : ffi.nullptr;
      desc.ref.layout = layout ?? ffi.nullptr;
      desc.ref.compute.nextInChain = ffi.nullptr;
      desc.ref.compute.module = shaderModule;
      desc.ref.compute.entryPoint = entryPoint.toNativeUtf8(allocator: arena);
      desc.ref.compute.constantCount = 0;
      desc.ref.compute.constants = ffi.nullptr;

      final pipeline = _wgpuDeviceCreateComputePipeline(device, desc);
      if (pipeline == ffi.nullptr) {
        throw GpuComputeException("Failed to create compute pipeline.");
      }
      return pipeline;
    });
  }

  /// Retrieves the bind group layout at index [groupIndex] for a pipeline.
  ffi.Pointer<ffi.Void> pipelineGetBindGroupLayout(
    ffi.Pointer<ffi.Void> pipeline,
    int groupIndex,
  ) {
    if (_wgpuComputePipelineGetBindGroupLayout == null) return ffi.nullptr;
    return _wgpuComputePipelineGetBindGroupLayout(pipeline, groupIndex);
  }

  /// Creates a `WGPUBindGroup` containing buffer resources.
  ffi.Pointer<ffi.Void> createBindGroup(
    ffi.Pointer<ffi.Void> device, {
    required ffi.Pointer<ffi.Void> layout,
    required List<WgpuBindGroupEntryData> entries,
    String? label,
  }) {
    return using((arena) {
      final entriesPtr = arena<WGPUBindGroupEntry>(entries.length);
      for (var i = 0; i < entries.length; i++) {
        final e = entries[i];
        final entryPtr = entriesPtr + i;
        entryPtr.ref.nextInChain = ffi.nullptr;
        entryPtr.ref.binding = e.binding;
        entryPtr.ref.buffer = e.buffer;
        entryPtr.ref.offset = e.offset;
        entryPtr.ref.size = e.size;
        entryPtr.ref.sampler = ffi.nullptr;
        entryPtr.ref.textureView = ffi.nullptr;
      }

      final desc = arena<WGPUBindGroupDescriptor>();
      desc.ref.nextInChain = ffi.nullptr;
      desc.ref.label = label != null ? label.toNativeUtf8(allocator: arena) : ffi.nullptr;
      desc.ref.layout = layout;
      desc.ref.entryCount = entries.length;
      desc.ref.entries = entriesPtr;

      final bg = _wgpuDeviceCreateBindGroup(device, desc);
      if (bg == ffi.nullptr) {
        throw GpuComputeException("Failed to create bind group.");
      }
      return bg;
    });
  }

  /// Creates a `WGPUCommandEncoder`.
  ffi.Pointer<ffi.Void> createCommandEncoder(
    ffi.Pointer<ffi.Void> device, {
    String? label,
  }) {
    return using((arena) {
      final desc = arena<WGPUCommandEncoderDescriptor>();
      desc.ref.nextInChain = ffi.nullptr;
      desc.ref.label = label != null ? label.toNativeUtf8(allocator: arena) : ffi.nullptr;

      final encoder = _wgpuDeviceCreateCommandEncoder(device, desc);
      if (encoder == ffi.nullptr) {
        throw GpuComputeException("Failed to create command encoder.");
      }
      return encoder;
    });
  }

  /// Begins a compute pass on a command encoder.
  ffi.Pointer<ffi.Void> commandEncoderBeginComputePass(
    ffi.Pointer<ffi.Void> encoder, {
    String? label,
  }) {
    return using((arena) {
      final desc = arena<WGPUComputePassDescriptor>();
      desc.ref.nextInChain = ffi.nullptr;
      desc.ref.label = label != null ? label.toNativeUtf8(allocator: arena) : ffi.nullptr;
      desc.ref.timestampWritesCount = 0;
      desc.ref.timestampWrites = ffi.nullptr;

      final pass = _wgpuCommandEncoderBeginComputePass(encoder, desc);
      if (pass == ffi.nullptr) {
        throw GpuComputeException("Failed to begin compute pass.");
      }
      return pass;
    });
  }

  /// Sets the active compute pipeline on a compute pass.
  void computePassSetPipeline(
    ffi.Pointer<ffi.Void> pass,
    ffi.Pointer<ffi.Void> pipeline,
  ) {
    _wgpuComputePassEncoderSetPipeline(pass, pipeline);
  }

  /// Sets the active bind group on a compute pass.
  void computePassSetBindGroup(
    ffi.Pointer<ffi.Void> pass,
    int groupIndex,
    ffi.Pointer<ffi.Void> bindGroup,
  ) {
    _wgpuComputePassEncoderSetBindGroup(pass, groupIndex, bindGroup, 0, ffi.nullptr);
  }

  /// Dispatches compute workgroups.
  void computePassDispatchWorkgroups(
    ffi.Pointer<ffi.Void> pass,
    int workgroupsX,
    int workgroupsY,
    int workgroupsZ,
  ) {
    _wgpuComputePassEncoderDispatchWorkgroups(pass, workgroupsX, workgroupsY, workgroupsZ);
  }

  /// Ends the compute pass.
  void computePassEnd(ffi.Pointer<ffi.Void> pass) {
    _wgpuComputePassEncoderEnd(pass);
  }

  /// Copies data between two GPU buffers within a command encoder.
  void commandEncoderCopyBufferToBuffer(
    ffi.Pointer<ffi.Void> encoder,
    ffi.Pointer<ffi.Void> source,
    int sourceOffset,
    ffi.Pointer<ffi.Void> destination,
    int destinationOffset,
    int size,
  ) {
    _wgpuCommandEncoderCopyBufferToBuffer?.call(
      encoder,
      source,
      sourceOffset,
      destination,
      destinationOffset,
      size,
    );
  }

  /// Finishes recording commands and returns a `WGPUCommandBuffer`.
  ffi.Pointer<ffi.Void> commandEncoderFinish(
    ffi.Pointer<ffi.Void> encoder, {
    String? label,
  }) {
    return using((arena) {
      final desc = arena<WGPUCommandBufferDescriptor>();
      desc.ref.nextInChain = ffi.nullptr;
      desc.ref.label = label != null ? label.toNativeUtf8(allocator: arena) : ffi.nullptr;

      final cmdBuf = _wgpuCommandEncoderFinish(encoder, desc);
      if (cmdBuf == ffi.nullptr) {
        throw GpuComputeException("Failed to finish command encoder.");
      }
      return cmdBuf;
    });
  }

  /// Submits command buffers to the GPU queue for execution.
  void queueSubmit(
    ffi.Pointer<ffi.Void> queue,
    List<ffi.Pointer<ffi.Void>> commandBuffers,
  ) {
    if (commandBuffers.isEmpty) return;
    using((arena) {
      final array = arena<ffi.Pointer<ffi.Void>>(commandBuffers.length);
      for (var i = 0; i < commandBuffers.length; i++) {
        array[i] = commandBuffers[i];
      }
      _wgpuQueueSubmit(queue, commandBuffers.length, array);
    });
  }

  /// Polls the device queue and event loop for completion.
  bool devicePoll(ffi.Pointer<ffi.Void> device, {bool wait = false}) {
    if (_wgpuDevicePoll == null) return true;
    return _wgpuDevicePoll(device, wait ? 1 : 0, ffi.nullptr) != 0;
  }

  /// Destroys a GPU buffer.
  void bufferDestroy(ffi.Pointer<ffi.Void> buffer) {
    if (buffer != ffi.nullptr) {
      _wgpuBufferDestroy(buffer);
    }
  }

  /// Destroys a WebGPU device.
  void deviceDestroy(ffi.Pointer<ffi.Void> device) {
    if (device != ffi.nullptr) {
      _wgpuDeviceDestroy?.call(device);
    }
  }

  /// Releases instance reference.
  void instanceRelease(ffi.Pointer<ffi.Void> instance) {
    if (instance != ffi.nullptr) {
      _wgpuInstanceRelease?.call(instance);
    }
  }

  /// Releases adapter reference.
  void adapterRelease(ffi.Pointer<ffi.Void> adapter) {
    if (adapter != ffi.nullptr) {
      _wgpuAdapterRelease?.call(adapter);
    }
  }

  /// Releases device reference.
  void deviceRelease(ffi.Pointer<ffi.Void> device) {
    if (device != ffi.nullptr) {
      _wgpuDeviceRelease?.call(device);
    }
  }

  /// Releases queue reference.
  void queueRelease(ffi.Pointer<ffi.Void> queue) {
    if (queue != ffi.nullptr) {
      _wgpuQueueRelease?.call(queue);
    }
  }

  /// Releases buffer reference.
  void bufferRelease(ffi.Pointer<ffi.Void> buffer) {
    if (buffer != ffi.nullptr) {
      _wgpuBufferRelease?.call(buffer);
    }
  }

  /// Releases shader module reference.
  void shaderModuleRelease(ffi.Pointer<ffi.Void> shaderModule) {
    if (shaderModule != ffi.nullptr) {
      _wgpuShaderModuleRelease?.call(shaderModule);
    }
  }

  /// Releases compute pipeline reference.
  void computePipelineRelease(ffi.Pointer<ffi.Void> computePipeline) {
    if (computePipeline != ffi.nullptr) {
      _wgpuComputePipelineRelease?.call(computePipeline);
    }
  }

  /// Releases bind group reference.
  void bindGroupRelease(ffi.Pointer<ffi.Void> bindGroup) {
    if (bindGroup != ffi.nullptr) {
      _wgpuBindGroupRelease?.call(bindGroup);
    }
  }

  /// Releases bind group layout reference.
  void bindGroupLayoutRelease(ffi.Pointer<ffi.Void> bindGroupLayout) {
    if (bindGroupLayout != ffi.nullptr) {
      _wgpuBindGroupLayoutRelease?.call(bindGroupLayout);
    }
  }

  /// Releases command encoder reference.
  void commandEncoderRelease(ffi.Pointer<ffi.Void> commandEncoder) {
    if (commandEncoder != ffi.nullptr) {
      _wgpuCommandEncoderRelease?.call(commandEncoder);
    }
  }

  /// Releases compute pass encoder reference.
  void computePassEncoderRelease(ffi.Pointer<ffi.Void> computePassEncoder) {
    if (computePassEncoder != ffi.nullptr) {
      _wgpuComputePassEncoderRelease?.call(computePassEncoder);
    }
  }

  /// Releases command buffer reference.
  void commandBufferRelease(ffi.Pointer<ffi.Void> commandBuffer) {
    if (commandBuffer != ffi.nullptr) {
      _wgpuCommandBufferRelease?.call(commandBuffer);
    }
  }

  // ===========================================================================
  // Shared Dynamic Library Resolver
  // ===========================================================================

  /// Attempts to load the native WebGPU dynamic shared library across supported platforms.
  static WgpuNativeLib? tryLoad({String? customPath}) {
    if (customPath != null && customPath.isNotEmpty) {
      try {
        final dylib = ffi.DynamicLibrary.open(customPath);
        return WgpuNativeLib(dylib, libraryPath: customPath);
      } catch (_) {
        return null;
      }
    }

    final candidatePaths = <String>[];

    final envPath = Platform.environment["WGPU_LIB_PATH"];
    if (envPath != null && envPath.isNotEmpty) {
      candidatePaths.add(envPath);
    }

    if (Platform.isLinux) {
      candidatePaths.addAll([
        "libwgpu_native.so",
        "libwgpu.so",
        "/usr/lib/libwgpu_native.so",
        "/usr/local/lib/libwgpu_native.so",
        "/usr/lib/x86_64-linux-gnu/libwgpu_native.so",
        "/usr/lib64/libwgpu_native.so",
      ]);
    } else if (Platform.isMacOS) {
      candidatePaths.addAll([
        "libwgpu_native.dylib",
        "libwgpu.dylib",
        "/usr/local/lib/libwgpu_native.dylib",
        "/opt/homebrew/lib/libwgpu_native.dylib",
      ]);
    } else if (Platform.isWindows) {
      candidatePaths.addAll([
        "wgpu_native.dll",
        "wgpu.dll",
      ]);
    }

    for (final path in candidatePaths) {
      try {
        final dylib = ffi.DynamicLibrary.open(path);
        return WgpuNativeLib(dylib, libraryPath: path);
      } catch (_) {
        // Continue searching candidates
      }
    }

    return null;
  }

  /// Loads the native WebGPU library or throws a [GpuException] if not found.
  static WgpuNativeLib load({String? customPath}) {
    final lib = tryLoad(customPath: customPath);
    if (lib == null) {
      throw GpuDeviceException(
        "Could not load native WebGPU dynamic library (libwgpu_native). "
        "Ensure libwgpu_native is installed or set the WGPU_LIB_PATH environment variable.",
      );
    }
    return lib;
  }
}

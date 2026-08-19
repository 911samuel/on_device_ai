import '../domain/backend_descriptor.dart';
import '../domain/model_spec.dart';
import '../domain/on_device_model.dart';
import '../domain/prediction.dart';
import 'litert_compiled_model_classifier.dart';
import 'litert_interpreter_classifier.dart';
import 'model_catalog.dart';

/// The inference configurations this PoC can run, and the only place that maps
/// a configuration to a concrete implementation.
///
/// The list is chosen so that each entry answers a question a reviewer will ask:
///
/// | entry                   | what it demonstrates                            |
/// |-------------------------|--------------------------------------------------|
/// | compiled_v2_gpu_cpu     | the recommended API's automatic GPU→CPU selection |
/// | compiled_v2_cpu         | a CPU-only baseline for the same API and model    |
/// | compiled_v2_npu_cpu     | what an NPU request actually does on this device  |
/// | interpreter_v2_xnnpack  | the same float weights on the older API           |
/// | interpreter_v1q_xnnpack | the quantized model, which CompiledModel can't run|
/// | interpreter_v1q_plain   | delegate off, i.e. what XNNPACK is actually worth  |
abstract final class BackendRegistry {
  static const BackendDescriptor compiledV2GpuCpu = BackendDescriptor(
    id: 'compiled_v2_gpu_cpu',
    label: 'CompiledModel · float32 · GPU→CPU',
    modelId: 'mobilenet_v2_float32',
    api: LiteRtApi.compiledModel,
    requestedUnits: {ComputeUnit.gpu, ComputeUnit.cpu},
    rationale: 'LiteRT Next with the permissive default set: try GPU, fall '
        'back to CPU if GPU compilation fails.',
  );

  static const BackendDescriptor compiledV2Cpu = BackendDescriptor(
    id: 'compiled_v2_cpu',
    label: 'CompiledModel · float32 · CPU only',
    modelId: 'mobilenet_v2_float32',
    api: LiteRtApi.compiledModel,
    requestedUnits: {ComputeUnit.cpu},
    rationale: 'Baseline for the same API and weights, with no accelerator in '
        'play. Any GPU/NPU claim has to beat this.',
  );

  static const BackendDescriptor compiledV2NpuCpu = BackendDescriptor(
    id: 'compiled_v2_npu_cpu',
    label: 'CompiledModel · float32 · NPU→CPU',
    modelId: 'mobilenet_v2_float32',
    api: LiteRtApi.compiledModel,
    requestedUnits: {ComputeUnit.npu, ComputeUnit.cpu},
    rationale: 'Requests the NPU (Core ML/ANE on iOS, vendor runtime on '
        'Android) with CPU fallback, and reports whether it was honoured.',
  );

  static const BackendDescriptor interpreterV2Xnnpack = BackendDescriptor(
    id: 'interpreter_v2_xnnpack',
    label: 'Interpreter · float32 · XNNPACK',
    modelId: 'mobilenet_v2_float32',
    api: LiteRtApi.interpreter,
    requestedUnits: {ComputeUnit.cpu},
    rationale: 'Identical weights on the classic API, so CompiledModel vs '
        'Interpreter is measured rather than argued.',
  );

  static const BackendDescriptor interpreterV1QuantXnnpack = BackendDescriptor(
    id: 'interpreter_v1q_xnnpack',
    label: 'Interpreter · uint8 · XNNPACK',
    modelId: 'mobilenet_v1_uint8',
    api: LiteRtApi.interpreter,
    requestedUnits: {ComputeUnit.cpu},
    rationale: 'The quantized model — 4 MB instead of 13 MB — which cannot use '
        'CompiledModel at all because that API is float32-only.',
  );

  static const BackendDescriptor interpreterV1QuantPlain = BackendDescriptor(
    id: 'interpreter_v1q_plain',
    label: 'Interpreter · uint8 · no delegate',
    modelId: 'mobilenet_v1_uint8',
    api: LiteRtApi.interpreter,
    requestedUnits: {ComputeUnit.cpu},
    rationale: 'Reference CPU kernels with no delegate, which is what makes '
        'the XNNPACK speed-up quantifiable.',
  );

  static const List<BackendDescriptor> all = [
    compiledV2GpuCpu,
    compiledV2Cpu,
    compiledV2NpuCpu,
    interpreterV2Xnnpack,
    interpreterV1QuantXnnpack,
    interpreterV1QuantPlain,
  ];

  static BackendDescriptor byId(String id) =>
      all.firstWhere((b) => b.id == id, orElse: () {
        throw ArgumentError.value(id, 'id', 'Unknown backend id');
      });

  /// Builds the implementation for [descriptor]. This is the single seam where
  /// the app becomes LiteRT-specific.
  static OnDeviceModel create(BackendDescriptor descriptor) {
    final spec = ModelCatalog.byId(descriptor.modelId);
    return switch (descriptor.id) {
      'compiled_v2_gpu_cpu' || 'compiled_v2_cpu' || 'compiled_v2_npu_cpu' =>
        LiteRtCompiledModelClassifier(
          spec: spec,
          requestedUnits: descriptor.requestedUnits,
        ),
      'interpreter_v2_xnnpack' || 'interpreter_v1q_xnnpack' =>
        LiteRtInterpreterClassifier(
          spec: spec,
          acceleration: InterpreterAcceleration.xnnpack,
        ),
      'interpreter_v1q_plain' => LiteRtInterpreterClassifier(
          spec: spec,
          acceleration: InterpreterAcceleration.none,
        ),
      _ => throw ArgumentError.value(
          descriptor.id,
          'descriptor.id',
          'No implementation registered',
        ),
    };
  }
}

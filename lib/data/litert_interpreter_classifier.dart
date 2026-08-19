import 'dart:typed_data';

import 'package:flutter_litert/native.dart' as litert;

import '../domain/input_image.dart';
import '../domain/ml_exceptions.dart';
import '../domain/model_spec.dart';
import '../domain/on_device_model.dart';
import '../domain/prediction.dart';
import 'asset_sources.dart';
import 'classification_postprocessor.dart';
import 'image_preprocessor.dart';
import 'tensor_data.dart';

/// Which delegate the classic `Interpreter` should attach.
///
/// Unlike `CompiledModel`, this API does not choose hardware for you: a
/// delegate is an explicit object you construct and attach, and if the delegate
/// declines an operator the interpreter silently runs that operator on the CPU.
enum InterpreterAcceleration {
  /// No delegate at all — reference CPU kernels. Slowest, and the only path
  /// that is unambiguously "plain CPU".
  none,

  /// XNNPACK: Google's optimised CPU kernel library (NEON on arm64). Still CPU,
  /// but typically several times faster than the reference kernels.
  xnnpack,

  /// GPU: OpenCL/OpenGL on Android, Metal on iOS. May decline a quantized graph.
  gpu,

  /// Core ML on iOS/macOS: the only route to the Apple Neural Engine from this
  /// API. arm64 devices only; the iOS simulator has no Neural Engine.
  coreMl,
}

/// [OnDeviceModel] backed by the **classic LiteRT `Interpreter`** API.
///
/// This exists for two reasons that are engineering facts rather than
/// preferences:
///
/// 1. It is the only API that accepts quantized (uint8) I/O, so the 4 MB
///    quantized model cannot run any other way.
/// 2. It exposes full tensor introspection — shape, dtype and quantization
///    parameters — so the model's contract can be asserted properly instead of
///    only by buffer size.
///
/// It is also generic over [ModelSpec], which lets the float model run here too.
/// That gives an apples-to-apples `CompiledModel` vs `Interpreter` comparison on
/// identical weights.
class LiteRtInterpreterClassifier implements OnDeviceModel {
  LiteRtInterpreterClassifier({
    required this.spec,
    required this.acceleration,
    this.numThreads = 4,
    AssetBytesLoader? bytesLoader,
    LabelRepository? labelRepository,
    this.preprocessor = const ImagePreprocessor(),
    this.postprocessor = const ClassificationPostprocessor(),
  })  : _loadBytes = bytesLoader ?? loadBundledBytes,
        _labelRepository = labelRepository ?? LabelRepository();

  @override
  final ModelSpec spec;

  final InterpreterAcceleration acceleration;
  final int numThreads;

  final ImagePreprocessor preprocessor;
  final ClassificationPostprocessor postprocessor;

  final AssetBytesLoader _loadBytes;
  final LabelRepository _labelRepository;

  litert.Interpreter? _interpreter;
  litert.Delegate? _delegate;
  litert.IsolateInterpreter? _isolate;
  List<String>? _labels;
  Duration? _initializationTime;
  RuntimeReport? _report;
  bool _disposed = false;

  @override
  bool get isInitialized => _interpreter != null && !_disposed;

  @override
  Duration? get initializationTime => _initializationTime;

  @override
  RuntimeReport get runtimeReport =>
      _report ??
      RuntimeReport(
        runtimeName: 'LiteRT (flutter_litert)',
        runtimeVersion: 'not initialised',
        apiName: 'Interpreter',
        executionMode: 'not initialised',
        accelerators: const AcceleratorReport.unknown(),
      );

  @override
  Future<void> initialize() async {
    if (_disposed) {
      throw const ModelLifecycleException(
        'This classifier was disposed and cannot be reinitialised.',
      );
    }
    if (_interpreter != null) return;

    final stopwatch = Stopwatch()..start();
    final bytes = await _loadBytes(spec.modelAsset);
    if (bytes.lengthInBytes != spec.fileSizeBytes) {
      throw TensorContractMismatchException(
        what: 'model asset size for ${spec.modelAsset}',
        expected: spec.fileSizeBytes,
        actual: bytes.lengthInBytes,
      );
    }

    // InterpreterFactory owns the per-platform delegate mapping and falls back
    // to plain options if a delegate cannot be constructed on this device.
    final (options, delegate) =
        litert.InterpreterFactory.create(_performanceConfig());

    final litert.Interpreter interpreter;
    try {
      interpreter = litert.Interpreter.fromBuffer(bytes, options: options);
    } on Object catch (error) {
      delegate?.delete();
      options.delete();
      throw ModelInitializationException(
        'Interpreter creation failed for ${spec.id} '
        '(acceleration: ${acceleration.name}).',
        cause: error,
      );
    }

    try {
      _assertTensorContract(interpreter);
    } on Object {
      interpreter.close();
      delegate?.delete();
      rethrow;
    }

    // A native interpreter cannot be shared across isolates once a delegate is
    // attached, so the binding only hands back an IsolateInterpreter when no
    // delegate is active. With a delegate, `invoke` runs on the calling
    // isolate — that is a real trade-off, reported honestly in executionMode.
    litert.IsolateInterpreter? isolate;
    try {
      isolate = await litert.InterpreterFactory.createIsolateIfNeeded(
        interpreter,
        delegate,
      );
    } on Object {
      isolate = null;
    }

    _labels = await _labelRepository.load(spec);
    _interpreter = interpreter;
    _delegate = delegate;
    _isolate = isolate;
    stopwatch.stop();
    _initializationTime = stopwatch.elapsed;

    final hasDelegate = interpreter.hasActiveDelegate;
    _report = RuntimeReport(
      runtimeName: 'LiteRT (flutter_litert)',
      runtimeVersion: _safeRuntimeVersion(),
      apiName: 'Interpreter (classic)',
      executionMode: isolate != null
          ? 'IsolateInterpreter — invoke on a background isolate'
          : 'invoke on the calling isolate '
              '(delegate attached: isolate sharing not available)',
      accelerators: AcceleratorReport(
        requested: _requestedUnits(),
        effective: hasDelegate ? _requestedUnits() : const {ComputeUnit.cpu},
        fullGraphAccelerated: false,
        delegateName: _delegateLabel(hasDelegate),
        accelerationProven: null,
        verificationSummary: null,
        notes: _notes(hasDelegate),
      ),
    );
  }

  @override
  Future<PredictionResult> predict(InputImage image, {int topK = 5}) async {
    final interpreter = _interpreter;
    final labels = _labels;
    if (interpreter == null || labels == null || _disposed) {
      throw const ModelLifecycleException(
        'predict() called before initialize() completed, or after dispose().',
      );
    }

    final preprocessWatch = Stopwatch()..start();
    final prepared = preprocessor.prepare(image, spec);
    preprocessWatch.stop();

    // Flat typed buffers in and out: the binding memcpys them straight into the
    // tensor, avoiding the nested-List conversion the old tflite_flutter
    // examples used (which allocates ~150k boxed doubles per frame).
    final input = switch (prepared.tensor) {
      Float32TensorData(:final values) => values as TypedData,
      Uint8TensorData(:final values) => values as TypedData,
    };
    final output = spec.outputType == TensorElementType.float32
        ? Float32List(spec.outputClassCount)
        : Uint8List(spec.outputClassCount);

    final inferenceWatch = Stopwatch()..start();
    Duration? nativeInference;
    try {
      final isolate = _isolate;
      if (isolate != null) {
        await isolate.run(input, output);
      } else {
        interpreter.run(input, output);
        nativeInference = Duration(
          microseconds: interpreter.lastInferenceDurationMicroseconds,
        );
      }
    } on Object catch (error) {
      throw InferenceException(
        'Interpreter.run failed for ${spec.id}.',
        cause: error,
      );
    }
    inferenceWatch.stop();

    final postprocessWatch = Stopwatch()..start();
    final predictions = postprocessor.decode(
      output: output is Float32List
          ? Float32TensorData(output)
          : Uint8TensorData(output as Uint8List),
      labels: labels,
      spec: spec,
      topK: topK,
    );
    postprocessWatch.stop();

    return PredictionResult(
      predictions: predictions,
      timings: PhaseTimings(
        preprocess: preprocessWatch.elapsed,
        inference: inferenceWatch.elapsed,
        postprocess: postprocessWatch.elapsed,
        nativeInference: nativeInference,
      ),
      runtime: runtimeReport,
      modelId: spec.id,
      imageSource: image.source,
    );
  }

  @override
  Future<void> dispose() async {
    _disposed = true;
    final isolate = _isolate;
    final interpreter = _interpreter;
    final delegate = _delegate;
    _isolate = null;
    _interpreter = null;
    _delegate = null;
    _labels = null;

    // Order matters: stop the worker before deleting what it points at.
    try {
      await isolate?.close();
    } on Object {
      /* ignore */
    }
    try {
      interpreter?.close();
    } on Object {
      /* ignore */
    }
    try {
      delegate?.delete();
    } on Object {
      /* ignore */
    }
  }

  // ---------------------------------------------------------------- internals

  litert.PerformanceConfig _performanceConfig() => switch (acceleration) {
        InterpreterAcceleration.none => litert.PerformanceConfig.disabled,
        InterpreterAcceleration.xnnpack =>
          litert.PerformanceConfig.xnnpack(numThreads: numThreads),
        InterpreterAcceleration.gpu =>
          litert.PerformanceConfig.gpu(numThreads: numThreads),
        InterpreterAcceleration.coreMl =>
          litert.PerformanceConfig.coreml(numThreads: numThreads),
      };

  /// The strong contract check: this API reports shape, dtype *and*
  /// quantization parameters, so all three are compared against [ModelSpec].
  /// Wrong quantization parameters would not crash — they would just make every
  /// confidence value wrong — so they are checked explicitly.
  void _assertTensorContract(litert.Interpreter interpreter) {
    final inputs = interpreter.getInputTensors();
    final outputs = interpreter.getOutputTensors();
    if (inputs.length != 1 || outputs.length != 1) {
      throw TensorContractMismatchException(
        what: 'tensor counts',
        expected: '1 input / 1 output',
        actual: '${inputs.length} input / ${outputs.length} output',
      );
    }

    final input = inputs.first;
    final output = outputs.first;

    _expectShape('input', input.shape, spec.inputShape);
    _expectShape('output', output.shape, spec.outputShape);
    _expectType('input', input.type, spec.inputType);
    _expectType('output', output.type, spec.outputType);
    _expectQuantization('input', input.params, spec.inputQuantization);
    _expectQuantization('output', output.params, spec.outputQuantization);

    if (input.numBytes() != spec.inputByteCount) {
      throw TensorContractMismatchException(
        what: 'input tensor byte size',
        expected: spec.inputByteCount,
        actual: input.numBytes(),
      );
    }
  }

  static void _expectShape(String what, List<int> actual, List<int> expected) {
    final same = actual.length == expected.length &&
        List.generate(actual.length, (i) => actual[i] == expected[i])
            .every((ok) => ok);
    if (!same) {
      throw TensorContractMismatchException(
        what: '$what shape',
        expected: expected,
        actual: actual,
      );
    }
  }

  static void _expectType(
    String what,
    litert.TensorType actual,
    TensorElementType expected,
  ) {
    final expectedNative = switch (expected) {
      TensorElementType.float32 => litert.TensorType.float32,
      TensorElementType.uint8 => litert.TensorType.uint8,
    };
    if (actual != expectedNative) {
      throw TensorContractMismatchException(
        what: '$what dtype',
        expected: expectedNative.name,
        actual: actual.name,
      );
    }
  }

  static void _expectQuantization(
    String what,
    litert.QuantizationParams actual,
    QuantizationSpec expected,
  ) {
    // Float tensors report scale 0 / zeroPoint 0, which QuantizationSpec.none
    // also encodes, so this single comparison covers both cases.
    const epsilon = 1e-9;
    final scaleMatches = (actual.scale - expected.scale).abs() < epsilon;
    if (!scaleMatches || actual.zeroPoint != expected.zeroPoint) {
      throw TensorContractMismatchException(
        what: '$what quantization parameters',
        expected: expected.toString(),
        actual: 'QuantizationSpec(scale: ${actual.scale}, '
            'zeroPoint: ${actual.zeroPoint})',
      );
    }
  }

  Set<ComputeUnit> _requestedUnits() => switch (acceleration) {
        InterpreterAcceleration.none ||
        InterpreterAcceleration.xnnpack =>
          const {ComputeUnit.cpu},
        InterpreterAcceleration.gpu => const {ComputeUnit.gpu},
        InterpreterAcceleration.coreMl => const {ComputeUnit.npu},
      };

  String _delegateLabel(bool hasDelegate) => switch (acceleration) {
        InterpreterAcceleration.none => 'none (reference CPU kernels)',
        InterpreterAcceleration.xnnpack =>
          hasDelegate ? 'XNNPACK ($numThreads threads)' : 'XNNPACK (declined)',
        InterpreterAcceleration.gpu =>
          hasDelegate ? 'GPU/Metal' : 'GPU (declined, running on CPU)',
        InterpreterAcceleration.coreMl =>
          hasDelegate ? 'Core ML' : 'Core ML (declined, running on CPU)',
      };

  String _notes(bool hasDelegate) {
    final notes = <String>[];
    if (acceleration != InterpreterAcceleration.none && !hasDelegate) {
      notes.add('The ${acceleration.name} delegate could not be attached on '
          'this device; execution fell back to CPU.');
    }
    if (acceleration == InterpreterAcceleration.xnnpack) {
      notes.add('XNNPACK is an optimised CPU kernel library, not a separate '
          'processor: this is still CPU execution.');
    }
    if (hasDelegate) {
      notes.add('A delegate may still decline individual operators, which then '
          'run on CPU; this API does not report per-operator placement.');
    }
    return notes.join(' ');
  }

  static String _safeRuntimeVersion() {
    try {
      return litert.Interpreter.version;
    } on Object {
      return 'unavailable';
    }
  }
}

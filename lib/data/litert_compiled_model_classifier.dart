import 'dart:io' show Platform;
import 'dart:typed_data';

// The native barrel. `package:flutter_litert/flutter_litert.dart` resolves to
// the WASM-safe web surface during static analysis, which hides the native-only
// symbols; this app targets Android and iOS only, so it imports the native
// surface directly and unconditionally.
import 'package:flutter_litert/native.dart' as litert;

import '../domain/input_image.dart';
import '../domain/ml_exceptions.dart';
import '../domain/model_spec.dart';
import '../domain/on_device_model.dart';
import '../domain/prediction.dart';
import '../domain/resize_strategy.dart';
import 'asset_sources.dart';
import 'classification_postprocessor.dart';
import 'image_preprocessor.dart';
import 'tensor_data.dart';

/// [OnDeviceModel] backed by the **LiteRT Next `CompiledModel`** API.
///
/// Why this API for the float model: it is the current recommended inference
/// path, it performs accelerator selection itself (NPU → GPU → CPU) instead of
/// making the app assemble delegates, and — critically for a PoC that must not
/// overclaim — it can report which accelerators it actually compiled for.
///
/// Why it is not used for everything: `CompiledModel` takes and returns
/// `Float32List` only. A uint8-quantized model cannot use it at all, which is
/// why [LiteRtInterpreterClassifier] also exists.
class LiteRtCompiledModelClassifier implements OnDeviceModel {
  LiteRtCompiledModelClassifier({
    required this.spec,
    required this.requestedUnits,
    AssetBytesLoader? bytesLoader,
    LabelRepository? labelRepository,
    this.preprocessor = const ImagePreprocessor(),
    this.postprocessor = const ClassificationPostprocessor(),
    this.verifyAgainstCpuReference = true,
  })  : _loadBytes = bytesLoader ?? loadBundledBytes,
        _labelRepository = labelRepository ?? LabelRepository();

  @override
  final ModelSpec spec;

  /// Run the binding's CPU-reference comparison at init. Costs one extra
  /// inference plus a second, plain-CPU interpreter; it is the only way to tell
  /// a working accelerator from a silent fallback, so it is on by default.
  final bool verifyAgainstCpuReference;

  /// Hardware the app asks for. What it gets is reported separately.
  final Set<ComputeUnit> requestedUnits;
  final ImagePreprocessor preprocessor;
  final ClassificationPostprocessor postprocessor;

  final AssetBytesLoader _loadBytes;
  final LabelRepository _labelRepository;

  litert.CompiledModel? _model;
  List<String>? _labels;
  Duration? _initializationTime;
  RuntimeReport? _report;
  bool _disposed = false;
  bool _useAsyncDispatch = true;

  @override
  bool get isInitialized => _model != null && !_disposed;

  @override
  Duration? get initializationTime => _initializationTime;

  @override
  RuntimeReport get runtimeReport =>
      _report ??
      RuntimeReport(
        runtimeName: 'LiteRT (flutter_litert)',
        runtimeVersion: 'not initialised',
        apiName: 'CompiledModel',
        executionMode: 'not initialised',
        accelerators: AcceleratorReport(
          requested: requestedUnits,
          effective: const {},
          fullGraphAccelerated: false,
          notes: 'Model not initialised yet.',
        ),
      );

  @override
  Future<void> initialize() async {
    if (_disposed) {
      throw const ModelLifecycleException(
        'This classifier was disposed and cannot be reinitialised.',
      );
    }
    if (_model != null) return;

    // Hard constraint of the API, not a stylistic choice: LiteRT Next's
    // CompiledModel exchanges Float32List buffers only.
    if (spec.inputType != TensorElementType.float32 ||
        spec.outputType != TensorElementType.float32) {
      throw UnsupportedPlatformException(
        'CompiledModel supports float32 I/O only, but ${spec.id} declares '
        'input ${spec.inputType.name} / output ${spec.outputType.name}. '
        'Use the classic Interpreter for quantized models.',
      );
    }

    final stopwatch = Stopwatch()..start();
    final bytes = await _loadBytes(spec.modelAsset);
    if (bytes.lengthInBytes != spec.fileSizeBytes) {
      throw TensorContractMismatchException(
        what: 'model asset size for ${spec.modelAsset}',
        expected: spec.fileSizeBytes,
        actual: bytes.lengthInBytes,
      );
    }

    final requested = _toAccelerators(requestedUnits);
    Object? gpuFallbackError;

    final litert.CompiledModel model;
    try {
      // `compiledModelFromBufferAuto` routes the permissive default
      // {gpu, cpu} through `fromBufferWithGpuFallback` (which retries CPU-only
      // if GPU compilation throws) and any explicit set through `fromBuffer`,
      // which treats the set as a hard requirement and does not degrade.
      model = litert.compiledModelFromBufferAuto(
        bytes,
        accelerators: requested,
        // fp32 is the binding's default: measured fp16 GPU output failed
        // CPU-reference parity for most models, so fp16 is an opt-in per model.
        precision: litert.Precision.fp32,
        onGpuFallback: (error) => gpuFallbackError = error,
      );
    } on Object catch (error) {
      throw ModelInitializationException(
        'CompiledModel compilation failed for ${spec.id} with accelerators '
        '${requested.map((a) => a.name).join('+')}.',
        cause: error,
      );
    }

    try {
      _assertTensorContract(model);
    } on Object {
      model.close();
      rethrow;
    }

    // Order matters: the verification verdict can only be interpreted once we
    // know which accelerators the runtime actually kept.
    final effectiveUnits = _toComputeUnits(model.accelerators);
    final verification = verifyAgainstCpuReference
        ? _verify(bytes, model, effectiveUnits)
        : null;

    // The binding documents that `runAsync` on thread-affine mobile GPU stacks
    // (some Android GL/CL drivers) is unvalidated, because the helper isolate
    // runs the model on a different thread than the one that compiled it.
    // Honour that: dispatch synchronously when a GPU backend is live on Android.
    _useAsyncDispatch =
        !(Platform.isAndroid && effectiveUnits.contains(ComputeUnit.gpu));

    _labels = await _labelRepository.load(spec);
    _model = model;
    stopwatch.stop();
    _initializationTime = stopwatch.elapsed;

    _report = RuntimeReport(
      runtimeName: 'LiteRT (flutter_litert)',
      runtimeVersion: _safeRuntimeVersion(),
      apiName: 'CompiledModel (LiteRT Next)',
      executionMode: _useAsyncDispatch
          ? 'runAsync — native call on a per-model helper isolate'
          : 'run — synchronous on the calling isolate (GPU thread affinity)',
      accelerators: AcceleratorReport(
        requested: requestedUnits,
        effective: effectiveUnits,
        fullGraphAccelerated: _safeIsFullyAccelerated(model),
        delegateName: null,
        accelerationProven: verification?.proven,
        verificationSummary: verification?.summary,
        notes: _buildNotes(
          requested: requestedUnits,
          effective: effectiveUnits,
          gpuFallbackError: gpuFallbackError,
        ),
      ),
    );
  }

  @override
  Future<PredictionResult> predict(
    InputImage image, {
    int topK = 5,
    ResizeStrategy resize = ResizeStrategy.stretch,
  }) async {
    final model = _model;
    final labels = _labels;
    if (model == null || labels == null || _disposed) {
      throw const ModelLifecycleException(
        'predict() called before initialize() completed, or after dispose().',
      );
    }

    final preprocessWatch = Stopwatch()..start();
    final prepared = preprocessor.prepare(image, spec, strategy: resize);
    preprocessWatch.stop();

    final tensor = prepared.tensor;
    if (tensor is! Float32TensorData) {
      throw TensorContractMismatchException(
        what: 'preprocessed tensor type',
        expected: 'Float32TensorData',
        actual: tensor.runtimeType,
      );
    }

    final inferenceWatch = Stopwatch()..start();
    final List<Float32List> outputs;
    try {
      outputs = _useAsyncDispatch
          ? await model.runAsync([tensor.values])
          : model.run([tensor.values]);
    } on Object catch (error) {
      throw InferenceException(
        'CompiledModel.${_useAsyncDispatch ? 'runAsync' : 'run'} failed for '
        '${spec.id}.',
        cause: error,
      );
    }
    inferenceWatch.stop();

    if (outputs.length != 1) {
      throw TensorContractMismatchException(
        what: 'output tensor count',
        expected: 1,
        actual: outputs.length,
      );
    }

    final postprocessWatch = Stopwatch()..start();
    final predictions = postprocessor.decode(
      output: Float32TensorData(outputs.first),
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
      ),
      runtime: runtimeReport,
      modelId: spec.id,
      imageSource: image.source,
    );
  }

  @override
  Future<void> dispose() async {
    _disposed = true;
    final model = _model;
    _model = null;
    _labels = null;
    if (model == null) return;
    try {
      model.close();
    } on Object {
      // Releasing native memory must never be the thing that crashes the app.
    }
  }

  // ---------------------------------------------------------------- internals

  /// `CompiledModel` exposes buffer *sizes* but not shapes or dtypes, so the
  /// contract is asserted in bytes. 1×224×224×3 float32 = 602,112 in,
  /// 1001 float32 = 4,004 out. A mismatch here means the bundled asset is not
  /// the model this code was written against.
  void _assertTensorContract(litert.CompiledModel model) {
    if (model.inputCount != 1) {
      throw TensorContractMismatchException(
        what: 'input tensor count',
        expected: 1,
        actual: model.inputCount,
      );
    }
    if (model.outputCount != 1) {
      throw TensorContractMismatchException(
        what: 'output tensor count',
        expected: 1,
        actual: model.outputCount,
      );
    }
    if (model.inputByteSizes.first != spec.inputByteCount) {
      throw TensorContractMismatchException(
        what: 'input tensor byte size',
        expected: spec.inputByteCount,
        actual: model.inputByteSizes.first,
      );
    }
    if (model.outputByteSizes.first != spec.outputByteCount) {
      throw TensorContractMismatchException(
        what: 'output tensor byte size',
        expected: spec.outputByteCount,
        actual: model.outputByteSizes.first,
      );
    }
  }

  /// Compares this compiled model against a bare-CPU interpreter built from the
  /// same bytes.
  ///
  /// Two independent things come out of one check:
  ///  * correctness — LiteRT Next has shipped defects where a compiled model
  ///    returns OK while producing wrong or never-written output;
  ///  * whether a compute path other than the reference CPU kernels ran — if the
  ///    result is *bit-identical* to the plain-CPU reference, nothing else did.
  ///
  /// [effectiveUnits] is required because a nonzero deviation on its own does
  /// NOT mean an accelerator ran. The reference uses `PerformanceConfig.disabled`
  /// (unoptimised CPU kernels), so an optimised CPU path deviates from it too.
  /// Measured on the Android emulator: GPU compilation failed, the model was
  /// rebuilt CPU-only, and the output still deviated by 0.0006% of range. Reading
  /// that as "GPU verified" would have been exactly the kind of overclaim this
  /// check exists to prevent — so when the effective set is CPU-only, the verdict
  /// is deliberately "undetermined" rather than "proven".
  _Verification? _verify(
    Uint8List bytes,
    litert.CompiledModel model,
    Set<ComputeUnit> effectiveUnits,
  ) {
    final litert.BackendVerification result;
    try {
      result = litert.verifyCompiledModel(bytes, model);
    } on Object catch (error) {
      return _Verification(null, 'verification could not run: $error');
    }

    if (result.skipped) {
      return _Verification(null, 'skipped: ${result.skippedReason}');
    }
    if (!result.agrees) {
      // Wrong output is not something to paper over: refuse the model.
      throw ModelInitializationException(
        'CompiledModel output disagrees with the plain-CPU reference for '
        '${spec.id} (${result.toString()}). Refusing to use a backend that '
        'computes the wrong thing.',
        cause: result.error,
      );
    }

    final deviation = result.absoluteDeviation;
    final effectivelyCpuOnly = effectiveUnits.every((u) => u == ComputeUnit.cpu);

    if (effectivelyCpuOnly) {
      return _Verification(
        null,
        'output agrees with the plain-CPU reference. The effective accelerator '
        'set is CPU-only, so this says nothing about acceleration — any '
        'deviation here (${(result.relativeDeviation * 100).toStringAsFixed(4)}% '
        'of range) is just optimised CPU kernels differing from the '
        'unoptimised reference ones.',
      );
    }
    if (deviation == 0) {
      return _Verification(
        false,
        'output is bit-identical to the plain-CPU reference, so the requested '
        'accelerator contributed nothing and this is a silent CPU fallback',
      );
    }
    return _Verification(
      true,
      'agrees with the CPU reference within tolerance while deviating by '
      '${(result.relativeDeviation * 100).toStringAsFixed(4)}% of output range, '
      'so a compute path other than the reference CPU kernels ran — consistent '
      'with the selected accelerator, but this does NOT identify which silicon '
      'executed the graph',
    );
  }

  String _buildNotes({
    required Set<ComputeUnit> requested,
    required Set<ComputeUnit> effective,
    required Object? gpuFallbackError,
  }) {
    final notes = <String>[];
    if (gpuFallbackError != null) {
      notes.add('GPU compilation failed and the model was rebuilt CPU-only: '
          '$gpuFallbackError');
    }
    final missing = requested.difference(effective);
    if (missing.isNotEmpty) {
      notes.add('Requested ${missing.map((u) => u.name.toUpperCase()).join('+')}'
          ' but the runtime narrowed the set, so it is not in use.');
    }
    notes.add('isFullyAccelerated is displayed for completeness only; the '
        'binding documents false as ambiguous (partial delegation also '
        'reports false), so it is not used to assert acceleration.');
    return notes.join(' ');
  }

  static Set<litert.Accelerator> _toAccelerators(Set<ComputeUnit> units) => {
        for (final unit in units)
          switch (unit) {
            ComputeUnit.cpu => litert.Accelerator.cpu,
            ComputeUnit.gpu => litert.Accelerator.gpu,
            ComputeUnit.npu => litert.Accelerator.npu,
          },
      };

  static Set<ComputeUnit> _toComputeUnits(Set<litert.Accelerator> a) => {
        for (final acc in a)
          switch (acc) {
            litert.Accelerator.cpu => ComputeUnit.cpu,
            litert.Accelerator.gpu => ComputeUnit.gpu,
            litert.Accelerator.npu => ComputeUnit.npu,
          },
      };

  static String _safeRuntimeVersion() {
    try {
      return litert.Interpreter.version;
    } on Object {
      return 'unavailable';
    }
  }

  static bool _safeIsFullyAccelerated(litert.CompiledModel model) {
    try {
      return model.isFullyAccelerated;
    } on Object {
      return false;
    }
  }
}

class _Verification {
  const _Verification(this.proven, this.summary);

  /// `true` = a non-CPU backend measurably contributed, `false` = silent CPU
  /// fallback, `null` = undetermined.
  final bool? proven;
  final String summary;
}

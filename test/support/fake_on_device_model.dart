import 'package:on_device_ai/domain/input_image.dart';
import 'package:on_device_ai/domain/resize_strategy.dart';
import 'package:on_device_ai/domain/ml_exceptions.dart';
import 'package:on_device_ai/domain/model_spec.dart';
import 'package:on_device_ai/domain/on_device_model.dart';
import 'package:on_device_ai/domain/prediction.dart';

/// A scriptable [OnDeviceModel] with no native dependency.
///
/// Its existence is itself a check on the architecture: if the controller could
/// only be driven by a real LiteRT model, the abstraction would not be doing its
/// job.
class FakeOnDeviceModel implements OnDeviceModel {
  FakeOnDeviceModel({
    required this.spec,
    this.initializeError,
    this.predictError,
    this.inferenceDurations = const [Duration(milliseconds: 10)],
    this.label = 'Labrador retriever',
    this.confidence = 0.94,
  });

  @override
  final ModelSpec spec;

  /// Thrown by [initialize] when set.
  final OnDeviceMlException? initializeError;

  /// Thrown by [predict] when set.
  final OnDeviceMlException? predictError;

  /// Cycled through, so a benchmark sees varying timings.
  final List<Duration> inferenceDurations;

  final String label;
  final double confidence;

  int initializeCalls = 0;
  int predictCalls = 0;
  int disposeCalls = 0;

  bool _initialized = false;

  @override
  bool get isInitialized => _initialized;

  @override
  Duration? get initializationTime =>
      _initialized ? const Duration(milliseconds: 120) : null;

  @override
  RuntimeReport get runtimeReport => const RuntimeReport(
        runtimeName: 'Fake',
        runtimeVersion: '0.0.0-test',
        apiName: 'Fake',
        executionMode: 'in-test',
        accelerators: AcceleratorReport.unknown(),
      );

  /// The strategy the last [predict] call received, so controller wiring can be
  /// asserted without a real model.
  ResizeStrategy? lastResize;

  @override
  Future<void> initialize() async {
    initializeCalls++;
    final error = initializeError;
    if (error != null) throw error;
    _initialized = true;
  }

  @override
  Future<PredictionResult> predict(
    InputImage image, {
    int topK = 5,
    ResizeStrategy resize = ResizeStrategy.stretch,
  }) async {
    predictCalls++;
    lastResize = resize;
    final error = predictError;
    if (error != null) throw error;
    if (!_initialized) {
      throw const ModelLifecycleException('predict before initialize');
    }
    final inference =
        inferenceDurations[(predictCalls - 1) % inferenceDurations.length];
    return PredictionResult(
      predictions: [
        Prediction(classIndex: 209, label: label, confidence: confidence),
        const Prediction(classIndex: 208, label: 'golden retriever',
            confidence: 0.03),
      ],
      timings: PhaseTimings(
        preprocess: const Duration(milliseconds: 4),
        inference: inference,
        postprocess: const Duration(milliseconds: 1),
      ),
      runtime: runtimeReport,
      modelId: spec.id,
      imageSource: image.source,
    );
  }

  @override
  Future<void> dispose() async {
    disposeCalls++;
    _initialized = false;
  }
}

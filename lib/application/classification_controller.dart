import 'package:flutter/foundation.dart';

import '../data/asset_sources.dart';
import '../data/backend_registry.dart';
import '../data/model_catalog.dart';
import '../domain/backend_descriptor.dart';
import '../domain/input_image.dart';
import '../domain/ml_exceptions.dart';
import '../domain/model_spec.dart';
import '../domain/on_device_model.dart';
import '../domain/prediction.dart';
import 'benchmark.dart';
import 'network_self_test.dart';

/// Builds an implementation for a descriptor. Injectable so widget tests and
/// controller tests can run without any native runtime.
typedef OnDeviceModelFactory = OnDeviceModel Function(BackendDescriptor);

/// Opens the platform image picker. Returns null when the user cancels.
typedef ImagePickerFn = Future<InputImage?> Function();

/// A bundled demo image.
class SampleImage {
  const SampleImage({
    required this.assetPath,
    required this.label,
    required this.expectation,
  });

  final String assetPath;
  final String label;

  /// What the reference (Python) run predicts for this image, so a live demo
  /// has a stated expectation instead of "whatever appears".
  final String expectation;
}

enum ControllerStatus { idle, initializing, ready, running, benchmarking, error }

/// Owns all UI state and the lifecycle of exactly one [OnDeviceModel].
///
/// This is the only layer that knows about both the UI and the ML service, and
/// it deliberately talks to the latter only through [OnDeviceModel]. It never
/// sees a LiteRT type.
class ClassificationController extends ChangeNotifier {
  ClassificationController({
    OnDeviceModelFactory? modelFactory,
    AssetBytesLoader? assetLoader,
    this.imagePicker,
    this.benchmarkRunner = const BenchmarkRunner(),
    this.networkSelfTest = const NetworkSelfTest(),
  })  : _modelFactory = modelFactory ?? BackendRegistry.create,
        _loadAsset = assetLoader ?? loadBundledBytes;

  static const List<SampleImage> samples = [
    SampleImage(
      assetPath: 'assets/images/grace_hopper.jpg',
      label: 'Grace Hopper',
      expectation: 'military uniform',
    ),
    SampleImage(
      assetPath: 'assets/images/labrador.jpg',
      label: 'Labrador',
      expectation: 'Labrador retriever (float) / kuvasz (quantized)',
    ),
    SampleImage(
      assetPath: 'assets/images/cat_on_snow.jpg',
      label: 'Cat on snow',
      expectation: 'lynx',
    ),
  ];

  final OnDeviceModelFactory _modelFactory;
  final AssetBytesLoader _loadAsset;
  /// Injected so the controller is testable without a platform channel.
  final ImagePickerFn? imagePicker;
  final BenchmarkRunner benchmarkRunner;
  final NetworkSelfTest networkSelfTest;

  OnDeviceModel? _model;
  BackendDescriptor _backend = BackendRegistry.all.first;
  ControllerStatus _status = ControllerStatus.idle;
  InputImage? _image;
  Uint8List? _imageBytesForDisplay;
  PredictionResult? _result;
  BenchmarkReport? _benchmark;
  NetworkProbeResult? _networkProbe;
  String? _errorMessage;
  String? _errorDetail;
  bool _disposed = false;

  List<BackendDescriptor> get backends => BackendRegistry.all;
  BackendDescriptor get backend => _backend;
  ModelSpec get spec => ModelCatalog.byId(_backend.modelId);
  ControllerStatus get status => _status;
  InputImage? get image => _image;
  Uint8List? get imageBytes => _imageBytesForDisplay;
  PredictionResult? get result => _result;
  BenchmarkReport? get benchmark => _benchmark;
  NetworkProbeResult? get networkProbe => _networkProbe;
  String? get errorMessage => _errorMessage;
  String? get errorDetail => _errorDetail;

  bool get isBusy =>
      _status == ControllerStatus.initializing ||
      _status == ControllerStatus.running ||
      _status == ControllerStatus.benchmarking;

  bool get canRun =>
      _status == ControllerStatus.ready && _image != null && _model != null;

  /// What the model reports about itself, once initialised.
  RuntimeReport? get runtimeReport => _model?.runtimeReport;

  Duration? get initializationTime => _model?.initializationTime;

  /// Loads the default backend and the first sample image.
  Future<void> start() async {
    await selectBackend(_backend);
    if (_status != ControllerStatus.error) {
      await loadSample(samples.first);
    }
  }

  /// Tears down the current model and initialises the chosen one.
  Future<void> selectBackend(BackendDescriptor descriptor) async {
    if (isBusy) return;
    _backend = descriptor;
    _result = null;
    _benchmark = null;
    _clearError();
    _setStatus(ControllerStatus.initializing);

    // Release the previous model's native memory before allocating the next.
    // Two live float32 models is ~28 MB of weights plus arenas.
    final previous = _model;
    _model = null;
    await previous?.dispose();

    try {
      final model = _modelFactory(descriptor);
      await model.initialize();
      _model = model;
      _setStatus(ControllerStatus.ready);
    } on OnDeviceMlException catch (error) {
      _fail(error.userMessage, error.toString());
    } on Object catch (error) {
      _fail(
        'The inference backend could not be initialised on this device.',
        error.toString(),
      );
    }
  }

  Future<void> loadSample(SampleImage sample) async {
    _clearError();
    try {
      final bytes = await _loadAsset(sample.assetPath);
      _setImage(
        InputImage(bytes: bytes, source: sample.assetPath),
        bytes,
      );
    } on OnDeviceMlException catch (error) {
      _fail(error.userMessage, error.toString());
    } on Object catch (error) {
      _fail('That sample image could not be loaded.', error.toString());
    }
  }

  Future<void> pickImage() async {
    final picker = imagePicker;
    if (picker == null) {
      _fail(
        'Image picking is not available in this build.',
        'No ImagePickerFn was provided to the controller.',
      );
      return;
    }
    _clearError();
    try {
      final picked = await picker();
      if (picked == null) return; // user cancelled
      _setImage(picked, picked.bytes);
    } on Object catch (error) {
      _fail('That image could not be opened.', error.toString());
    }
  }

  /// Runs one prediction.
  Future<void> classify() async {
    final model = _model;
    final image = _image;
    if (model == null || image == null || isBusy) return;
    _clearError();
    _setStatus(ControllerStatus.running);
    try {
      _result = await model.predict(image);
      _setStatus(ControllerStatus.ready);
    } on OnDeviceMlException catch (error) {
      _fail(error.userMessage, error.toString());
    } on Object catch (error) {
      _fail('Inference failed.', error.toString());
    }
  }

  /// Runs [iterations] predictions and summarises cold vs warm timings.
  Future<void> runBenchmark({int iterations = 30}) async {
    final model = _model;
    final image = _image;
    if (model == null || image == null || isBusy) return;
    _clearError();
    _setStatus(ControllerStatus.benchmarking);
    try {
      _benchmark = await benchmarkRunner.run(
        model: model,
        image: image,
        backendLabel: _backend.label,
        iterations: iterations,
      );
      _result = await model.predict(image);
      _setStatus(ControllerStatus.ready);
    } on OnDeviceMlException catch (error) {
      _fail(error.userMessage, error.toString());
    } on Object catch (error) {
      _fail('The benchmark could not complete.', error.toString());
    }
  }

  /// Asks the OS for an outbound socket; see [NetworkSelfTest].
  Future<void> runNetworkSelfTest() async {
    _networkProbe = await networkSelfTest.probe();
    _notify();
  }

  @override
  void dispose() {
    _disposed = true;
    final model = _model;
    _model = null;
    // Fire-and-forget: ChangeNotifier.dispose is synchronous, and the native
    // handles must still be released.
    model?.dispose();
    super.dispose();
  }

  // ---------------------------------------------------------------- internals

  void _setImage(InputImage image, Uint8List displayBytes) {
    _image = image;
    _imageBytesForDisplay = displayBytes;
    _result = null;
    _benchmark = null;
    _notify();
  }

  void _setStatus(ControllerStatus status) {
    _status = status;
    _notify();
  }

  void _clearError() {
    _errorMessage = null;
    _errorDetail = null;
  }

  void _fail(String message, String detail) {
    _errorMessage = message;
    _errorDetail = detail;
    _status = ControllerStatus.error;
    _notify();
  }

  void _notify() {
    if (_disposed) return;
    notifyListeners();
  }
}

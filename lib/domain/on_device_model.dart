import 'input_image.dart';
import 'model_spec.dart';
import 'prediction.dart';
import 'resize_strategy.dart';

/// The application's entire view of "a model that runs locally".
///
/// This is the seam that keeps the app decoupled from the inference runtime.
/// Nothing above this interface knows that LiteRT exists: swapping in ONNX
/// Runtime, ML Kit, or a remote endpoint means adding an implementation, not
/// touching the UI or the controller. Both bundled implementations
/// (`CompiledModel` and classic `Interpreter`) sit behind it, which is what
/// proves the abstraction is real rather than decorative.
abstract interface class OnDeviceModel {
  /// The I/O contract this implementation was built against.
  ModelSpec get spec;

  /// Whether [initialize] has completed and [dispose] has not been called.
  bool get isInitialized;

  /// How this backend describes itself once initialised. Valid after
  /// [initialize]; before that it reports what will be *attempted*.
  RuntimeReport get runtimeReport;

  /// Time spent loading and compiling the model, available after [initialize].
  Duration? get initializationTime;

  /// Loads the model and labels, verifies the tensor contract, and prepares the
  /// runtime. Idempotent: calling it twice does not reload the model.
  ///
  /// Throws [ModelAssetMissingException], [ModelInitializationException],
  /// [TensorContractMismatchException], [LabelSetException] or
  /// [UnsupportedPlatformException] (all from `ml_exceptions.dart`).
  Future<void> initialize();

  /// Runs the full pipeline for [image] and returns the top [topK] classes.
  ///
  /// [resize] selects how a non-square source is fitted to the model's square
  /// input. It lives here rather than in the constructor because it is a
  /// per-prediction choice the user can flip to compare results.
  ///
  /// Throws [ImageDecodeException] for unusable input, [InferenceException] if
  /// the runtime fails, [ModelLifecycleException] if not initialised.
  Future<PredictionResult> predict(
    InputImage image, {
    int topK = 5,
    ResizeStrategy resize = ResizeStrategy.stretch,
  });

  /// Releases native memory. Safe to call more than once, and safe to call
  /// without a preceding [initialize].
  Future<void> dispose();
}

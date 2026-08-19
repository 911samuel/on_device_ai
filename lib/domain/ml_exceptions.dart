/// Failure modes of the on-device inference pipeline.
///
/// Every one of these is reachable in practice, and each carries a message that
/// is safe and useful to put in front of a user. The UI renders [userMessage];
/// logs get [toString], which keeps the underlying cause.
library;

/// Base type for anything the ML layer can fail with.
sealed class OnDeviceMlException implements Exception {
  const OnDeviceMlException(this.message, {this.cause});

  /// Engineer-facing detail.
  final String message;

  /// The lower-level error, if this wraps one.
  final Object? cause;

  /// Short, non-technical sentence for the UI.
  String get userMessage;

  @override
  String toString() {
    final suffix = cause == null ? '' : ' (cause: $cause)';
    return '$runtimeType: $message$suffix';
  }
}

/// The model asset is absent from the bundle, or unreadable.
final class ModelAssetMissingException extends OnDeviceMlException {
  const ModelAssetMissingException(this.assetPath, {super.cause})
      : super('Model asset "$assetPath" could not be loaded from the bundle.');

  final String assetPath;

  @override
  String get userMessage =>
      'The model file is missing from this build. Reinstall the app.';
}

/// The runtime refused to build a model from the asset, or the asset is not a
/// valid flatbuffer.
final class ModelInitializationException extends OnDeviceMlException {
  const ModelInitializationException(super.message, {super.cause});

  @override
  String get userMessage =>
      'The model could not be initialised on this device.';
}

/// The model's real I/O does not match what [ModelSpec] declares.
///
/// This is the check that turns "wrong normalisation / wrong shape" from a
/// silent accuracy bug into a loud startup failure.
final class TensorContractMismatchException extends OnDeviceMlException {
  const TensorContractMismatchException({
    required this.what,
    required this.expected,
    required this.actual,
  }) : super('Tensor contract mismatch for $what: '
            'expected $expected, model reports $actual');

  final String what;
  final Object expected;
  final Object actual;

  @override
  String get userMessage =>
      'This build bundles a model that does not match the code that reads it.';
}

/// Labels are missing, empty, or the wrong length for the output tensor.
final class LabelSetException extends OnDeviceMlException {
  const LabelSetException(super.message, {super.cause});

  @override
  String get userMessage => 'The model\'s label list is missing or invalid.';
}

/// The bytes handed in were not a decodable image.
final class ImageDecodeException extends OnDeviceMlException {
  const ImageDecodeException(super.message, {super.cause});

  @override
  String get userMessage =>
      'That file could not be read as an image. Try a JPEG or PNG.';
}

/// Inference itself failed inside the native runtime.
final class InferenceException extends OnDeviceMlException {
  const InferenceException(super.message, {super.cause});

  @override
  String get userMessage => 'Inference failed on this device.';
}

/// The requested backend cannot run on this platform at all.
final class UnsupportedPlatformException extends OnDeviceMlException {
  const UnsupportedPlatformException(super.message, {super.cause});

  @override
  String get userMessage => 'This inference backend is not supported here.';
}

/// A method was called before `initialize()` or after `dispose()`.
final class ModelLifecycleException extends OnDeviceMlException {
  const ModelLifecycleException(super.message);

  @override
  String get userMessage => 'The model is not ready yet.';
}

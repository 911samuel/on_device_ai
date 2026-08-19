import 'dart:typed_data';

import 'package:image/image.dart' as img;

import '../domain/input_image.dart';
import '../domain/ml_exceptions.dart';
import '../domain/model_spec.dart';
import 'tensor_data.dart';

/// Outcome of preprocessing, including what the source looked like.
class PreprocessedImage {
  const PreprocessedImage({
    required this.tensor,
    required this.sourceWidth,
    required this.sourceHeight,
    required this.resizedWidth,
    required this.resizedHeight,
    required this.interpolation,
  });

  final TensorData tensor;
  final int sourceWidth;
  final int sourceHeight;
  final int resizedWidth;
  final int resizedHeight;

  /// Which resampling filter was used, so it can be reported and tested.
  final String interpolation;

  /// True when the resize changed the aspect ratio (see the note on
  /// [ImagePreprocessor] about stretch vs centre-crop).
  bool get distortedAspectRatio =>
      sourceWidth * resizedHeight != sourceHeight * resizedWidth;
}

/// Turns encoded image bytes into the exact tensor a [ModelSpec] declares.
///
/// Pure Dart on purpose: no `dart:ui`, no LiteRT, no platform channels. That
/// makes the arithmetic — which is where classification bugs actually live —
/// unit-testable on the host, and it is validated bit-for-bit against the
/// Python reference in `test/image_preprocessor_test.dart`.
///
/// Pipeline: **decode → force 8-bit RGB → resize to 224×224 → normalise → flat
/// buffer**.
///
/// Resize policy is a plain stretch to the model's square input, with no
/// letterbox and no centre crop. That is a deliberate, documented choice: it is
/// the simplest thing to explain and it keeps the whole frame visible. The cost
/// is aspect-ratio distortion on non-square inputs, which measurably lowers
/// confidence on elongated subjects; the standard ImageNet evaluation recipe
/// instead centre-crops to 87.5% before resizing.
/// [PreprocessedImage.distortedAspectRatio] surfaces when this applies.
///
/// The *filter* is chosen by scale factor, which is a measured decision rather
/// than a default. Downsampling a 512x600 photo to 224x224 throws away ~84% of
/// the pixels; a plain bilinear tap reads only the 4 nearest source pixels and
/// therefore aliases, while a box/area filter averages every pixel that maps
/// into the destination pixel. Measured against the Pillow BILINEAR reference
/// (which antialiases on downscale) over 66 sampled bytes per image:
///
/// | image                | scale  | nearest | linear | cubic | average |
/// |----------------------|--------|---------|--------|-------|---------|
/// | grace_hopper 512x600 | down 2.3x | 10.89 | 7.69   | 8.18  | **4.00** |
/// | labrador 700x577     | down 3.1x |  9.86 | 7.95   | 8.62  | **3.22** |
/// | cat_on_snow 320x213  | ~1:1      |  4.29 | **2.00** | 2.42 | 4.23   |
///
/// So: area-average when genuinely downscaling, bilinear when close to 1:1 or
/// upscaling (where a box filter degenerates to nearest-neighbour). Getting this
/// wrong does not crash anything — it reorders the low-confidence tail of the
/// top-5, which is exactly the class of bug that ships unnoticed.
class ImagePreprocessor {
  const ImagePreprocessor();

  PreprocessedImage prepare(InputImage image, ModelSpec spec) {
    if (image.bytes.isEmpty) {
      throw const ImageDecodeException('Image data is empty (0 bytes).');
    }

    final img.Image? decoded;
    try {
      decoded = img.decodeImage(image.bytes);
    } on Object catch (error) {
      throw ImageDecodeException(
        'Decoder threw while reading "${image.source}".',
        cause: error,
      );
    }
    if (decoded == null) {
      throw ImageDecodeException(
        'Unsupported or corrupt image format for "${image.source}" '
        '(${image.byteLength} bytes). Supported: JPEG, PNG, GIF, BMP, TIFF, WebP.',
      );
    }

    // Force a predictable 8-bit, 3-channel representation before touching
    // pixels. Without this, a 16-bit PNG or a palette/greyscale image would
    // yield a buffer with the wrong element width or channel count.
    final rgb8 = decoded.numChannels == 3 && decoded.format == img.Format.uint8
        ? decoded
        : decoded.convert(format: img.Format.uint8, numChannels: 3);

    final alreadyExact =
        rgb8.width == spec.inputWidth && rgb8.height == spec.inputHeight;
    final interpolation = selectInterpolation(
      sourceWidth: rgb8.width,
      sourceHeight: rgb8.height,
      targetWidth: spec.inputWidth,
      targetHeight: spec.inputHeight,
    );
    final resized = alreadyExact
        ? rgb8
        : img.copyResize(
            rgb8,
            width: spec.inputWidth,
            height: spec.inputHeight,
            interpolation: interpolation,
          );

    // Explicit channel order. Getting RGB vs BGR wrong does not crash and does
    // not look obviously broken — it just quietly degrades accuracy — so it is
    // stated rather than inherited from a default.
    final rgbBytes = resized.getBytes(order: img.ChannelOrder.rgb);
    if (rgbBytes.length != spec.inputElementCount) {
      throw TensorContractMismatchException(
        what: 'preprocessed RGB byte count',
        expected: spec.inputElementCount,
        actual: rgbBytes.length,
      );
    }

    return PreprocessedImage(
      tensor: normalize(rgbBytes, spec),
      sourceWidth: decoded.width,
      sourceHeight: decoded.height,
      resizedWidth: resized.width,
      resizedHeight: resized.height,
      interpolation: alreadyExact ? 'none' : interpolation.name,
    );
  }

  /// Minimum shrink factor on either axis before area-averaging is used.
  ///
  /// Below this, a box filter covers barely more than one source pixel per
  /// destination pixel and degenerates towards nearest-neighbour, which measured
  /// worse than bilinear on the near-1:1 sample.
  static const double antialiasShrinkThreshold = 1.5;

  /// Picks the resampling filter from the scale factor. See the class comment
  /// for the measurements behind the threshold.
  static img.Interpolation selectInterpolation({
    required int sourceWidth,
    required int sourceHeight,
    required int targetWidth,
    required int targetHeight,
  }) {
    final shrinkX = sourceWidth / targetWidth;
    final shrinkY = sourceHeight / targetHeight;
    final downscaling = shrinkX >= antialiasShrinkThreshold ||
        shrinkY >= antialiasShrinkThreshold;
    return downscaling ? img.Interpolation.average : img.Interpolation.linear;
  }

  /// Maps 0..255 RGB bytes into the numeric range the model was trained on.
  ///
  /// Kept separate and static so tests can drive it directly with known bytes.
  static TensorData normalize(Uint8List rgbBytes, ModelSpec spec) {
    switch (spec.normalization) {
      case InputNormalization.rawUint8:
        // No arithmetic at all: the model's input quantization parameters
        // (scale 1/128, zero-point 128) do the scaling inside the graph.
        return Uint8TensorData(rgbBytes);
      case InputNormalization.minusOneToOne:
        final out = Float32List(rgbBytes.length);
        for (var i = 0; i < rgbBytes.length; i++) {
          out[i] = rgbBytes[i] / 127.5 - 1.0;
        }
        return Float32TensorData(out);
      case InputNormalization.zeroToOne:
        final out = Float32List(rgbBytes.length);
        for (var i = 0; i < rgbBytes.length; i++) {
          out[i] = rgbBytes[i] / 255.0;
        }
        return Float32TensorData(out);
    }
  }
}

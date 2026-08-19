import 'dart:typed_data';

import '../domain/model_spec.dart';

/// A flat, typed buffer ready to be handed to the runtime.
///
/// **Why a tensor and not a Flutter image object.** A `ui.Image` is an opaque
/// handle to a GPU/CPU raster owned by the Flutter engine: its memory layout,
/// colour space and channel order are engine implementation details, and it may
/// not even live in addressable host memory. A model is a fixed sequence of
/// arithmetic kernels compiled against an exact buffer contract — here
/// `[1, 224, 224, 3]` NHWC, 602,112 bytes of little-endian float32 for the float
/// model. The runtime does a single `memcpy` into that buffer, so the caller must
/// supply precisely those bytes in precisely that order. "Tensor" is just the
/// name for that flat buffer plus the shape used to index it: element
/// `(n, y, x, c)` lives at `((n*H + y)*W + x)*C + c`.
sealed class TensorData {
  const TensorData();

  /// Number of scalar elements.
  int get length;

  /// Size of the underlying buffer in bytes.
  int get byteLength;

  /// The element type this buffer satisfies.
  TensorElementType get elementType;
}

/// float32 buffer, for models with float input/output.
final class Float32TensorData extends TensorData {
  const Float32TensorData(this.values);

  final Float32List values;

  @override
  int get length => values.length;

  @override
  int get byteLength => values.lengthInBytes;

  @override
  TensorElementType get elementType => TensorElementType.float32;
}

/// uint8 buffer, for quantized models. Note that this is *also* the natural
/// representation of raw RGB pixels, which is why a quantized model needs no
/// floating-point normalisation pass at all.
final class Uint8TensorData extends TensorData {
  const Uint8TensorData(this.values);

  final Uint8List values;

  @override
  int get length => values.length;

  @override
  int get byteLength => values.lengthInBytes;

  @override
  TensorElementType get elementType => TensorElementType.uint8;
}

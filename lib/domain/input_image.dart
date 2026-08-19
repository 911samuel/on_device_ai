import 'dart:typed_data';

/// An image on its way into the model, in its original encoded form.
///
/// Deliberately *not* a Flutter `Image`, `ImageProvider` or `ui.Image`: those
/// are presentation-layer types tied to the Flutter engine's rasteriser. The ML
/// layer needs pixels it can address numerically, so the boundary type is the
/// encoded file bytes plus enough provenance to show in the UI and in logs.
class InputImage {
  const InputImage({
    required this.bytes,
    required this.source,
  });

  /// Raw contents of a JPEG/PNG/etc. file. Decoding happens in the
  /// preprocessing stage, which is where format errors are detected.
  final Uint8List bytes;

  /// Where this came from, e.g. `assets/images/labrador.jpg` or
  /// `gallery:IMG_0042.HEIC`. Shown in the UI; never parsed.
  final String source;

  int get byteLength => bytes.length;

  @override
  String toString() => 'InputImage($source, $byteLength bytes)';
}

import 'package:flutter/services.dart';

import '../domain/ml_exceptions.dart';
import '../domain/model_spec.dart';

/// Loads binary assets (the `.tflite` files).
typedef AssetBytesLoader = Future<Uint8List> Function(String assetPath);

/// Loads text assets (the label file).
typedef AssetTextLoader = Future<String> Function(String assetPath);

/// Reads a bundled asset as bytes via the Flutter asset bundle.
///
/// `rootBundle` reads out of the application package on disk — the APK's
/// `assets/flutter_assets/` or the iOS `.app` bundle. No network is involved,
/// which is the mechanical reason this PoC works offline.
Future<Uint8List> loadBundledBytes(String assetPath) async {
  try {
    final data = await rootBundle.load(assetPath);
    // A view, not a copy: LiteRT copies it into native memory itself.
    return data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
  } on Object catch (error) {
    throw ModelAssetMissingException(assetPath, cause: error);
  }
}

Future<String> loadBundledText(String assetPath) =>
    rootBundle.loadString(assetPath);

/// Loads and caches class labels, and refuses to hand back a list that would
/// silently misalign predictions.
class LabelRepository {
  LabelRepository({AssetTextLoader? loader})
      : _loader = loader ?? loadBundledText;

  final AssetTextLoader _loader;
  final Map<String, List<String>> _cache = {};

  /// Returns the labels for [spec], validating the count against the model's
  /// output size.
  Future<List<String>> load(ModelSpec spec) async {
    final cached = _cache[spec.labelsAsset];
    if (cached != null) return cached;

    final String raw;
    try {
      raw = await _loader(spec.labelsAsset);
    } on Object catch (error) {
      throw LabelSetException(
        'Label asset "${spec.labelsAsset}" could not be loaded.',
        cause: error,
      );
    }

    final lines = raw.split('\n');
    // Trailing newline at end of file is normal; a blank line in the middle is
    // not, because it would shift every subsequent class.
    while (lines.isNotEmpty && lines.last.trim().isEmpty) {
      lines.removeLast();
    }
    final labels = lines.map((line) => line.trim()).toList(growable: false);

    if (labels.isEmpty) {
      throw LabelSetException(
        'Label asset "${spec.labelsAsset}" is empty.',
      );
    }
    if (labels.length != spec.outputClassCount) {
      throw LabelSetException(
        'Label asset "${spec.labelsAsset}" has ${labels.length} entries but '
        '${spec.id} produces ${spec.outputClassCount} classes. Off-by-one '
        'label files are the classic cause of confidently wrong predictions.',
      );
    }
    final blankIndex = labels.indexWhere((l) => l.isEmpty);
    if (blankIndex >= 0) {
      throw LabelSetException(
        'Label asset "${spec.labelsAsset}" has a blank entry at index '
        '$blankIndex, which would misalign the classes after it.',
      );
    }

    _cache[spec.labelsAsset] = labels;
    return labels;
  }
}

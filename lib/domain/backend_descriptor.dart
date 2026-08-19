import 'model_spec.dart';
import 'prediction.dart';

/// A selectable inference configuration, described in pure data.
///
/// The UI renders these and passes the chosen one back; it never learns which
/// class implements it. That is what keeps the presentation layer independent of
/// the runtime.
class BackendDescriptor {
  const BackendDescriptor({
    required this.id,
    required this.label,
    required this.modelId,
    required this.api,
    required this.requestedUnits,
    required this.rationale,
  });

  /// Stable key; also the switch key used by the factory in `data/`.
  final String id;

  /// Short human-readable name for the selector.
  final String label;

  /// Which [ModelSpec] this configuration loads.
  final String modelId;

  /// Which LiteRT API executes it.
  final LiteRtApi api;

  /// What hardware the app will ask for. What it *gets* is reported separately
  /// by [AcceleratorReport], because the two are not the same thing.
  final Set<ComputeUnit> requestedUnits;

  /// Why this configuration is in the list — each one demonstrates something
  /// specific.
  final String rationale;

  @override
  String toString() => 'BackendDescriptor($id)';
}

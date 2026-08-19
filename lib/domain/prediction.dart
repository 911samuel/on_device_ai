/// One scored class from the classifier's output.
class Prediction {
  const Prediction({
    required this.classIndex,
    required this.label,
    required this.confidence,
  });

  /// Index into the output tensor, and therefore into the label file.
  final int classIndex;

  final String label;

  /// Probability in [0, 1]. Both bundled models end in a softmax, so this is a
  /// genuine probability rather than an unbounded logit.
  final double confidence;

  /// Confidence as a percentage string, e.g. `94.2%`.
  String get confidencePercent =>
      '${(confidence * 100).toStringAsFixed(1)}%';

  @override
  String toString() =>
      'Prediction(#$classIndex $label ${confidence.toStringAsFixed(4)})';
}

/// Wall-clock cost of each stage of a single prediction.
///
/// Measured separately because they have completely different scaling
/// behaviour: preprocessing is Dart-side CPU work proportional to source image
/// size, inference is native work proportional to model complexity, and
/// postprocessing is a sort over the class count.
class PhaseTimings {
  const PhaseTimings({
    required this.preprocess,
    required this.inference,
    required this.postprocess,
    this.nativeInference,
  });

  static const PhaseTimings zero = PhaseTimings(
    preprocess: Duration.zero,
    inference: Duration.zero,
    postprocess: Duration.zero,
  );

  /// Decode + resize + normalise + tensor fill.
  final Duration preprocess;

  /// Time around the runtime's inference call, as observed from Dart. Includes
  /// the FFI hop and, on the async path, an isolate round trip.
  final Duration inference;

  /// Dequantise + sort + label mapping.
  final Duration postprocess;

  /// Time the native runtime reports for the invoke itself, when the API
  /// exposes it (classic `Interpreter` only). Null means "not reported by this
  /// backend" — never a fabricated value.
  final Duration? nativeInference;

  Duration get total => preprocess + inference + postprocess;

  Map<String, Duration> asMap() => {
        'preprocess': preprocess,
        'inference': inference,
        'postprocess': postprocess,
        'total': total,
      };

  @override
  String toString() => 'PhaseTimings(pre: ${preprocess.inMicroseconds}us, '
      'inf: ${inference.inMicroseconds}us, '
      'post: ${postprocess.inMicroseconds}us)';
}

/// Compute units a backend can run on, mirrored in the domain layer so the UI
/// never imports the LiteRT package.
enum ComputeUnit { cpu, gpu, npu }

/// What actually happened, hardware-wise — reported rather than assumed.
///
/// The distinction between [requested] and [effective] matters: LiteRT narrows
/// the accelerator set silently when a device cannot honour the request, and a
/// PoC that prints the request would be claiming acceleration it never got.
class AcceleratorReport {
  const AcceleratorReport({
    required this.requested,
    required this.effective,
    required this.fullGraphAccelerated,
    required this.notes,
    this.delegateName,
    this.accelerationProven,
    this.verificationSummary,
  });

  const AcceleratorReport.unknown()
      : requested = const {},
        effective = const {},
        fullGraphAccelerated = false,
        notes = 'Not reported by this backend.',
        delegateName = null,
        accelerationProven = null,
        verificationSummary = null;

  /// What the app asked for.
  final Set<ComputeUnit> requested;

  /// What the runtime says it compiled for, after any narrowing.
  final Set<ComputeUnit> effective;

  /// Whether the runtime claims the whole graph was taken by an accelerator.
  ///
  /// Treated as a weak signal only: the LiteRT binding documents `false` as
  /// ambiguous (partial delegation also reports false), so this is displayed
  /// but never used to assert that acceleration happened.
  final bool fullGraphAccelerated;

  /// Name of the delegate applied on the classic-`Interpreter` path.
  final String? delegateName;

  /// Tri-state: `true` when output differs measurably from a bare-CPU
  /// reference (so a non-CPU backend demonstrably contributed), `false` when it
  /// was bit-identical (i.e. a silent CPU fallback), `null` when the check did
  /// not run. Never inferred from timings.
  final bool? accelerationProven;

  /// Human-readable outcome of that comparison.
  final String? verificationSummary;

  /// Caveats worth showing next to the numbers.
  final String notes;

  String describe(Set<ComputeUnit> units) =>
      units.isEmpty ? '—' : units.map((u) => u.name.toUpperCase()).join(' + ');

  bool get wasNarrowed => !_setEquals(requested, effective);

  static bool _setEquals(Set<ComputeUnit> a, Set<ComputeUnit> b) =>
      a.length == b.length && a.every(b.contains);
}

/// Everything the UI needs to describe *how* a prediction was produced.
class RuntimeReport {
  const RuntimeReport({
    required this.runtimeName,
    required this.runtimeVersion,
    required this.apiName,
    required this.executionMode,
    required this.accelerators,
  });

  /// e.g. `LiteRT (via flutter_litert)`.
  final String runtimeName;

  /// Version string reported by the native runtime, when available.
  final String runtimeVersion;

  /// `CompiledModel` or `Interpreter`.
  final String apiName;

  /// e.g. `background isolate (runAsync)`.
  final String executionMode;

  final AcceleratorReport accelerators;
}

/// Result of one prediction: the ranked classes, the cost, and the provenance.
class PredictionResult {
  const PredictionResult({
    required this.predictions,
    required this.timings,
    required this.runtime,
    required this.modelId,
    required this.imageSource,
  });

  /// Top-K, highest confidence first.
  final List<Prediction> predictions;

  final PhaseTimings timings;
  final RuntimeReport runtime;
  final String modelId;
  final String imageSource;

  Prediction get top => predictions.first;

  /// Confidence at or above which the top-1 label is treated as a real answer.
  ///
  /// 0.50 is not invented here: it is the same threshold
  /// `integration_test/on_device_inference_test.dart` uses to decide whether the
  /// reference prediction is decisive enough to assert an exact label match.
  /// Reusing it keeps the UI and the test suite telling the same story.
  static const double decisiveConfidence = 0.50;

  /// Below this gap between top-1 and top-2, the model is effectively spreading
  /// its probability rather than choosing.
  static const double indecisiveMargin = 0.10;

  /// True when the top-1 confidence clears [decisiveConfidence].
  bool get isDecisive => top.confidence >= decisiveConfidence;

  /// How far ahead the winner is. Small values mean a near-tie.
  double get topTwoMargin => predictions.length < 2
      ? top.confidence
      : top.confidence - predictions[1].confidence;

  /// True when the result looks like "the right answer is not on the menu".
  ///
  /// These models choose among a fixed 1,001 ImageNet classes and have no way to
  /// abstain, so an out-of-vocabulary subject — a person, a building, a phone, a
  /// plate of food, none of which are ImageNet classes — still produces a
  /// confident-looking label. The tell is a low top-1 *and* a crowded field: the
  /// probability is spread across several nearest-neighbours rather than
  /// concentrated on one.
  ///
  /// This is a heuristic, deliberately reported as a caveat rather than used to
  /// suppress the prediction.
  bool get probablyOutsideVocabulary =>
      !isDecisive && topTwoMargin < indecisiveMargin;

  /// A short, honest description of how much weight the top-1 label carries.
  ConfidenceVerdict get verdict {
    if (isDecisive) return ConfidenceVerdict.decisive;
    if (probablyOutsideVocabulary) return ConfidenceVerdict.inconclusive;
    return ConfidenceVerdict.weak;
  }
}

/// How much the top-1 label should be trusted. See
/// [PredictionResult.verdict].
enum ConfidenceVerdict {
  /// Top-1 clears 0.50. Treat it as the model's answer.
  decisive,

  /// Top-1 is low but clearly ahead of the runner-up: a plausible answer on a
  /// hard or ambiguous image.
  weak,

  /// Top-1 is low *and* the field is crowded. Most often this means the subject
  /// is not one of the model's classes at all.
  inconclusive;

  String get headline => switch (this) {
        ConfidenceVerdict.decisive => 'Confident',
        ConfidenceVerdict.weak => 'Low confidence',
        ConfidenceVerdict.inconclusive => 'Inconclusive',
      };

  /// Plain-language explanation, safe to show a user.
  String get explanation => switch (this) {
        ConfidenceVerdict.decisive =>
          'The model concentrated its probability on one class.',
        ConfidenceVerdict.weak =>
          'Below 50%, but clearly ahead of the runner-up. Typical for an '
              'ambiguous photo, or one where the subject is partly obscured.',
        ConfidenceVerdict.inconclusive =>
          'Low score and a crowded field. This model can only choose among '
              '1,001 ImageNet classes — which include no person, building, '
              'road or food class — so a subject outside that list still gets '
              'a label, just an unconvincing one.',
      };
}

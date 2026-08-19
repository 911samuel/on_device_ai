import '../domain/input_image.dart';
import '../domain/on_device_model.dart';
import '../domain/prediction.dart';

/// Summary statistics over a set of measured durations.
///
/// Mean alone is not enough: on a phone, a single scheduler preemption or a
/// thermal step shows up as an outlier that drags the mean while the median
/// barely moves. Reporting min/median/p90/max makes that visible instead of
/// hiding it.
class DurationStats {
  const DurationStats({
    required this.count,
    required this.min,
    required this.max,
    required this.mean,
    required this.median,
    required this.p90,
  });

  factory DurationStats.from(List<Duration> samples) {
    if (samples.isEmpty) {
      throw ArgumentError('Cannot summarise an empty sample set.');
    }
    final micros = samples.map((d) => d.inMicroseconds).toList()..sort();
    final total = micros.fold<int>(0, (a, b) => a + b);
    return DurationStats(
      count: micros.length,
      min: Duration(microseconds: micros.first),
      max: Duration(microseconds: micros.last),
      mean: Duration(microseconds: (total / micros.length).round()),
      median: Duration(microseconds: _percentile(micros, 0.50)),
      p90: Duration(microseconds: _percentile(micros, 0.90)),
    );
  }

  final int count;
  final Duration min;
  final Duration max;
  final Duration mean;
  final Duration median;
  final Duration p90;

  /// Nearest-rank percentile on a pre-sorted, non-empty list.
  static int _percentile(List<int> sorted, double fraction) {
    final rank = (fraction * (sorted.length - 1)).round();
    return sorted[rank.clamp(0, sorted.length - 1)];
  }

  String describeMs() => 'min ${_ms(min)} / med ${_ms(median)} / '
      'mean ${_ms(mean)} / p90 ${_ms(p90)} / max ${_ms(max)}';

  static String _ms(Duration d) =>
      '${(d.inMicroseconds / 1000).toStringAsFixed(1)}ms';
}

/// Result of a benchmark run, with the cold measurement kept separate.
class BenchmarkReport {
  const BenchmarkReport({
    required this.backendLabel,
    required this.modelId,
    required this.iterations,
    required this.initialization,
    required this.cold,
    required this.warmPreprocess,
    required this.warmInference,
    required this.warmPostprocess,
    required this.warmTotal,
    required this.topLabel,
    required this.topConfidence,
  });

  final String backendLabel;
  final String modelId;

  /// Total predictions executed, including the cold one.
  final int iterations;

  /// Model load + compile + contract check + label load.
  final Duration initialization;

  /// The very first prediction after initialization.
  ///
  /// Reported on its own because it is not the same operation as a warm run:
  /// the first invoke pays lazy kernel/delegate setup, first-touch page faults
  /// on freshly allocated tensor arenas, instruction-cache misses on code paths
  /// never executed before, and — on a big.LITTLE phone — a CPU governor still
  /// at a low frequency. Averaging it in would inflate every number and hide
  /// the steady-state latency that actually matters for a UX budget.
  final PhaseTimings cold;

  final DurationStats warmPreprocess;
  final DurationStats warmInference;
  final DurationStats warmPostprocess;
  final DurationStats warmTotal;

  /// Sanity anchor: the prediction produced during the benchmark, so a fast but
  /// wrong configuration is not mistaken for a good one.
  final String topLabel;
  final double topConfidence;

  /// How much slower the cold run was than the warm median.
  double get coldPenaltyFactor => warmTotal.median.inMicroseconds == 0
      ? double.nan
      : cold.total.inMicroseconds / warmTotal.median.inMicroseconds;
}

/// Runs a fixed number of predictions and summarises them.
class BenchmarkRunner {
  const BenchmarkRunner();

  /// [model] must already be initialised. [iterations] includes the cold run,
  /// so 30 means 1 cold + 29 warm.
  Future<BenchmarkReport> run({
    required OnDeviceModel model,
    required InputImage image,
    required String backendLabel,
    int iterations = 30,
  }) async {
    if (iterations < 2) {
      throw ArgumentError.value(
        iterations,
        'iterations',
        'Need at least 2 runs to separate cold from warm.',
      );
    }

    final first = await model.predict(image);
    final warm = <PhaseTimings>[];
    var last = first;
    for (var i = 1; i < iterations; i++) {
      last = await model.predict(image);
      warm.add(last.timings);
      // Yield to the event loop so a long benchmark cannot starve the UI.
      await Future<void>.delayed(Duration.zero);
    }

    return BenchmarkReport(
      backendLabel: backendLabel,
      modelId: model.spec.id,
      iterations: iterations,
      initialization: model.initializationTime ?? Duration.zero,
      cold: first.timings,
      warmPreprocess: DurationStats.from([for (final t in warm) t.preprocess]),
      warmInference: DurationStats.from([for (final t in warm) t.inference]),
      warmPostprocess: DurationStats.from([for (final t in warm) t.postprocess]),
      warmTotal: DurationStats.from([for (final t in warm) t.total]),
      topLabel: last.top.label,
      topConfidence: last.top.confidence,
    );
  }
}

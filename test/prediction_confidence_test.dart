import 'package:flutter_test/flutter_test.dart';
import 'package:on_device_ai/domain/prediction.dart';

/// The confidence verdict exists because these models cannot abstain: they
/// always return one of 1,001 ImageNet classes, so an out-of-vocabulary subject
/// still gets a label. These tests pin the thresholds so the UI and the
/// integration suite keep telling the same story.
void main() {
  PredictionResult resultWith(List<double> confidences) => PredictionResult(
        predictions: [
          for (var i = 0; i < confidences.length; i++)
            Prediction(
              classIndex: i,
              label: 'class$i',
              confidence: confidences[i],
            ),
        ],
        timings: PhaseTimings.zero,
        runtime: const RuntimeReport(
          runtimeName: 'test',
          runtimeVersion: 'test',
          apiName: 'test',
          executionMode: 'test',
          accelerators: AcceleratorReport.unknown(),
        ),
        modelId: 'test',
        imageSource: 'test',
      );

  test('the decisive threshold matches the one the integration suite uses', () {
    expect(PredictionResult.decisiveConfidence, 0.50);
  });

  group('decisive', () {
    test('a clear winner above 0.50 is decisive', () {
      final r = resultWith([0.8750, 0.013, 0.010]);
      expect(r.isDecisive, isTrue);
      expect(r.verdict, ConfidenceVerdict.decisive);
      expect(r.probablyOutsideVocabulary, isFalse);
    });

    test('exactly 0.50 counts as decisive', () {
      expect(resultWith([0.50, 0.10]).isDecisive, isTrue);
    });
  });

  group('weak but plausible', () {
    test('below 0.50 yet clearly ahead is weak, not inconclusive', () {
      // The real Labrador measurement: 0.387 top-1, next well behind.
      final r = resultWith([0.3867, 0.12, 0.08]);
      expect(r.isDecisive, isFalse);
      expect(r.probablyOutsideVocabulary, isFalse);
      expect(r.verdict, ConfidenceVerdict.weak);
    });
  });

  group('inconclusive — probably not in the label set', () {
    test('a low score in a crowded field is flagged', () {
      final r = resultWith([0.09, 0.08, 0.07, 0.06]);
      expect(r.probablyOutsideVocabulary, isTrue);
      expect(r.verdict, ConfidenceVerdict.inconclusive);
    });

    test('the margin, not just the score, decides it', () {
      final tight = resultWith([0.30, 0.29]);
      final clear = resultWith([0.30, 0.05]);
      expect(tight.verdict, ConfidenceVerdict.inconclusive);
      expect(clear.verdict, ConfidenceVerdict.weak);
    });
  });

  test('topTwoMargin falls back to the top score for a single prediction', () {
    expect(resultWith([0.42]).topTwoMargin, 0.42);
  });

  test('every verdict has user-facing text', () {
    for (final v in ConfidenceVerdict.values) {
      expect(v.headline, isNotEmpty);
      expect(v.explanation, isNotEmpty);
    }
  });
}

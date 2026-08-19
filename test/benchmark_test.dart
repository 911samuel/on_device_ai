import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:on_device_ai/application/benchmark.dart';
import 'package:on_device_ai/data/model_catalog.dart';
import 'package:on_device_ai/domain/input_image.dart';

import 'support/fake_on_device_model.dart';

void main() {
  group('DurationStats', () {
    test('computes min, max, mean, median and p90 over known samples', () {
      final stats = DurationStats.from([
        for (final ms in [10, 12, 11, 30, 13, 12, 11, 12, 14, 12])
          Duration(milliseconds: ms),
      ]);
      expect(stats.count, 10);
      expect(stats.min, const Duration(milliseconds: 10));
      expect(stats.max, const Duration(milliseconds: 30));
      // sorted: 10 11 11 12 12 12 12 13 14 30 -> mean 13.7ms
      expect(stats.mean.inMicroseconds, 13700);
      expect(stats.median, const Duration(milliseconds: 12));
      expect(stats.p90, const Duration(milliseconds: 14));
    });

    test('median resists the outlier that drags the mean', () {
      final stats = DurationStats.from([
        for (final ms in [10, 10, 10, 10, 200]) Duration(milliseconds: ms),
      ]);
      expect(stats.median, const Duration(milliseconds: 10));
      expect(stats.mean, const Duration(milliseconds: 48));
    });

    test('handles a single sample', () {
      final stats = DurationStats.from([const Duration(milliseconds: 7)]);
      expect(stats.min, stats.max);
      expect(stats.median, const Duration(milliseconds: 7));
    });

    test('rejects an empty sample set instead of returning zeros', () {
      expect(() => DurationStats.from([]), throwsA(isA<ArgumentError>()));
    });
  });

  group('BenchmarkRunner', () {
    final image = InputImage(bytes: Uint8List(1), source: 'test');

    test('separates the cold run from the warm distribution', () async {
      // Run 1 is deliberately slow, like a real first inference.
      final model = FakeOnDeviceModel(
        spec: ModelCatalog.mobileNetV2Float32,
        inferenceDurations: const [
          Duration(milliseconds: 90),
          Duration(milliseconds: 10),
          Duration(milliseconds: 11),
          Duration(milliseconds: 12),
        ],
      );
      await model.initialize();

      final report = await const BenchmarkRunner().run(
        model: model,
        image: image,
        backendLabel: 'fake',
        iterations: 4,
      );

      expect(report.iterations, 4);
      expect(model.predictCalls, 4);
      expect(report.cold.inference, const Duration(milliseconds: 90));
      // Warm stats must exclude the cold run entirely.
      expect(report.warmInference.count, 3);
      expect(report.warmInference.min, const Duration(milliseconds: 10));
      expect(report.warmInference.max, const Duration(milliseconds: 12));
      expect(report.coldPenaltyFactor, greaterThan(3));
    });

    test('carries the prediction through so a wrong-but-fast run is visible',
        () async {
      final model = FakeOnDeviceModel(
        spec: ModelCatalog.mobileNetV1Uint8,
        label: 'kuvasz',
        confidence: 0.18,
      );
      await model.initialize();
      final report = await const BenchmarkRunner()
          .run(model: model, image: image, backendLabel: 'fake', iterations: 3);
      expect(report.topLabel, 'kuvasz');
      expect(report.topConfidence, closeTo(0.18, 1e-9));
      expect(report.modelId, 'mobilenet_v1_uint8');
    });

    test('refuses to run with too few iterations to split cold from warm',
        () async {
      final model = FakeOnDeviceModel(spec: ModelCatalog.mobileNetV2Float32);
      await model.initialize();
      expect(
        () => const BenchmarkRunner()
            .run(model: model, image: image, backendLabel: 'f', iterations: 1),
        throwsA(isA<ArgumentError>()),
      );
    });
  });
}

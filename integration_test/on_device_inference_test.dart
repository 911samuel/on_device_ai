import 'dart:convert';
import 'dart:io';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:on_device_ai/application/benchmark.dart';
import 'package:on_device_ai/data/asset_sources.dart';
import 'package:on_device_ai/data/backend_registry.dart';
import 'package:on_device_ai/data/model_catalog.dart';
import 'package:on_device_ai/domain/backend_descriptor.dart';
import 'package:on_device_ai/domain/input_image.dart';
import 'package:on_device_ai/domain/on_device_model.dart';

/// Real inference on a real device, validated against the Python reference.
///
/// This is the layer the host-side unit tests cannot reach: FFI, the native
/// runtime, delegate selection and actual arithmetic on device hardware. It runs
/// every registered backend over every sample image and checks the result
/// against `test/fixtures/reference_predictions.json`, which was produced by the
/// Python LiteRT reference interpreter on the same model files.
///
/// Tolerances are deliberate, not slack. Two independent pipelines resize with
/// different implementations of the same filter family, so they do not see
/// byte-identical pixels (measured mean absolute difference: 3.2-4.0 levels out
/// of 255 — see `test/image_resize_test.dart`). The assertions are therefore
/// graded by how decisive the reference itself is:
///
///  * Always: the Dart top-1 must appear in the reference top-5 and vice versa.
///    Any real preprocessing bug — wrong normalisation, BGR instead of RGB, a
///    transposed tensor — moves the top-1 far outside the reference top-5.
///  * When the reference top-1 is decisive (>= 0.50): the label must match
///    exactly and the confidence to within 0.20.
///  * When the reference is undecided (< 0.50, e.g. `labrador.jpg` where the
///    quantized model's top-1 is only 0.18 and several dog breeds are within a
///    few 1/256 steps of each other): rank order may legitimately swap, so only
///    the containment rule applies. Asserting an exact label there would be
///    testing the resampling filter, not the pipeline.
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  const sampleImages = [
    'assets/images/grace_hopper.jpg',
    'assets/images/labrador.jpg',
    'assets/images/cat_on_snow.jpg',
  ];

  late Map<String, dynamic> fixture;

  setUpAll(() async {
    fixture = jsonDecode(
      await rootBundle.loadString('test/fixtures/reference_predictions.json'),
    ) as Map<String, dynamic>;
    _log('platform: ${Platform.operatingSystem} '
        '${Platform.operatingSystemVersion}');
    _log('processors: ${Platform.numberOfProcessors}');
  });

  List<Map<String, dynamic>> referenceTop5(String imageAsset, String modelId) {
    final imageName = imageAsset.split('/').last;
    final predictions = fixture['predictions'] as Map<String, dynamic>;
    final perModel = (predictions[imageName] as Map)[modelId] as Map;
    return (perModel['top5'] as List).cast<Map<String, dynamic>>();
  }

  for (final backend in BackendRegistry.all) {
    group(backend.id, () {
      test('initialises, reports its hardware, and matches the reference',
          () async {
        final OnDeviceModel model;
        try {
          model = BackendRegistry.create(backend);
        } on Object catch (error) {
          fail('could not construct ${backend.id}: $error');
        }

        try {
          await model.initialize();
        } on Object catch (error) {
          // A backend that cannot exist on this device is a documented outcome
          // (no GPU on an emulator, no Neural Engine on a simulator), not a
          // test failure — but it must be visible in the log, never silent.
          _log('SKIP ${backend.id}: initialise failed -> $error');
          markTestSkipped('${backend.id} unavailable on this device: $error');
          return;
        }

        expect(model.isInitialized, isTrue);
        _logRuntime(backend, model);

        for (final asset in sampleImages) {
          final bytes = await loadBundledBytes(asset);
          final result =
              await model.predict(InputImage(bytes: bytes, source: asset));
          final expected = referenceTop5(asset, model.spec.id);

          final referenceTopScore = (expected.first['score'] as num).toDouble();
          final referenceTopLabel = expected.first['label'] as String;
          final decisive = referenceTopScore >= 0.50;
          final actualLabels = result.predictions.map((p) => p.label).toList();
          final expectedLabels =
              expected.map((e) => e['label'] as String).toList();

          _log('${backend.id} | ${asset.split('/').last} | '
              '${decisive ? 'decisive' : 'ambiguous'} | '
              'dart="${result.top.label}" '
              '${result.top.confidence.toStringAsFixed(4)} '
              'ref="$referenceTopLabel" '
              '${referenceTopScore.toStringAsFixed(4)} | '
              'dart_top5=$actualLabels ref_top5=$expectedLabels | '
              'pre ${_us(result.timings.preprocess)} '
              'inf ${_us(result.timings.inference)} '
              'post ${_us(result.timings.postprocess)}');

          expect(result.predictions, hasLength(5));

          // Containment in both directions: the strong, always-on bug detector.
          expect(
            expectedLabels, contains(result.top.label),
            reason: 'the on-device top-1 "${result.top.label}" is not even in '
                'the reference top-5 for $asset on ${backend.id}; that is a '
                'preprocessing or decoding bug, not a rounding difference',
          );
          expect(
            actualLabels, contains(referenceTopLabel),
            reason: 'the reference top-1 "$referenceTopLabel" is missing from '
                'the on-device top-5 for $asset on ${backend.id}',
          );

          if (decisive) {
            expect(
              result.top.label, referenceTopLabel,
              reason: 'the reference is confident '
                  '(${referenceTopScore.toStringAsFixed(2)}) for $asset, so the '
                  'top-1 label must match exactly on ${backend.id}',
            );
            expect(
              result.top.confidence,
              closeTo(referenceTopScore, 0.20),
              reason: 'top-1 confidence is further from the reference than '
                  'resampling differences can explain',
            );
          }

          // The shared head must overlap even when the tail reshuffles.
          expect(
            actualLabels.toSet().intersection(expectedLabels.toSet()).length,
            greaterThanOrEqualTo(2),
            reason: 'top-5 sets barely overlap: $actualLabels vs $expectedLabels',
          );

          // Both bundled graphs end in softmax, so confidences are ordered
          // probabilities in [0, 1].
          for (var i = 1; i < result.predictions.length; i++) {
            expect(
              result.predictions[i].confidence,
              lessThanOrEqualTo(result.predictions[i - 1].confidence),
            );
          }
          expect(result.top.confidence, inInclusiveRange(0.0, 1.0));
        }

        await model.dispose();
        expect(model.isInitialized, isFalse);
        // dispose() must be idempotent: a second call is a no-op, not a crash.
        await model.dispose();
      });

      test('benchmark: cold vs warm over 30 runs', () async {
        final model = BackendRegistry.create(backend);
        try {
          await model.initialize();
        } on Object catch (error) {
          _log('SKIP BENCH ${backend.id}: $error');
          markTestSkipped('${backend.id} unavailable: $error');
          return;
        }

        final bytes = await loadBundledBytes(sampleImages.first);
        final report = await const BenchmarkRunner().run(
          model: model,
          image: InputImage(bytes: bytes, source: sampleImages.first),
          backendLabel: backend.label,
          iterations: 30,
        );

        // Single machine-readable line per backend, so the numbers in
        // docs/BENCHMARKS.md are copied from a real run rather than estimated.
        _log('BENCH ${backend.id} '
            'model=${report.modelId} '
            'init=${_us(report.initialization)} '
            'cold_total=${_us(report.cold.total)} '
            'warm_total_min=${_us(report.warmTotal.min)} '
            'warm_total_med=${_us(report.warmTotal.median)} '
            'warm_total_mean=${_us(report.warmTotal.mean)} '
            'warm_total_p90=${_us(report.warmTotal.p90)} '
            'warm_total_max=${_us(report.warmTotal.max)} '
            'warm_pre_med=${_us(report.warmPreprocess.median)} '
            'warm_inf_min=${_us(report.warmInference.min)} '
            'warm_inf_med=${_us(report.warmInference.median)} '
            'warm_inf_max=${_us(report.warmInference.max)} '
            'warm_post_med=${_us(report.warmPostprocess.median)} '
            'cold_factor=${report.coldPenaltyFactor.toStringAsFixed(2)} '
            'top=${report.topLabel} '
            'conf=${report.topConfidence.toStringAsFixed(4)}');

        expect(report.warmTotal.count, 29);
        expect(report.warmTotal.median.inMicroseconds, greaterThan(0));
        // Sanity: a warm run should not be slower than a fresh cold start.
        expect(
          report.warmTotal.median.inMicroseconds,
          lessThanOrEqualTo(report.cold.total.inMicroseconds * 2),
        );
        await model.dispose();
      });
    });
  }

  group('error handling on device', () {
    test('a non-image file is rejected without crashing the runtime', () async {
      final model = BackendRegistry.create(BackendRegistry.compiledV2Cpu);
      await model.initialize();
      await expectLater(
        model.predict(InputImage(
          bytes: Uint8List.fromList('not an image'.codeUnits),
          source: 'bad.txt',
        )),
        throwsA(isA<Exception>()),
      );
      // The model must still work after a rejected input.
      final bytes = await loadBundledBytes(sampleImages.first);
      final ok =
          await model.predict(InputImage(bytes: bytes, source: 'recovery'));
      expect(ok.top.label, 'military uniform');
      await model.dispose();
    });

    test('predict() after dispose() throws instead of using freed memory',
        () async {
      final model = BackendRegistry.create(BackendRegistry.compiledV2Cpu);
      await model.initialize();
      await model.dispose();
      final bytes = await loadBundledBytes(sampleImages.first);
      await expectLater(
        model.predict(InputImage(bytes: bytes, source: 'after-dispose')),
        throwsA(isA<Exception>()),
      );
    });

    test('the shipped assets are readable from the app bundle', () async {
      for (final spec in ModelCatalog.all) {
        final bytes = await loadBundledBytes(spec.modelAsset);
        expect(bytes.lengthInBytes, spec.fileSizeBytes);
      }
    });
  });
}

String _us(Duration d) => '${(d.inMicroseconds / 1000).toStringAsFixed(2)}ms';

void _logRuntime(BackendDescriptor backend, OnDeviceModel model) {
  final r = model.runtimeReport;
  final a = r.accelerators;
  _log('RUNTIME ${backend.id} '
      'api="${r.apiName}" '
      'litert="${r.runtimeVersion}" '
      'exec="${r.executionMode}" '
      'requested=${a.describe(a.requested)} '
      'effective=${a.describe(a.effective)} '
      'fullGraph=${a.fullGraphAccelerated} '
      'delegate=${a.delegateName ?? '-'} '
      'proven=${a.accelerationProven} '
      'verify="${a.verificationSummary ?? '-'}" '
      'init=${model.initializationTime == null ? '-' : _us(model.initializationTime!)} '
      'notes="${a.notes}"');
}

/// `debugPrint` truncates long lines; the device log is the only channel back
/// from an integration test, so use stdout directly.
void _log(String message) {
  // ignore: avoid_print
  print('[poc] $message');
}

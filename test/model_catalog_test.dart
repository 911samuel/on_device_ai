import 'package:crypto/crypto.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:on_device_ai/data/backend_registry.dart';
import 'package:on_device_ai/data/model_catalog.dart';
import 'package:on_device_ai/domain/model_spec.dart';

import 'support/fixture.dart';

/// Closes the loop between three independent sources of truth:
/// the `.tflite` files on disk, the Python reference fixture generated from
/// them, and the [ModelSpec] constants the Dart code relies on. If any of the
/// three drifts, this fails.
void main() {
  final fixture = ReferenceFixture.load();

  group('shipped assets match their declared metadata', () {
    for (final spec in ModelCatalog.all) {
      test('${spec.id}: size and SHA-256 are as declared', () {
        final bytes = readRepoFile(spec.modelAsset);
        expect(bytes.length, spec.fileSizeBytes,
            reason: '${spec.modelAsset} is not the file this code expects');
        expect(sha256.convert(bytes).toString(), spec.sha256,
            reason: 'digest mismatch: the model asset was swapped or corrupted');
      });
    }

    test('the label asset is present and has 1001 entries', () {
      expect(readLabels(), hasLength(1001));
    });
  });

  group('tensor arithmetic in the spec', () {
    test('float32 input is 602,112 bytes and output 4,004 bytes', () {
      final spec = ModelCatalog.mobileNetV2Float32;
      expect(spec.inputElementCount, 224 * 224 * 3);
      expect(spec.inputByteCount, 602112);
      expect(spec.outputByteCount, 4004);
      expect(spec.inputShape, [1, 224, 224, 3]);
      expect(spec.outputShape, [1, 1001]);
    });

    test('the quantized model uses a quarter of the input bytes', () {
      final quant = ModelCatalog.mobileNetV1Uint8;
      expect(quant.inputByteCount, 150528);
      expect(quant.outputByteCount, 1001);
      expect(
        ModelCatalog.mobileNetV2Float32.inputByteCount ~/ quant.inputByteCount,
        4,
      );
    });

    test('the quantized model is ~3.3x smaller on disk', () {
      final ratio = ModelCatalog.mobileNetV2Float32.fileSizeBytes /
          ModelCatalog.mobileNetV1Uint8.fileSizeBytes;
      expect(ratio, closeTo(3.27, 0.05));
    });
  });

  group('spec agrees with the reference fixture', () {
    test('normalisation choices match what the Python reference used', () {
      final models = fixture.json['models'] as Map<String, dynamic>;
      expect(
        (models['mobilenet_v2_float32'] as Map)['normalization'],
        'minus_one_to_one',
      );
      expect(
        (models['mobilenet_v1_uint8'] as Map)['normalization'],
        'raw_uint8',
      );
      expect(ModelCatalog.mobileNetV2Float32.normalization,
          InputNormalization.minusOneToOne);
      expect(ModelCatalog.mobileNetV1Uint8.normalization,
          InputNormalization.rawUint8);
    });

    test('the normalisation probe justifies the float model\'s -1..1 choice',
        () {
      final probe = fixture.normalizationProbe;
      double top(String key) =>
          (((probe[key] as Map)['top3'] as List).first as Map)['score'] as double;
      String label(String key) =>
          (((probe[key] as Map)['top3'] as List).first as Map)['label']
              as String;

      expect(label('minus_one_to_one'), 'military uniform');
      expect(top('minus_one_to_one'), greaterThan(top('zero_to_one')));
      expect(label('zero_to_255'), isNot('military uniform'));
    });

    test('the float graph outputs a proper softmax distribution', () {
      // Sums to exactly 1.0, confirming SOFTMAX is inside the graph and must
      // not be applied a second time in Dart.
      for (final image in ['grace_hopper.jpg', 'labrador.jpg']) {
        expect(_probSum(fixture, image, 'mobilenet_v2_float32'),
            closeTo(1.0, 1e-3));
      }
      expect(ModelCatalog.mobileNetV2Float32.outputIsProbability, isTrue);
    });

    test('the quantized graph\'s probabilities sum to slightly under 1', () {
      // A real and instructive quantization artefact: the output step is
      // 1/256, so every class whose true probability is below 1/512 rounds to
      // zero. Across 1001 classes that lost mass is measurable — the sums are
      // 0.98 / 0.95 / 0.99 for the three sample images. It is still a softmax
      // distribution (so no second softmax), just a coarsely stored one, and
      // this is why quantized confidence values should not be read as
      // calibrated probabilities.
      for (final image in ['grace_hopper.jpg', 'labrador.jpg', 'cat_on_snow.jpg']) {
        final sum = _probSum(fixture, image, 'mobilenet_v1_uint8');
        expect(sum, lessThan(1.0));
        expect(sum, greaterThan(0.90),
            reason: 'losing more than 10% of the mass would suggest a '
                'dequantisation bug rather than rounding');
      }
      expect(ModelCatalog.mobileNetV1Uint8.outputIsProbability, isTrue);
    });
  });

  group('backend registry', () {
    test('every descriptor points at a model in the catalogue', () {
      for (final backend in BackendRegistry.all) {
        expect(() => ModelCatalog.byId(backend.modelId), returnsNormally);
      }
    });

    test('no CompiledModel backend is registered for the quantized model', () {
      // The API is float32-only; a registry entry pairing them would be a bug.
      final offenders = BackendRegistry.all.where((b) =>
          b.api == LiteRtApi.compiledModel &&
          ModelCatalog.byId(b.modelId).inputType != TensorElementType.float32);
      expect(offenders, isEmpty);
    });

    test('descriptor ids are unique and resolvable', () {
      final ids = BackendRegistry.all.map((b) => b.id).toSet();
      expect(ids, hasLength(BackendRegistry.all.length));
      for (final id in ids) {
        expect(BackendRegistry.byId(id).id, id);
      }
    });

    test('an unknown id is rejected', () {
      expect(() => BackendRegistry.byId('nope'), throwsA(isA<ArgumentError>()));
      expect(() => ModelCatalog.byId('nope'), throwsA(isA<ArgumentError>()));
    });
  });
}

double _probSum(ReferenceFixture fixture, String image, String modelId) {
  final preds = fixture.json['predictions'] as Map<String, dynamic>;
  final entry = (preds[image] as Map)[modelId] as Map<String, dynamic>;
  return (entry['prob_sum'] as num).toDouble();
}

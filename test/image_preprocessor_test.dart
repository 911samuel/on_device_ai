import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:on_device_ai/data/image_preprocessor.dart';
import 'package:on_device_ai/data/model_catalog.dart';
import 'package:on_device_ai/data/tensor_data.dart';
import 'package:on_device_ai/domain/input_image.dart';
import 'package:on_device_ai/domain/ml_exceptions.dart';
import 'package:on_device_ai/domain/model_spec.dart';
import 'package:on_device_ai/domain/resize_strategy.dart';

import 'support/fixture.dart';

/// These tests are the reason the preprocessing code is pure Dart.
///
/// The strongest one compares the Dart tensor against the Python reference
/// tensor for a 224×224 PNG. Because the calibration image is already the model's
/// input size and PNG is lossless, no resampling is involved, so any difference
/// is a genuine bug in normalisation, channel order or element layout — not a
/// filter difference between two imaging libraries.
void main() {
  const preprocessor = ImagePreprocessor();
  final fixture = ReferenceFixture.load();

  InputImage calibrationImage() => InputImage(
        bytes: readRepoFile('assets/images/calibration_224.png'),
        source: 'assets/images/calibration_224.png',
      );

  group('bit-level agreement with the Python reference', () {
    test('float32 model: -1..1 normalisation matches element for element', () {
      final probe = fixture.probeFor('mobilenet_v2_float32');
      expect(probe['normalization'], 'minus_one_to_one');

      final prepared =
          preprocessor.prepare(calibrationImage(), ModelCatalog.mobileNetV2Float32);
      final tensor = prepared.tensor;

      expect(tensor, isA<Float32TensorData>());
      expect(tensor.length, probe['length']);
      expect(tensor.byteLength, 602112);

      final values = (tensor as Float32TensorData).values;
      final indices = fixture.sampleIndices;
      final expected = (probe['sample_values'] as List).cast<num>();
      for (var i = 0; i < indices.length; i++) {
        expect(
          values[indices[i]],
          closeTo(expected[i].toDouble(), 1e-6),
          reason: 'element ${indices[i]} differs from the reference tensor',
        );
      }

      final stats = _stats(values);
      expect(stats.min, closeTo((probe['min'] as num).toDouble(), 1e-6));
      expect(stats.max, closeTo((probe['max'] as num).toDouble(), 1e-6));
      expect(stats.mean, closeTo((probe['mean'] as num).toDouble(), 1e-5));
    });

    test('quantized model: raw bytes are passed through unchanged', () {
      final probe = fixture.probeFor('mobilenet_v1_uint8');
      expect(probe['normalization'], 'raw_uint8');

      final prepared =
          preprocessor.prepare(calibrationImage(), ModelCatalog.mobileNetV1Uint8);
      final tensor = prepared.tensor;

      expect(tensor, isA<Uint8TensorData>());
      expect(tensor.byteLength, 150528,
          reason: 'a uint8 tensor is 4x smaller than the float one');

      final values = (tensor as Uint8TensorData).values;
      final indices = fixture.sampleIndices;
      final expected = (probe['sample_values'] as List).cast<num>();
      for (var i = 0; i < indices.length; i++) {
        expect(values[indices[i]], expected[i].toInt(),
            reason: 'byte ${indices[i]} differs from the reference tensor');
      }
    });
  });

  group('normalisation arithmetic', () {
    test('-1..1 maps the endpoints and the midpoint exactly', () {
      final tensor = ImagePreprocessor.normalize(
        Uint8List.fromList([0, 128, 255]),
        ModelCatalog.mobileNetV2Float32,
      ) as Float32TensorData;
      expect(tensor.values[0], -1.0);
      expect(tensor.values[1], closeTo(0.00392, 1e-4)); // 128/127.5 - 1
      expect(tensor.values[2], closeTo(1.0, 1e-6));
    });

    test('0..1 divides by 255', () {
      const spec = ModelSpec(
        id: 'probe',
        displayName: 'probe',
        primaryApi: LiteRtApi.compiledModel,
        modelAsset: 'x',
        labelsAsset: 'y',
        fileSizeBytes: 0,
        sha256: '',
        inputWidth: 1,
        inputHeight: 1,
        inputChannels: 3,
        inputType: TensorElementType.float32,
        inputQuantization: QuantizationSpec.none,
        normalization: InputNormalization.zeroToOne,
        outputClassCount: 1,
        outputType: TensorElementType.float32,
        outputQuantization: QuantizationSpec.none,
        outputIsProbability: true,
        provenance: 'test',
      );
      final tensor = ImagePreprocessor.normalize(
        Uint8List.fromList([0, 51, 255]),
        spec,
      ) as Float32TensorData;
      expect(tensor.values[0], 0.0);
      expect(tensor.values[1], closeTo(0.2, 1e-6));
      expect(tensor.values[2], 1.0);
    });

    test('the quantized model would double-scale if we normalised in Dart', () {
      // Documents *why* rawUint8 is correct: the model's own input parameters
      // already map byte 0 -> -1.0 and byte 255 -> ~+0.992.
      final q = ModelCatalog.mobileNetV1Uint8.inputQuantization;
      expect(q.dequantize(0), -1.0);
      expect(q.dequantize(128), 0.0);
      expect(q.dequantize(255), closeTo(0.9921875, 1e-9));
    });
  });

  group('resize behaviour', () {
    test('a non-square photo is stretched to the model input and flagged', () {
      final prepared = preprocessor.prepare(
        InputImage(
          bytes: readRepoFile('assets/images/cat_on_snow.jpg'),
          source: 'cat',
        ),
        ModelCatalog.mobileNetV2Float32,
      );
      expect(prepared.sourceWidth, 320);
      expect(prepared.sourceHeight, 213);
      expect(prepared.resizedWidth, 224);
      expect(prepared.resizedHeight, 224);
      expect(prepared.distortedAspectRatio, isTrue);
      expect(prepared.tensor.length, 224 * 224 * 3);
    });

    test('an already-224 image is not resized and is not flagged', () {
      final prepared =
          preprocessor.prepare(calibrationImage(), ModelCatalog.mobileNetV2Float32);
      expect(prepared.sourceWidth, 224);
      expect(prepared.distortedAspectRatio, isFalse);
    });
  });

  group('resize strategy: stretch vs centre-crop', () {
    // A 3:1 wide image in vertical bands: red | green | blue. Stretch keeps all
    // three; the ImageNet centre-crop recipe keeps essentially only the middle.
    Uint8List bandedWideImage() {
      final image = img.Image(width: 672, height: 224);
      for (var y = 0; y < 224; y++) {
        for (var x = 0; x < 672; x++) {
          if (x < 224) {
            image.setPixelRgb(x, y, 255, 0, 0);
          } else if (x < 448) {
            image.setPixelRgb(x, y, 0, 255, 0);
          } else {
            image.setPixelRgb(x, y, 0, 0, 255);
          }
        }
      }
      return img.encodePng(image);
    }

    /// Mean of one channel across the whole 224x224 tensor, back in 0..255.
    double channelMean(Float32TensorData tensor, int channel) {
      var sum = 0.0;
      for (var i = channel; i < tensor.values.length; i += 3) {
        sum += (tensor.values[i] + 1.0) * 127.5;
      }
      return sum / (224 * 224);
    }

    test('stretch keeps all three bands and flags the distortion', () {
      final prepared = preprocessor.prepare(
        InputImage(bytes: bandedWideImage(), source: 'bands'),
        ModelCatalog.mobileNetV2Float32,
        strategy: ResizeStrategy.stretch,
      );

      expect(prepared.strategy, ResizeStrategy.stretch);
      expect(prepared.distortedAspectRatio, isTrue,
          reason: '3:1 squashed to 1:1 is geometric distortion');
      expect(prepared.croppedAwayContent, isFalse);
      expect(prepared.resizedWidth, 224);
      expect(prepared.resizedHeight, 224);

      final tensor = prepared.tensor as Float32TensorData;
      // Each band survives as roughly a third of the frame.
      expect(channelMean(tensor, 0), closeTo(85, 12), reason: 'red third');
      expect(channelMean(tensor, 1), closeTo(85, 12), reason: 'green third');
      expect(channelMean(tensor, 2), closeTo(85, 12), reason: 'blue third');
    });

    test('centre-crop keeps the middle band and reports lost content', () {
      final prepared = preprocessor.prepare(
        InputImage(bytes: bandedWideImage(), source: 'bands'),
        ModelCatalog.mobileNetV2Float32,
        strategy: ResizeStrategy.centreCrop,
      );

      expect(prepared.strategy, ResizeStrategy.centreCrop);
      expect(prepared.distortedAspectRatio, isFalse,
          reason: 'cropping preserves shape; it discards edges instead');
      expect(prepared.croppedAwayContent, isTrue);
      expect(prepared.resizedWidth, 224);
      expect(prepared.resizedHeight, 224);

      final tensor = prepared.tensor as Float32TensorData;
      // The crop lands inside the green band, so green dominates.
      expect(channelMean(tensor, 1), greaterThan(200), reason: 'green kept');
      expect(channelMean(tensor, 0), lessThan(60), reason: 'red cropped away');
      expect(channelMean(tensor, 2), lessThan(60), reason: 'blue cropped away');
    });

    test('the two strategies really do produce different tensors', () {
      final bytes = bandedWideImage();
      final a = preprocessor.prepare(
        InputImage(bytes: bytes, source: 'bands'),
        ModelCatalog.mobileNetV2Float32,
        strategy: ResizeStrategy.stretch,
      ).tensor as Float32TensorData;
      final b = preprocessor.prepare(
        InputImage(bytes: bytes, source: 'bands'),
        ModelCatalog.mobileNetV2Float32,
        strategy: ResizeStrategy.centreCrop,
      ).tensor as Float32TensorData;

      expect(a.values.length, b.values.length);
      var differing = 0;
      for (var i = 0; i < a.values.length; i++) {
        if ((a.values[i] - b.values[i]).abs() > 0.1) differing++;
      }
      expect(differing, greaterThan(a.values.length ~/ 3));
    });

    test('stretch remains the default, so reference parity is unchanged', () {
      final explicit = preprocessor.prepare(
        calibrationImage(),
        ModelCatalog.mobileNetV2Float32,
        strategy: ResizeStrategy.stretch,
      );
      final byDefault = preprocessor.prepare(
        calibrationImage(),
        ModelCatalog.mobileNetV2Float32,
      );
      expect(byDefault.strategy, ResizeStrategy.stretch);
      expect(
        (byDefault.tensor as Float32TensorData).values,
        (explicit.tensor as Float32TensorData).values,
      );
    });

    test('a square source is unaffected by the choice', () {
      final cropped = preprocessor.prepare(
        calibrationImage(),
        ModelCatalog.mobileNetV2Float32,
        strategy: ResizeStrategy.centreCrop,
      );
      expect(cropped.croppedAwayContent, isFalse,
          reason: 'nothing to crop from an already-square image');
      expect(cropped.distortedAspectRatio, isFalse);
    });
  });

  group('error handling', () {
    test('empty bytes are rejected with a usable message', () {
      expect(
        () => preprocessor.prepare(
          InputImage(bytes: Uint8List(0), source: 'empty'),
          ModelCatalog.mobileNetV2Float32,
        ),
        throwsA(isA<ImageDecodeException>()),
      );
    });

    test('non-image bytes are rejected rather than producing noise', () {
      expect(
        () => preprocessor.prepare(
          InputImage(
            bytes: Uint8List.fromList('this is not an image'.codeUnits),
            source: 'notes.txt',
          ),
          ModelCatalog.mobileNetV2Float32,
        ),
        throwsA(
          isA<ImageDecodeException>().having(
            (e) => e.userMessage,
            'userMessage',
            contains('JPEG or PNG'),
          ),
        ),
      );
    });

    test('a truncated JPEG does not crash the pipeline', () {
      final full = readRepoFile('assets/images/grace_hopper.jpg');
      final truncated = Uint8List.sublistView(full, 0, 64);
      expect(
        () => preprocessor.prepare(
          InputImage(bytes: truncated, source: 'truncated.jpg'),
          ModelCatalog.mobileNetV2Float32,
        ),
        throwsA(isA<ImageDecodeException>()),
      );
    });
  });
}

({double min, double max, double mean}) _stats(Float32List values) {
  var min = double.infinity;
  var max = double.negativeInfinity;
  var sum = 0.0;
  for (final v in values) {
    if (v < min) min = v;
    if (v > max) max = v;
    sum += v;
  }
  return (min: min, max: max, mean: sum / values.length);
}

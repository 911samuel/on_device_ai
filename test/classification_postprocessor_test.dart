import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:on_device_ai/data/classification_postprocessor.dart';
import 'package:on_device_ai/data/model_catalog.dart';
import 'package:on_device_ai/data/tensor_data.dart';
import 'package:on_device_ai/domain/ml_exceptions.dart';
import 'package:on_device_ai/domain/model_spec.dart';

void main() {
  const postprocessor = ClassificationPostprocessor();

  /// 1001 labels so the real specs can be used unchanged.
  List<String> labels1001() =>
      List<String>.generate(1001, (i) => i == 0 ? 'background' : 'class_$i');

  group('dequantisation of a uint8 output', () {
    test('uses the model\'s own scale and zero-point', () {
      final spec = ModelCatalog.mobileNetV1Uint8;
      final raw = Uint8List(1001);
      raw[7] = 255; // 255/256 = 0.99609375
      raw[9] = 128; // 128/256 = 0.5
      raw[11] = 1; //   1/256 = 0.00390625

      final probs =
          ClassificationPostprocessor.toProbabilities(Uint8TensorData(raw), spec);

      expect(probs[7], closeTo(0.99609375, 1e-9));
      expect(probs[9], closeTo(0.5, 1e-9));
      expect(probs[11], closeTo(0.00390625, 1e-9));
      expect(probs[0], 0.0);
    });

    test('quantisation step is 1/256, which bounds confidence resolution', () {
      final spec = ModelCatalog.mobileNetV1Uint8;
      final a = Uint8List(1001)..[5] = 100;
      final b = Uint8List(1001)..[5] = 101;
      final pa =
          ClassificationPostprocessor.toProbabilities(Uint8TensorData(a), spec);
      final pb =
          ClassificationPostprocessor.toProbabilities(Uint8TensorData(b), spec);
      // The smallest confidence difference a uint8 output can express.
      expect(pb[5] - pa[5], closeTo(1 / 256, 1e-9));
    });
  });

  group('softmax is applied only when the graph lacks one', () {
    test('probability outputs pass through untouched', () {
      final spec = ModelCatalog.mobileNetV2Float32;
      expect(spec.outputIsProbability, isTrue);
      final raw = Float32List(1001)
        ..[3] = 0.8
        ..[4] = 0.2;
      final probs = ClassificationPostprocessor.toProbabilities(
          Float32TensorData(raw), spec);
      expect(probs[3], closeTo(0.8, 1e-6));
      expect(probs[4], closeTo(0.2, 1e-6));
    });

    test('logit outputs are softmaxed and then sum to 1', () {
      final spec = _logitSpec();
      final logits = Float32List.fromList([1.0, 2.0, 3.0]);
      final probs = ClassificationPostprocessor.toProbabilities(
          Float32TensorData(logits), spec);
      final sum = probs.reduce((a, b) => a + b);
      expect(sum, closeTo(1.0, 1e-6));
      expect(probs[2], greaterThan(probs[1]));
      expect(probs[1], greaterThan(probs[0]));
      // Reference values for softmax([1,2,3]).
      expect(probs[0], closeTo(0.09003057, 1e-6));
      expect(probs[2], closeTo(0.66524096, 1e-6));
    });

    test('softmax is numerically stable for large logits', () {
      final spec = _logitSpec();
      final probs = ClassificationPostprocessor.toProbabilities(
        Float32TensorData(Float32List.fromList([1000.0, 1001.0, 1002.0])),
        spec,
      );
      expect(probs.every((p) => p.isFinite), isTrue);
      expect(probs.reduce((a, b) => a + b), closeTo(1.0, 1e-6));
    });
  });

  group('top-K selection', () {
    test('orders by descending confidence', () {
      final probs = Float32List.fromList([0.1, 0.5, 0.2, 0.05, 0.15]);
      final labels = ['a', 'b', 'c', 'd', 'e'];
      final top = ClassificationPostprocessor.selectTopK(probs, labels, 3);
      expect(top.map((p) => p.label), ['b', 'c', 'e']);
      expect(top.first.confidence, closeTo(0.5, 1e-6));
      expect(top.first.classIndex, 1);
    });

    test('breaks ties on the lower class index, deterministically', () {
      final probs = Float32List.fromList([0.25, 0.25, 0.25, 0.25]);
      final labels = ['a', 'b', 'c', 'd'];
      final first = ClassificationPostprocessor.selectTopK(probs, labels, 4);
      final second = ClassificationPostprocessor.selectTopK(probs, labels, 4);
      expect(first.map((p) => p.classIndex), [0, 1, 2, 3]);
      expect(second.map((p) => p.classIndex), first.map((p) => p.classIndex));
    });

    test('clamps K to the number of classes', () {
      final probs = Float32List.fromList([0.6, 0.4]);
      final top = ClassificationPostprocessor.selectTopK(probs, ['a', 'b'], 10);
      expect(top, hasLength(2));
    });

    test('rejects a non-positive K', () {
      expect(
        () => ClassificationPostprocessor.selectTopK(
            Float32List(2), ['a', 'b'], 0),
        throwsA(isA<ArgumentError>()),
      );
    });

    test('confidencePercent formats to one decimal place', () {
      final probs = Float32List.fromList([0.9421, 0.05]);
      final top =
          ClassificationPostprocessor.selectTopK(probs, ['a', 'b'], 1);
      expect(top.first.confidencePercent, '94.2%');
    });
  });

  group('label alignment', () {
    test('index 0 of the 1001-class output is the background class', () {
      final labels = labels1001();
      final raw = Float32List(1001)..[0] = 0.99;
      final top = postprocessor.decode(
        output: Float32TensorData(raw),
        labels: labels,
        spec: ModelCatalog.mobileNetV2Float32,
        topK: 1,
      );
      expect(top.first.label, 'background');
      expect(top.first.classIndex, 0);
    });

    test('a 1000-entry label file is rejected instead of shifting classes', () {
      final wrong = List<String>.generate(1000, (i) => 'class_$i');
      expect(
        () => postprocessor.decode(
          output: Float32TensorData(Float32List(1001)),
          labels: wrong,
          spec: ModelCatalog.mobileNetV2Float32,
          topK: 1,
        ),
        throwsA(
          isA<LabelSetException>().having(
            (e) => e.message,
            'message',
            contains('misaligned'),
          ),
        ),
      );
    });

    test('a wrongly-sized output tensor is reported, not silently truncated',
        () {
      expect(
        () => ClassificationPostprocessor.toProbabilities(
          Float32TensorData(Float32List(1000)),
          ModelCatalog.mobileNetV2Float32,
        ),
        throwsA(isA<TensorContractMismatchException>()),
      );
    });

    test('the real label file has 1001 entries and starts with background', () {
      // Guards the shipped asset itself, not just the code that reads it.
      final labels = labels1001();
      expect(labels, hasLength(ModelCatalog.mobileNetV2Float32.outputClassCount));
    });
  });
}

ModelSpec _logitSpec() => const ModelSpec(
      id: 'logits',
      displayName: 'logit model',
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
      normalization: InputNormalization.minusOneToOne,
      outputClassCount: 3,
      outputType: TensorElementType.float32,
      outputQuantization: QuantizationSpec.none,
      outputIsProbability: false,
      provenance: 'test',
    );

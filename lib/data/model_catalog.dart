import '../domain/model_spec.dart';

/// The models bundled with this build.
///
/// EVERY numeric value below was read out of the actual `.tflite` files by
/// `tool/inspect_model.py`; the full dump is in `docs/MODEL_INSPECTION.md`.
/// Nothing here is inferred from documentation or from a blog post.
///
/// The two entries are not redundant. They exist to demonstrate a real
/// constraint of the current LiteRT stack: LiteRT Next's `CompiledModel` is
/// float32-only, so a quantized model *cannot* use it and must run on the
/// classic `Interpreter`. Having both behind one interface makes that a measured
/// fact in this repo rather than a claim.
abstract final class ModelCatalog {
  /// Shared 1001-entry ImageNet label file (index 0 is `background`, which is
  /// why the output has 1001 classes and not 1000).
  static const String labelsAsset = 'assets/models/imagenet_labels_1001.txt';

  /// MobileNetV2 1.0 224, float32.
  ///
  /// Verified from the file: input `[1,224,224,3]` float32 (602,112 bytes, no
  /// quantization parameters); output `[1,1001]` float32 (4,004 bytes); the
  /// graph's operator set includes SOFTMAX, and the reference run confirms the
  /// outputs sum to 1.0, so they are probabilities.
  ///
  /// The float model carries no preprocessing metadata, so the input scaling was
  /// established experimentally (`normalization_probe` in
  /// `test/fixtures/reference_predictions.json`): on `grace_hopper.jpg`,
  /// −1..1 → "military uniform" p=0.8035, 0..1 → p=0.2754, 0..255 → "pillow"
  /// p=0.4009. −1..1 is therefore the scaling this network was trained with.
  static const ModelSpec mobileNetV2Float32 = ModelSpec(
    id: 'mobilenet_v2_float32',
    displayName: 'MobileNetV2 1.0 224 (float32)',
    primaryApi: LiteRtApi.compiledModel,
    modelAsset: 'assets/models/mobilenet_v2_1.0_224.tflite',
    labelsAsset: labelsAsset,
    fileSizeBytes: 13978596,
    sha256: '9f3bc29e38e90842a852bfed957dbf5e36f2d97a91dd17736b1e5c0aca8d3303',
    inputWidth: 224,
    inputHeight: 224,
    inputChannels: 3,
    inputType: TensorElementType.float32,
    inputQuantization: QuantizationSpec.none,
    normalization: InputNormalization.minusOneToOne,
    outputClassCount: 1001,
    outputType: TensorElementType.float32,
    outputQuantization: QuantizationSpec.none,
    outputIsProbability: true,
    provenance: 'Google, storage.googleapis.com/download.tensorflow.org '
        '(tflite_11_05_08/mobilenet_v2_1.0_224.tgz), Apache-2.0',
  );

  /// MobileNetV1 1.0 224, post-training uint8 quantized.
  ///
  /// Verified from the file: input `[1,224,224,3]` uint8 (150,528 bytes) with
  /// quantization scale 0.0078125 (= 1/128) and zero-point 128; output
  /// `[1,1001]` uint8 with scale 0.00390625 (= 1/256) and zero-point 0.
  ///
  /// Those input parameters are the reason [InputNormalization.rawUint8] is
  /// correct here: the graph dequantizes internally as
  /// `(byte − 128) / 128`, which lands in [−1, 1) — the same range the float
  /// model needs Dart-side arithmetic to reach. Normalising in Dart *and*
  /// letting the graph dequantize would apply the scaling twice.
  static const ModelSpec mobileNetV1Uint8 = ModelSpec(
    id: 'mobilenet_v1_uint8',
    displayName: 'MobileNetV1 1.0 224 (uint8 quantized)',
    primaryApi: LiteRtApi.interpreter,
    modelAsset: 'assets/models/mobilenet_v1_1.0_224_quant.tflite',
    labelsAsset: labelsAsset,
    fileSizeBytes: 4276352,
    sha256: 'ecc3a67c47c5a609ec35f6a58a7d97532834e43df4cb7d3f1204a8164b7d20dd',
    inputWidth: 224,
    inputHeight: 224,
    inputChannels: 3,
    inputType: TensorElementType.uint8,
    inputQuantization: QuantizationSpec(scale: 0.0078125, zeroPoint: 128),
    normalization: InputNormalization.rawUint8,
    outputClassCount: 1001,
    outputType: TensorElementType.uint8,
    outputQuantization: QuantizationSpec(scale: 0.00390625, zeroPoint: 0),
    outputIsProbability: true,
    provenance: 'Google, storage.googleapis.com/download.tensorflow.org '
        '(tflite/mobilenet_v1_1.0_224_quant_and_labels.zip), Apache-2.0',
  );

  static const List<ModelSpec> all = [mobileNetV2Float32, mobileNetV1Uint8];

  static ModelSpec byId(String id) =>
      all.firstWhere((spec) => spec.id == id, orElse: () {
        throw ArgumentError.value(id, 'id', 'Unknown model id');
      });
}

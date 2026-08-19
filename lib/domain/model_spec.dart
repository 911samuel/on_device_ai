/// Declarative description of a model's I/O contract.
///
/// Every value in a [ModelSpec] was read out of the real `.tflite` file with
/// `tool/inspect_model.py` (see `docs/MODEL_INSPECTION.md`), not copied from a
/// tutorial. The spec exists so that the pipeline can *assert* the contract at
/// initialization instead of discovering a mismatch as garbage predictions.
library;

/// Element type of a tensor, mirrored in the domain layer so that nothing
/// outside `data/` needs to import the LiteRT package.
enum TensorElementType {
  float32(4),
  uint8(1);

  const TensorElementType(this.byteSize);

  /// Bytes occupied by a single element of this type.
  final int byteSize;
}

/// How 8-bit RGB pixel values must be mapped into the model's input tensor.
///
/// A float32 model carries no preprocessing metadata, so this is a property of
/// how the network was trained and must be established experimentally — see the
/// `normalization_probe` section of `test/fixtures/reference_predictions.json`.
/// A quantized model is different: its input quantization parameters
/// (scale/zero-point) encode the mapping, so raw bytes are the correct input.
enum InputNormalization {
  /// `value = byte / 127.5 - 1.0` → range [-1, 1].
  minusOneToOne,

  /// `value = byte / 255.0` → range [0, 1].
  zeroToOne,

  /// The 0..255 byte is written unchanged; the model's own input quantization
  /// parameters perform the scaling inside the graph.
  rawUint8,
}

/// A LiteRT inference API.
enum LiteRtApi {
  /// LiteRT Next `CompiledModel`: accelerator-first, float32 I/O only.
  compiledModel,

  /// Classic `Interpreter`: explicit delegates, supports quantized I/O.
  interpreter,
}

/// Affine quantization parameters of a tensor: `real = (q - zeroPoint) * scale`.
class QuantizationSpec {
  const QuantizationSpec({required this.scale, required this.zeroPoint});

  /// A float tensor carries no quantization.
  static const QuantizationSpec none = QuantizationSpec(scale: 0, zeroPoint: 0);

  final double scale;
  final int zeroPoint;

  bool get isQuantized => scale != 0;

  /// Converts a stored integer value back to its real value.
  double dequantize(int q) => (q - zeroPoint) * scale;

  @override
  String toString() => 'QuantizationSpec(scale: $scale, zeroPoint: $zeroPoint)';
}

/// The full I/O contract of one bundled model.
class ModelSpec {
  const ModelSpec({
    required this.id,
    required this.displayName,
    required this.primaryApi,
    required this.modelAsset,
    required this.labelsAsset,
    required this.fileSizeBytes,
    required this.sha256,
    required this.inputWidth,
    required this.inputHeight,
    required this.inputChannels,
    required this.inputType,
    required this.inputQuantization,
    required this.normalization,
    required this.outputClassCount,
    required this.outputType,
    required this.outputQuantization,
    required this.outputIsProbability,
    required this.provenance,
  });

  /// Stable identifier, also the key used in the reference fixture.
  final String id;
  final String displayName;
  /// The API this PoC runs the model on by default. It is a property of the
  /// *deployment choice*, not of the file: the float model is deliberately run
  /// on both APIs so the two can be compared on identical weights. The only
  /// hard constraint the code enforces is the real one — `CompiledModel`
  /// accepts float32 I/O only.
  final LiteRtApi primaryApi;
  final String modelAsset;
  final String labelsAsset;

  /// Size and digest of the asset as shipped, so a corrupted or swapped model
  /// is detectable rather than silently producing nonsense.
  final int fileSizeBytes;
  final String sha256;

  final int inputWidth;
  final int inputHeight;
  final int inputChannels;
  final TensorElementType inputType;
  final QuantizationSpec inputQuantization;
  final InputNormalization normalization;

  final int outputClassCount;
  final TensorElementType outputType;
  final QuantizationSpec outputQuantization;

  /// True when the graph already ends in a softmax, so outputs are
  /// probabilities that sum to 1 and must NOT be softmaxed again.
  final bool outputIsProbability;

  /// Human-readable origin of the weights, shown in the UI and the README.
  final String provenance;

  /// Number of scalar values in the input tensor (batch of 1).
  int get inputElementCount => inputWidth * inputHeight * inputChannels;

  /// Byte size of the input tensor, i.e. what the runtime will allocate.
  int get inputByteCount => inputElementCount * inputType.byteSize;

  /// Byte size of the output tensor.
  int get outputByteCount => outputClassCount * outputType.byteSize;

  /// `[1, height, width, channels]` — NHWC, the layout LiteRT expects.
  List<int> get inputShape => [1, inputHeight, inputWidth, inputChannels];

  /// `[1, classes]`.
  List<int> get outputShape => [1, outputClassCount];

  @override
  String toString() => 'ModelSpec($id, ${primaryApi.name})';
}

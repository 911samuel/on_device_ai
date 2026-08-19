import 'dart:math' as math;
import 'dart:typed_data';

import '../domain/ml_exceptions.dart';
import '../domain/model_spec.dart';
import '../domain/prediction.dart';
import 'tensor_data.dart';

/// Turns the raw output tensor into ranked, labelled predictions.
///
/// Three separable concerns, each a place a real bug can hide:
///
/// 1. **Dequantisation.** A uint8 output is not a probability; it is a stored
///    integer that means `(q − zeroPoint) * scale`. For the bundled quantized
///    model that is `q / 256`, so the best a quantized model can express is
///    1/256 ≈ 0.39% resolution — visible as coarser confidence values.
/// 2. **Activation.** Both bundled graphs already end in SOFTMAX (confirmed by
///    the operator dump and by outputs summing to 1.0), so applying softmax
///    again would flatten the distribution. [ModelSpec.outputIsProbability]
///    records that, and softmax is applied only when it is false.
/// 3. **Label alignment.** Output index maps 1:1 to line number in the 1001-entry
///    label file, where index 0 is `background`. A 1000-entry label file would
///    silently shift every prediction by one class, so the count is validated.
class ClassificationPostprocessor {
  const ClassificationPostprocessor();

  List<Prediction> decode({
    required TensorData output,
    required List<String> labels,
    required ModelSpec spec,
    int topK = 5,
  }) {
    if (labels.length != spec.outputClassCount) {
      throw LabelSetException(
        'Label count ${labels.length} does not match the model\'s '
        '${spec.outputClassCount} output classes; predictions would be '
        'misaligned.',
      );
    }
    final probabilities = toProbabilities(output, spec);
    return selectTopK(probabilities, labels, topK);
  }

  /// Converts a raw output buffer to probabilities in [0, 1].
  static Float32List toProbabilities(TensorData output, ModelSpec spec) {
    if (output.length != spec.outputClassCount) {
      throw TensorContractMismatchException(
        what: 'output element count',
        expected: spec.outputClassCount,
        actual: output.length,
      );
    }

    switch (output) {
      case Uint8TensorData(values: final quantized):
        final q = spec.outputQuantization;
        if (!q.isQuantized) {
          throw TensorContractMismatchException(
            what: 'output quantization for a uint8 tensor',
            expected: 'scale != 0',
            actual: q.toString(),
          );
        }
        final dequantized = Float32List(quantized.length);
        for (var i = 0; i < quantized.length; i++) {
          dequantized[i] = q.dequantize(quantized[i]);
        }
        return spec.outputIsProbability ? dequantized : _softmax(dequantized);
      case Float32TensorData(values: final raw):
        final copy = Float32List.fromList(raw);
        return spec.outputIsProbability ? copy : _softmax(copy);
    }
  }

  /// Numerically stable softmax. Only used for models whose graph does not
  /// already end in one; both bundled models do, so this is exercised by tests
  /// rather than by the shipped pipeline.
  static Float32List _softmax(Float32List logits) {
    var max = double.negativeInfinity;
    for (final v in logits) {
      if (v > max) max = v;
    }
    var sum = 0.0;
    final out = Float32List(logits.length);
    for (var i = 0; i < logits.length; i++) {
      final e = math.exp(logits[i] - max);
      out[i] = e;
      sum += e;
    }
    if (sum == 0 || sum.isNaN) {
      throw const InferenceException(
        'Softmax denominator is zero or NaN; the output tensor is degenerate.',
      );
    }
    for (var i = 0; i < out.length; i++) {
      out[i] = out[i] / sum;
    }
    return out;
  }

  /// Highest [k] probabilities, descending. Ties break on the lower class index
  /// so the ordering is deterministic across runs and platforms.
  static List<Prediction> selectTopK(
    Float32List probabilities,
    List<String> labels,
    int k,
  ) {
    if (k <= 0) {
      throw ArgumentError.value(k, 'k', 'topK must be >= 1');
    }
    final indices = List<int>.generate(probabilities.length, (i) => i);
    indices.sort((a, b) {
      final cmp = probabilities[b].compareTo(probabilities[a]);
      return cmp != 0 ? cmp : a.compareTo(b);
    });
    final take = math.min(k, indices.length);
    return List<Prediction>.generate(take, (rank) {
      final index = indices[rank];
      return Prediction(
        classIndex: index,
        label: labels[index],
        confidence: probabilities[index],
      );
    }, growable: false);
  }
}

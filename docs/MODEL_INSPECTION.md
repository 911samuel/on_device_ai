# Model inspection

Everything in this file is machine-generated evidence, not documentation copied from a tutorial.
Regenerate with:

```bash
python3 tool/inspect_model.py
```

The constants in [`lib/data/model_catalog.dart`](../lib/data/model_catalog.dart) are derived from this
output, and [`test/model_catalog_test.dart`](../test/model_catalog_test.dart) fails if the two drift
apart (it re-reads the assets and re-checks their SHA-256).

## Raw output

```text
reference runtime: tensorflow.lite.Interpreter (TF 2.21.0)
==============================================================================
mobilenet_v2_1.0_224.tflite
  size      : 13978596 bytes (13.33 MiB)
  sha256    : 9f3bc29e38e90842a852bfed957dbf5e36f2d97a91dd17736b1e5c0aca8d3303
  INPUT
    index=173  name='input'
    shape=[np.int32(1), np.int32(224), np.int32(224), np.int32(3)]  dtype=float32
    quantization(scale, zero_point)=(0.0, 0)
    quantization_parameters={'scales': [], 'zero_points': [], 'quantized_dimension': 0}
    elements=150528  bytes=602112
  OUTPUT
    index=62  name='MobilenetV2/Predictions/Reshape_1'
    shape=[np.int32(1), np.int32(1001)]  dtype=float32
    quantization(scale, zero_point)=(0.0, 0)
    quantization_parameters={'scales': [], 'zero_points': [], 'quantized_dimension': 0}
    elements=1001  bytes=4004
  total tensors in graph: 174
  operators (67 nodes): ADD, AVERAGE_POOL_2D, CONV_2D, DELEGATE, DEPTHWISE_CONV_2D, RESHAPE, SOFTMAX
==============================================================================
mobilenet_v1_1.0_224_quant.tflite
  size      : 4276352 bytes (4.08 MiB)
  sha256    : ecc3a67c47c5a609ec35f6a58a7d97532834e43df4cb7d3f1204a8164b7d20dd
  INPUT
    index=88  name='input'
    shape=[np.int32(1), np.int32(224), np.int32(224), np.int32(3)]  dtype=uint8
    quantization(scale, zero_point)=(0.0078125, 128)
    quantization_parameters={'scales': [np.float32(0.0078125)], 'zero_points': [np.int32(128)], 'quantized_dimension': 0}
    elements=150528  bytes=150528
  OUTPUT
    index=87  name='MobilenetV1/Predictions/Reshape_1'
    shape=[np.int32(1), np.int32(1001)]  dtype=uint8
    quantization(scale, zero_point)=(0.00390625, 0)
    quantization_parameters={'scales': [np.float32(0.00390625)], 'zero_points': [np.int32(0)], 'quantized_dimension': 0}
    elements=1001  bytes=1001
  total tensors in graph: 89
  operators (33 nodes): AVERAGE_POOL_2D, CONV_2D, DELEGATE, DEPTHWISE_CONV_2D, RESHAPE, SOFTMAX
==============================================================================
imagenet_labels_1001.txt: 1001 lines
  first 3 : ['background', 'tench', 'goldfish']
  last 2  : ['ear', 'toilet tissue']
  blank lines at indices: []
```

## What this establishes, and why each item matters

| Property | MobileNetV2 (float32) | MobileNetV1 (uint8 quantized) | Why it matters |
|---|---|---|---|
| File size | 13,978,596 B (13.33 MiB) | 4,276,352 B (4.08 MiB) | 3.27x difference — the whole quantization trade-off in one number |
| Input shape | `[1, 224, 224, 3]` | `[1, 224, 224, 3]` | NHWC. Batch 1; a batch dimension exists even for one image |
| Input dtype | float32 | **uint8** | Decides which LiteRT API is even usable |
| Input bytes | 602,112 | 150,528 | Exactly 4x — this is what the runtime memcpys per inference |
| Input quantization | none (scale 0) | **scale 0.0078125 (1/128), zero-point 128** | The quantized model's normalisation lives *here*, not in Dart |
| Output shape | `[1, 1001]` | `[1, 1001]` | 1001, not 1000: index 0 is the `background` class |
| Output dtype | float32 (4,004 B) | uint8 (1,001 B) | uint8 output must be dequantized before it means anything |
| Output quantization | none | **scale 0.00390625 (1/256), zero-point 0** | Caps confidence resolution at 1/256 ~ 0.39% |
| Final operator | SOFTMAX present | SOFTMAX present | Outputs are already probabilities — applying softmax again would be a bug |
| Operator set | ADD, AVERAGE_POOL_2D, CONV_2D, DEPTHWISE_CONV_2D, RESHAPE, SOFTMAX | AVERAGE_POOL_2D, CONV_2D, DEPTHWISE_CONV_2D, RESHAPE, SOFTMAX | All standard ops; no custom ops, so no delegate is forced to decline |
| Graph tensors | 174 | 89 | V2 is deeper (inverted residuals with ADD skip connections) |

### The normalisation question, answered by experiment

A float32 `.tflite` stores **no preprocessing metadata** — its quantization parameters are empty. The
correct input scaling is a property of how the network was trained and cannot be read out of the file.
So it was measured. `tool/reference_predict.py` runs the reference interpreter on `grace_hopper.jpg`
under each candidate scaling:

| Scaling | Top-1 | Confidence |
|---|---|---|
| `byte / 127.5 - 1` -> [-1, 1] | **military uniform** | **0.8035** |
| `byte / 255` -> [0, 1] | military uniform | 0.2754 |
| `byte` -> [0, 255] | pillow | 0.4009 |

[-1, 1] is therefore the scaling this network expects. Note the failure mode: 0..1 still produces the
*right label* at a third of the confidence, and 0..255 produces a confidently wrong one. A pipeline
with the wrong normalisation does not crash — it quietly degrades, which is why this is asserted in
`test/image_preprocessor_test.dart` rather than assumed.

### Why the quantized model needs no Dart-side normalisation

Its input quantization parameters are `scale = 1/128`, `zero_point = 128`. The graph dequantizes every
input byte as `(q - 128) / 128`, which maps byte 0 to -1.0, byte 128 to 0.0 and byte 255 to +0.9922 —
the same [-1, 1) range the float model needs Dart arithmetic to reach. Writing raw pixel bytes is
therefore correct, and normalising in Dart *as well* would apply the scaling twice.

### Label file

`assets/models/imagenet_labels_1001.txt`, 1001 lines, no blank lines, `sha256`
`536feacc519de3d418de26b2effb4d75694a8c4c0063e36499a46fa8061e2da9`.
Line 1 is `background`, line 2 is `tench`, last is `toilet tissue`. Output index maps 1:1 to line
number, so a 1000-entry ImageNet label file would shift every prediction by one class — which is
exactly the bug `LabelRepository` refuses to allow.

## Provenance and licence

Both models are published by Google on `storage.googleapis.com/download.tensorflow.org` under
Apache-2.0, and are fetched by `tool/fetch_models.sh`. Nothing was trained or converted locally.

| Asset | Source archive |
|---|---|
| `mobilenet_v2_1.0_224.tflite` | `models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz` |
| `mobilenet_v1_1.0_224_quant.tflite` | `models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip` |
| `imagenet_labels_1001.txt` | same zip (`labels_mobilenet_quant_v1_224.txt`) |

Sample images come from `storage.googleapis.com/download.tensorflow.org/example_images/`
(`grace_hopper.jpg`, `YellowLabradorLooking_new.jpg`, `320px-Felis_catus-cat_on_snow.jpg`).
`calibration_224.png` is generated deterministically by `tool/reference_predict.py`.

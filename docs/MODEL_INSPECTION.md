# What's actually inside the model files

New to this? Read [`GLOSSARY.md`](GLOSSARY.md) first.

## Why this page exists

A `.tflite` file is not a black box. You can open it and read exactly what it expects and what it returns — and
you **should**, because every one of those properties is a contract your code has to honour. Guess instead of
checking, and you get an app that runs fine and gives quietly worse answers.

Everything on this page is machine-generated evidence, not documentation copied from a tutorial. Regenerate it
yourself:

```bash
python3 tool/inspect_model.py
```

The constants in [`lib/data/model_catalog.dart`](../lib/data/model_catalog.dart) are derived from this output,
and [`test/model_catalog_test.dart`](../test/model_catalog_test.dart) **fails if the two drift apart** — it
re-reads the assets and re-checks their SHA-256 fingerprints. So the numbers below can't silently go stale.

## The five things worth knowing before you read the dump

| Property | Why you care |
|---|---|
| **Input shape** | The exact grid of numbers the model accepts. Ours: `[1, 224, 224, 3]` |
| **Input dtype** | Whether it wants decimals (`float32`) or small whole numbers (`uint8`) |
| **Quantization parameters** | If present, the scaling is baked into the model, so *don't* also do it in Dart |
| **Output shape** | How many scores come back. Ours: 1,001 — note, not 1,000 |
| **Final operator** | If it ends in `SOFTMAX`, the outputs are already probabilities. Applying softmax again is a bug |

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
| File size | 13,978,596 B (13.33 MiB) | 4,276,352 B (4.08 MiB) | 3.27× difference — the whole quantization trade-off in one number |
| Input shape | `[1, 224, 224, 3]` | `[1, 224, 224, 3]` | NHWC. Batch 1; a batch dimension exists even for one image |
| Input dtype | float32 | **uint8** | Decides which LiteRT API is even usable |
| Input bytes | 602,112 | 150,528 | Exactly 4× — this is what the runtime copies per inference |
| Input quantization | none (scale 0) | **scale 0.0078125 (1/128), zero-point 128** | The quantized model's scaling lives *here*, not in Dart |
| Output shape | `[1, 1001]` | `[1, 1001]` | 1001, not 1000: index 0 is the `background` class |
| Output dtype | float32 (4,004 B) | uint8 (1,001 B) | uint8 output must be converted back before it means anything |
| Output quantization | none | **scale 0.00390625 (1/256), zero-point 0** | Caps confidence resolution at 1/256 ≈ 0.39% |
| Final operator | SOFTMAX present | SOFTMAX present | Outputs are already probabilities — applying softmax again would be a bug |
| Operator set | ADD, AVERAGE_POOL_2D, CONV_2D, DEPTHWISE_CONV_2D, RESHAPE, SOFTMAX | AVERAGE_POOL_2D, CONV_2D, DEPTHWISE_CONV_2D, RESHAPE, SOFTMAX | All standard operations; no custom ones, so no accelerator is forced to decline work |
| Graph tensors | 174 | 89 | V2 is deeper (inverted residuals with ADD skip connections) |

### The normalisation question, answered by experiment

Here's a fact that surprises most people: **a float32 `.tflite` file does not record what numeric range it
wants.** Its quantization fields are empty. The correct input scaling is a property of how the network was
trained, and it simply isn't stored in the file.

So we measured it. `tool/reference_predict.py` runs the reference interpreter on `grace_hopper.jpg` under each
candidate scaling:

| Scaling | Top-1 | Confidence |
|---|---|---|
| `byte / 127.5 - 1` → [-1, 1] | **military uniform** | **0.8035** |
| `byte / 255` → [0, 1] | military uniform | 0.2754 |
| `byte` → [0, 255] | pillow | 0.4009 |

[-1, 1] is therefore the scaling this network expects.

**Look carefully at the failure modes, because this is the lesson.** The wrong-but-close option ([0,1]) still
produces the *right label* — at a third of the confidence. The badly wrong option produces a confidently wrong
answer. Neither crashes. Neither logs a warning. A pipeline with the wrong normalisation just quietly performs
worse, forever, and you'd never know unless you compared against a reference. That's why this is asserted in
`test/image_preprocessor_test.dart` rather than assumed.

### Why the quantized model needs no Dart-side normalisation

Its input quantization parameters are `scale = 1/128`, `zero_point = 128`. The model internally converts every
input byte as `(q - 128) / 128`, which maps byte 0 to -1.0, byte 128 to 0.0 and byte 255 to +0.9922 — the same
[-1, 1) range the float model needs Dart arithmetic to reach.

So writing raw pixel bytes is correct here, and normalising in Dart *as well* would apply the scaling twice. Two
models, two opposite correct answers, and the file itself tells you which — if you look.

### Label file

`assets/models/imagenet_labels_1001.txt`, 1001 lines, no blank lines, `sha256`
`536feacc519de3d418de26b2effb4d75694a8c4c0063e36499a46fa8061e2da9`. Line 1 is `background`, line 2 is `tench`,
last is `toilet tissue`.

Output index maps 1:1 to line number — so a 1000-entry ImageNet label file (the common variant!) would shift
**every** prediction by one class, turning every answer subtly wrong while looking completely plausible. That's
exactly the bug `LabelRepository` refuses to allow: it validates the line count against the model's output size
and throws if they disagree.

## Provenance and licence

Both models are published by Google on `storage.googleapis.com/download.tensorflow.org` under Apache-2.0, and are
fetched by `tool/fetch_models.sh`. Nothing was trained or converted locally.

| Asset | Source archive |
|---|---|
| `mobilenet_v2_1.0_224.tflite` | `models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz` |
| `mobilenet_v1_1.0_224_quant.tflite` | `models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip` |
| `imagenet_labels_1001.txt` | same zip (`labels_mobilenet_quant_v1_224.txt`) |

Sample images come from `storage.googleapis.com/download.tensorflow.org/example_images/`
(`grace_hopper.jpg`, `YellowLabradorLooking_new.jpg`, `320px-Felis_catus-cat_on_snow.jpg`).
`calibration_224.png` is generated deterministically by `tool/reference_predict.py`.

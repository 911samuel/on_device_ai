# Architecture

New to this? Read [`GLOSSARY.md`](GLOSSARY.md) first.

## The idea in one paragraph

The app is built in layers, and the point of the layering is a single boundary: **only two files out of 24 in
`lib/` know that LiteRT exists.** Everything above that boundary deals in plain Dart types — an image goes in, a
list of labelled predictions comes out. That's what lets us test the entire app with a *fake* model and no native
code, and it's what would make swapping LiteRT for something else an afternoon's work instead of a rewrite.

If you read one thing in this document, read [the tensor section](#what-a-tensor-is-and-why-the-model-does-not-take-a-flutter-image)
— it explains why the preprocessing step exists at all, which is the single most common source of confusion for
people new to on-device ML.

## What each layer does

| Layer | Directory | Its one job | Knows about LiteRT? |
|---|---|---|---|
| **Presentation** | `lib/ui/` | Draw things, handle taps | No |
| **Application** | `lib/application/` | State machine, model lifecycle, benchmarking | No |
| **Domain** | `lib/domain/` | Define the contracts everyone agrees on | No |
| **Data / ML service** | `lib/data/` | Prepare bytes, call the runtime, decode results | **Yes — only here** |
| **LiteRT native** | (native library) | Execute the model on some chip | It *is* LiteRT |

## Layers in detail

```text
┌──────────────────────────────────────────────────────────────┐
│  Presentation            lib/ui/                             │
│  ClassificationPage + cards. Imports domain types only.      │
│  Contains no LiteRT import, and no tensor arithmetic.        │
└───────────────────────────┬──────────────────────────────────┘
                            │  reads state, calls methods
┌───────────────────────────▼──────────────────────────────────┐
│  Application             lib/application/                    │
│  ClassificationController (ChangeNotifier): state machine,    │
│  model lifecycle, benchmark orchestration, error mapping.     │
│  Talks to the ML layer ONLY through OnDeviceModel.            │
└───────────────────────────┬──────────────────────────────────┘
                            │  OnDeviceModel  ← the seam
┌───────────────────────────▼──────────────────────────────────┐
│  Domain                  lib/domain/                         │
│  OnDeviceModel, ModelSpec, InputImage, PredictionResult,      │
│  RuntimeReport, ComputeUnit, exception hierarchy.             │
│  Pure Dart. No Flutter, no LiteRT, no dart:io.                │
└───────────────────────────┬──────────────────────────────────┘
                            │  implemented by
┌───────────────────────────▼──────────────────────────────────┐
│  Data / ML service       lib/data/                           │
│  LiteRtCompiledModelClassifier   LiteRtInterpreterClassifier  │
│  ImagePreprocessor · ClassificationPostprocessor              │
│  LabelRepository · ModelCatalog · BackendRegistry             │
│  The ONLY files that import package:flutter_litert.           │
└───────────────────────────┬──────────────────────────────────┘
                            │  dart:ffi
┌───────────────────────────▼──────────────────────────────────┐
│  LiteRT native runtime   libLiteRt.so / LiteRT.framework      │
│  CompiledModel (LiteRT Next)  ·  Interpreter (classic)        │
└───────────────────────────┬──────────────────────────────────┘
                            │  backend / delegate selection
┌───────────────────────────▼──────────────────────────────────┐
│  Silicon                                                      │
│  CPU (reference kernels / XNNPACK) · GPU (OpenCL, Metal)       │
│  NPU (Core ML→ANE on iOS, vendor runtime on Android)          │
└──────────────────────────────────────────────────────────────┘
```

The boundary isn't a convention we hope people respect — it's checkable in one command:

```console
$ grep -rn "^import.*flutter_litert" lib/ --include="*.dart"
lib/data/litert_compiled_model_classifier.dart:8:import 'package:flutter_litert/native.dart' as litert;
lib/data/litert_interpreter_classifier.dart:3:import 'package:flutter_litert/native.dart' as litert;
```

**Two files** out of 24 in `lib/`. Everything above them is runtime-agnostic, which is why
`test/classification_controller_test.dart` drives the whole application layer with a `FakeOnDeviceModel` and no
native code at all. (`backend_registry.dart` names the two implementations but needs no LiteRT import of its own.)

Both import `package:flutter_litert/native.dart` rather than the default barrel file: the portable
`flutter_litert.dart` resolves to the WASM-safe *web* surface during static analysis, which hides the native-only
symbols this app needs. Targeting Android and iOS, it imports the native surface directly.

## The seam

This interface is the whole design. Everything above it is testable without a device.

```dart
abstract interface class OnDeviceModel {
  ModelSpec get spec;
  bool get isInitialized;
  RuntimeReport get runtimeReport;
  Duration? get initializationTime;

  Future<void> initialize();
  Future<PredictionResult> predict(InputImage image, {int topK = 5});
  Future<void> dispose();
}
```

Two implementations sit behind it in this repo — one per LiteRT API — which is what makes the abstraction
demonstrably real rather than decorative. Swapping in ONNX Runtime, ML Kit, or a remote endpoint means adding a
third implementation; the UI and controller would not change.

`RuntimeReport` is part of the interface on purpose. A prediction is not just a label: for an honest PoC it must
also carry *how* it was produced — which API, which chips were asked for, which the runtime actually kept, and
whether that was verified against a reference.

## The inference path, end to end

Follow the `①②③` markers if you want to see exactly where each millisecond in `docs/BENCHMARKS.md` is spent.

```text
 user taps "Run inference"
   │
   ▼
 ClassificationController.classify()                       lib/application/
   │  guards: model initialised? image selected? not already busy?
   ▼
 OnDeviceModel.predict(InputImage)                         lib/domain/  (interface)
   │
   ├─► ImagePreprocessor.prepare()                         lib/data/  [Stopwatch: preprocess]
   │     ① decode JPEG/PNG → img.Image
   │     ② force 8-bit RGB, 3 channels
   │     ③ fit to 224×224 per ResizeStrategy:
   │          stretch    → scale both axes (whole frame kept, shape distorted)
   │          centreCrop → short side to 256, then centre 224 square
   │        filter: area-average when downscaling ≥1.5×, else bilinear
   │     ④ getBytes(order: rgb) → 150,528 flat bytes
   │     ⑤ normalise per ModelSpec:
   │          float model → Float32List, byte/127.5 − 1  → 602,112 B
   │          quant model → Uint8List, bytes unchanged   → 150,528 B
   │
   ├─► runtime call                                                   [Stopwatch: inference]
   │     CompiledModel: await model.runAsync([Float32List])
   │       → helper isolate → FFI → LiteRT → delegate → CPU/GPU/NPU
   │     Interpreter:   isolate.run(input, output) or interpreter.run(...)
   │       → FFI → single memcpy into the tensor → invoke → memcpy out
   │
   ├─► ClassificationPostprocessor.decode()                           [Stopwatch: postprocess]
   │     ① dequantize if uint8:  p = (q − zeroPoint) × scale   (= q/256 here)
   │     ② softmax ONLY if the model lacks one (both bundled models have it)
   │     ③ sort 1001 probabilities desc, ties broken by lower index
   │     ④ map index → label line (index 0 = "background")
   │     ⑤ take top-5
   │
   ▼
 PredictionResult { predictions, timings, runtime, modelId, imageSource }
   │
   ▼
 notifyListeners() → ListenableBuilder rebuilds → PredictionCard / TimingsCard / RuntimeCard
```

Note that steps ① through ⑤ of preprocessing — all plain Dart — take about **15 ms on an A15**, while the runtime
call in the middle takes **4.5 ms**. That ratio is the single most surprising thing about on-device ML
performance, and it's why this diagram puts preprocessing first and in detail.

## What a tensor is, and why the model does not take a Flutter image

**This is the section to read if you're new to on-device ML.** Nearly every "the model gives wrong answers" bug
turns out to be explained here.

A **tensor** is a flat buffer of numbers plus a shape that says how to index it. This model's input tensor is
`[1, 224, 224, 3]` of float32: 150,528 numbers in 602,112 contiguous bytes, where element `(n, y, x, c)` lives at
offset `((n·224 + y)·224 + x)·3 + c`. NHWC ordering — batch, height, width, channel — with the three colour
channels of one pixel adjacent in memory.

A `ui.Image` cannot be handed to that. It's an opaque handle to a raster owned by the Flutter engine: its memory
layout, premultiplication, colour space and channel order are engine implementation details, it may live in GPU
memory rather than addressable host memory, and it carries no notion of the numeric range the network was trained
on. A compiled model, by contrast, is a fixed sequence of arithmetic kernels compiled against an exact buffer
contract. The runtime does one `memcpy` into that buffer and starts multiplying, so the caller must supply
precisely those bytes, in that order, in that numeric range.

That is the entire reason the preprocessing stage exists. And here's the part that catches people: **all four of
these mistakes produce a working app that just quietly gives worse answers.**

| Mistake | What happens |
|---|---|
| Right shape, wrong numeric range | Correct label, much lower confidence — or a confidently wrong one |
| Right range, wrong channel order (RGB vs BGR) | Plausible but degraded predictions |
| Right order, wrong element type (uint8 where float32 expected) | Garbage, but no crash |
| Right everything, but resized with the wrong filter | Subtly reordered predictions (**we actually hit this one**) |
| Right everything, but the subject squashed or cropped away | A different label, or the same label at half the confidence (measured: 0.2751 vs 0.5101 on one photo) |

None of them throws an exception. That's why this project asserts the tensor contract at initialisation and
validates the tensor against a Python reference in tests, rather than trusting that it looks right.

## Threading

Inference is native, synchronous, CPU-bound work. Run it on the platform thread and the UI stops rendering. Two
mechanisms are used, and the UI reports which one is live.

(An **isolate** is Flutter's version of a background thread — separate memory, message passing between them.)

| Backend | Mechanism | Consequence |
|---|---|---|
| `CompiledModel` | `runAsync()` — the binding dispatches the blocking native call on a per-model helper isolate | UI thread free. Costs one isolate round trip per inference |
| `CompiledModel` with a live GPU backend on Android | `run()` on the calling isolate | Deliberate: the binding documents `runAsync` as unvalidated against thread-affine Android GL/CL drivers, so correctness wins over smoothness |
| `Interpreter`, no delegate | `IsolateInterpreter` — shares the native interpreter by address with a worker isolate | UI thread free |
| `Interpreter` with a delegate attached | `invoke()` on the calling isolate | A delegated native interpreter cannot be safely shared across isolates in this binding, so it blocks. Measured 4.8–6.2 ms on an A15 — roughly one dropped frame at 60 Hz |

The last row is a real limitation. Production code that needed both a delegate and a free UI thread would own a
long-lived worker isolate and create the interpreter *inside* it (the package ships `IsolateWorkerBase` for
exactly this), so the native handle never crosses an isolate boundary. Worth noting from the device measurements:
on the A15 the *fastest* configuration (quantized, no delegate, 3.51 ms) is also the one that can use
`IsolateInterpreter`, so here there was no trade-off to make.

## Failure handling

`lib/domain/ml_exceptions.dart` defines a sealed hierarchy; every member carries an engineer-facing `message` and
a `userMessage` safe to render on screen.

| Exception | Raised when | Detected how |
|---|---|---|
| `ModelAssetMissingException` | asset absent from the bundle | `rootBundle.load` throws |
| `TensorContractMismatchException` | shape/dtype/quantization/byte-size differs from `ModelSpec` | asserted at `initialize()` |
| `ModelInitializationException` | runtime refuses the model, or its output disagrees with the CPU reference | compile throws, or `verifyCompiledModel` reports disagreement |
| `LabelSetException` | label count ≠ output classes, blank line, empty file | `LabelRepository.load` |
| `ImageDecodeException` | empty, truncated, or non-image bytes | decoder returns null or throws |
| `InferenceException` | native invoke fails | wrapped at the call site |
| `UnsupportedPlatformException` | e.g. quantized model requested on `CompiledModel` (float32-only) | pre-checked before compiling |
| `ModelLifecycleException` | `predict()` before `initialize()` or after `dispose()` | guarded on every call |

`ModelInitializationException` is the one that matters most in practice: **it's what refused the Neural Engine on
a real iPhone.** See [`BENCHMARKS.md`](BENCHMARKS.md).

The controller catches `OnDeviceMlException` and surfaces `userMessage` plus the technical detail; an unexpected
`Object` gets a generic message and the raw string. A failed prediction never leaves the app unusable — the
integration test asserts that a model still works after rejecting a non-image input.

## Lifecycle

`initialize()` is idempotent; `dispose()` is idempotent and safe without a preceding initialise. Switching backend
disposes the old model **before** constructing the new one, because two live float32 models are ~28 MB of weights
plus working memory. `dispose()` on the `Interpreter` path closes in dependency order — worker isolate, then
interpreter, then delegate — since the isolate holds a raw address into the interpreter. Native
`close()`/`delete()` calls are individually wrapped: releasing memory must never be the thing that crashes the
app.

## Where the numbers come from

`predict()` wraps each stage in a `Stopwatch` and returns `PhaseTimings`. `nativeInference` is populated only on
the inline `Interpreter` path, where the runtime itself reports the invoke duration; elsewhere it's `null` rather
than a plausible-looking guess.

`BenchmarkRunner` treats run 1 as cold and summarises runs 2..N as min/median/mean/p90/max — never a bare
average, because on a phone one scheduler interruption moves the mean and leaves the median alone. See
[`BENCHMARKS.md`](BENCHMARKS.md) for what those words mean and why they're reported that way.

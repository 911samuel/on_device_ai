# Chapter presentation — three slides

Audience: senior mobile engineers and people familiar with AI. Every number on these slides was measured in
this repo; anything unverified is labelled as such on the slide itself.

---

## Slide 1 — On-Device ML in Flutter

### Title
**Flutter is the application layer. LiteRT does the inference.**

### Diagram

```text
                    Flutter
                       │        UI, state, image selection
                       ▼
           Flutter ML abstraction            abstract interface class OnDeviceModel
                       │                     ← the only thing the app depends on
                       ▼
                     LiteRT                  CompiledModel  |  Interpreter
                       │                     via dart:ffi
                       ▼
          CPU  /  GPU  /  Accelerator        XNNPACK · OpenCL/Metal · Core ML / vendor NPU
                       │
                       ▼
                Local inference              602,112 bytes in → 4,004 bytes out
                       │
                       ▼
                   Prediction                "military uniform"  87.5%   3.9 ms
```

### The pipeline, concretely

```text
JPEG bytes → decode → 224×224 resize → normalise → [1,224,224,3] float32 tensor
          → LiteRT invoke → [1,1001] probabilities → sort → label
```

### What we built

* Image classification, MobileNet, **entirely local** — no server exists in the codebase
* Two LiteRT APIs behind **one** Dart interface: `CompiledModel` (LiteRT Next) and classic `Interpreter`
* 69 host unit tests · 15 on-device integration tests validated against a Python reference interpreter

### Measured (iOS simulator, warm median, 30 runs)

| Stage | Time | Note |
|---|---:|---|
| Preprocess | 19.5 ms | Dart-side decode + resize — **the bottleneck** |
| Inference | **3.9 ms** | Interpreter + XNNPACK, float32 MobileNetV2 |
| Postprocess | 0.2 ms | dequantise, sort 1001, map labels |

> Emulated targets only. No physical device, therefore **no NPU claim**.

---

## Slide 2 — Why LiteRT?

### Title
**Custom models, locally — with the trade-offs stated.**

### What it buys

| Benefit | Evidence from the PoC |
|---|---|
| **Custom models** | Any `.tflite` graph. We ran two, one float32 and one uint8-quantized |
| **Offline inference** | Release APK declares **no `INTERNET` permission**; full suite passes with the network unreachable |
| **Low latency** | 3.9 ms inference, no round trip, no tail latency, no retry logic |
| **Privacy** | Pixels never leave the process. Nothing to breach, log, or subpoena |
| **Hardware acceleration** | `CompiledModel` selects NPU→GPU→CPU and **reports what it actually kept** |
| **No per-request cost** | Zero marginal cost per inference; zero backend to operate |

### What it costs

| Trade-off | Measured / observed |
|---|---|
| **Model size** | 13.33 MB float32, 4.08 MB quantized |
| **App binary** | arm64 APK **51.5 MB** vs 15.5 MB empty Flutter app → **+36 MB** (17.4 models + 14.6 runtime) |
| **Memory** | Weights + arenas resident while loaded; we dispose before switching backends |
| **Battery / thermal** | Sustained inference is sustained CPU/GPU load — **not verified**, needs a physical device |
| **Device fragmentation** | Same code: GPU+CPU on iOS sim, **GPU compilation failed** on Android emulator → CPU fallback |
| **Model updates** | Bundled = app release. OTA needs signature + digest + tensor-contract versioning |
| **Runtime compatibility** | `flutter_litert` is **community-maintained**, not a Google package |

### The result that keeps us honest

> Same weights, same device: classic `Interpreter` + XNNPACK ran inference in **3.93 ms**;
> LiteRT Next `CompiledModel` needed **11.41 ms**.
>
> The newer, accelerator-first API was **2.9× slower here**. It buys automatic backend selection and a
> path to the NPU — not present-day speed on this hardware. **Measure per target; don't assume.**

And preprocessing (19.5 ms) cost 5× more than inference (3.9 ms). On-device ML performance work is
often image-pipeline work, not model work.

---

## Slide 3 — Recommended Flutter architecture

### Title
**Keep Flutter for the app. Isolate inference behind a Dart abstraction.**

### Diagram

```text
             Flutter App
                  │
                  ▼
          ML Service Interface          abstract interface class OnDeviceModel
                  │                     { initialize · predict · dispose }
                  ▼
             LiteRT Layer               CompiledModel | Interpreter
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
      Android              iOS
        │                   │
     CPU/GPU/            CPU/GPU/
     Accelerator         Neural Engine
        │                   │
        └─────────┬─────────┘
                  ▼
             Local Model
                  │
                  ▼
              Inference
                  │
                  ▼
               Result
                  │
                  ▼
              Flutter UI
```

### The recommendation

> For custom on-device ML in Flutter, keep Flutter responsible for application and UI concerns, and
> isolate the inference implementation behind a Dart abstraction backed by LiteRT.

### Which tool, when

| Use | When |
|---|---|
| **ML Kit** | The capability already exists as a high-level API — OCR, barcode, face detection, pose, generic labelling |
| **LiteRT** | Your own model, run locally, with full control of the tensor pipeline |
| **ONNX Runtime** | You already have an ONNX pipeline, or need one artefact across mobile/desktop/server |
| **Cloud** | Model too large or costly to run locally, or you need server-side state and daily updates |

### Non-negotiables if you do this in production

1. **One interface, no leakage.** Only 2 of 25 files in `lib/` import the runtime. That is what let us test
   the entire app layer with a fake model and zero native code.
2. **Assert the tensor contract at startup.** Shape, dtype, quantization params, byte sizes. Wrong
   normalisation doesn't crash — it silently degrades accuracy.
3. **Validate against a reference.** We compare on-device output to a Python LiteRT run on the same file.
   This caught a real resampling defect that had quietly reordered predictions.
4. **Report hardware, never assume it.** Requested ≠ effective. Verify against a CPU reference, and say
   "not verified" when you cannot.
5. **Keep inference off the UI thread**, and know when your binding won't let you.
6. **Version the model as an API.** The tensor contract is part of it.

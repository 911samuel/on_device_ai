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
                   Prediction                "military uniform"  87.5%   4.5 ms
```

### The pipeline, concretely

```text
JPEG bytes → decode → 224×224 resize → normalise → [1,224,224,3] float32 tensor
          → LiteRT invoke → [1,1001] probabilities → sort → label
```

### What we built

* Image classification, MobileNet, **entirely local** — no server exists in the codebase
* Two LiteRT APIs behind **one** Dart interface: `CompiledModel` (LiteRT Next) and classic `Interpreter`
* 69 host unit tests · 15 integration tests validated against a Python reference interpreter

### Measured on a real iPhone 13 Pro (A15), warm median of 30 runs

| Stage | Time | Note |
|---|---:|---|
| Preprocess | 15.1 ms | Dart-side decode + resize — **75–81% of total** |
| Inference | **4.5 ms** | CompiledModel + Metal GPU, float32 MobileNetV2 |
| Postprocess | 0.2 ms | dequantise, sort 1001, map labels |

> Physical device. Also measured on the iOS simulator and Android emulator — and those disagreed with
> real hardware, which is the subject of slide 2.

---

## Slide 2 — Why LiteRT?

### Title
**Custom models, locally — with the trade-offs stated.**

### What it buys

| Benefit | Evidence from the PoC |
|---|---|
| **Custom models** | Any `.tflite` graph. We ran two, one float32 and one uint8-quantized |
| **Offline inference** | Release APK declares **no `INTERNET` permission**; full suite passes with the network unreachable |
| **Low latency** | 4.5 ms inference, no round trip, no tail latency, no retry logic |
| **Privacy** | Pixels never leave the process. Nothing to breach, log, or subpoena |
| **Hardware acceleration** | Metal GPU **verified**: 4.53 ms vs 9.54 ms CPU-only = **2.11× faster**, same API and weights |
| **No per-request cost** | Zero marginal cost per inference; zero backend to operate |

### What it costs

| Trade-off | Measured / observed |
|---|---|
| **Model size** | 13.33 MB float32, 4.08 MB quantized |
| **App binary** | arm64 APK **51.5 MB** vs 15.5 MB empty Flutter app → **+36 MB** (17.4 models + 14.6 runtime) |
| **Memory** | Weights + arenas resident while loaded; we dispose before switching backends |
| **Battery / thermal** | Sustained inference is sustained CPU/GPU load — **not measured** |
| **Device fragmentation** | Same code, three targets: Metal verified on A15; **GPU compile failure** on Android emulator; **NPU numerically wrong** on the A15 |
| **Accelerator correctness** | An accelerator can be *engaged and wrong*. Must be verified, not assumed |
| **Model updates** | Bundled = app release. OTA needs signature + digest + tensor-contract versioning |
| **Runtime compatibility** | `flutter_litert` is **community-maintained**, not a Google package |

### The result that keeps us honest

> On the real A15, requesting the **Neural Engine** produced output that deviated **4.946% of range** from a
> plain-CPU reference — reproducible bit-for-bit, against 0.0005% for healthy configurations. Consistent
> with fp16 computation on the ANE.
>
> **The app refused the backend rather than serve wrong predictions.**
>
> The same configuration reported *healthy* on the iOS simulator (0.0002%), because Core ML there runs on
> the host Mac and never touches a Neural Engine.

Two corollaries a senior audience should take away:

1. **"Accelerated" and "correct" are independent properties.** Verify accelerator output against a CPU
   reference at startup. We would have shipped silently wrong predictions without it.
2. **Simulator benchmarks produce wrong conclusions.** An earlier draft of this deck claimed the classic
   `Interpreter` API was 2.9× faster than `CompiledModel`, from simulator data. On real hardware the
   ordering reverses — the simulator simply had no mobile GPU. We corrected the claim.

And preprocessing (15.1 ms) costs ~3× inference (4.5 ms). On-device ML performance work is usually
image-pipeline work, not model work. Note too that XNNPACK made the *quantized* model **76% slower** on
this device — delegates are not automatically a win.

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
3. **Validate against a reference — twice over.** Compare the pipeline against a host reference run (this
   caught a resampling defect that had quietly reordered predictions), and compare each accelerator against
   a plain-CPU reference at startup (this caught the ANE returning wrong output on a real phone).
4. **Report hardware, never assume it.** Requested ≠ effective ≠ correct. Say "not verified" when you
   cannot verify, and **test on physical devices** — the simulator called a broken backend healthy.
5. **Keep inference off the UI thread**, and know when your binding won't let you.
6. **Version the model as an API.** The tensor contract is part of it.

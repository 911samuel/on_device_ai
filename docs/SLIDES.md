# Chapter presentation — four slides

**Audience:** a mixed room. Some colleagues are senior mobile engineers; others have never touched machine
learning and are here to find out whether it's worth their curiosity. The deck is built so **both leave with
something**: Slide 0 gives everyone the vocabulary, and Slides 1–3 go progressively deeper.

Every number on these slides was measured in this repo. Anything unverified is labelled as such **on the slide
itself** — that's the habit worth modelling for the room.

Each slide has a **"Say it in plain words"** box. If you only have ten minutes, or the room turns out to be
mostly beginners, read those boxes and skip the tables.

Suggested timing: 3 min slide 0, 4 min slide 1, 6 min slide 2, 4 min slide 3, rest Q&A.

---

## Slide 0 — What "on-device AI" actually means

### Title
**The model is a file. Your phone does the maths.**

### The whole idea in four steps

```text
   ┌─────────────┐    ┌──────────────────┐    ┌─────────────┐    ┌──────────────┐
   │  1. A photo │ →  │ 2. Turn it into  │ →  │ 3. Multiply │ →  │ 4. Read the  │
   │             │    │    numbers the   │    │  it through │    │    winning   │
   │             │    │  model expects   │    │   the model │    │    label     │
   └─────────────┘    └──────────────────┘    └─────────────┘    └──────────────┘
        JPEG              602,112 bytes            4.5 ms          "military
                          224×224×3                                 uniform" 87.5%

         └──────────── all of this happens on the phone ────────────┘
```

### Say it in plain words

> "An AI model is not a program that thinks. It's a **file full of numbers** — ours is 13 MB — that somebody
> trained once on a million photos and then froze. Running it means replaying a fixed sequence of
> multiplications on new input. That's cheap: a few milliseconds on a phone.
>
> Training the model is the expensive part, and it happened in a data centre long before we got involved. We
> only *run* it. The word for that is **inference**, and it's the only AI operation in this entire project."

### Three words that carry the rest of the talk

| Word | What it means | Why you care |
|---|---|---|
| **Inference** | Running an already-trained model on new input | It's cheap and fast. Training is the expensive thing, and we don't do it |
| **Tensor** | A grid of numbers with a known shape | The model accepts *exactly one* shape. Getting it wrong is the #1 bug source |
| **Accelerator** | A chip other than the main CPU — GPU or NPU | Where the speed comes from, and where the surprises live |

### Why anyone would do this instead of calling an API

| | On-device | Cloud API |
|---|---|---|
| Works offline | **Yes** | No |
| Photo leaves the device | **Never** | Always |
| Cost per prediction | **Zero** | Per call, forever |
| Latency | **~20 ms** | Network round trip |
| Model updates | App release | Instant |
| Model size limit | Phone-sized | Effectively none |

The last two rows are the honest reasons you'd still choose cloud. This is a trade, not a win.

---

## Slide 1 — On-device ML in Flutter

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

**LiteRT** is the runtime that executes the model — it used to be called TensorFlow Lite. If the model is a
document, LiteRT is the app that opens it.

### The pipeline, concretely

```text
JPEG bytes → decode → 224×224 resize → normalise → [1,224,224,3] float32 tensor
          → LiteRT invoke → [1,1001] probabilities → sort → label
```

### Say it in plain words

> "A JPEG is not something a model can read. The model wants exactly 602,112 bytes: 224 by 224 pixels, three
> colour values each, every number scaled to between −1 and 1. Everything between 'the user picked a photo' and
> 'the model can run' is our responsibility — and that's the part that bites you.
>
> Out the other end come 1,001 numbers, one score per category the model knows. We sort them and look up the
> winner's name. That's it. No magic anywhere in the chain."

### What we built

* Image classification, MobileNet, **entirely local** — no server exists in the codebase
* Two LiteRT APIs behind **one** Dart interface: `CompiledModel` (newer, picks chips for you) and the classic
  `Interpreter` (more manual, supports small quantized models)
* 86 host unit tests · 15 integration tests validated against an independent Python reference

### Measured on a real iPhone 13 Pro (A15), warm median of 30 runs

| Stage | Time | Note |
|---|---:|---|
| Preprocess | 15.1 ms | Dart-side decode + resize — **75–81% of total** |
| Inference | **4.5 ms** | The actual AI. CompiledModel + Metal GPU, float32 MobileNetV2 |
| Postprocess | 0.2 ms | dequantise, sort 1001, map labels |

**The punchline of this slide:** the AI is the *fastest* part. Resizing the photo takes three times longer than
running the model. If someone asked us to speed this up, we wouldn't touch the model.

> Physical device. Also measured on the iOS simulator and Android emulator — and those disagreed with real
> hardware, which is the subject of slide 2.

---

## Slide 2 — Why LiteRT, and what it costs

### Title
**Custom models, locally — with the trade-offs stated.**

### What it buys

| Benefit | Evidence from the PoC |
|---|---|
| **Custom models** | Any `.tflite` file. We ran two, one full-precision and one compressed |
| **Offline inference** | Release APK declares **no `INTERNET` permission** — the OS kernel blocks connections. Full suite passes with the network unreachable |
| **Low latency** | 4.5 ms inference, no round trip, no tail latency, no retry logic |
| **Privacy** | Pixels never leave the process. Nothing to breach, log, or subpoena |
| **Hardware acceleration** | Metal GPU **verified**: ~4.5–4.9 ms vs 9.54 ms CPU-only = **≈2× faster**, same API and weights, three runs |
| **No per-request cost** | Zero marginal cost per inference; zero backend to operate |

### What it costs

| Trade-off | Measured / observed |
|---|---|
| **Model size** | 13.33 MB full-precision, 4.08 MB compressed |
| **App binary** | arm64 APK **51.5 MB** vs 15.5 MB empty Flutter app → **+36 MB** (17.4 models + 14.6 runtime) |
| **Memory** | Weights stay in RAM while loaded; we free one model before loading another |
| **Battery / thermal** | Sustained inference is sustained chip load — **not measured** |
| **Device fragmentation** | Same code, three targets: Metal verified on A15; **GPU compile failure** on Android emulator; **NPU numerically wrong** on the A15 |
| **Accelerator correctness** | An accelerator can be *engaged and wrong*. Must be verified, not assumed |
| **Model updates** | Bundled = app release. Over-the-air needs signature + digest + tensor-contract versioning |
| **Runtime compatibility** | `flutter_litert` is **community-maintained**, not a Google package |

### The result that keeps us honest

> On the real A15, requesting the **Neural Engine** — Apple's dedicated AI chip — produced output that deviated
> **4.946% of range** from a plain-CPU reference. Reproducible bit-for-bit, against 0.0005% for healthy
> configurations. Consistent with the chip computing in reduced precision (fp16).
>
> **The app refused the backend rather than serve wrong predictions.**
>
> The same configuration reported *healthy* on the iOS simulator (0.0002%), because Core ML there runs on the
> host Mac and never touches a Neural Engine.

### Say it in plain words

> "We asked the phone's dedicated AI chip to do the work. It said yes, did the work, and gave us answers that
> were about five percent off — the same five percent off, every single time. Not random noise: that chip
> genuinely computes differently, because it uses fewer digits per number.
>
> Our app checks every chip against the plain CPU before trusting it, so it caught this and refused. Without
> that check, we'd have shipped an app that was quietly a bit wrong. It wouldn't crash. Nobody would report it.
>
> And here's the part that should worry you: the *simulator* said that same setup was perfectly fine — because
> a simulator has no AI chip and quietly used the Mac's processors instead."

### Two takeaways for the engineers in the room

1. **"Accelerated" and "correct" are independent properties.** Verify accelerator output against a CPU reference
   at startup. We would have shipped silently wrong predictions without it.
2. **Simulator benchmarks produce wrong conclusions.** An earlier version of this deck claimed the classic
   `Interpreter` API was 2.9× faster than `CompiledModel`, from simulator data. On real hardware the ordering
   reverses — the simulator simply had no mobile GPU. We corrected the claim rather than dropping it.

Two supporting facts: preprocessing (15.1 ms) costs ~3× inference (4.5 ms), so on-device ML performance work is
usually image-pipeline work. And **XNNPACK — a library whose entire purpose is to make things faster — made the
compressed model 76% slower** on this device. Measure the boring baseline too.

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

### Say it in plain words

> "One interface, about eight lines: initialise, predict, dispose. Everything above that line has no idea LiteRT
> exists — in this repo, only 2 files out of 24 mention it.
>
> That isn't architectural neatness for its own sake. It's what let us test the entire app with a *fake* model
> and no native code at all. And it means swapping LiteRT for something else later is adding a class, not a
> rewrite.
>
> Below the line, the platforms genuinely diverge — Apple's Neural Engine on iOS, a completely different stack
> on Android. That difference belongs down there, hidden, not in your widgets."

### Which tool, when

| Use | When |
|---|---|
| **ML Kit** | The capability already exists as a ready-made API — OCR, barcode, face detection, pose, generic labelling |
| **LiteRT** | Your own model, run locally, with full control of the pipeline |
| **ONNX Runtime** | You already have an ONNX pipeline, or need one model file across mobile/desktop/server |
| **Cloud** | Model too large or costly to run locally, or you need server-side state and daily updates |

If your problem is on the ML Kit row, use ML Kit. Reimplementing barcode scanning with LiteRT would be worse
engineering, and saying so is part of being trustworthy about the rest.

### Non-negotiables if you do this in production

1. **One interface, no leakage.** Only 2 of 24 files in `lib/` import the runtime. That is what let us test the
   entire app layer with a fake model and zero native code.
2. **Assert the tensor contract at startup.** Shape, dtype, quantization parameters, byte sizes. Wrong
   normalisation doesn't crash — it silently degrades accuracy.
3. **Validate against a reference — twice over.** Compare the pipeline against a host reference run (this caught
   a resizing defect that had quietly reordered predictions), and compare each accelerator against a plain-CPU
   reference at startup (this caught the Neural Engine returning wrong output on a real phone).
4. **Report hardware, never assume it.** Requested ≠ effective ≠ correct. Say "not verified" when you cannot
   verify, and **test on physical devices** — the simulator called a broken backend healthy.
5. **Keep inference off the UI thread**, and know when your binding won't let you.
6. **Version the model as an API.** The tensor contract is part of it.

---

## Appendix — jargon card for the presenter

Keep this visible. If someone looks lost, these are the definitions to reach for. Full list in
[`GLOSSARY.md`](GLOSSARY.md).

| Term | One-line answer |
|---|---|
| **Inference** | Running a trained model on new input. The only AI operation here |
| **Model / weights** | A file of numbers, frozen after training. Read-only, identical for every user |
| **Tensor** | A grid of numbers with a fixed shape. Ours: `[1, 224, 224, 3]` |
| **LiteRT** | The runtime that executes the model. Formerly TensorFlow Lite |
| **Quantization** | Storing numbers as small integers. ~4× smaller, slightly less accurate |
| **float32 / fp16 / uint8** | How many digits per number. Fewer digits = faster, less precise |
| **CPU / GPU / NPU** | Three chips in the phone. NPU = dedicated AI chip (Apple: Neural Engine) |
| **Delegate** | A plug-in that runs part of the model on faster code. XNNPACK is one |
| **Core ML / Metal** | Apple's ML framework / Apple's GPU framework |
| **Cold vs warm** | First run vs later runs. The first is always slower |
| **Median** | Middle measurement. More honest than an average for latency |
| **ms** | Millisecond. A 60 fps screen draws a frame every 16.7 ms |

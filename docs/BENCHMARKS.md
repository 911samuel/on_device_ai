# Measured performance

Every number here was produced by `integration_test/on_device_inference_test.dart`, which prints one
machine-readable `BENCH` line per backend. Raw logs are the source; nothing is estimated, rounded up,
or carried over from another device.

```bash
flutter test integration_test/on_device_inference_test.dart -d <device>
```

Protocol: 30 predictions per backend on `grace_hopper.jpg` (512x600 JPEG). Run 1 is reported separately
as **cold**; runs 2–30 are summarised as **warm**. Timings come from `Stopwatch` around each pipeline
stage inside `predict()`.

## Read this before quoting any of it

**Both targets are emulated, and neither is representative of production phone hardware.**

* The **iOS simulator** does not emulate an iPhone SoC. It executes arm64 code natively on this Mac's
  Apple Silicon CPU/GPU. There is **no Neural Engine in the simulator**, so no number here says
  anything about ANE performance.
* The **Android emulator** runs arm64 on the same host CPU and has **no working OpenCL**, which is why
  GPU compilation fails outright (see below).

A real phone differs in every direction that matters: fewer/slower cores, a big.LITTLE scheduler, a DVFS
governor that ramps up over the first few inferences, thermal throttling under sustained load, and a real
mobile GPU/NPU. Treat these as *relative* comparisons between backends on identical hardware, which is
what they are good for, and not as latency budgets for a shipping app.

## iOS Simulator — iPhone 17, iOS 26.2 (23C54), 8 logical CPUs, LiteRT native `2.20.0-dev0+selfbuilt`

All times in ms.

| Backend | Model | Init | Cold total | Warm total med | Warm inference min / med / max | Warm preprocess med | Warm postprocess med | Cold ÷ warm |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CompiledModel · GPU→CPU | V2 float32 | 124.7 | 41.6 | 31.3 | 7.26 / **11.47** / 24.73 | 19.54 | 0.23 | 1.33× |
| CompiledModel · CPU only | V2 float32 | 33.7 | 32.0 | 30.9 | 10.57 / **11.41** / 15.06 | 19.31 | 0.22 | 1.04× |
| CompiledModel · NPU→CPU | V2 float32 | 588.6 | 39.2 | 36.6 | 14.97 / **16.80** / 24.97 | 19.55 | 0.22 | 1.07× |
| Interpreter · XNNPACK | V2 float32 | 6.8 | 25.8 | 23.9 | 3.66 / **3.93** / 4.84 | 19.63 | 0.23 | 1.08× |
| Interpreter · XNNPACK | V1 uint8 | 12.7 | 25.9 | 24.4 | 4.47 / **4.67** / 5.37 | 19.43 | 0.20 | 1.06× |
| Interpreter · no delegate | V1 uint8 | 2.6 | 25.3 | 24.9 | 2.38 / **4.95** / 8.08 | 19.55 | 0.19 | 1.01× |

## Android Emulator — Pixel 8, Android build `CE2A.260420.019`, arm64-v8a, 4 logical CPUs, LiteRT native `2.22.0-dev0+selfbuilt`

| Backend | Model | Init | Cold total | Warm total med | Warm inference min / med / max | Warm preprocess med | Warm postprocess med | Cold ÷ warm |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CompiledModel · GPU→CPU **(GPU failed → CPU)** | V2 float32 | 308.8 | 81.1 | 40.5 | 10.74 / **12.24** / 21.71 | 27.07 | 0.27 | 2.00× |
| CompiledModel · CPU only | V2 float32 | 57.0 | 44.1 | 43.2 | 10.83 / **13.18** / 28.27 | 29.30 | 0.27 | 1.02× |
| CompiledModel · NPU→CPU **(NPU narrowed → CPU)** | V2 float32 | 152.9 | 141.2 | 39.3 | 10.48 / **12.02** / 19.39 | 26.70 | 0.25 | 3.59× |
| Interpreter · XNNPACK | V2 float32 | 7.1 | 32.6 | 39.6 | 3.01 / **6.17** / 50.36 | 29.04 | 0.37 | 0.82× |
| Interpreter · XNNPACK | V1 uint8 | 17.2 | 32.2 | 36.6 | 4.97 / **8.48** / 23.37 | 27.11 | 0.22 | 0.88× |
| Interpreter · no delegate | V1 uint8 | 7.4 | 58.9 | 36.0 | 5.54 / **10.88** / 52.95 | 25.37 | 0.21 | 1.64× |

The Android `max` values are 4–8× the medians and two `cold ÷ warm` ratios are **below 1.0** — a cold run
came out *faster* than the warm median. That is host-scheduler noise on a virtualised device, and it is
the reason this project reports min/median/p90/max instead of a single mean. On a noisy target, a mean is
a number that describes the noise.

## Findings worth defending

### 1. Preprocessing dominates, not inference

On iOS, decode + resize + normalise costs **~19.5 ms** while the fastest inference path costs **3.9 ms** —
preprocessing is ~5× the model. On Android it is 25–29 ms against 3–12 ms. Preprocessing also scales with
the *source* image, not the model: measured per-image on iOS, 11.4 ms for the 320×213 cat, 19.5 ms for the
512×600 portrait, 26 ms for the 700×577 Labrador.

This is pure Dart work (JPEG decode plus an area-average resample). If this pipeline needed to hit 30 fps,
the model is not what to optimise — the correct move is to stop producing JPEGs at all and feed camera
frames directly, converting YUV→RGB and resizing on the GPU or in native code.

### 2. The newer API was not the faster one here

Same weights, same device, same image: `Interpreter` + XNNPACK ran inference in **3.93 ms** median while
`CompiledModel` needed **11.41 ms** CPU-only and **11.47 ms** with GPU+CPU — roughly 2.9×. Two caveats
that keep this honest:

* The `CompiledModel` figures include a helper-isolate round trip, because this project uses `runAsync`
  to keep the UI thread free. The `Interpreter` + XNNPACK figures are measured inline on the calling
  isolate. Some of the gap is that hop, not the runtime. The isolate cost is bounded by
  `Interpreter · no delegate`, which *does* go through `IsolateInterpreter`: 4.95 ms median. So the hop is
  a few ms, not 7.5.
* `CompiledModel` defaults to fp32 precision in this binding (3.8.0 changed it from fp16 for accuracy),
  and the binding's own changelog records that this costs 24–43% GPU latency.

The defensible conclusion is narrow and useful: **on this hardware, for this model, the accelerator-first
API did not beat a well-configured CPU path.** Choosing `CompiledModel` here buys future NPU access and
automatic backend selection, not present-day speed — and that trade should be measured per target device,
not assumed.

### 3. XNNPACK is worth having, and it is still CPU

Quantized V1, same API and device: **4.67 ms** with XNNPACK vs **4.95 ms** without on iOS, and **8.48 ms**
vs **10.88 ms** on Android. A modest gain here, larger on the noisier target. XNNPACK is an optimised CPU
kernel library — NEON SIMD, better blocking — not a separate processor. Reporting it as "hardware
acceleration" would be misleading, so the UI labels it explicitly as CPU.

### 4. The shared LiteRT environment makes the *first* model expensive

Initialisation was measured twice per backend in the same process (once per test). The first
`CompiledModel` creation of the process paid a much larger cost than the second:

| Backend | 1st init in process | 2nd init | iOS |
|---|---:|---:|---|
| CompiledModel · GPU→CPU | **1782.1** | 124.7 | 14× cheaper the second time |
| CompiledModel · NPU→CPU | 776.8 | 588.6 | Core ML compilation dominates |
| CompiledModel · CPU only | 80.7 | 33.7 | |
| Interpreter · XNNPACK (V2) | 21.4 | 6.8 | |

Creating a LiteRT environment spins up the full GPU stack — adapter enumeration, device and context
creation, kernel cache — and the binding shares one per isolate. So the first model an app loads absorbs
that cost and later models do not. Practical consequence: **warm up off the critical path** (this app does
it in a post-frame callback so the UI is on screen first), and do not benchmark a single cold start and
call it your latency.

### 5. Cold vs warm, and why they differ

Cold run 1 pays: lazy delegate/kernel setup, first-touch page faults on freshly allocated tensor arenas,
i-cache misses on code paths never executed, and on a real phone a CPU governor still at a low frequency.
Measured cold penalty ranged from 1.01× to 3.59×. The largest (Android NPU→CPU, 141.2 ms cold vs 39.3 ms
warm median) is also the case doing the most first-run setup work.

### 6. Postprocessing is free

0.19–0.37 ms to dequantize (when applicable), sort 1001 values and map labels. A full sort is used for
determinism and clarity; at this size the optimisation is not worth the complexity.

## Binary size, measured

`flutter build apk --release --split-per-abi`, compared against a bare `flutter create` app built the same
way on the same toolchain:

| APK | This PoC | Empty Flutter app | Delta |
|---|---:|---:|---:|
| arm64-v8a | **51.5 MB** | 15.5 MB | **+36.0 MB** |
| armeabi-v7a | 41.5 MB | 12.7 MB | +28.8 MB |
| x86_64 | 57.5 MB | 16.9 MB | +40.6 MB |
| universal (all 3 ABIs) | 112.5 MB | — | — |

Where the arm64 delta goes:

| Component | Size |
|---|---:|
| `mobilenet_v2_1.0_224.tflite` | 13.33 MB |
| `mobilenet_v1_1.0_224_quant.tflite` | 4.08 MB |
| `libLiteRt.so` | 5.14 MB |
| `libtensorflowlite_jni.so` | 4.27 MB |
| `libLiteRtClGlAccelerator.so` | 2.70 MB |
| `libtensorflowlite_gpu_jni.so` | 2.47 MB |
| `libtflite_custom_ops.so` | 0.01 MB |
| **LiteRT native total** | **14.59 MB** |
| Dart/Flutter code for the app itself | remainder (~4 MB) |

So the runtime costs ~14.6 MB per ABI and the models 17.4 MB. Levers, in order of effect: ship one model
instead of two (−13.3 MB by dropping the float one), always split per ABI or use an App Bundle (−61 MB
against the universal APK), set `flutterLitert.bundleGpuAccelerator=false` if GPU is not needed (~−3 MB
per the package's own docs, **not verified here**), and deliver models out-of-band rather than bundling
them (see the model-update discussion in the README).

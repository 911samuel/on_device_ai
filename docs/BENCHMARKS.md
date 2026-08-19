# Measured performance

Every number here was produced by `integration_test/on_device_inference_test.dart`, which prints one
machine-readable `BENCH` line per backend. Nothing is estimated or carried over between devices.

```bash
flutter test integration_test/on_device_inference_test.dart -d <device>
```

Protocol: 30 predictions per backend on `grace_hopper.jpg` (512×600 JPEG). Run 1 is reported separately as
**cold**; runs 2–30 are summarised as **warm**. Timings come from `Stopwatch` around each pipeline stage
inside `predict()`. The physical-device suite was run twice; figures below are from the first run and the
second agreed within ~0.1 ms on every median.

---

## 1. Physical device — iPhone 13 Pro (A15 Bionic, 6 CPUs, 16-core Neural Engine), iOS 26.6 (23G71), LiteRT native `2.20.0-dev0+selfbuilt`

**This is the only target that represents real mobile hardware. Prefer these numbers.** All times in ms.

| Backend | Model | Init | Cold total | Warm total med | Warm inference min / med / max | Warm preprocess med | Warm postproc med | Cold ÷ warm |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CompiledModel · GPU→CPU | V2 float32 | 91.3 | 21.8 | 20.3 | 3.76 / **4.53** / 7.35 | 15.14 | 0.18 | 1.07× |
| CompiledModel · CPU only | V2 float32 | 29.5 | 26.4 | 24.9 | 9.48 / **9.54** / 12.89 | 15.13 | 0.17 | 1.06× |
| CompiledModel · NPU→CPU | V2 float32 | — | — | — | **REFUSED — wrong output, see §3** | — | — | — |
| Interpreter · XNNPACK | V2 float32 | 4.1 | 21.2 | 20.1 | 4.64 / **4.79** / 5.30 | 14.99 | 0.18 | 1.06× |
| Interpreter · XNNPACK | V1 uint8 | 9.9 | 21.7 | 21.2 | 6.11 / **6.17** / 7.59 | 14.90 | 0.14 | 1.03× |
| Interpreter · no delegate | V1 uint8 | 1.2 | 19.3 | 18.8 | 3.32 / **3.51** / 4.18 | 15.01 | 0.18 | 1.03× |

Real hardware is markedly *more* stable than the emulated targets: cold-to-warm ratios of 1.01–1.08 and
max values within ~1.6× of the median, versus 4–8× spreads on the Android emulator.

---

## 2. Headline findings from real hardware

### GPU (Metal) is verified, and it is the fastest float path

`{gpu, cpu}` reported `effective = GPU + CPU`, and the CPU-reference comparison found a deviation of
**0.0005% of output range** — non-zero, so a compute path other than the reference CPU kernels genuinely ran,
and well inside the 1% correctness tolerance. Against the same API's CPU-only configuration:

| | Inference median |
|---|---:|
| CompiledModel · GPU→CPU | **4.53 ms** |
| CompiledModel · CPU only | 9.54 ms |
| **Speed-up** | **2.11×** |

This is a like-for-like comparison: same API, same weights, same device, same measurement path (both use
`runAsync`, so both pay the identical helper-isolate hop). GPU acceleration on iOS is therefore **verified**,
not assumed.

### The Neural Engine produced numerically wrong output, and was refused

Requesting `{npu, cpu}` compiles successfully on the A15 — and then fails the correctness check:

```text
ModelInitializationException: CompiledModel output disagrees with the plain-CPU reference
  for mobilenet_v2_float32
  BackendVerification(agrees: false,
                      deviation: 4.946% of range,
                      absolute: 0.008670523762702942,
                      range: 0.17528938692953488)
  Refusing to use a backend that computes the wrong thing.
```

Reproduced **bit-for-bit identical** across two independent runs, so this is a deterministic numerical
property, not noise. For scale: healthy configurations on this device deviate by 0.0005%, and the binding's
tolerance is 1% — this is ~10,000× the healthy deviation and ~5× the tolerance.

The most likely cause is precision. The binding documents that "NPU accuracy [is] bounded by fp16 — Apple
Neural Engine and Qualcomm HTP compute in reduced precision," and Core ML's
CPU-and-Neural-Engine policy is free to run the graph in fp16. A ~5% shift in a probability distribution is
consistent with fp16 accumulation through a 174-tensor graph.

Two things this establishes, and one it does not:

* **It does establish** that Core ML was genuinely engaged — the output changed substantially, so something
  other than the CPU kernels computed the graph.
* **It does establish** that the PoC's verification is load-bearing rather than decorative: without it, this
  configuration would have shipped silently wrong predictions on real hardware. Note that the *simulator*
  reported this same configuration as healthy (0.0002% deviation), because Core ML there runs on the host
  Mac and never touches a Neural Engine. Simulator testing would have missed this entirely.
* **It does not establish** ANE throughput or efficiency. We refused the backend, so there is no latency
  figure for it — and quoting one would mean quoting a wrong-output configuration. Whether an fp16-tolerant
  model (or a quantization-aware-trained one) would pass here is **not tested**.

### CompiledModel vs Interpreter: the simulator's verdict was wrong

On the iOS simulator, `Interpreter` + XNNPACK beat `CompiledModel` by ~2.9×, and an earlier draft of this
document drew a conclusion from that. On real hardware the ordering reverses:

| Configuration | Simulator | **iPhone 13 Pro** |
|---|---:|---:|
| CompiledModel · GPU→CPU | 11.47 ms | **4.53 ms** |
| Interpreter · XNNPACK (float32) | 3.93 ms | 4.79 ms |

With a real GPU available, the accelerator-first API is the faster one — narrowly. The simulator's result
was an artifact of having no mobile GPU: `CompiledModel` fell back to a CPU path while XNNPACK ran optimised
CPU kernels, so the comparison was never about the API.

**Do not benchmark on a simulator and draw architectural conclusions.** That is the most useful methodological
lesson in this project, and it cost a wrong claim to learn.

### XNNPACK made the quantized model slower on real hardware

| Configuration (V1 uint8) | Simulator | **iPhone 13 Pro** |
|---|---:|---:|
| Interpreter · XNNPACK | 4.67 ms | 6.17 ms |
| Interpreter · no delegate | 4.95 ms | **3.51 ms** |

On the A15, attaching XNNPACK to the uint8 graph cost **76% more** inference time than the reference kernels.
The plain quantized path is also the fastest inference measured anywhere in this project. Delegates are not
free and not always a win — measure per model *and* per device, including the "no delegate" baseline.

(A secondary consequence: the no-delegate path is the only `Interpreter` configuration that can use
`IsolateInterpreter`, so on this device the fastest option is also the one that keeps the UI thread free.)

### Preprocessing still dominates

~15 ms of Dart-side decode + resize against 3.5–9.5 ms of inference. Inference got ~4× faster moving from
the simulator to real hardware; preprocessing improved only ~1.3× (19.5 → 15 ms), because it is
single-threaded Dart work, not accelerated kernels. On real hardware preprocessing is now **75–81% of total
latency** for the fastest configurations.

### Initialisation and the shared LiteRT environment

Init was measured twice per backend in the same process. The first `CompiledModel` creation pays for the
LiteRT environment (GPU stack, adapter enumeration, kernel cache), which the binding then shares per isolate:

| Backend | 1st init in process | 2nd init |
|---|---:|---:|
| CompiledModel · GPU→CPU | **687.6** | 91.3 |
| CompiledModel · CPU only | 35.5 | 29.5 |
| Interpreter · XNNPACK (V2) | 235.1 | 4.1 |
| Interpreter · no delegate (V1) | 2.9 | 1.2 |

688 ms for the first GPU-backed model is a real startup cost on a real phone. Warm up off the critical path —
this app initialises in a post-frame callback so the UI is on screen first.

---

## 3. Emulated targets, for comparison only

Retained because they show how misleading emulated measurement can be. **Neither represents phone hardware:**
the iOS simulator executes arm64 natively on the host Mac and has no Neural Engine; the Android emulator has
no working OpenCL, so GPU compilation fails outright.

### iOS Simulator — iPhone 17, iOS 26.2, 8 CPUs, LiteRT `2.20.0-dev0+selfbuilt`

| Backend | Model | Init | Cold | Warm total med | Warm inference min / med / max | Preproc med |
|---|---|---:|---:|---:|---:|---:|
| CompiledModel · GPU→CPU | V2 f32 | 124.7 | 41.6 | 31.3 | 7.26 / 11.47 / 24.73 | 19.54 |
| CompiledModel · CPU only | V2 f32 | 33.7 | 32.0 | 30.9 | 10.57 / 11.41 / 15.06 | 19.31 |
| CompiledModel · NPU→CPU | V2 f32 | 588.6 | 39.2 | 36.6 | 14.97 / 16.80 / 24.97 | 19.55 |
| Interpreter · XNNPACK | V2 f32 | 6.8 | 25.8 | 23.9 | 3.66 / 3.93 / 4.84 | 19.63 |
| Interpreter · XNNPACK | V1 u8 | 12.7 | 25.9 | 24.4 | 4.47 / 4.67 / 5.37 | 19.43 |
| Interpreter · no delegate | V1 u8 | 2.6 | 25.3 | 24.9 | 2.38 / 4.95 / 8.08 | 19.55 |

Note row 3: the simulator ran the NPU configuration and reported it **healthy** (0.0002% deviation). The same
configuration is numerically broken on the real A15. This is the clearest possible argument for
physical-device validation.

### Android Emulator — Pixel 8, arm64-v8a, 4 CPUs, LiteRT `2.22.0-dev0+selfbuilt`

| Backend | Model | Init | Cold | Warm total med | Warm inference min / med / max | Preproc med |
|---|---|---:|---:|---:|---:|---:|
| CompiledModel · GPU→CPU **(GPU failed → CPU)** | V2 f32 | 308.8 | 81.1 | 40.5 | 10.74 / 12.24 / 21.71 | 27.07 |
| CompiledModel · CPU only | V2 f32 | 57.0 | 44.1 | 43.2 | 10.83 / 13.18 / 28.27 | 29.30 |
| CompiledModel · NPU→CPU **(narrowed → CPU)** | V2 f32 | 152.9 | 141.2 | 39.3 | 10.48 / 12.02 / 19.39 | 26.70 |
| Interpreter · XNNPACK | V2 f32 | 7.1 | 32.6 | 39.6 | 3.01 / 6.17 / 50.36 | 29.04 |
| Interpreter · XNNPACK | V1 u8 | 17.2 | 32.2 | 36.6 | 4.97 / 8.48 / 23.37 | 27.11 |
| Interpreter · no delegate | V1 u8 | 7.4 | 58.9 | 36.0 | 5.54 / 10.88 / 52.95 | 25.37 |

Android GPU compilation failed with
`LiteRtCreateManagedTensorBufferFromRequirements input[0] failed with LiteRtStatus=3
(kLiteRtStatusErrorRuntimeFailure)` and correctly fell back to CPU. Max values run 4–8× the medians and two
cold-to-warm ratios come out below 1.0 — host-scheduler noise, and the reason this project reports
min/median/p90/max rather than a single mean.

**Android GPU and vendor-NPU acceleration remain NOT VERIFIED.** No physical Android device was available.

---

## 4. Binary size, measured

`flutter build apk --release --split-per-abi`, against a bare `flutter create` app on the same toolchain:

| APK | This PoC | Empty Flutter app | Delta |
|---|---:|---:|---:|
| arm64-v8a | **51.5 MB** | 15.5 MB | **+36.0 MB** |
| armeabi-v7a | 41.5 MB | 12.7 MB | +28.8 MB |
| x86_64 | 57.5 MB | 16.9 MB | +40.6 MB |
| universal (3 ABIs) | 112.5 MB | — | — |

iOS device build: `Runner.app` is 58.2 MB (unsigned, before App Store thinning).

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

Levers, by effect: ship one model instead of two (−13.3 MB by dropping the float one — though on this device
the float+GPU path is the fastest correct float option, so that trade is real); always split per ABI or use
an App Bundle (−61 MB vs the universal APK); set `flutterLitert.bundleGpuAccelerator=false` if GPU is unused
(~−3 MB per the package docs, **not verified here**); or deliver models out-of-band.

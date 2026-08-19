# Measured performance

New to this? Read [`GLOSSARY.md`](GLOSSARY.md) first — it defines every term used here in plain language.

Every number on this page was produced by `integration_test/on_device_inference_test.dart`, which prints one
machine-readable `BENCH` line per backend. Nothing is estimated, and nothing is carried over from one device to
another.

```bash
flutter test integration_test/on_device_inference_test.dart -d <device>
```

## How to read this page

**What we measured.** We classify the same photo (`grace_hopper.jpg`, a 512×600 JPEG) 30 times on each
*backend*. A "backend" is one combination of API + model + chip — for example "the newer API, float model, run
on the GPU". Six combinations, three devices.

**Why the run is split into three stages.** Getting an answer takes more than the model:

| Stage | What actually happens | Is it "AI"? |
|---|---|---|
| **Preprocess** | Decode the JPEG, resize it to 224×224, arrange the pixels into the exact box of numbers the model expects | No — ordinary image code |
| **Inference** | The model does its multiplications | Yes — this is the model |
| **Postprocess** | Turn 1,001 raw scores into "military uniform, 87.5%" | No — sorting and a lookup |

Splitting them matters because, as you'll see, the "AI" is not the slow part.

**Why there are two speed columns.** The very first run is always slower — caches are cold, memory isn't
allocated, the GPU driver is waking up. So run 1 is reported as **cold** and runs 2–30 as **warm**. We give you
both rather than quoting whichever flatters us.

**Why median and not average.** The *median* is the middle measurement once you sort them. One unlucky slow run
drags an average around; a median ignores it. We also show min and max so you can see the spread yourself.

**A reference point for all of these numbers:** a 60 fps screen draws a frame every 16.7 ms. Anything below
that feels instant to a human.

The physical-device suite has now been run **three times**. The tables below are from run 1. Runs 1 and 2 agreed
within ~0.1 ms on every median; run 3 drifted slightly higher on the GPU path (4.90 ms vs 4.53 ms) while the
CPU-only path was identical to 0.01 ms. Where that spread matters to a conclusion, the range is given rather than
the best number — see [Finding 3](#finding-3-the-gpu-is-genuinely-verifiably-about-twice-as-fast).

---

## 1. The real phone — iPhone 13 Pro (A15 Bionic, 6 CPU cores, 16-core Neural Engine), iOS 26.6

**This is the only target that represents real mobile hardware. If you read one table, read this one.** All
times in milliseconds.

| Backend | Model | Init | Cold total | Warm total med | Warm inference min / med / max | Warm preprocess med | Warm postproc med | Cold ÷ warm |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CompiledModel · GPU→CPU | V2 float32 | 91.3 | 21.8 | 20.3 | 3.76 / **4.53** / 7.35 | 15.14 | 0.18 | 1.07× |
| CompiledModel · CPU only | V2 float32 | 29.5 | 26.4 | 24.9 | 9.48 / **9.54** / 12.89 | 15.13 | 0.17 | 1.06× |
| CompiledModel · NPU→CPU | V2 float32 | — | — | — | **REFUSED — wrong output, see below** | — | — | — |
| Interpreter · XNNPACK | V2 float32 | 4.1 | 21.2 | 20.1 | 4.64 / **4.79** / 5.30 | 14.99 | 0.18 | 1.06× |
| Interpreter · XNNPACK | V1 uint8 | 9.9 | 21.7 | 21.2 | 6.11 / **6.17** / 7.59 | 14.90 | 0.14 | 1.03× |
| Interpreter · no delegate | V1 uint8 | 1.2 | 19.3 | 18.8 | 3.32 / **3.51** / 4.18 | 15.01 | 0.18 | 1.03× |

**In plain terms:** every one of these configurations classifies a photo in about 19–25 ms end to end, of which
only 3.5–9.5 ms is the model. The fastest model run is the small quantized one with no delegate at all
(3.51 ms). The `×` column tells you the first run is only 1.03–1.07× slower than later ones — so on real
hardware there is barely any warm-up penalty once the model is loaded. (Across all three runs of the suite that
ratio spans 1.01–1.14×, so treat it as "a little slower", not a fixed figure.)

Real hardware is also markedly *steadier* than the emulated targets: the slowest single run is within ~1.6–1.8×
of the median here, versus 4–8× on the Android emulator.

---

## 2. The four findings worth telling your colleagues

### Finding 1: a dedicated AI chip computed the wrong answer

This is the most valuable result in the project, and the least expected.

We asked LiteRT for the Neural Engine (`{npu, cpu}`). It accepted the model, compiled it successfully — and
then failed our startup correctness check:

```text
ModelInitializationException: CompiledModel output disagrees with the plain-CPU reference
  for mobilenet_v2_float32
  BackendVerification(agrees: false,
                      deviation: 4.946% of range,
                      absolute: 0.008670523762702942,
                      range: 0.17528938692953488)
  Refusing to use a backend that computes the wrong thing.
```

**What that means.** Before trusting any accelerator, the app runs the same image through the plain CPU and
compares. A healthy accelerator differs from the CPU by a tiny rounding amount. This one differed by **4.946%
of the output range** — and produced *the identical wrong numbers* on **three separate runs**, down to the last
decimal place of `absolute: 0.008670523762702942`. That is not random noise; it is a stable, reproducible
property of that chip's arithmetic.

For scale: healthy configurations on this same phone deviate by **0.0005%**. This is roughly 10,000× that, and
5× the 1% tolerance the library considers acceptable.

**Why it probably happens.** Precision. The Neural Engine computes in **fp16** — fewer digits per number than
the float32 the model was designed for. The `flutter_litert` docs say so explicitly: NPU accuracy is "bounded
by fp16" on both the Apple Neural Engine and Qualcomm's chips. A ~5% shift in a probability distribution is
what you'd expect from lower-precision arithmetic accumulating through a 174-tensor model.

**Three careful conclusions:**

* **The Neural Engine really was engaged.** The output changed substantially, which the CPU path cannot
  explain. So this isn't "the request was ignored".
* **The verification code earns its keep.** Without it, this configuration would have shipped and been
  silently, consistently slightly wrong. It wouldn't have crashed. Nobody would have filed a bug.
* **We cannot tell you how fast the Neural Engine is.** We refused it, so there is no timing — and quoting a
  speed for a configuration that computes the wrong answer would be worse than quoting nothing. Whether a model
  *designed* to tolerate fp16 would pass here is untested.

### Finding 2: the simulator lied about exactly this

The same NPU configuration, on the iPhone simulator, reported **healthy** — 0.0002% deviation. Because Core ML
on a simulator runs on the Mac's own processors and never touches a Neural Engine at all.

Simulator-only testing would have missed a real, reproducible correctness bug. That's the strongest argument
for physical-device validation this project can offer, and we got it by accident.

### Finding 3: the GPU is genuinely, verifiably about twice as fast

Measured three times, so you get the spread rather than the flattering single number:

| Run | GPU→CPU inference median | CPU-only inference median | Speed-up |
|---|---:|---:|---:|
| 1 | 4.53 ms | 9.54 ms | 2.11× |
| 2 | 4.57 ms | 9.57 ms | 2.09× |
| 3 | 4.90 ms | 9.54 ms | 1.95× |
| **Summary** | **4.53–4.90 ms** | **9.54–9.57 ms** | **≈2× (1.95–2.11×)** |

Note which side moves. The CPU-only path is almost perfectly repeatable (9.54, 9.57, 9.54); the GPU path is the
noisy one, which is what you'd expect from a shared resource with its own clock and queue. **So the honest claim
is "about 2×", not "2.11×"** — quoting the best of three runs would be exactly the kind of thing this project
exists to avoid.

The comparison itself is fair: identical API, identical model file, identical measurement path (both use
`runAsync`, so both pay the same background-thread cost). The GPU also deviated only 0.0005% from the CPU
reference — non-zero, so a genuinely different chip did the work, but far inside tolerance, so it's correct.

**iOS GPU acceleration is therefore verified, not assumed.** That's a sentence we couldn't write before
testing on a real phone.

### Finding 4: our simulator benchmark produced a wrong conclusion — and a "performance" delegate made things slower

Two versions of the same lesson.

**The API comparison reversed.** On the simulator, the classic `Interpreter` API looked ~2.9× faster than the
newer `CompiledModel`, and an earlier draft of this document drew a conclusion from that. On real hardware:

| Configuration | Simulator | **iPhone 13 Pro** |
|---|---:|---:|
| CompiledModel · GPU→CPU | 11.47 ms | **4.53 ms** |
| Interpreter · XNNPACK (float32) | 3.93 ms | 4.79 ms |

The ordering flips. The simulator result was an artifact of having no mobile GPU: `CompiledModel` fell back to
a CPU path while XNNPACK ran optimised CPU code, so the comparison was never really about the two APIs.

**But don't over-read the device column either.** Repeating the suite three times shows these two are
effectively tied:

| Run | CompiledModel · GPU→CPU | Interpreter · XNNPACK | Winner |
|---|---:|---:|---|
| 1 | 4.53 ms | 4.79 ms | GPU by 0.26 ms |
| 2 | 4.57 ms | 5.12 ms | GPU by 0.55 ms |
| 3 | 4.90 ms | 4.85 ms | XNNPACK by 0.05 ms |

The GPU path wins two of three, by margins that sit inside the run-to-run variance. The correct conclusion is
**"these two are equivalent on this device"**, not "the newer API is faster". Declaring a winner from run 1 alone
would have repeated the exact mistake this section is about — one measurement is an anecdote.

What *is* robust across all three runs: the quantized no-delegate path was fastest every time (3.51 / 3.56 /
3.54 ms), and it's the most repeatable figure in the project.

**The delegate comparison also reversed.** XNNPACK is a plug-in whose whole purpose is to make things faster:

| Configuration (V1 uint8) | Simulator | **iPhone 13 Pro** |
|---|---:|---:|
| Interpreter · XNNPACK | 4.67 ms | 6.17 ms |
| Interpreter · no delegate | 4.95 ms | **3.51 ms** |

On the A15, attaching XNNPACK to the quantized model cost **76% more** time than using nothing at all. The
plain path is the fastest inference measured anywhere in this project.

**The lesson for a beginner, and it's the most useful one here:** do not benchmark on a simulator and draw
architectural conclusions, and do not assume a component named "accelerator" or "optimiser" actually
accelerates *your* model on *your* device. Always measure the boring baseline too. This cost us a wrong claim
to learn, which is why it's written down instead of quietly deleted.

### And the thing that dominates everything: image handling

~15 ms of decoding and resizing the JPEG in Dart, against 3.5–9.5 ms of actual model work.

Moving from the simulator to real hardware made inference ~4× faster but preprocessing only ~1.3× faster
(19.5 → 15 ms) — because preprocessing is plain single-threaded Dart code, not accelerated kernels. On real
hardware, preprocessing is now **75–81% of the total time**.

If someone asked us to make this app faster, we would not touch the model.

### One more practical detail: the first model load is expensive

We measured initialisation twice per backend in the same app session. The first `CompiledModel` creation pays
to set up the whole LiteRT environment (GPU stack, adapter enumeration, kernel cache), which is then shared:

| Backend | 1st init in the process | 2nd init |
|---|---:|---:|
| CompiledModel · GPU→CPU | **687.6–922.1** | 89.8–91.3 |
| CompiledModel · CPU only | 35.5 | 29.5 |
| Interpreter · XNNPACK (V2) | 235.1 | 4.1 |
| Interpreter · no delegate (V1) | 2.9 | 1.2 |

Between 0.7 and 0.9 seconds for the first GPU-backed model is a real, user-visible delay on a real phone — and
notice it's the *least* repeatable number in the whole project (687.6 ms in run 1, 922.1 ms in run 3), because
it depends on driver and cache state the app doesn't control. The fix is to warm up off the critical path — this
app initialises in a post-frame callback so the interface appears first.

The very first preprocess of a session is similarly expensive: 286 ms in run 3, against a 15 ms warm median,
because Dart's image code is still warming up. Another reason to do one throwaway prediction before the user
sees anything.

---

## 3. The emulated targets, kept only as a cautionary tale

These are retained to show how misleading emulated measurement can be. **Neither represents phone hardware:**
the iOS simulator runs on the Mac's own processors and has no Neural Engine; the Android emulator has no
working OpenCL, so GPU compilation fails outright.

### iOS Simulator — iPhone 17, iOS 26.2, 8 CPUs

| Backend | Model | Init | Cold | Warm total med | Warm inference min / med / max | Preproc med |
|---|---|---:|---:|---:|---:|---:|
| CompiledModel · GPU→CPU | V2 f32 | 124.7 | 41.6 | 31.3 | 7.26 / 11.47 / 24.73 | 19.54 |
| CompiledModel · CPU only | V2 f32 | 33.7 | 32.0 | 30.9 | 10.57 / 11.41 / 15.06 | 19.31 |
| CompiledModel · NPU→CPU | V2 f32 | 588.6 | 39.2 | 36.6 | 14.97 / 16.80 / 24.97 | 19.55 |
| Interpreter · XNNPACK | V2 f32 | 6.8 | 25.8 | 23.9 | 3.66 / 3.93 / 4.84 | 19.63 |
| Interpreter · XNNPACK | V1 u8 | 12.7 | 25.9 | 24.4 | 4.47 / 4.67 / 5.37 | 19.43 |
| Interpreter · no delegate | V1 u8 | 2.6 | 25.3 | 24.9 | 2.38 / 4.95 / 8.08 | 19.55 |

Look at row 3. The simulator ran the NPU configuration and pronounced it **healthy** (0.0002% deviation). The
same configuration is numerically broken on the real A15.

### Android Emulator — Pixel 8, arm64-v8a, 4 CPUs

| Backend | Model | Init | Cold | Warm total med | Warm inference min / med / max | Preproc med |
|---|---|---:|---:|---:|---:|---:|
| CompiledModel · GPU→CPU **(GPU failed → CPU)** | V2 f32 | 308.8 | 81.1 | 40.5 | 10.74 / 12.24 / 21.71 | 27.07 |
| CompiledModel · CPU only | V2 f32 | 57.0 | 44.1 | 43.2 | 10.83 / 13.18 / 28.27 | 29.30 |
| CompiledModel · NPU→CPU **(narrowed → CPU)** | V2 f32 | 152.9 | 141.2 | 39.3 | 10.48 / 12.02 / 19.39 | 26.70 |
| Interpreter · XNNPACK | V2 f32 | 7.1 | 32.6 | 39.6 | 3.01 / 6.17 / 50.36 | 29.04 |
| Interpreter · XNNPACK | V1 u8 | 17.2 | 32.2 | 36.6 | 4.97 / 8.48 / 23.37 | 27.11 |
| Interpreter · no delegate | V1 u8 | 7.4 | 58.9 | 36.0 | 5.54 / 10.88 / 52.95 | 25.37 |

The GPU request failed with `LiteRtCreateManagedTensorBufferFromRequirements input[0] failed with
LiteRtStatus=3 (kLiteRtStatusErrorRuntimeFailure)` and correctly fell back to the CPU — which is the *right*
behaviour, and worth seeing once.

Notice how noisy this is: the worst runs are 4–8× the medians, and two cold-to-warm ratios come out *below* 1.0
(the first run appearing faster than later ones, which is physically implausible and simply means the host
machine's scheduling dominates). That noise is exactly why this project reports min/median/p90/max instead of a
single number.

**Android GPU and vendor-NPU acceleration remain NOT VERIFIED** — no physical Android device was available.

---

## 4. Binary size, measured

How much bigger does shipping on-device AI make your app? Built with
`flutter build apk --release --split-per-abi`, compared against a bare `flutter create` app on the same
toolchain:

| APK | This PoC | Empty Flutter app | Added |
|---|---:|---:|---:|
| arm64-v8a | **51.5 MB** | 15.5 MB | **+36.0 MB** |
| armeabi-v7a | 41.5 MB | 12.7 MB | +28.8 MB |
| x86_64 | 57.5 MB | 16.9 MB | +40.6 MB |
| universal (3 ABIs) | 112.5 MB | — | — |

iOS device build: `Runner.app` is 58.2 MB unsigned, before App Store thinning.

Where those 36 MB go — two models plus the runtime that executes them:

| Component | Size |
|---|---:|
| `mobilenet_v2_1.0_224.tflite` (float model) | 13.33 MB |
| `mobilenet_v1_1.0_224_quant.tflite` (quantized model) | 4.08 MB |
| `libLiteRt.so` | 5.14 MB |
| `libtensorflowlite_jni.so` | 4.27 MB |
| `libLiteRtClGlAccelerator.so` | 2.70 MB |
| `libtensorflowlite_gpu_jni.so` | 2.47 MB |
| `libtflite_custom_ops.so` | 0.01 MB |
| **LiteRT native total** | **14.59 MB** |

Ways to shrink it, biggest lever first: ship one model instead of two (−13.3 MB by dropping the float one,
though on this device the float+GPU path is the fastest *correct* float option, so it's a genuine trade-off);
always split per ABI or ship an App Bundle (−61 MB versus that universal APK); set
`flutterLitert.bundleGpuAccelerator=false` if you never use the GPU (~−3 MB per the package docs, **not
verified here**); or download the models after install instead of bundling them.

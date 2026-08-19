# Q&A preparation

New to this? Read [`GLOSSARY.md`](GLOSSARY.md) first, then **Part A** below.

Every technical answer opens with a **short answer** in bold — enough for a beginner or a quick skim — followed
by the detail you'd need if someone pushes. Figures are from a physical iPhone 13 Pro (A15) unless stated.
Where something wasn't measured, the answer says so.

* **Part A** — 6 questions beginners actually ask
* **Part B** — 22 questions senior engineers are likely to ask

---

# Part A — beginner questions

### A1. Is this "real AI", or a trick?

Real machine learning, and also not magic. Nobody wrote rules like "if it has fur, guess dog" — the numbers in
the model file were *learned* from about a million example photos. But there's no reasoning happening either:
it's pattern-matching arithmetic. That's precisely why it will confidently misidentify a coffee mug. It has no
concept of being wrong, and no way to say "I don't know".

### A2. Does the app learn from my photos?

**No.** The model file is read-only and identical for every user. Learning (training) happened at Google, before
we ever downloaded the file. This app only *runs* the frozen model — an operation called inference. Nothing you
do in the app changes it.

### A3. Why bother, when you could just call a cloud API?

Four reasons this project demonstrates: it works with no network at all; the photo never leaves the device, so
there's no privacy review to pass; there's no per-prediction cost; and there's no network round trip, so you wait
about 20 ms instead of a few hundred.

And two honest reasons to choose cloud instead: you can update a server-side model instantly, and a server can
run models far too large for a phone.

### A4. Do I need to know maths to work on this?

**Not to use a model, which is what this project does.** Everything in this repo is file loading, image resizing,
buffer shapes, threading and measurement — ordinary engineering. Maths becomes necessary when you want to
*train* or modify a model, which is a different job.

The one concept worth internalising isn't maths: **the model accepts exactly one shape of input, and getting it
wrong degrades your results silently instead of crashing.**

### A5. What's a "tensor", really?

A grid of numbers with a known shape. A single number is 0-D, a list is 1-D, a spreadsheet is 2-D, a colour image
is 3-D (width × height × 3 colours). It's just the word ML uses for "a box of numbers whose shape everyone
agrees on".

Ours is `[1, 224, 224, 3]`: one image, 224 pixels tall, 224 wide, 3 colour values each — 150,528 numbers, which
as 32-bit floats is 602,112 bytes.

### A5b. Why is the confidence so low on my own photos?

**Short answer: usually because the thing you photographed isn't one of the model's 1,001 categories — and the
model has no way to say "I don't know".**

Three causes, in order of how often they're the real one:

1. **The subject isn't in the label set.** These are the categories ImageNet-1k does *not* contain:

   ```text
   person   man   woman   building   road   food   smartphone
   ```

   There are 120 dog breeds and no "person". So a photo of a colleague, a room, your lunch, or a screen
   **cannot** be classified correctly. The model still returns its best guess, because it always returns
   something — you get the nearest thing it knows, at 10–30%.

2. **Low confidence is normal for this model anyway.** The bundled examples score 0.39 (Labrador) and 0.43
   (cat) on a real device. Only the curated Grace Hopper photo reaches 0.87. **0.87 is the outlier, not the
   baseline.** MobileNetV2 is a small 2018 model with ~71.8% top-1 accuracy on ImageNet itself.

3. **Framing.** By default the app *stretches* your photo to a square, and phone photos are 4:3 or 16:9, so they
   distort much more than the near-square bundled samples. Use the **Centre-crop** toggle to switch to the
   standard ImageNet recipe — measured at nearly double the confidence on a centred subject (Labrador
   0.28 → 0.51). It isn't a universal win; see question 19b.

The app now flags this for you: below 50% it shows *"Low confidence"*, and when the top score is low **and** the
runner-up is close it shows *"Inconclusive"*, which usually means the subject is outside the vocabulary.

### A6. Where should I start if I want to try this myself?

Clone the repo, run `./tool/fetch_models.sh`, then `flutter run`. Then point the app at something *not* in its
1,001 categories — a charger, a houseplant, your desk — and watch it confidently guess the nearest thing it
knows. That's the single most instructive five minutes available here.
[`GLOSSARY.md`](GLOSSARY.md) lists more experiments.

---

# Part B — engineering questions

### 1. Why LiteRT instead of ML Kit?

**Short answer: because ML Kit solves a different problem — ready-made features, not your own model.**

ML Kit gives you tuned, ready-made capabilities — OCR, barcode, face detection, pose, generic labelling — with
the preprocessing already handled. If our task were any of those, using LiteRT to reimplement it would be worse
engineering.

Our premise is a **custom model** with full control of the pipeline, which is LiteRT's job. Also worth knowing:
ML Kit's own custom-model paths still constrain you to a specific input contract, and ML Kit is Android/iOS only.
One thing that is *not* a differentiator: both Flutter bindings are community-maintained. The `google_mlkit_*`
packages state explicitly that they are not sponsored or maintained by Google.

### 2. Why LiteRT instead of ONNX Runtime?

**Short answer: on mobile, LiteRT's toolchain is the more travelled path — but if you already emit ONNX, use
ONNX Runtime.**

If you have an ONNX pipeline, converting to `.tflite` adds a lossy step you must re-validate every release.
Genuine reasons to prefer LiteRT on mobile: the quantization and delegate toolchain is better trodden, vendor NPU
runtimes target it first, and the Flutter binding situation is less fragmented — the most-referenced
`onnxruntime` plugin appears unmaintained and has been forked (`onnxruntime_v2`, plus `flutter_onnxruntime` and
`fonnx`).

Not a reason: "ONNX is more standard." On mobile specifically that is not the deciding factor.

### 3. Is LiteRT actually using the NPU?

**Short answer: on a real iPhone, yes — Core ML was engaged, it computed the wrong answer, and we refused it.**

Requesting `{npu, cpu}` compiled successfully, then failed the startup correctness check with a deviation of
**4.946% of output range** against a plain-CPU reference, reproducible bit-for-bit across independent runs.
Healthy configurations on the same device deviate 0.0005%. That the output moved so far is itself evidence Core ML
genuinely ran the model; it's also evidence the result is unusable. The most likely cause is fp16 precision — the
binding documents that the Apple Neural Engine and Qualcomm HTP compute in reduced precision.

Consequences stated carefully: we have **no Neural Engine latency figure**, because quoting one would mean
quoting a configuration that computes the wrong answer. And note the trap — the **iOS simulator reported this
same configuration as healthy** (0.0002%), because Core ML there runs on the host Mac and never touches a Neural
Engine. Simulator-only testing would have shipped this.

On Android, no vendor NPU runtime is installed, so the request is silently narrowed to CPU and the UI shows
`effective = CPU`.

### 4. How do we know inference is happening locally?

**Short answer: the release app has no internet permission, so the OS kernel physically won't let it connect.**

Four independent pieces of evidence, in descending strength:

1. The **release APK declares no `INTERNET` permission**. On Android that's install-time and kernel-enforced —
   the process cannot open a socket. Verified with `aapt2 dump permissions` on the built artifact.
2. The full integration suite (15 tests, 6 backends, 3 images) **passes with the network unreachable** — airplane
   mode on, `ping` returning `Network is unreachable`, no routes at all.
3. The app's user id owns **zero TCP/UDP connections** while running with both models loaded, verified against a
   positive control showing the same query does detect another app's connection.
4. The `.tflite` files are inside the APK, byte-for-byte the sizes the code validates at startup, loaded via
   `rootBundle`. There's no HTTP client in the dependency graph.

Details, including which checks are weak and one we couldn't complete, are in
[`OFFLINE_VERIFICATION.md`](OFFLINE_VERIFICATION.md).

### 5. GPU acceleration — did that actually work?

**Short answer: yes, verified — about 2× faster than the same API on CPU only, measured three times.**

This is the one acceleration claim this PoC makes cleanly. On the A15, `{gpu, cpu}` reported
`effective = GPU + CPU`, deviated 0.0005% from the CPU reference (non-zero, so a different compute path genuinely
ran; well inside tolerance, so it's correct), and measured **4.53 ms against 9.54 ms** for the same API and
weights CPU-only. Repeated three times: 2.11×, 2.09×, 1.95× — so the defensible claim is **≈2×**, not the best
single figure. Interestingly the CPU side barely moves (9.54, 9.57, 9.54) while the GPU side does (4.53, 4.57,
4.90), which is what you'd expect from a shared resource with its own clock and queue. Same measurement path on
both sides, including the background-thread hop, so it's like-for-like.

Android GPU is **not verified**: the emulator has no working OpenCL and compilation fails outright with
`kLiteRtStatusErrorRuntimeFailure`, correctly falling back to CPU.

### 6. What happens if the device doesn't have an NPU?

**Short answer: it falls back to a slower chip, and you only find out if you check.**

With `CompiledModel` the behaviour is well defined: a permissive request like `{gpu, cpu}` retries CPU-only if GPU
compilation throws; a *strict* single-accelerator request throws instead of degrading. On Android, `{npu, cpu}`
silently drops `npu` when no vendor runtime is installed.

We observed both on the Android emulator: GPU compilation failed with
`LiteRtCreateManagedTensorBufferFromRequirements … kLiteRtStatusErrorRuntimeFailure` and rebuilt CPU-only; the NPU
request was narrowed to CPU. The app compares *requested* against *effective* and displays the difference,
because a PoC that printed only the request would be claiming acceleration it never got.

### 7. What happens to battery consumption?

**Short answer: not measured. One prediction is negligible; sustained use is the real question.**

Measuring it properly needs sustained runs on physical hardware with power instrumentation, which was out of
scope.

From first principles and the numbers we do have: a single inference is 3.5–9.5 ms of compute on an A15, which is
nothing. The concern is *sustained* inference — a live camera pipeline at 30 fps means continuous chip load, which
raises temperature, triggers thermal throttling, and drains the battery. NPUs exist largely because they do this
work at far better energy-per-prediction than a CPU, which is a reason to care that our Neural Engine path failed
correctness. Also relevant: in this pipeline the JPEG decode and resize cost ~3× the inference, so a naive camera
loop would burn most of its power on image handling, not on the model.

### 8. How big can the model realistically be?

**Short answer: tens of MB is routine. Hundreds of MB means downloading it separately and managing memory
carefully.**

The constraints are memory and app size rather than a hard limit. Our 13.3 MB float32 model loads in ~91 ms on an
A15 (second load in-process; the first pays 690–920 ms to set up the shared LiteRT/GPU environment). Practical
guidance:

* Weights stay resident in RAM while loaded, plus working memory. Two float32 models at once is ~28 MB, which is
  why we free one before loading another.
* Android kills background apps long before foreground ones; a few hundred MB of weights is a bad idea on
  low-end devices.
* App size is the harder ceiling in practice. Our two models add 17.4 MB; the arm64 APK is 51.5 MB against
  15.5 MB for an empty Flutter app.
* Quantization is the main lever: our uint8 model is 3.27× smaller than the float one.

### 9. Can we update the model without releasing a new app version?

**Short answer: not as built — ours are bundled. A production system would download them, and that needs
signature verification.**

A real over-the-air path needs:

* Download to app-private storage; `CompiledModel.fromFile`/`fromBuffer` and `Interpreter.fromBuffer` accept
  bytes from anywhere, so the runtime side is trivial.
* **Verify a signature and digest before use.** A model file is executable intent; treat it like code.
* Keep the bundled model as a fallback and roll back on failure.
* **Version the tensor contract alongside the model.** A new model with a different input size or class count is
  a breaking API change to the app. Our `ModelSpec` is exactly that contract, and `initialize()` already
  validates size, shape, dtype and quantization — the same checks an OTA path needs.
* Staged rollout: a model regression is a behaviour regression you cannot hotfix from the client.

### 10. How does quantization affect accuracy?

**Short answer: typically ~1% top-1 on models like these — but our two models differ in architecture too, so be
careful what you attribute to quantization.**

Our quantized model is MobileNet**V1** while the float one is V2, so architecture and precision both differ.
State that when you show it.

What is purely a quantization artefact: the uint8 output has scale 1/256, so confidence resolution is 0.39% and
any class below 1/512 rounds to **zero**. Across 1,001 classes the dequantized distribution therefore sums to
0.95–0.99 instead of 1.0. So quantized confidences are not calibrated probabilities — relevant if you threshold
on them.

On behaviour: on the decisive sample both models agree ("military uniform", 0.875 float vs 0.859 quantized). On
the ambiguous Labrador photo the float model says "Labrador retriever" (0.39) and the quantized one says "Eskimo
dog" (0.29) with kuvasz and Labrador next — the reference implementation shows the same instability, so this is a
genuinely uncertain input where a coarse output grid flips near-ties.

Rule of thumb from the wider literature: post-training uint8 quantization typically costs ~1% top-1 on
ImageNet-class models, and quantization-aware training recovers most of it. Always validate on *your* data — and
validate on-device, since NPUs may compute in fp16 regardless.

### 11. How does this work differently on Android and iOS?

**Short answer: same model file, genuinely different chips and frameworks underneath — which is why the
abstraction exists.**

| | Android | iOS |
|---|---|---|
| Runtime library | `libLiteRt.so`, `libtensorflowlite_jni.so` etc. per architecture | LiteRT framework linked into the app |
| CPU acceleration | XNNPACK | XNNPACK |
| GPU | OpenCL/OpenGL — driver-dependent, **failed on the emulator** | Metal — **verified on device** |
| NPU | Vendor runtime you must ship (e.g. Qualcomm HTP), API 31+ arm64 | Core ML → Neural Engine, iOS 13+, arm64 only — **rejected for wrong output** |
| Offline proof | `INTERNET` permission can be omitted → kernel-enforced | No equivalent install-time permission; must argue from architecture |
| Native version observed | `2.22.0-dev0+selfbuilt` | `2.20.0-dev0+selfbuilt` |

Note the last row: the same package version shipped **different native runtime versions** per platform. Also
Android GPU has an extra wrinkle — the binding documents `runAsync` as unvalidated against thread-affine GL/CL
drivers, so our code deliberately dispatches synchronously when a GPU backend is live on Android.

### 12. How does Flutter communicate with native ML runtimes?

**Short answer: through `dart:ffi` — direct C calls, no serialisation.**

`flutter_litert` binds the LiteRT C API directly, so a Dart call becomes a C call with no message hop to the
platform thread. Input goes in as a `Float32List`/`Uint8List` whose bytes are `memcpy`'d straight into the tensor
buffer.

That matters for performance: the old pattern of passing nested `List<List<List<double>>>` through a platform
channel would allocate ~150,000 boxed doubles per frame and serialise them all. We pass flat typed data and the
binding does one memcpy.

### 13. Does Dart perform the actual inference?

**Short answer: no. Dart prepares bytes and reads results; native code does the maths.**

Dart does three things: normalisation (a loop over 150,528 bytes), the FFI call, and the final
dequantise-and-sort. The convolutions run in compiled native kernels — hand-optimised NEON on CPU, Metal/OpenCL
shaders on GPU, or a vendor NPU library. If Dart were doing the matrix multiplies, inference would be orders of
magnitude slower.

### 14. What happens if inference takes too long?

**Short answer: this PoC has a partial policy — it can't hang the app, but it can't cancel either.**

What exists: inference is awaited off the UI thread, the controller has an explicit `running` state that disables
the buttons, and failures surface as user-facing errors rather than a hang.

What's missing for production: a timeout with cancellation. That's harder than it looks —
`CompiledModel.runAsync` and `IsolateInterpreter.run` don't expose cancellation, and LiteRT invocations aren't
interruptible mid-model. Realistic strategies: bound work by throttling input frames rather than cancelling
inference; pick a smaller model for low-end tiers; or run inference in an isolate you own and kill the isolate if
it exceeds budget, accepting the cost of rebuilding the interpreter.

### 15. Can inference block the UI thread?

**Short answer: yes, and in one of our configurations it does — deliberately, and the UI says so.**

`CompiledModel.runAsync()` dispatches the blocking native call on a helper isolate, so the UI thread stays free.
On the classic `Interpreter` path we use `IsolateInterpreter` — but only when no delegate is attached, because
this binding cannot safely share a delegated native interpreter across isolates. So the XNNPACK configurations
run `invoke()` on the calling isolate, blocking it for 4.8–6.2 ms — roughly one dropped frame at 60 Hz.

The production fix is to own a long-lived worker isolate and construct the interpreter *inside* it, so the native
handle never crosses an isolate boundary; the package ships `IsolateWorkerBase` for this. Note also that isolates
give you concurrency, not free parallelism: a synchronous `run()` on the platform thread stalls rendering no
matter how the Dart code is structured.

Worth noting from the device measurements: on the A15 the *fastest* configuration (quantized, no delegate,
3.51 ms) is also the one that can use `IsolateInterpreter` — so here there was no trade-off to make.

### 16. Is it safe to process sensitive information this way?

**Short answer: structurally much better than sending it to a server — but "on-device" is not automatically
"secure".**

The pixels never leave the process, so there's no transport to intercept, no server logs, no retention policy,
and nothing to subpoena. For regulated data that's a materially easier compliance story, and our release build
cannot even open a socket.

Things that still need attention:

* The **model file is readable** in the app bundle. Anyone can extract it. If the model itself is intellectual
  property, bundling it in plaintext gives it away, and obfuscation only raises the cost.
* **Model inversion / membership inference** are real attack classes if the model memorised training data.
* **Anything you persist** — results, cached crops, logs — is back to ordinary data protection.
* **Analytics leakage** is the common own-goal: sending the *label* to your analytics backend undoes the privacy
  claim entirely.
* Device backups may include app storage unless you exclude it.

### 17. Why two models and two APIs? Isn't that redundant?

**Short answer: it turns a documented constraint into a demonstrated fact, and proves the abstraction is real.**

LiteRT Next's `CompiledModel` is **float32-only**, so the uint8-quantized model literally cannot run on it — it
must use the classic `Interpreter`. Having both behind one interface makes that a fact in the repo (enforced in
code, covered by a test), gives an apples-to-apples comparison of the two APIs on identical weights, and produces
the measured quantization trade-off. It also proves the abstraction works: two very different implementations,
one interface, no changes above it.

### 18. Which API should we actually use — CompiledModel or Interpreter?

**Short answer: measure both on your target device. On our phone the answer differed from the simulator's.**

As of this PoC:

* **`CompiledModel`** if your model is float32 and you want the runtime to handle GPU/NPU selection and report
  what it kept. On the real A15 with Metal available it was **tied** with `Interpreter`+XNNPACK for the fastest
  correct float path (4.53–4.90 ms vs 4.79–4.85 ms over three runs — inside run-to-run noise).
* **`Interpreter`** if you need quantized/integer input/output, named signatures, custom operators or on-device
  training. On the A15 the quantized model with **no delegate** gave the fastest inference measured anywhere in
  this project (3.51 ms), and it's the only `Interpreter` configuration that can use `IsolateInterpreter`.

A cautionary tale attached to this question: on the iOS **simulator**, `Interpreter`+XNNPACK looked ~2.9× faster
than `CompiledModel`, and an earlier version of this material concluded from that. On real hardware the ordering
reverses, because the simulator has no mobile GPU. The methodological answer is more valuable than the API
answer: **benchmark on the device tiers you ship to.**

### 19. Why is preprocessing slower than inference, and what would you do about it?

**Short answer: because we decode and resize a JPEG in pure Dart on every prediction. The fix is to stop
producing JPEGs.**

On the A15 that's ~15 ms of preprocessing against 4.5 ms of inference — 75–81% of total latency. It also scales
with the *source* image size, while inference is constant because the model always sees 224×224. (Measured on the
iOS simulator, where total preprocessing was 19.5 ms: 11.4 ms for a 320×213 photo versus 26 ms for a 700×577 one.
The relationship holds on device; only the absolute numbers shrink.)

What we'd do: take camera frames directly (YUV420), do the colour conversion and resize on the GPU or in native
code, and reuse a preallocated buffer instead of allocating a new `Float32List` per frame. `flutter_litert` ships
camera-frame helpers for exactly this path. Only after that would model optimisation be worth anything.

### 19b. Stretch or centre-crop — which should the app use?

**Short answer: neither wins everywhere, so the app ships both and lets you compare. Stretch is the default
because it matches the reference fixture.**

Measured against the Python reference on the three bundled samples:

| image | stretch | resize-256 + centre-crop-224 |
|---|---|---|
| labrador.jpg | Labrador retriever 0.2751 | Labrador retriever **0.5101** |
| cat_on_snow.jpg | lynx 0.3524 | Egyptian cat 0.4477 |
| grace_hopper.jpg | military uniform **0.8035** | mortarboard 0.5195 |

The pattern is about *where the identifying detail sits*. Cropping nearly doubles confidence on the Labrador,
whose subject is centred and fills the frame. It **loses** Grace Hopper, because the crop removes the uniform and
keeps the hat — so the label changes to "mortarboard", confidently.

Why stretch remains the default: it can never crop the subject out of view, and it is what
`tool/reference_predict.py` does, so the committed fixture and the bit-level parity tests describe that path.
Switching the default would mean regenerating the fixture and re-validating every backend, which is a real cost
for a trade that doesn't clearly pay.

Published accuracy figures for these models are measured with the centre-crop recipe, so it's the more faithful
comparison **when your subject is where the recipe assumes it is** — centred, filling the frame. For a real
product, crop to the subject (with a detector) rather than to the centre.

### 20. How do you know your preprocessing is correct? It seems easy to get wrong.

**Short answer: it's extremely easy to get wrong, so it's tested at two levels — and the tests caught a real
bug.**

1. **Bit-level.** A deterministic 224×224 PNG needs no resizing, so Dart's tensor is compared element-by-element
   against the Python reference tensor. That isolates normalisation, channel order and layout with resizing
   removed from the equation.
2. **End-to-end.** On-device predictions are compared against the Python LiteRT reference on the same model
   files, with assertions graded by how decisive the reference is.

This caught a real bug. Dart's `Interpolation.linear` doesn't antialias when shrinking an image, while the
reference's bilinear does. Mean error was 7.7 levels out of 255 and the top-5 tail was reordered. Switching to
area-averaging for ≥1.5× shrink cut the error to 4.0 and fixed a wrong top answer on the Labrador photo. A
regression test now asserts the chosen filter beats every alternative.

The general lesson, and it's the one worth repeating to beginners: **preprocessing parity matters as much as the
model, and the failure mode is silent degradation, not a crash.**

### 21. What would you do differently for production?

**Short answer: validate on real devices across tiers, ship one model, move preprocessing off Dart, and build
the OTA pipeline properly.**

1. Validate on **physical devices** across tiers — this PoC already shows why: a backend the simulator called
   healthy is numerically broken on a real A15. Extend to an Android phone, which remains unmeasured.
2. Ship **one** model, chosen by measurement, and split per architecture or use an App Bundle.
3. Move preprocessing to native/GPU and feed camera frames rather than JPEGs.
4. Own a worker isolate for inference so a delegate never forces work onto the UI thread.
5. Build the OTA model pipeline: signed, digest-verified, versioned against the tensor contract, with rollback.
6. Put the reference-comparison check in CI, per model and per target architecture.
7. Add telemetry for what actually happened per device — effective accelerators, warm latency percentiles —
   because chip selection varies in the field in ways you cannot predict from your desk.
8. Decide the timeout/cancellation policy properly, and pin the `flutter_litert` version given it's
   community-maintained.

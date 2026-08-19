# Q&A preparation

Twenty-one questions senior engineers are likely to ask, with answers that are technically accurate and,
where something was not measured, say so. Figures are from a physical iPhone 13 Pro (A15) unless stated.

---

### 1. Why LiteRT instead of ML Kit?

Because ML Kit solves a different problem. ML Kit gives you tuned, ready-made capabilities — OCR, barcode,
face detection, pose, generic labelling — with the preprocessing already handled. If our task were any of
those, using LiteRT to reimplement it would be worse engineering.

Our premise is a **custom model** with full control of the tensor pipeline, which is LiteRT's job. Also worth
knowing: ML Kit's own custom-model paths still constrain you to a specific input contract, and ML Kit is
Android/iOS only. One thing that is *not* a differentiator: both Flutter bindings are community-maintained.
The `google_mlkit_*` packages state explicitly that they are not sponsored or maintained by Google.

### 2. Why LiteRT instead of ONNX Runtime?

If you already have an ONNX pipeline, ONNX Runtime is a reasonable choice and converting to `.tflite` would
add a lossy step you must re-validate every release. Genuine reasons to prefer LiteRT on mobile: the
quantization and delegate toolchain is the more travelled path, vendor NPU runtimes target it first, and the
Flutter binding situation is less fragmented — the most-referenced `onnxruntime` plugin appears unmaintained
and has been forked (`onnxruntime_v2`, plus `flutter_onnxruntime` and `fonnx` as alternatives).

Not a reason: "ONNX is more standard." On mobile specifically that is not the deciding factor.

### 3. Is LiteRT actually using the NPU?

On a real iPhone 13 Pro: **yes, Core ML was engaged — and it computed the wrong answer, so we refused it.**

Requesting `{npu, cpu}` compiled successfully, then failed the startup correctness check with a deviation of
**4.946% of output range** against a plain-CPU reference, reproducible bit-for-bit across independent runs.
Healthy configurations on the same device deviate 0.0005%. That the output moved so far is itself evidence
Core ML genuinely ran the graph; it is also evidence the result is unusable. The most likely cause is fp16
precision — the binding documents that the Apple Neural Engine and Qualcomm HTP compute in reduced precision.

Consequences I state carefully: we have **no ANE latency figure**, because quoting one would mean quoting a
configuration that computes the wrong answer. And note the trap — the **iOS simulator reported this same
configuration as healthy** (0.0002%), because Core ML there runs on the host Mac and never touches a Neural
Engine. Simulator-only testing would have shipped this.

On Android, no vendor NPU runtime is installed, so the request is silently narrowed to CPU and the UI shows
`effective = CPU`.

### 4. How do we know inference is happening locally?

Four independent pieces of evidence, in descending strength:

1. The **release APK declares no `INTERNET` permission**. On Android that is install-time and kernel-enforced
   — the process cannot open a socket. Verified with `aapt2 dump permissions` on the built artifact.
2. The full integration suite (15 tests, 6 backends, 3 images) **passes with the network unreachable** —
   airplane mode on, `ping` returning `Network is unreachable`, no routes at all.
3. The app's uid owns **zero TCP/UDP sockets** while running with both models loaded, verified against a
   positive control showing the same query does detect another app's established socket.
4. The `.tflite` files are inside the APK, byte-for-byte the sizes the code validates at startup, loaded via
   `rootBundle`. There is no HTTP client in the dependency graph.

Details, including which checks are weak and one I could not complete, are in `docs/OFFLINE_VERIFICATION.md`.

### 4b. GPU acceleration — did that work?

Yes, and it is the one acceleration claim this PoC can make cleanly. On the A15, `{gpu, cpu}` reported
`effective = GPU + CPU`, deviated 0.0005% from the CPU reference (non-zero, so a different compute path ran;
well inside tolerance, so it is correct), and measured **4.53 ms against 9.54 ms** for the same API and
weights CPU-only — a **2.11× speed-up**. Same measurement path on both sides, including the isolate hop, so
it is like-for-like.

Android GPU is **not verified**: the emulator has no working OpenCL and compilation fails outright with
`kLiteRtStatusErrorRuntimeFailure`, correctly falling back to CPU.

### 5. What happens if the device doesn't have an NPU?

It falls back, and you find out only if you look. With `CompiledModel` the behaviour is well defined: a
permissive request like `{gpu, cpu}` routes through a fallback path that retries CPU-only if GPU compilation
throws; a *strict* single-accelerator request throws instead of degrading. On Android, `{npu, cpu}` silently
drops `npu` when no vendor runtime is installed.

We observed both on the Android emulator: GPU compilation failed with
`LiteRtCreateManagedTensorBufferFromRequirements … kLiteRtStatusErrorRuntimeFailure` and rebuilt CPU-only;
the NPU request was narrowed to CPU. The app compares *requested* against *effective* and displays the
difference, because a PoC that printed the request would be claiming acceleration it never got.

### 6. What happens to battery consumption?

**Not measured** — that needs sustained runs on physical hardware with power instrumentation, which was out
of scope.

What I can say from first principles and from the numbers I do have: a single inference is 3.5–9.5 ms of
compute on an A15, which is negligible. The concern is *sustained* inference — a live camera pipeline at
30 fps means continuous CPU or GPU load, which raises SoC temperature, triggers thermal throttling, and
increases drain. NPUs exist largely because they do this work at far better joules-per-inference than a CPU,
which is a reason to care that our ANE path failed correctness. Also relevant: in this pipeline the JPEG
decode and resize cost ~3× the inference, so a naive camera loop would burn most of its power on image
handling, not on the model.

### 7. How big can the model realistically be?

Constraints are memory and app size rather than a hard limit. Our 13.3 MB float32 model loads in ~91 ms on
an A15 (second load in-process; the first pays ~688 ms for the shared LiteRT/GPU environment). Practical guidance:

* Weights are resident in RAM while loaded, plus a tensor arena. Two float32 models at once is ~28 MB, which
  is why we dispose before switching.
* Android will kill a background app well before it kills a foreground one; a few hundred MB of weights is a
  bad idea on low-end devices.
* App size is the harder ceiling in practice. Our two models add 17.4 MB; the arm64 APK is 51.5 MB against
  15.5 MB for an empty Flutter app.
* Quantization is the main lever: our uint8 model is 3.27× smaller than the float one.

Models in the tens of MB are routine on mobile. Hundreds of MB means out-of-band delivery and careful
lifecycle management.

### 8. Can we update the model without releasing a new app version?

Yes, but not as it stands — ours are bundled assets, so today it takes an app release. A production OTA path
needs:

* Download to app-private storage; `CompiledModel.fromFile`/`fromBuffer` and `Interpreter.fromBuffer` accept
  bytes from anywhere, so the runtime side is trivial.
* **Verify a signature and digest before use.** A model file is executable intent; treat it like code.
* Keep the bundled model as a fallback and roll back on failure.
* **Version the tensor contract alongside the model.** A new model with a different input size or output
  class count is a breaking API change to the app. Our `ModelSpec` is exactly that contract, and
  `initialize()` already validates size, shape, dtype and quantization — the same checks an OTA path needs.
* Consider staged rollout: a model regression is a behaviour regression you cannot hotfix from the client.

### 9. How does quantization affect accuracy?

We have measured effects, though ours conflates two variables — our quantized model is MobileNet**V1** while
the float one is V2, so architecture and precision both differ. State that when you show it.

What is purely a quantization artefact: the uint8 output has scale 1/256, so confidence resolution is
0.39% and any class below 1/512 rounds to **zero**. Across 1001 classes the dequantized distribution
therefore sums to 0.95–0.99 instead of 1.0. So quantized confidences are not calibrated probabilities —
relevant if you threshold on them.

On behaviour: on the decisive sample both models agree ("military uniform", 0.875 float vs 0.859 quantized).
On the ambiguous Labrador photo the float model says "Labrador retriever" (0.39) and the quantized one says
"Eskimo dog" (0.29) with kuvasz and Labrador next — the reference implementation shows the same instability,
so this is a genuinely uncertain input where a coarse output grid flips near-ties.

Rule of thumb from the wider literature: post-training uint8 quantization typically costs ~1% top-1 on
ImageNet-class models, and quantization-aware training recovers most of it. Always validate on *your* data —
and validate on-device, since NPUs may compute in fp16 regardless.

### 10. How does this work differently on Android and iOS?

| | Android | iOS |
|---|---|---|
| Runtime library | `libLiteRt.so`, `libtensorflowlite_jni.so` etc. per ABI | LiteRT framework linked into the app |
| CPU acceleration | XNNPACK | XNNPACK |
| GPU | OpenCL/OpenGL — driver-dependent, **failed on the emulator** | Metal |
| NPU | Vendor runtime you must ship (e.g. Qualcomm HTP), API 31+ arm64 | Core ML → Neural Engine, iOS 13+, arm64 only |
| Offline proof | `INTERNET` permission can be omitted → kernel-enforced | No equivalent install-time permission; must argue from architecture |
| Native version observed | `2.22.0-dev0+selfbuilt` | `2.20.0-dev0+selfbuilt` |

Note the last row: the same package version shipped **different native runtime versions** per platform. Also
Android GPU has an extra wrinkle — the binding documents `runAsync` as unvalidated against thread-affine
GL/CL drivers, so our code deliberately dispatches synchronously when a GPU backend is live on Android.

### 11. How does Flutter communicate with native ML runtimes?

Through **`dart:ffi`**, not platform channels. `flutter_litert` binds the LiteRT C API directly, so a Dart
call becomes a C call with no serialisation and no message hop to the platform thread. Input goes in as a
`Float32List`/`Uint8List` whose bytes are `memcpy`'d straight into the tensor buffer.

That matters for performance: the old pattern of passing nested `List<List<List<double>>>` through a
platform channel would allocate ~150,000 boxed doubles per frame and serialise them. We pass flat typed data
and the binding does one memcpy.

### 12. Does Dart perform the actual inference?

No. Dart does three things: normalisation (a loop over 150,528 bytes), the FFI call, and the final
dequantise-and-sort. The convolutions run in compiled native kernels — XNNPACK's hand-tuned NEON on CPU, or
Metal/OpenCL shaders, or a vendor NPU library. If Dart were doing the matrix multiplies, inference would be
orders of magnitude slower.

### 13. What happens if inference takes too long?

You need a policy, and the honest answer for this PoC is that it has a partial one. What exists: inference is
awaited off the UI thread, the controller has an explicit `running` state that disables the buttons, and
failures surface as user-facing errors rather than a hang.

What is missing and would be needed in production: a timeout with cancellation. That is harder than it looks
— `CompiledModel.runAsync` and `IsolateInterpreter.run` do not expose cancellation, and LiteRT invocations
are not interruptible mid-graph. Realistic strategies: bound work by throttling input frames rather than
cancelling inference; choose a smaller/quantized model for low-end tiers; or run inference in an isolate you
own and kill the isolate if it exceeds budget, accepting the cost of rebuilding the interpreter.

### 14. Can inference block the UI thread?

Yes, and in one of our configurations it does — deliberately, and reported in the UI.

`CompiledModel.runAsync()` dispatches the blocking native call on a helper isolate, so the UI thread stays
free. On the classic `Interpreter` path we use `IsolateInterpreter` — but only when no delegate is attached,
because this binding cannot safely share a delegated native interpreter across isolates. So the XNNPACK
configurations run `invoke()` on the calling isolate, blocking it for 4–9 ms (one or two dropped frames).

The production fix is to own a long-lived worker isolate and construct the interpreter *inside* it, so the
native handle never crosses an isolate boundary; the package ships `IsolateWorkerBase` for this. Note also
that isolates give you concurrency, not free parallelism: a synchronous `run()` on the platform thread stalls
rendering no matter how the Dart code is structured.

### 15. Is it safe to process sensitive information this way?

It is *structurally* better than the alternative — the pixels never leave the process, so there is no
transport to intercept, no server logs, no retention policy, and nothing to subpoena. For regulated data
that is a materially easier compliance story, and our release build cannot even open a socket.

But "on-device" is not automatically "secure". Things that still need attention:

* The **model file is readable** in the app bundle. Anyone can extract it. If the model itself is IP, bundling
  it plaintext gives it away, and obfuscation only raises the cost.
* **Model inversion / membership inference** are real attack classes if the model memorised training data.
* **Anything you persist** — inference results, cached crops, logs — is back to ordinary data protection.
* **Analytics leakage** is the common own-goal: sending the *label* to your analytics backend undoes the
  privacy claim.
* Device backups may include app storage unless you exclude it.

### 16. Why two models and two APIs? Isn't that redundant?

They demonstrate a real constraint rather than assert it. LiteRT Next's `CompiledModel` is **float32-only**,
so the uint8-quantized model literally cannot run on it — it must use the classic `Interpreter`. Having both
behind one interface makes that a fact in the repo (enforced in code and covered by a test), gives an
apples-to-apples comparison of the two APIs on identical weights, and produces the measured quantization
trade-off. It also proves the abstraction is real: two very different implementations, one interface, no
changes above it.

### 17. Which API should we actually use — CompiledModel or Interpreter?

Decide per model and per target, and measure on **physical hardware**. As of this PoC:

* **`CompiledModel`** if your model is float32 and you want the runtime to handle GPU/NPU selection and
  report what it kept. On the real A15 with Metal available it was the fastest correct float path (4.53 ms).
* **`Interpreter`** if you need quantized/integer I/O, named signatures, custom ops or on-device training.
  On the A15 the quantized model with **no delegate** gave the fastest inference measured anywhere in this
  project (3.51 ms) — and it is also the only `Interpreter` configuration that can use `IsolateInterpreter`.

A cautionary tale attached to this question: on the iOS **simulator**, `Interpreter`+XNNPACK looked ~2.9×
faster than `CompiledModel`, and an earlier version of this material concluded from that. On real hardware
the ordering reverses, because the simulator has no mobile GPU. The methodological answer is more valuable
than the API answer: **benchmark on the device tiers you ship to.**

### 18. Why is preprocessing slower than inference, and what would you do about it?

Because we decode a 512×600 JPEG and resample it to 224×224 in pure Dart on every prediction: ~19.5 ms
against 3.9 ms of inference. It also scales with the *source* image — 11.4 ms for a 320×213 photo, 26 ms for
a 700×577 one — while inference is constant, because the model always sees 224×224.

What I would do: stop producing JPEGs. Take camera frames directly (YUV420), do the YUV→RGB conversion and
the resize on the GPU or in native code, and reuse a preallocated buffer instead of allocating a new
`Float32List` per frame. `flutter_litert` ships camera-frame helpers for exactly this path. Only after that
would model optimisation be worth anything.

### 19. How do you know your preprocessing is correct? It seems easy to get wrong.

It is extremely easy to get wrong, which is why correctness here is tested rather than assumed, at two
levels:

1. **Bit-level.** A deterministic 224×224 PNG needs no resizing, so Dart's tensor is compared
   element-by-element against the Python reference tensor. That isolates normalisation, channel order and
   element layout with resampling removed from the equation.
2. **End-to-end.** On-device predictions are compared against the Python LiteRT reference on the same model
   files, with assertions graded by how decisive the reference is.

This caught a real bug. Dart's `Interpolation.linear` does not antialias when downscaling, while the
reference's bilinear does. Mean error was 7.7 levels out of 255 and the top-5 tail was reordered. Switching
to area-averaging for ≥1.5× shrink cut the error to 4.0 and fixed a top-1 disagreement on the Labrador
photo. A regression test now asserts the chosen filter beats every alternative.

The general lesson: preprocessing parity between training and inference is as important as the model, and
the failure mode is silent degradation, not a crash.

### 20. What would you do differently for production?

1. Validate on **physical devices** across tiers — this PoC already shows why: a backend the simulator called
   healthy is numerically broken on a real A15. Extend to an Android phone, which remains unmeasured.
2. Ship **one** model, chosen by measurement, and split per ABI or use an App Bundle.
3. Move preprocessing to native/GPU and feed camera frames rather than JPEGs.
4. Own a worker isolate for inference so a delegate never forces work onto the UI thread.
5. Build the OTA model pipeline: signed, digest-verified, versioned against the tensor contract, with rollback.
6. Put the reference-comparison check in CI, per model and per target ABI.
7. Add telemetry for what actually happened per device — effective accelerators, warm latency percentiles —
   because backend selection varies in the field in ways you cannot predict from your desk.
8. Decide the timeout/cancellation policy properly, and pin the `flutter_litert` version given it is
   community-maintained.

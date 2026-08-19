# On-device ML options for Flutter — an engineering comparison

Scope note: this compares options **as reachable from Flutter**, which is a different question from
comparing the underlying runtimes. Every claim below is either verified in this repo, verified against
current package documentation (checked 2026-08-18), or explicitly marked as not verified.

## The table

| Option | Best for | Custom models | On-device | Complexity | Flutter binding |
|---|---|---|---|---|---|
| **ML Kit** | Capabilities already exposed as a high-level mobile API | Limited — only where the API accepts one (e.g. custom image-labelling / object-detection models); most APIs are fixed | Yes (several APIs also have cloud variants) | **Low** | `google_mlkit_*`, community-maintained, **explicitly not sponsored or maintained by Google** |
| **LiteRT** | Custom on-device inference, full control of the tensor pipeline | **Yes** — any `.tflite` graph | Yes | **Medium** | `flutter_litert`, community-maintained (publisher `hugo.ml`); Google ships no first-party Flutter binding |
| **ONNX Runtime** | Teams with an existing ONNX pipeline, or needing cross-framework portability | **Yes** — any `.onnx` graph | Yes | **Medium** | Several competing community plugins; the original `onnxruntime` appears unmaintained, forked as `onnxruntime_v2`; also `flutter_onnxruntime`, `fonnx` |
| **Cloud AI** | Models too large or too expensive to run locally; anything needing server-side state or frequent updates | Yes | **No** | **Low client / high backend** | Any HTTP client |

The complexity column is about *what you must get right*, not lines of code. ML Kit hands you a labelled
result. LiteRT hands you a buffer contract: you own resize, colour order, normalisation, dequantisation,
label alignment and threading — this repo is essentially a catalogue of those decisions. Cloud is trivial
on the client and moves the whole problem to infrastructure, auth, rate limiting, cost and privacy review.

## The finding that surprised me most

**Every on-device option depends on a community-maintained Flutter binding.** ML Kit's Flutter plugins state
plainly that they are not sponsored or maintained by Google. The most-referenced ONNX Runtime plugin is
unmaintained enough to have spawned forks. `flutter_litert` is one person's package — an actively developed,
unusually well-documented one, but not a Google deliverable.

So "is the binding official?" is not a discriminator between these options; it is a shared risk. The useful
questions are: is it actively maintained, does it document its own failure modes, and can we verify its
behaviour ourselves? On the third point `flutter_litert` scored well in this PoC — it ships
`verifyCompiledModel`, which compares a compiled model against a plain-CPU reference precisely because
"LiteRT Next has shipped defects where CompiledModel returns OK while producing output that is wrong."
A binding that hands you a tool to catch its own upstream's bugs is being honest with you.

Mitigations if the binding is the concern: pin the version and vendor it; keep inference behind an interface
(this repo's `OnDeviceModel`) so the binding is replaceable; validate every model on every target with a
reference comparison in CI; and be prepared to write the platform channel yourself, which is a bounded
amount of work for one model with a fixed contract.

## When each is the right answer

### ML Kit

Use it when your problem is *already* one of ML Kit's APIs: text recognition/OCR, barcode scanning, face
detection and mesh, pose detection, selfie segmentation, language ID, translation, smart reply, or generic
image labelling. You get tuned models, tuned preprocessing, and per-frame camera plumbing you would
otherwise write. Reaching for LiteRT to reimplement barcode scanning would be a straightforwardly worse
engineering decision.

Constraints to know: Android and iOS only; it adds meaningful binary weight (bundled models) or a Play
Services dependency (unbundled); and where an API accepts a custom model, it expects a specific input
contract, so you have less freedom than raw LiteRT.

### LiteRT

Use it when the model is yours — a classifier trained on your own labels, a domain-specific detector, an
embedding model for on-device search. You want direct control of the tensor pipeline and per-device
accelerator selection, and you accept owning preprocessing correctness. This PoC is that case.

Also use it when the alternative would be shipping user data off-device for something a 4 MB model can do
locally in single-digit milliseconds.

### ONNX Runtime

Use it when your organisation's model pipeline already emits ONNX and converting to `.tflite` would add a
lossy, hard-to-validate step for every release; when you need one model artefact shared across mobile,
desktop and server; or when your architecture uses operators with better ONNX support. The trade is that the
Flutter binding situation is messier than LiteRT's and Android NNAPI/vendor-NPU access from ONNX Runtime via
Flutter is **not verified** here.

Do not pick it merely because ONNX is "more standard" — on mobile specifically, the LiteRT toolchain
(quantization, delegates, vendor NPU runtimes) is the more travelled path.

### Cloud inference

Use it when the model cannot run locally at a sane cost: large generative models, anything needing
retrieval over a corpus you cannot ship, ensembles, or workloads where you must update the model daily and
observe aggregate behaviour. Also use it when you need centralised auditing of every inference.

The costs are the obvious ones plus two often forgotten: it does not work offline or on a bad connection,
and every inference is a data-egress event that needs a privacy answer. A hybrid is frequently correct —
cheap local model gates or pre-filters, cloud handles the hard tail.

## What this PoC actually established about LiteRT

| Claim | Status |
|---|---|
| Real local inference from Flutter, matching an independent reference implementation | **Verified** — 15 integration tests, 2 platforms, 6 backend configurations, 3 images |
| Works with no network | **Verified** — release APK declares no INTERNET permission; full suite passes with the network unreachable |
| `CompiledModel` selects and reports accelerators | **Verified** — including a narrowing GPU→CPU failure on the Android emulator, correctly reported |
| `CompiledModel` is float32-only, so quantized models need the classic `Interpreter` | **Verified** — enforced in code, covered by a test |
| Quantized model is 3.3× smaller (4.08 vs 13.33 MB) | **Verified** |
| iOS GPU (Metal) acceleration | **Verified on an iPhone 13 Pro** — 4.53 ms vs 9.54 ms CPU-only on the same API and weights, a 2.11× speed-up |
| Apple Neural Engine is usable for this model | **Refuted** — Core ML engaged but deviated 4.946% of output range from a plain-CPU reference, reproducible bit-for-bit. The app refuses the backend. No ANE latency figure exists as a result |
| Newer API is faster | **Depends on the device, and the simulator misled us.** Simulator: `Interpreter`+XNNPACK ~2.9× faster. Real A15: `CompiledModel`+GPU (4.53 ms) narrowly beats `Interpreter`+XNNPACK (4.79 ms) |
| Delegates always help | **Refuted** — XNNPACK made the quantized model 76% slower on the A15 (6.17 vs 3.51 ms) |
| Android GPU / vendor NPU acceleration | **Not verified** — GPU compilation fails on the emulator (no OpenCL); no vendor NPU runtime; no physical Android device available |
| Battery and thermal behaviour | **Not measured** — needs sustained runs with power instrumentation |

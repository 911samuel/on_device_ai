# Choosing an on-device ML option for Flutter

New to this? Read [`GLOSSARY.md`](GLOSSARY.md) first.

There are four realistic ways to add machine learning to a Flutter app. This page explains what each one is,
who it's for, and what we actually verified about the one we chose.

**Scope note:** this compares the options **as reachable from Flutter**, which is a different question from
comparing the underlying runtimes. Every claim is either verified in this repo, verified against current
package documentation (checked 2026-08-18), or explicitly marked as not verified.

## The four options, in one sentence each

* **ML Kit** — Google's ready-made features (scan a barcode, read text, detect a face). You get an answer, not
  a model. Fastest route *if* your problem is already on their list.
* **LiteRT** — run your own model file. Full control, and full responsibility for feeding it correctly. **This
  is what this project uses.**
* **ONNX Runtime** — same idea as LiteRT, different model file format. Makes sense if your team already
  produces `.onnx` files.
* **Cloud AI** — send the data to a server and get a prediction back. Simple client, everything else becomes
  someone's infrastructure problem.

## The table

| Option | Best for | Custom models | On-device | Complexity | Flutter binding |
|---|---|---|---|---|---|
| **ML Kit** | Capabilities already exposed as a high-level mobile API | Limited — only where the API accepts one (e.g. custom image-labelling / object-detection); most APIs are fixed | Yes (several APIs also have cloud variants) | **Low** | `google_mlkit_*`, community-maintained, **explicitly not sponsored or maintained by Google** |
| **LiteRT** | Custom on-device inference, full control of the tensor pipeline | **Yes** — any `.tflite` model | Yes | **Medium** | `flutter_litert`, community-maintained (publisher `hugo.ml`); Google ships no first-party Flutter binding |
| **ONNX Runtime** | Teams with an existing ONNX pipeline, or needing cross-framework portability | **Yes** — any `.onnx` model | Yes | **Medium** | Several competing community plugins; the original `onnxruntime` appears unmaintained, forked as `onnxruntime_v2`; also `flutter_onnxruntime`, `fonnx` |
| **Cloud AI** | Models too large or expensive to run locally; anything needing server-side state or frequent updates | Yes | **No** | **Low client / high backend** | Any HTTP client |

**About that "complexity" column** — it's about *what you must get right*, not how much code you write. ML Kit
hands you a labelled result. LiteRT hands you a contract: you own resizing, colour order, normalisation,
converting the output back to numbers you can read, label alignment, and threading. Get any one wrong and you
get confident nonsense rather than an error. This repo is essentially a catalogue of those decisions. Cloud is
trivial on the client and moves the whole problem to infrastructure, auth, rate limiting, cost and privacy
review.

## The finding that surprised us most

**Every on-device option depends on a community-maintained Flutter binding.** ML Kit's Flutter plugins state
plainly that they are not sponsored or maintained by Google. The most-referenced ONNX Runtime plugin is
unmaintained enough to have spawned forks. `flutter_litert` is one person's package — actively developed and
unusually well-documented, but not a Google deliverable.

So "is the binding official?" doesn't separate these options; it's a shared risk. The useful questions are: is
it actively maintained, does it document its own failure modes, and can we check its behaviour ourselves? On the
third point `flutter_litert` did well here — it ships `verifyCompiledModel`, which compares a compiled model
against a plain-CPU reference precisely because, in its own words, LiteRT Next has shipped defects where a model
reports success while producing wrong output. **A library that hands you a tool for catching its own upstream's
bugs is being honest with you** — and in this project that tool caught a real problem on a real phone.

If the binding is your concern: pin the version and vendor it; keep inference behind an interface (this repo's
`OnDeviceModel`) so it's replaceable; validate every model on every target with a reference comparison in CI;
and know that writing the platform channel yourself is a bounded amount of work for one model with a fixed
contract.

## When each is the right answer

### ML Kit

Use it when your problem is *already* one of ML Kit's APIs: text recognition/OCR, barcode scanning, face
detection and mesh, pose detection, selfie segmentation, language ID, translation, smart reply, or generic image
labelling. You get tuned models, tuned preprocessing, and per-frame camera plumbing you'd otherwise write
yourself. Reaching for LiteRT to reimplement barcode scanning would be straightforwardly worse engineering.

Constraints: Android and iOS only; it adds meaningful app weight (bundled models) or a Play Services dependency
(unbundled); and where an API does accept a custom model, it expects a specific input contract, so you have less
freedom than raw LiteRT.

### LiteRT

Use it when the model is yours — a classifier trained on your own labels, a domain-specific detector, an
embedding model for on-device search. You want direct control of the pipeline and per-device accelerator
choice, and you accept owning correctness. **This PoC is that case.**

Also use it when the alternative would be shipping user data off-device for something a 4 MB model can do
locally in single-digit milliseconds.

### ONNX Runtime

Use it when your organisation's pipeline already emits ONNX and converting to `.tflite` would add a lossy,
hard-to-validate step to every release; when you need one model artefact shared across mobile, desktop and
server; or when your architecture uses operators with better ONNX support. The trade is that the Flutter binding
situation is messier than LiteRT's, and Android vendor-NPU access from ONNX Runtime via Flutter is **not
verified** here.

Don't pick it just because ONNX sounds "more standard" — on mobile specifically, the LiteRT toolchain
(quantization, delegates, vendor NPU runtimes) is the more travelled path.

### Cloud inference

Use it when the model genuinely can't run locally: large generative models, anything needing retrieval over a
corpus you can't ship, ensembles, or workloads you must update daily and observe in aggregate. Also when you
need centralised auditing of every prediction.

The costs are the obvious ones plus two people forget: it doesn't work offline or on a bad connection, and every
prediction is a data-egress event that needs a privacy answer. A hybrid is often the right design — a cheap
local model filters or gates, the cloud handles the hard cases.

## What this PoC actually established about LiteRT

This is the honest scorecard. "Verified" means we ran it and measured it; "not verified" means we didn't, and we
won't pretend otherwise.

| Claim | Status |
|---|---|
| Real local inference from Flutter, matching an independent reference implementation | **Verified** — 15 integration tests, 2 platforms, 6 backend configurations, 3 images |
| Works with no network | **Verified** — release APK declares no INTERNET permission; the full suite passes with the network unreachable |
| `CompiledModel` selects and reports accelerators | **Verified** — including a GPU→CPU failure on the Android emulator, correctly reported rather than hidden |
| `CompiledModel` is float32-only, so quantized models need the classic `Interpreter` | **Verified** — enforced in code, covered by a test |
| Quantized model is 3.3× smaller (4.08 vs 13.33 MB) | **Verified** |
| iOS GPU (Metal) acceleration | **Verified on an iPhone 13 Pro** — 4.53–4.90 ms vs 9.54 ms CPU-only on the same API and weights, a ≈2× speed-up (1.95–2.11× over three runs) |
| Apple Neural Engine is usable for this model | **Refuted** — Core ML was engaged but deviated 4.946% of output range from a plain-CPU reference, reproducibly. The app refuses the backend, so no ANE speed figure exists |
| Newer API is faster | **No — they're tied, and the simulator misled us.** Simulator: `Interpreter`+XNNPACK looked ~2.9× faster. Real A15 over three runs: `CompiledModel`+GPU 4.53/4.57/4.90 ms vs `Interpreter`+XNNPACK 4.79/5.12/4.85 ms — GPU wins 2 of 3 by a margin inside run-to-run noise |
| Delegates always help | **Refuted** — XNNPACK made the quantized model 76% slower on the A15 (6.17 vs 3.51 ms) |
| Android GPU / vendor NPU acceleration | **Not verified** — GPU compilation fails on the emulator (no OpenCL), no vendor NPU runtime, and no physical Android device was available |
| Battery and thermal behaviour | **Not measured** — needs sustained runs with power instrumentation |

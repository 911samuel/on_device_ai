# Speaker notes

Written for an audience of senior mobile engineers and people who know AI. Assume they will push on the
claims — the notes below are structured so the strongest thing you can do is show a measurement.

Rough timing: 4 min slide 1, 6 min slide 2, 4 min slide 3, rest Q&A.

---

## Slide 1 — On-Device ML in Flutter

### What to say

> "The one-line summary: Flutter is the application layer, LiteRT does the inference. Dart never performs
> the arithmetic — it prepares a buffer, hands it across FFI to a native runtime, and interprets what comes
> back.
>
> I built a working PoC rather than a demo. It classifies an image with MobileNet, entirely on-device. It
> ships two models and two different LiteRT APIs behind a single Dart interface, and every number I show you
> came out of this repo on hardware I can name.
>
> Read the diagram top to bottom, because that is the actual call path. Flutter selects an image. My ML
> abstraction — one Dart interface, `OnDeviceModel` — is the only thing the rest of the app depends on.
> Under it sits LiteRT, reached over `dart:ffi`. LiteRT decides which backend runs the graph, and the result
> comes back up as a labelled prediction.
>
> The pipeline line is worth dwelling on, because it is where the engineering actually is. A JPEG is not a
> tensor. That model takes exactly 602,112 bytes — a `[1,224,224,3]` float32 buffer, NHWC, RGB, normalised
> to minus-one-to-one — and returns 4,004 bytes, which is 1001 float32 probabilities. Everything between
> 'user picked a photo' and 'model can run' is my responsibility, and that is the part that bites you.
>
> And that shows up in the timings, measured on a real iPhone 13 Pro. Inference is 4.5 milliseconds.
> Preprocessing is 15. Three times the model, and about 75% of the total. If someone asked me to make this
> pipeline faster, I would not touch the model.""

### What the diagram means

* Each arrow is a real boundary. Flutter→abstraction is a Dart call. Abstraction→LiteRT is where the runtime
  type first appears. LiteRT→silicon is a delegate decision the app does not make.
* The byte counts are the contract. Say them out loud once — it establishes that you know the model's shape
  rather than trusting a tutorial.

### Terminology to define as you go

* **LiteRT** — the current name for TensorFlow Lite. Same `.tflite` format; the rename is branding, not a
  new file format. "LiteRT Next" is the newer `CompiledModel` API layered on the same runtime.
* **Tensor** — a flat buffer plus a shape that tells you how to index it. Element `(n,y,x,c)` at
  `((n·H+y)·W+x)·C+c`.
* **NHWC** — batch, height, width, channel. The channel values for one pixel are adjacent in memory.
* **Delegate** — an object that offers to execute part of the graph on other hardware. It may decline
  operators, which then run on CPU.
* **Quantization** — storing weights/activations as 8-bit integers with an affine mapping
  `real = (q − zero_point) × scale`.
* **FFI** — Dart's foreign function interface; direct C calls, no platform-channel serialisation.

### Likely questions here

**"Does Dart run the model?"**
No. Dart writes a byte buffer and calls a C function. The kernels are compiled native code — hand-tuned
NEON in XNNPACK's case. Dart's only arithmetic in this pipeline is normalisation and the final sort.

**"Is this a real model or a toy?"**
MobileNetV2, Google's published ImageNet checkpoint, 1001 classes, 13.3 MB. Not trained by me, not
fine-tuned. The point of the PoC is the integration, so I deliberately used a known-good model whose
expected outputs I can verify.

**"How do you know the predictions are actually right?"**
Two independent comparisons. The pipeline is compared against the Python LiteRT reference interpreter on the
same model files — on the decisive sample, "military uniform" at 0.875 on device against 0.804 in the
reference. And separately, every compiled model is compared against a plain-CPU reference at startup, which
is what caught the Neural Engine returning wrong output. Both are tests, not one-offs.

---

## Slide 2 — Why LiteRT?

### What to say

> "Why LiteRT and not something else. The honest answer is: because the premise was a custom model running
> locally, and that is precisely what LiteRT is for. If the task were OCR or barcode scanning I would use
> ML Kit and be done in an afternoon. If we already had an ONNX pipeline I would look hard at ONNX Runtime.
> If the model were too big to ship, cloud. None of those was the case.
>
> The left column is what it buys, and I can back each line. Offline: the release APK does not declare the
> INTERNET permission at all. That is kernel-enforced on Android — the process cannot open a socket. I also
> put the emulator in airplane mode, confirmed the network was unreachable, and re-ran the whole suite. It
> passed.
>
> Latency: 3.9 milliseconds, no round trip, no tail latency, no retry logic, no offline handling. Privacy:
> the pixels never leave the process, so there is nothing to breach or subpoena. And there is no marginal
> cost per inference and no backend to operate.
>
> Now the right column, because I am not here to sell this. Thirty-six megabytes added to the app: seventeen
> of models, fifteen of runtime, per ABI. Battery and thermals I did not measure, so I am not going to tell
> you about them.
>
> And then the result at the bottom, which is the most valuable thing in this PoC.
>
> On the real A15 I asked for the Neural Engine. It compiled successfully. And then its output deviated
> almost five percent of the output range from a plain-CPU reference — reproducible bit-for-bit across runs,
> where a healthy configuration deviates five ten-thousandths of a percent. That is consistent with the ANE
> computing in fp16. So the app refused the backend and told me why.
>
> Here is the part that should worry you. The same configuration reported perfectly healthy on the iOS
> simulator, because Core ML on a simulator runs on this Mac and never touches a Neural Engine. If I had
> tested only on the simulator, I would have shipped a configuration that is numerically wrong on real
> phones — and it would not have crashed, it would just have been quietly worse.
>
> Two things follow. First: 'accelerated' and 'correct' are independent properties, so verify accelerator
> output against a CPU reference at startup. Second, and I'll own this one: an earlier version of this deck
> claimed the old `Interpreter` API was almost three times faster than `CompiledModel`. That came from
> simulator numbers. On real hardware the ordering reverses, because the simulator has no mobile GPU. I
> corrected the claim rather than dropping it, because the methodological lesson is the point: do not draw
> architectural conclusions from emulated measurement."

### What the tables mean

* Left column = the standard on-device pitch, but each row is tied to something in the repo. If challenged,
  open `docs/OFFLINE_VERIFICATION.md` or `docs/BENCHMARKS.md`.
* Right column exists so nobody has to extract the downsides from you. Volunteering them is what makes the
  left column credible.

### Terminology

* **XNNPACK** — Google's optimised CPU kernel library (NEON SIMD, cache blocking). Still the CPU. Calling it
  "hardware acceleration" is misleading.
* **NPU / ANE / HTP** — dedicated neural accelerators. Apple's is the Neural Engine, reachable only via
  Core ML. Qualcomm's needs a vendor runtime shipped with the app.
* **Delegate narrowing** — the runtime silently reducing your requested accelerator set when the device
  cannot honour it. The reason "requested" and "effective" are separate rows in my UI.
* **Cold vs warm** — first inference pays lazy kernel setup, page faults on fresh arenas, i-cache misses,
  and on a real phone a CPU governor still ramping.

### Likely questions

**"Is it actually using the NPU?"**
On the real A15, Core ML was demonstrably engaged — the output changed by ~5% of range, which the CPU path
cannot explain. But it was *wrong*, so we refused it. I have no ANE latency figure, and I am not going to
quote one from a configuration that computes the wrong answer. On Android there is no vendor NPU runtime, so
the request was narrowed to CPU.

**"Couldn't that 5% just be fp16 rounding, which might be fine?"**
It might be acceptable for some models — that is a per-model decision, and I have not tested an
fp16-tolerant or quantization-aware-trained model here. But 4.9% of output range on a probability
distribution is not something I would ship untested, and the tolerance that flagged it (1%) is the
binding's own, calibrated against measurements where healthy configurations sat under 0.07%.

**"What would you do to actually use the ANE?"**
Validate the specific model at fp16 offline, compare top-1 agreement and confidence drift on a real
evaluation set, and only then enable it — per device family, behind a remote flag, with the CPU-reference
check still running at startup.

**"How can you tell acceleration from a silent CPU fallback?"**
Compare output against a plain-CPU reference. If it is bit-identical, nothing else touched the graph. If it
deviates slightly but agrees within tolerance, a different compute path ran. The binding ships that check
because LiteRT Next has shipped bugs where a compiled model returns OK and produces wrong output. Worth
noting: my first version of this logic reported a false positive on Android — GPU had fallen back to CPU,
yet I claimed acceleration, because optimised CPU kernels also deviate from unoptimised reference kernels.
Cross-platform testing caught it, and I fixed the interpretation.

**"Why is CompiledModel slower? That seems backwards."**
Three contributors I can name: it defaults to fp32 in this binding version (changed from fp16 for accuracy,
and the changelog says that costs 24–43% GPU latency); my measurement includes a helper-isolate round trip;
and the CPU path does not appear to use XNNPACK the way the classic API does. I would not ship the
conclusion "always use Interpreter" — I would benchmark both on the actual device tier.

**"Is 36 MB acceptable?"**
Depends on the app, and it is reducible. Ship one model instead of two, drop 13 MB. Split per ABI — the
universal APK is 112 MB versus 51 MB for arm64. Skip the GPU accelerator library if unused. Or deliver the
model out-of-band and don't bundle it at all.

---

## Slide 3 — Recommended architecture

### What to say

> "This is what I would actually recommend, and the shape matters more than the specific runtime.
>
> Flutter owns the app. Immediately below it, one Dart interface: initialise, predict, dispose. Everything
> above that line is runtime-agnostic. In this repo, only two files out of twenty-five in `lib` import LiteRT
> at all — and that is not a style preference, it is what let me test the entire application layer with a
> fake model and no native code. It also means replacing LiteRT with ONNX Runtime is adding an
> implementation, not a refactor.
>
> Below the interface the paths diverge by platform, and they genuinely do diverge — Core ML and the Neural
> Engine on iOS, OpenCL or a vendor runtime on Android — which is exactly why that difference belongs behind
> the abstraction rather than in your widgets.
>
> The 'which tool when' table is there so nobody thinks I am claiming LiteRT wins everything. It doesn't.
>
> And the non-negotiables are the things I would insist on in review. Assert the tensor contract at startup,
> because wrong normalisation does not crash — it quietly makes your product worse, and my own experiment
> proved it: feeding zero-to-one instead of minus-one-to-one still produced the right label at a third of
> the confidence. Validate against a reference implementation, because that is what caught a real
> resampling bug in my preprocessing. Report hardware, never assume it. Keep inference off the UI thread,
> and know when your binding won't let you. And treat the model as a versioned API, because the tensor
> contract is part of that API."

### What the diagram means

* The single funnel point below "ML Service Interface" is the argument: divergence happens *below* the seam,
  so the app never branches on platform.
* Both platform branches converge on "Local Model" — same weights, same file, different silicon.

### Likely questions

**"Isn't the abstraction over-engineering for one model?"**
It paid for itself inside this PoC. I have two implementations behind it (the two LiteRT APIs), the
controller is fully unit-testable without native code, and swapping backends is a dropdown. The interface is
about 15 lines.

**"Where would you put this in a real app's architecture?"**
The ML service is a repository-shaped dependency: inject it, keep it a singleton per model, own its lifecycle
explicitly. It is not a widget concern and it is not global state.

**"What breaks first at scale?"**
Model lifecycle and memory. Two float32 models loaded at once is ~28 MB of weights plus arenas, so I dispose
before switching. Second is preprocessing throughput once you go to live camera frames.

---

## Closing line

> "The takeaway I would like to leave you with: on-device ML in Flutter is not hard because of the model.
> It is hard because of everything around the model — the byte contract, preprocessing parity, threading,
> and knowing which hardware actually ran your graph. LiteRT does the inference part well. The engineering
> is in being able to prove what you just claimed."

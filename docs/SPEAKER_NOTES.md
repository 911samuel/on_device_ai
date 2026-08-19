# Speaker notes

Written for a **mixed room**: some senior mobile engineers, some colleagues who have never touched ML and are
deciding whether this is worth their curiosity. The notes are structured so you can serve both — and so that
when someone pushes on a claim, the strongest thing you can do is show a measurement.

Rough timing: 3 min slide 0, 4 min slide 1, 6 min slide 2, 4 min slide 3, rest Q&A.

**Two rules that make this talk land:**

1. **Never say a number you can't source.** Every figure in the deck is in this repo. If challenged, open
   `docs/BENCHMARKS.md` on screen. Saying "I didn't measure that" costs you nothing and buys you everything.
2. **Define a word the first time you use it, in six words or less.** The jargon card at the end of
   `SLIDES.md` exists for this. Beginners disengage silently, so don't wait to be asked.

---

## Slide 0 — What "on-device AI" actually means

*Skip this slide only if you're certain the whole room has ML experience. If in doubt, run it — the engineers
won't be bored by three minutes, and the beginners will follow everything afterwards.*

### What to say

> "Before anything else, let me kill the mystery, because 'AI on your phone' sounds like more than it is.
>
> An AI model is a **file full of numbers**. Ours is 13 megabytes. Somebody at Google showed a network a million
> photos, adjusted those numbers until it got good at naming things, and then froze the file. That's training,
> it cost a fortune, and it happened years before I got involved.
>
> What we do is **replay** that file: take a new photo, push it through a fixed sequence of multiplications, and
> read the answer. That's called **inference**. It's the only AI operation in this entire project, and it takes
> four and a half milliseconds.
>
> So when I say 'on-device AI', I mean: the file is on the phone, the multiplications happen on the phone, and
> nothing goes to a server. Four steps — photo, turn it into numbers, multiply, read the label.
>
> The last table is why you'd bother: it works offline, the photo never leaves the device, and it costs nothing
> per prediction. And the two rows where cloud wins are real — you can update a server model instantly, and a
> server can run models far too big for a phone. This is a trade, not a victory."

### If someone asks the obvious beginner question

**"So it's not really intelligent?"**
Not in any sense you'd recognise. It's pattern-matching arithmetic learned from examples. Nobody wrote a rule
like "if it has fur, guess dog" — that pattern was *learned* — but there's no reasoning happening. Which is
exactly why it will confidently tell you a coffee mug is a soup bowl: it has no concept of being wrong.

**"Does it learn from my photos as I use it?"**
No. The file is read-only and identical for every user. Learning happens before shipping, elsewhere. This is a
common and reasonable assumption, so answer it before someone worries about privacy for the wrong reason.

---

## Slide 1 — On-device ML in Flutter

### What to say

> "The one-line summary: Flutter is the application layer, LiteRT does the inference. Dart never performs the
> arithmetic — it prepares a buffer of bytes, hands it to a native runtime, and interprets what comes back.
>
> LiteRT is the thing that runs the model. You may know it as TensorFlow Lite; it was renamed. Same file format.
>
> I built a working proof of concept rather than a demo. It classifies an image with MobileNet, entirely
> on-device. It ships two models and two different LiteRT APIs behind a single Dart interface, and every number I
> show you came out of this repo on hardware I can name.
>
> Read the diagram top to bottom, because that's the actual call path. Flutter picks an image. My ML abstraction
> — one Dart interface called `OnDeviceModel` — is the only thing the rest of the app depends on. Under it sits
> LiteRT. LiteRT decides which chip runs the model, and the result comes back up as a labelled prediction.
>
> Now the pipeline line, because this is where the engineering actually is. **A JPEG is not a tensor.** That
> model takes exactly 602,112 bytes — 224 by 224 pixels, three colours each, as 32-bit floats, scaled to between
> minus one and one — and returns 4,004 bytes, which is 1,001 probabilities. Everything between 'user picked a
> photo' and 'model can run' is my responsibility, and it's the part that bites you.
>
> And that shows up in the timings, measured on a real iPhone 13 Pro. Inference is 4.5 milliseconds.
> Preprocessing is 15. **The AI is the fastest part of the AI feature.** If someone asked me to make this
> pipeline faster, I would not touch the model."

### What the diagram means

* Each arrow is a real boundary. Flutter→abstraction is a Dart call. Abstraction→LiteRT is where the runtime
  type first appears. LiteRT→silicon is a chip decision the app does not make.
* The byte counts are the contract. Say them out loud once — it establishes that you know the model's shape
  rather than trusting a tutorial.

### Terminology to define as you go

Keep each of these to one sentence unless asked for more.

* **LiteRT** — the current name for TensorFlow Lite. Same `.tflite` format; the rename is branding, not a new
  format. "LiteRT Next" is the newer `CompiledModel` API on the same runtime.
* **Tensor** — a flat buffer of numbers plus a shape telling you how to index it. Element `(n,y,x,c)` sits at
  `((n·H+y)·W+x)·C+c`.
* **NHWC** — batch, height, width, channel. The three colour values for one pixel are next to each other in
  memory. (Only mention this if someone asks about layout.)
* **Delegate** — a component that offers to run part of the model on other hardware or faster code. It may
  decline parts, which then fall back to CPU.
* **Quantization** — storing numbers as 8-bit integers with a scale factor: `real = (stored − zero_point) ×
  scale`. Smaller and often faster, slightly less precise.
* **FFI** — Dart's foreign function interface. Direct C calls, no serialisation overhead.

### Likely questions here

**"Does Dart run the model?"**
No. Dart writes a byte buffer and calls a C function. The heavy maths runs in compiled native code —
hand-optimised for the chip. Dart's only arithmetic in this pipeline is the normalisation loop and the final
sort.

**"Is this a real model or a toy?"**
MobileNetV2, Google's published ImageNet checkpoint, 1,001 classes, 13.3 MB. Not trained by me, not fine-tuned.
The point of the PoC is the integration, so I deliberately used a known-good model whose expected outputs I can
verify against a reference.

**"How do you know the predictions are actually right?"**
Two independent comparisons. The pipeline is compared against the Python LiteRT reference interpreter on the
same model files — on the decisive sample, "military uniform" at 0.875 on device against 0.804 in the reference.
And separately, every compiled model is compared against a plain-CPU reference at startup, which is what caught
the Neural Engine returning wrong output. Both are automated tests, not one-offs.

---

## Slide 2 — Why LiteRT, and what it costs

### What to say

> "Why LiteRT and not something else. The honest answer: because the premise was a custom model running locally,
> and that's precisely what LiteRT is for. If the task were OCR or barcode scanning I'd use ML Kit and be done in
> an afternoon. If we already had an ONNX pipeline I'd look hard at ONNX Runtime. If the model were too big to
> ship, cloud. None of those was the case.
>
> The left column is what it buys, and I can back every line. Offline: the release APK doesn't declare the
> INTERNET permission at all. On Android that's enforced by the kernel — the process physically cannot open a
> connection. I also put the emulator in airplane mode, confirmed the network was unreachable, and re-ran the
> whole suite. It passed.
>
> Latency: 4.5 milliseconds, no round trip, no retry logic, no offline handling to write. Privacy: the pixels
> never leave the process, so there's nothing to breach or subpoena. And there's no marginal cost per prediction
> and no backend to operate.
>
> Now the right column, because I'm not here to sell this. Thirty-six megabytes added to the app: seventeen of
> models, fifteen of runtime, per architecture. Battery and thermals I did not measure, so I'm not going to tell
> you about them.
>
> And then the result at the bottom, which is the most valuable thing in this PoC.
>
> On the real A15 I asked for the Neural Engine — Apple's dedicated AI chip. It compiled successfully. And then
> its output deviated almost five percent of the output range from a plain-CPU reference. Reproducible
> bit-for-bit across runs, where a healthy configuration deviates five ten-thousandths of a percent. That's
> consistent with the Neural Engine computing in half precision — fewer digits per number. So the app refused
> the backend and told me why.
>
> Here's the part that should worry you. The same configuration reported perfectly healthy on the iOS simulator,
> because Core ML on a simulator runs on this Mac and never touches a Neural Engine. If I'd tested only on the
> simulator, I'd have shipped a configuration that is numerically wrong on real phones — and it wouldn't have
> crashed, it would just have been quietly worse.
>
> Two things follow. First: 'accelerated' and 'correct' are independent properties, so verify accelerator output
> against a CPU reference at startup. Second, and I'll own this one: an earlier version of this deck claimed the
> old `Interpreter` API was almost three times faster than `CompiledModel`. That came from simulator numbers. On
> real hardware the ordering reverses, because the simulator has no mobile GPU. I corrected the claim rather than
> dropping it, because the methodological lesson is the actual point: **do not draw architectural conclusions
> from emulated measurement.**"

### If the room is mostly beginners

Compress the whole slide to this:

> "We asked the phone's dedicated AI chip to do the work. It said yes, did it, and gave answers that were five
> percent off — the same five percent off every time, because that chip uses fewer digits per number. Our app
> checks every chip against the plain CPU before trusting it, so it caught this and refused to use it.
>
> Without that check we'd have shipped something quietly wrong that never crashed and nobody would have
> reported. And the simulator told us it was fine, because a simulator doesn't have that chip."

That version needs no jargon at all and carries the same lesson.

### What the tables mean

* Left column = the standard on-device pitch, but every row is tied to something in the repo. If challenged, open
  `docs/OFFLINE_VERIFICATION.md` or `docs/BENCHMARKS.md`.
* Right column exists so nobody has to extract the downsides from you. Volunteering them is what makes the left
  column credible.

### Terminology

* **XNNPACK** — Google's optimised CPU maths library. Still the CPU. Calling it "hardware acceleration" is
  misleading, and on our quantized model it was actually *slower*.
* **NPU / ANE / HTP** — dedicated neural chips. Apple's is the Neural Engine, reachable only via Core ML.
  Qualcomm's needs a vendor runtime you ship yourself.
* **Delegate narrowing** — the runtime quietly reducing your requested chip list when the device can't honour
  it. The reason "requested" and "effective" are separate rows in my UI.
* **Cold vs warm** — the first inference pays for lazy setup, fresh memory pages, and a CPU still ramping up its
  clock speed. Later runs don't.

### Likely questions

**"Is it actually using the NPU?"**
On the real A15, Core ML was demonstrably engaged — the output changed by ~5% of range, which the CPU path cannot
explain. But it was *wrong*, so we refused it. I have no Neural Engine latency figure, and I'm not going to quote
one from a configuration that computes the wrong answer. On Android there's no vendor NPU runtime installed, so
the request was narrowed to CPU.

**"Couldn't that 5% just be rounding, which might be fine?"**
It might be acceptable for some models — that's a per-model decision, and I haven't tested a model designed to
tolerate half precision. But 4.9% of output range on a probability distribution isn't something I'd ship
untested, and the 1% tolerance that flagged it is the binding's own, calibrated against measurements where
healthy configurations sat under 0.07%.

**"What would you do to actually use the Neural Engine?"**
Validate that specific model at fp16 offline, compare top-1 agreement and confidence drift on a real evaluation
set, and only then enable it — per device family, behind a remote flag, with the CPU-reference check still
running at startup.

**"How can you tell real acceleration from a silent fallback to CPU?"**
Compare output against a plain-CPU reference. Bit-identical means nothing else touched it. Slightly different but
within tolerance means a different compute path ran. The binding ships that check because LiteRT Next has shipped
bugs where a model reports success and produces wrong output. Worth volunteering: my first version of this logic
gave a false positive on Android — the GPU had fallen back to CPU, yet I claimed acceleration, because optimised
CPU kernels *also* deviate from unoptimised reference kernels. Cross-platform testing caught it and I fixed the
interpretation.

**"Which API is faster, then?"**
Honestly? On this phone those two are **tied**. Across three runs `CompiledModel`+GPU measured 4.53, 4.57 and
4.90 ms while `Interpreter`+XNNPACK measured 4.79, 5.12 and 4.85 — so the GPU path won two of three, and lost the
third by 0.05 ms. That difference is inside run-to-run noise, and I'm not going to pretend otherwise.

What *is* robust: the compressed model with no delegate at all was fastest every single time (3.51, 3.56,
3.54 ms) and it's the most repeatable number in the project. And the ordering is device-specific anyway — on the
simulator the ranking was completely different. So the answer I'd defend is "benchmark on the tiers you actually
ship to", not "always use X".

**"Is 36 MB acceptable?"**
Depends on the app, and it's reducible. Ship one model instead of two, drop 13 MB. Split per architecture — the
universal APK is 112 MB versus 51 MB for arm64. Skip the GPU library if unused. Or download the model after
install instead of bundling it.

---

## Slide 3 — Recommended architecture

### What to say

> "This is what I'd actually recommend, and the shape matters more than the specific runtime.
>
> Flutter owns the app. Immediately below it, one Dart interface: initialise, predict, dispose. Everything above
> that line is runtime-agnostic. In this repo only two files out of twenty-four in `lib` import LiteRT at all —
> and that's not a style preference, it's what let me test the entire application layer with a fake model and no
> native code. It also means replacing LiteRT with ONNX Runtime is adding an implementation, not a refactor.
>
> Below the interface the paths genuinely diverge — Core ML and the Neural Engine on iOS, OpenCL or a vendor
> runtime on Android. That divergence is real, and it's exactly why it belongs behind the abstraction rather
> than in your widgets.
>
> The 'which tool when' table is there so nobody thinks I'm claiming LiteRT wins everything. It doesn't. If your
> problem is barcode scanning, use ML Kit.
>
> And the non-negotiables are what I'd insist on in review. Assert the tensor contract at startup, because wrong
> normalisation doesn't crash — it quietly makes your product worse, and my own experiment proved it: feeding
> zero-to-one instead of minus-one-to-one still produced the right label at a *third* of the confidence.
> Validate against a reference implementation, because that's what caught a real image-resizing bug in my
> preprocessing. Report hardware, never assume it. Keep inference off the UI thread, and know when your binding
> won't let you. And treat the model as a versioned API, because the tensor contract is part of that API."

### What the diagram means

* The single funnel point below "ML Service Interface" is the argument: divergence happens *below* the seam, so
  the app never branches on platform.
* Both platform branches converge on "Local Model" — same weights, same file, different silicon.

### Likely questions

**"Isn't the abstraction over-engineering for one model?"**
It paid for itself inside this PoC. I have two implementations behind it (the two LiteRT APIs), the controller is
fully unit-testable without native code, and swapping backends is a dropdown. The interface is about 15 lines.

**"Where would you put this in a real app's architecture?"**
The ML service is a repository-shaped dependency: inject it, keep one instance per model, own its lifecycle
explicitly. It's not a widget concern and it's not global state.

**"What breaks first at scale?"**
Model lifecycle and memory. Two full-precision models loaded at once is ~28 MB of weights plus working memory,
so I free one before loading the other. Second is preprocessing throughput once you move to live camera frames.

---

## Closing line

> "The takeaway I'd like to leave you with: on-device ML in Flutter is not hard because of the model. It's hard
> because of everything *around* the model — the byte contract, preprocessing parity, threading, and knowing
> which hardware actually ran your graph. LiteRT does the inference part well.
>
> The engineering is in being able to prove what you just claimed. And the most useful thing I found in three
> days of measuring is that a dedicated AI chip gave me the wrong answer and only a correctness check caught it.
> If you take one habit from this: **verify, then claim.**"

### If you want to leave people curious rather than impressed

Point them at `docs/GLOSSARY.md` and give them the experiment: run the app, point it at something that isn't in
its 1,001 categories — a laptop charger, a houseplant — and watch it confidently pick the nearest thing it
knows. That single moment teaches more about how these models actually behave than any tutorial, and it's the
thing most likely to make someone want to go further.

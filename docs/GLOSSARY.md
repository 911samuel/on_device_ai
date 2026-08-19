# Start here: on-device AI in plain language

New to AI or to on-device machine learning? Read this page first. Everything else in `docs/` assumes the
vocabulary defined here. No maths, no prior ML experience needed.

---

## The 60-second version

This app takes a photo, and a small file of numbers tells it what's in the photo — **entirely on the phone**.
No internet, no server, no account.

That's it. That's on-device AI. The rest is detail about *how fast*, *on which chip*, and *how do we know it's
actually right*.

A useful mental picture: the "AI" here is a **recipe made of multiplications**. Someone trained it once on a
million photos, froze the result into a file, and now our phone just replays that arithmetic on new pictures.
Training is expensive and happens in a data centre. **Running** the frozen recipe is cheap — a few
milliseconds — and that's what a phone can do.

The word for "running the frozen recipe" is **inference**. If you learn one term today, learn that one.

---

## The words you'll hear

### About the model

**Model** — a file containing millions of numbers (called *weights*) plus a description of what order to
multiply them in. Ours is `mobilenet_v2_1.0_224.tflite`, 13.3 MB. You can copy it, email it, delete it. It is
just a file, and it does not change while the app runs.

**Inference** — one run of the model on one input. "Inference took 4.5 ms" means the phone did all those
multiplications in four and a half thousandths of a second.

**Tensor** — a grid of numbers. A single number is a 0-D tensor, a list is 1-D, a spreadsheet is 2-D, and a
colour image is 3-D (width × height × 3 colour channels). It's just the word ML uses for "a box of numbers of
a known shape". Our model wants a tensor shaped `[1, 224, 224, 3]`: one image, 224 pixels tall, 224 wide,
3 colours.

**MobileNet** — the specific model design we use. Built for phones: small and fast, slightly less accurate
than the giant models that run on servers. It knows 1,001 categories of everyday objects.

**ImageNet labels** — the list of those 1,001 category names ("golden retriever", "espresso", "military
uniform"). The model doesn't output words; it outputs 1,001 scores, and we look up which name goes with the
highest score.

**Quantization** — storing the weights as small whole numbers (0–255, "uint8") instead of high-precision
decimals ("float32"). The model gets ~4× smaller and often faster, at a small cost in accuracy. We ship both
versions to show the trade-off: 4.08 MB quantized vs 13.33 MB float.

**float32 / uint8 / fp16** — how precisely each number is stored. float32 is 32 bits per number and very
precise; uint8 is 8 bits and coarse; fp16 sits in between. **This matters more than it sounds** — one of the
main findings in this project is a chip that computed in fp16 and got a visibly different answer.

### About the hardware

Modern phone chips contain several different processors, and you can choose which one does the maths:

**CPU** — the general-purpose processor. Runs anything, moderately fast. Always available.

**GPU** — originally for graphics, but it's excellent at doing thousands of multiplications simultaneously,
which is exactly what a model needs. On iPhones the GPU is programmed through Apple's **Metal** framework.

**NPU** — "Neural Processing Unit", a chip designed *only* for ML maths. Apple calls theirs the **Neural
Engine** (ANE). Very fast and very power-efficient — and, as we discovered, not always accurate enough.

**Accelerator** — umbrella term for "GPU or NPU", i.e. anything that isn't the plain CPU.

**Delegate** — a plug-in that hands part of the model to a faster implementation. **XNNPACK** is a delegate
containing hand-optimised CPU code. Attaching one usually helps. On our test phone, one of them made things
*76% slower*, which is a good lesson: measure, don't assume.

**Core ML** — Apple's own ML framework. LiteRT can hand a model to Core ML, which then decides whether to use
the CPU, GPU, or Neural Engine.

### About the software

**LiteRT** — the runtime that actually executes the model file. Formerly called TensorFlow Lite. Think of the
model as a document and LiteRT as the app that opens it.

**`flutter_litert`** — the Flutter package that lets Dart code talk to LiteRT. Community-maintained, not an
official Google product.

**Interpreter vs CompiledModel** — two APIs for running a model. `Interpreter` is the classic one (works with
quantized models, more manual control). `CompiledModel` is the newer, accelerator-first one (picks GPU/NPU for
you and reports what it chose, but float32 only). We measured both, and which one wins **depends on the
device** — see `BENCHMARKS.md`.

**Isolate** — Flutter's version of a background thread. Heavy work must move off the main isolate or the UI
visibly stutters. Relevant here because a model run is heavy work.

### About speed

**Cold vs warm** — the first run is slower than the ones after it: caches are empty, memory isn't allocated
yet, the GPU driver is waking up. So we report run 1 separately (**cold**) from runs 2–30 (**warm**). Quoting
only the warm number would flatter the app; quoting only the cold one would libel it.

**Median (p50)** — the middle value when you sort all the measurements. We prefer it over the average because
a single unlucky slow run drags an average around, whereas a median shrugs it off.

**p90** — 90% of runs were at least this fast. It answers "how bad does it get on a bad day?", which users
feel more than they feel the average.

**Latency vs throughput** — latency is how long *one* run takes (what a user waits for). Throughput is how
many runs per second you can sustain (what a server cares about). This project is about latency.

**ms** — millisecond, one thousandth of a second. For reference: a 60 fps screen draws a new frame every
16.7 ms, so anything under ~16 ms feels instant.

### About being sure it works

**Reference implementation** — a second, independent way to compute the same thing, used as an answer key. We
ran the same model files through Python's LiteRT on a laptop and compared. If the phone and the laptop agree,
the phone probably isn't lying.

**Verification** — this project checks each accelerator's output against the plain CPU's output *at startup*.
If they disagree by too much, the app refuses that accelerator. This is the check that caught the Neural
Engine returning wrong answers on a real iPhone.

**Deviation** — how far two results are apart, as a percentage of the output range. Healthy backends on our
phone differ by 0.0005%, which is harmless rounding. The Neural Engine differed by **4.946%**, which is not.

---

## Five things that surprised us (the fun part)

These are the findings a curious colleague will enjoy most. All measured on a real iPhone 13 Pro.

1. **A dedicated AI chip gave the wrong answer.** We asked for the Neural Engine. It accepted the job, ran it,
   and returned results 4.946% off from the CPU's — the same wrong amount every single time. Almost certainly
   because the Neural Engine computes in fp16. Our app refused to use it. Without that check, we'd have
   shipped an app that was quietly, consistently a bit wrong.

2. **The simulator lied to us.** That same broken configuration reported *perfectly healthy* on the iPhone
   simulator on a Mac — because Core ML there runs on the Mac's own chips and never touches a Neural Engine.
   If we'd only tested on a simulator, we'd never have known.

3. **Resizing the photo costs 3× more than the AI.** Inference is 4.5 ms; decoding the JPEG and scaling it to
   224×224 is 15 ms. Most of the "AI" time isn't AI — it's image plumbing. Speeding up on-device ML is usually
   an image-pipeline job.

4. **The GPU really is about twice as fast.** 4.5–4.9 ms vs 9.54 ms for the identical model on the identical
   API — a ≈2× win, measured three times rather than assumed.

5. **A "performance" delegate made things slower.** XNNPACK on the quantized model: 6.17 ms with it, 3.51 ms
   without. The fastest configuration we measured anywhere was the *simplest* one.

6. **How you crop the photo matters as much as which chip runs it.** Switching from a stretch to the standard
   ImageNet centre-crop nearly doubled confidence on one sample photo (0.28 → 0.51) — and destroyed it on
   another, by cropping away the very detail that identified the class. There's a toggle in the app; try both on
   the same photo and watch the number move.

---

## Try it yourself

The fastest way to get curious is to break something and watch what happens. All of these are safe.

Run the benchmark suite on a connected device:

```bash
flutter test integration_test/on_device_inference_test.dart
```

Run the plain unit tests (no device needed):

```bash
flutter test
```

Then try these experiments:

* **Feed it something it has never seen.** Point it at a mug, a cable, a face. The 1,001 categories don't
  include "laptop charger", so watch what it confidently guesses instead. This teaches more about ML than any
  tutorial: the model always answers, even when the right answer isn't available to it.
* **Compare the two models.** Switch between the quantized (4 MB) and float (13 MB) model in the app and watch
  both the confidence numbers and the timings move.
* **Switch accelerators** and watch the latency card change. Try asking for the NPU on an iPhone and read the
  refusal message.
* **Turn on airplane mode** and run it again. Nothing changes — which is the whole point.

---

## Questions beginners actually ask

**Why does it give low confidence on my own photos?**
Usually because your subject isn't one of the 1,001 categories. ImageNet has **no** "person", "building", "road"
or "food" class — but it does have 120 dog breeds. The model can't abstain, so it returns the nearest thing it
knows at a low score. That's the correct output for an impossible question, and the app now labels it
*"Inconclusive"* when it detects the pattern. Also try the **Centre-crop** toggle: phone photos are 4:3 or 16:9
and the default stretches them to a square, which costs accuracy (measured: 0.28 → 0.51 on the Labrador photo).

**Is this "real AI" or a trick?**
It's real machine learning, and it's also not magic. Nobody wrote rules like "if it has fur, guess dog". The
numbers in that file were *learned* from examples. But it isn't reasoning either — it's pattern-matching
arithmetic, which is why it can be confidently wrong about a coffee mug.

**Does it learn from my photos?**
No. The model file is read-only and identical for every user. Learning (training) happens elsewhere, before
shipping. This app only *runs* the frozen model.

**Why bother, when I could just call an API?**
Four reasons this project demonstrates: it works with no network at all; the photo never leaves the device, so
there's no privacy review to pass; there's no per-request cost; and there's no network round trip, so latency
is a few milliseconds instead of a few hundred.

**What's the catch?**
Size and honesty. The two models plus the runtime add ~36 MB to the app. Accelerator behaviour differs per
device, so you must test on real hardware. And you get no server-side logging, so you can't see how the model
behaves in the wild unless you build that yourself.

**Do I need to know maths to work on this?**
Not to *use* a model, which is what this project does. Everything here is file loading, image resizing, buffer
shapes, and measurement — ordinary engineering. Maths becomes necessary when you want to train or modify a
model.

**Where should I go next?**

| If you want to… | Read |
|---|---|
| Know what this app is and how to run it | [`../README.md`](../README.md) |
| See the numbers, explained | [`BENCHMARKS.md`](BENCHMARKS.md) |
| Understand how the code is put together | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| Choose between LiteRT / ML Kit / ONNX / cloud | [`COMPARISON.md`](COMPARISON.md) |
| See what's really inside a model file | [`MODEL_INSPECTION.md`](MODEL_INSPECTION.md) |
| Check the offline claim yourself | [`OFFLINE_VERIFICATION.md`](OFFLINE_VERIFICATION.md) |
| Read the hard questions with honest answers | [`QA.md`](QA.md) — Part A is written for beginners |
| Present this to your team | [`SLIDES.md`](SLIDES.md) + [`SPEAKER_NOTES.md`](SPEAKER_NOTES.md) |

And if something here was still confusing, that's a documentation bug worth reporting — this page exists to make
the rest readable.

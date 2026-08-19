#!/usr/bin/env python3
"""Builds the ground-truth fixture the Dart pipeline is validated against.

Three jobs:

1. NORMALIZATION PROBE. A float32 .tflite carries no preprocessing metadata (its quantization
   params are empty), so the correct input scaling cannot be read out of the file. We therefore
   determine it empirically: run the reference interpreter with each candidate scaling and see
   which produces a coherent prediction. The evidence is written into the fixture rather than
   asserted from a tutorial.

2. PREPROCESS PROBE. Generates a deterministic 224x224 PNG (no resize needed, lossless codec) and
   records the exact tensor the reference preprocessing produces for it. The Dart unit tests
   reproduce this tensor bit-for-bit, which validates normalization + channel order with the
   resampling filter taken out of the equation.

3. END-TO-END PREDICTIONS. Top-5 per sample image per model, used by the on-device integration test.

Usage: python3 tool/reference_predict.py
"""
import hashlib
import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
from PIL import Image

try:
    from ai_edge_litert.interpreter import Interpreter
    RUNTIME = "ai_edge_litert.interpreter.Interpreter"
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
    RUNTIME = f"tensorflow.lite.Interpreter (TF {tf.__version__})"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "assets", "models")
IMAGE_DIR = os.path.join(ROOT, "assets", "images")
FIXTURE = os.path.join(ROOT, "test", "fixtures", "reference_predictions.json")
SIDE = 224
CALIBRATION = "calibration_224.png"
SAMPLES = ["grace_hopper.jpg", "labrador.jpg", "cat_on_snow.jpg"]

MODELS = {
    # id                      file                                normalization applied in Dart
    "mobilenet_v2_float32": ("mobilenet_v2_1.0_224.tflite", "minus_one_to_one"),
    "mobilenet_v1_uint8": ("mobilenet_v1_1.0_224_quant.tflite", "raw_uint8"),
}

NORMALIZERS = {
    "minus_one_to_one": lambda a: (a.astype(np.float32) / 127.5) - 1.0,
    "zero_to_one": lambda a: a.astype(np.float32) / 255.0,
    "zero_to_255": lambda a: a.astype(np.float32),
    "raw_uint8": lambda a: a.astype(np.uint8),
}


def labels():
    with open(os.path.join(MODEL_DIR, "imagenet_labels_1001.txt")) as fh:
        return fh.read().splitlines()


def make_calibration_png(path):
    """Deterministic RGB pattern covering the full 0..255 range with distinct per-channel values."""
    a = np.zeros((SIDE, SIDE, 3), dtype=np.uint8)
    xs = np.arange(SIDE)
    a[:, :, 0] = ((xs * 255) // (SIDE - 1))[None, :]          # red varies along x
    a[:, :, 1] = ((xs * 255) // (SIDE - 1))[:, None]          # green varies along y
    a[:, :, 2] = ((xs[None, :] + xs[:, None]) * 255) // (2 * (SIDE - 1))
    Image.fromarray(a, mode="RGB").save(path, format="PNG", optimize=True)
    return a


def rgb_tensor(path):
    """Decode -> RGB -> bilinear stretch to 224x224. Mirrors what the Dart preprocessor does."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.size != (SIDE, SIDE):
            im = im.resize((SIDE, SIDE), Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)


def top_k(probs, names, k=5):
    order = np.argsort(probs)[::-1][:k]
    return [{"index": int(i), "label": names[int(i)], "score": round(float(probs[i]), 6)} for i in order]


def run(interp, tensor):
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    interp.set_tensor(inp["index"], tensor[np.newaxis, ...])
    interp.invoke()
    raw = interp.get_tensor(out["index"])[0]
    if out["dtype"] == np.uint8:                     # dequantize with the model's own parameters
        scale, zero = out["quantization"]
        return raw.astype(np.float32) * scale - zero * scale, raw
    return raw.astype(np.float32), raw


def sample_positions(n=64, total=SIDE * SIDE * 3):
    pos = {0, total - 1}
    pos.update((i * total) // n for i in range(n))
    return sorted(pos)


def main():
    names = labels()
    os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
    calib_path = os.path.join(IMAGE_DIR, CALIBRATION)
    make_calibration_png(calib_path)
    calib_rgb = rgb_tensor(calib_path)
    digest = hashlib.sha256(open(calib_path, "rb").read()).hexdigest()
    print(f"wrote {calib_path} (sha256 {digest[:16]}...)")

    interps = {}
    for mid, (fname, _) in MODELS.items():
        it = Interpreter(model_path=os.path.join(MODEL_DIR, fname))
        it.allocate_tensors()
        interps[mid] = it

    # ---- 1. normalization probe (float model only; the uint8 model's scaling is in its quant params)
    probe = {}
    probe_img = rgb_tensor(os.path.join(IMAGE_DIR, "grace_hopper.jpg"))
    print("\n=== normalization probe: mobilenet_v2_float32 on grace_hopper.jpg ===")
    for key in ("minus_one_to_one", "zero_to_one", "zero_to_255"):
        probs, _ = run(interps["mobilenet_v2_float32"], NORMALIZERS[key](probe_img))
        best = top_k(probs, names, 3)
        probe[key] = {"top3": best, "prob_sum": round(float(probs.sum()), 6)}
        print(f"  {key:17s} -> {best[0]['label']!r} {best[0]['score']:.4f}   (sum={probs.sum():.4f})")

    # ---- 2. preprocess probe on the calibration PNG
    idx = sample_positions()
    pre = {"image": f"assets/images/{CALIBRATION}", "sample_indices": idx, "sha256": digest}
    for mid, (_, norm) in MODELS.items():
        t = NORMALIZERS[norm](calib_rgb)
        flat = t.reshape(-1)
        pre[mid] = {
            "normalization": norm,
            "dtype": str(flat.dtype),
            "length": int(flat.size),
            "sample_values": [float(flat[i]) for i in idx],
            "min": float(flat.min()), "max": float(flat.max()), "mean": round(float(flat.mean()), 6),
        }
        probs, _ = run(interps[mid], t)
        pre[mid]["top5"] = top_k(probs, names)

    # ---- 2b. resize probe: the resized uint8 RGB bytes for a non-square photo.
    # Lets a host-side test measure how closely Dart's resampling matches the
    # reference filter, instead of assuming the two agree.
    resize = {"filter": "Pillow BILINEAR", "images": {}}
    for img in SAMPLES:
        rgb = rgb_tensor(os.path.join(IMAGE_DIR, img)).reshape(-1)
        resize["images"][img] = {
            "sample_indices": idx,
            "sample_values": [int(rgb[i]) for i in idx],
            "mean": round(float(rgb.mean()), 6),
        }

    # ---- 3. end-to-end predictions per sample image
    preds = {}
    print("\n=== reference predictions ===")
    for img in SAMPLES:
        rgb = rgb_tensor(os.path.join(IMAGE_DIR, img))
        preds[img] = {}
        for mid, (_, norm) in MODELS.items():
            probs, _ = run(interps[mid], NORMALIZERS[norm](rgb))
            preds[img][mid] = {"top5": top_k(probs, names), "prob_sum": round(float(probs.sum()), 6)}
            t1 = preds[img][mid]["top5"][0]
            print(f"  {img:18s} {mid:22s} -> {t1['label']!r} {t1['score']:.4f}")

    fixture = {
        "_comment": "GENERATED by tool/reference_predict.py - do not edit by hand.",
        "reference_runtime": RUNTIME,
        "resize": {"library": "Pillow", "filter": "BILINEAR", "target": [SIDE, SIDE],
                   "mode": "stretch to square, no crop, no letterbox"},
        "models": {mid: {"asset": f"assets/models/{f}", "normalization": n} for mid, (f, n) in MODELS.items()},
        "normalization_probe": probe,
        "preprocess_probe": pre,
        "resize_probe": resize,
        "predictions": preds,
    }
    with open(FIXTURE, "w") as fh:
        json.dump(fixture, fh, indent=2)
        fh.write("\n")
    print(f"\nwrote {os.path.relpath(FIXTURE, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

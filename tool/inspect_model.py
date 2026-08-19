#!/usr/bin/env python3
"""Ground truth for the tensor contract.

Loads each bundled .tflite with the Python LiteRT/TFLite reference interpreter and prints the exact
input/output index, name, shape, dtype and quantization parameters. Every constant used by the Dart
preprocessing/postprocessing code is derived from THIS output (see docs/MODEL_INSPECTION.md) so that
nothing about the model's I/O format is guessed.

Usage: python3 tool/inspect_model.py
"""
import hashlib
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

try:
    from ai_edge_litert.interpreter import Interpreter  # LiteRT standalone runtime, if present
    RUNTIME = "ai_edge_litert.interpreter.Interpreter"
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
    RUNTIME = f"tensorflow.lite.Interpreter (TF {tf.__version__})"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = ["mobilenet_v2_1.0_224.tflite", "mobilenet_v1_1.0_224_quant.tflite"]


def describe(details):
    q = details["quantization"]          # (scale, zero_point) legacy tuple
    qp = details.get("quantization_parameters", {})
    return (
        f"    index={details['index']}  name={details['name']!r}\n"
        f"    shape={list(details['shape'])}  dtype={details['dtype'].__name__}\n"
        f"    quantization(scale, zero_point)={q}\n"
        f"    quantization_parameters={{'scales': {list(qp.get('scales', []))}, "
        f"'zero_points': {list(qp.get('zero_points', []))}, "
        f"'quantized_dimension': {qp.get('quantized_dimension')}}}"
    )


def main():
    print(f"reference runtime: {RUNTIME}\n")
    for name in MODELS:
        path = os.path.join(ROOT, "assets", "models", name)
        raw = open(path, "rb").read()
        print("=" * 78)
        print(f"{name}")
        print(f"  size      : {len(raw)} bytes ({len(raw)/1024/1024:.2f} MiB)")
        print(f"  sha256    : {hashlib.sha256(raw).hexdigest()}")
        it = Interpreter(model_path=path)
        it.allocate_tensors()
        for label, dets in (("INPUT", it.get_input_details()), ("OUTPUT", it.get_output_details())):
            for d in dets:
                n = 1
                for dim in d["shape"]:
                    n *= int(dim)
                itemsize = d["dtype"]().itemsize
                print(f"  {label}")
                print(describe(d))
                print(f"    elements={n}  bytes={n * itemsize}")
        print(f"  total tensors in graph: {len(it.get_tensor_details())}")
        try:                                    # op list is a private API; best effort only
            ops = it._get_ops_details()
            kinds = sorted({o["op_name"] for o in ops})
            print(f"  operators ({len(ops)} nodes): {', '.join(kinds)}")
        except Exception as exc:                # noqa: BLE001 - diagnostic only
            print(f"  operators: unavailable ({type(exc).__name__})")
    print("=" * 78)
    labels = os.path.join(ROOT, "assets", "models", "imagenet_labels_1001.txt")
    with open(labels) as fh:
        lines = fh.read().splitlines()
    print(f"imagenet_labels_1001.txt: {len(lines)} lines")
    print(f"  first 3 : {lines[:3]}")
    print(f"  last 2  : {lines[-2:]}")
    trailing_blank = [i for i, l in enumerate(lines) if not l.strip()]
    print(f"  blank lines at indices: {trailing_blank}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

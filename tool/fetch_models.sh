#!/usr/bin/env bash
# Downloads the two pretrained LiteRT (.tflite) models + the ImageNet label set into assets/models/.
# Both archives are published by Google on storage.googleapis.com/download.tensorflow.org (Apache-2.0).
# Nothing here is generated or trained locally: we ship upstream artefacts and record their SHA-256.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/assets/models"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$DEST"

V1_ZIP="https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip"
V2_TGZ="https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz"

echo "==> MobileNetV1 1.0 224 quant (uint8) + labels"
curl -fsSL "$V1_ZIP" -o "$TMP/v1.zip"
unzip -o -j -q "$TMP/v1.zip" mobilenet_v1_1.0_224_quant.tflite labels_mobilenet_quant_v1_224.txt -d "$TMP"
cp "$TMP/mobilenet_v1_1.0_224_quant.tflite" "$DEST/"
cp "$TMP/labels_mobilenet_quant_v1_224.txt" "$DEST/imagenet_labels_1001.txt"

echo "==> MobileNetV2 1.0 224 (float32)  [~85 MB archive, only the .tflite is kept]"
curl -fsSL "$V2_TGZ" -o "$TMP/v2.tgz"
tar -xzf "$TMP/v2.tgz" -C "$TMP" mobilenet_v2_1.0_224.tflite
cp "$TMP/mobilenet_v2_1.0_224.tflite" "$DEST/"

echo
echo "==> assets/models contents"
ls -l "$DEST"
echo
echo "==> SHA-256 (record these in docs/MODEL_INSPECTION.md)"
shasum -a 256 "$DEST"/*

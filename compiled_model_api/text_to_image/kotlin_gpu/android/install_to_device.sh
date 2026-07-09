#!/usr/bin/env bash
# Stage the Z-Image LiteRT graphs + precomputed host inputs into the app's external files
# dir (far too big to bundle). Build them with the conversion scripts, then run:
#   ./install_to_device.sh <dir-with-the-tflites-and-gen_bins>   (default: current dir)
set -e
PKG=com.google.ai.edge.examples.text_to_image
DIR="${1:-.}"
DST="/sdcard/Android/data/$PKG/files"

GRAPHS=(
    z_embx.tflite z_embc.tflite z_refx.tflite z_refc.tflite
    zc_main0.tflite zc_main1.tflite zc_main2.tflite
    zc_main3.tflite zc_main4.tflite zc_main5.tflite
    zc_final.tflite zvae_int8_256.tflite
)

adb shell mkdir -p "$DST/gen_bins"
for g in "${GRAPHS[@]}"; do
    echo "pushing $g ..."
    adb push "$DIR/$g" "$DST/$g"
done
echo "pushing gen_bins/ ..."
adb push "$DIR/gen_bins/." "$DST/gen_bins/"
adb shell "chmod -R 666 $DST/gen_bins/*.bin" || true
echo "done — launch the Z-Image app and tap Generate."

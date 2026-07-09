# Object Detection with LiteRT — RF-DETR Nano (on-device, fully-GPU)

An Android sample that runs **RF-DETR Nano** ([roboflow/rf-detr](https://github.com/roboflow/rf-detr),
Roboflow 2025, an LW-DETR derivative, Apache-2.0) end-to-end on device with the LiteRT `CompiledModel` API.
RF-DETR is a **transformer** detector (windowed DINOv2-S backbone + deformable-attention DETR decoder) — a
family that doesn't ride the GPU API off-the-shelf (deformable `grid_sample` → `GATHER_ND`, windowed
attention → 5D/6D tensors, two-stage query selection → `TOPK`/`GATHER`). Here every one of those is
re-authored or split out, so the whole detector runs on the GPU. The app detects objects in a bundled
image and any image picked from the gallery, drawing the boxes + COCO labels.

## Model (two-graph split)

| Graph | In → Out | Delegate (Pixel 8a) |
| :-- | :-- | :--: |
| Graph A — backbone + encoder + proposal heads | image[1,3,384,384] → enc_class[1,576,91], enc_coord[1,576,4], memory[1,576,256] | **GPU** `1381/1381` |
| Graph B — two-stage combine + decoder + heads | (memory, refpoint_ts[1,300,4]) → boxes[1,300,4], logits[1,300,91] | **GPU** `404/404` |

fp16, converted with [litert-torch](https://github.com/google-ai-edge/litert) — per-graph tflite-vs-torch
corr **1.0**. On a Pixel 8a (Tensor G3): both graphs fully on `LITERT_CL`, **≈27 ms** model time; on a real
image the device chain reproduces the PyTorch detections at **IoU 0.98–0.99** (same class, matching score).

## How it splits (and why it's fully GPU)

RF-DETR is a two-stage DETR. The query selection (top-300 proposals by class score) is `TOPK`/`GATHER`,
which have no GPU op — but the proposal grid is image-independent, so the model splits at exactly that point
(the standard two-stage-DETR edge split):

```
image →[GPU Graph A]→ enc_class, enc_coord, memory
      →[host: top-300 by max class score → gather coords]→ refpoint_ts
      →[GPU Graph B (memory, refpoint_ts)]→ boxes, logits
      →[host: sigmoid + threshold + cxcywh→xyxy + per-class NMS]→ detections
```

The enc auxiliary outputs (`memory_ts`/`boxes_ts`) are dead at inference, so the host step is just a topk +
coord-gather. See [`conversion/`](conversion/) for the re-authoring (windowed attention, deformable
`grid_sample` → tent-matmul, MSDeformAttn ≤4D, baked sine pos-embed) and the fp16-safe LayerNorm fixes the
Mali delegate needs.

## Output & decode

Graph B emits `boxes` (cxcywh, normalized `[0,1]`) and `logits` (91-way, index = COCO category id). The host
(`RfDetr.kt`) applies sigmoid + score threshold + `cxcywh→xyxy` + per-class NMS (light — removes the fp16
near-duplicate queries). Preprocessing: square resize to 384×384, RGB, ImageNet mean/std, NCHW.

## Run

1. Build the two tflites with [`conversion/build_rfdetr_split.py`](conversion/build_rfdetr_split.py), or get
   them from [litert-community/RF-DETR-Nano-LiteRT](https://huggingface.co/litert-community/RF-DETR-Nano-LiteRT).
2. Build/install the app and push the models:
   ```bash
   cd android
   ./gradlew :app:installDebug
   ./install_to_device.sh <dir-with-the-two-tflites>
   ```
   (`test_image.jpg` + `coco_labels.txt` are bundled.)
3. Launch **RF-DETR** — it compiles the GPU shaders (~1 s/graph on first launch), then detects objects.

## App architecture

The app follows the same MVVM + Jetpack Compose structure as the
[image_segmentation](../../image_segmentation/kotlin_cpu_gpu) sample: `MainActivity` is a thin Compose
host, `MainViewModel` owns the detector and exposes a single `UiState`, and all inference stays in
`RfDetr`. The view model confines every model call to one worker, since `RfDetr` reuses native buffers.

## Files

| File | Description |
|------|-------------|
| `android/.../rf_detr/RfDetr.kt` | Both GPU graphs on CompiledModel + host topk/gather + decode + per-class NMS |
| `android/.../rf_detr/MainActivity.kt` | Thin Compose host: wires the gallery picker to the view model |
| `android/.../rf_detr/MainViewModel.kt` | Owns `RfDetr`, runs detection off the main thread, exposes `UiState` |
| `android/.../rf_detr/UiState.kt` | Immutable UI state + the display-ready `DetectionBox` |
| `android/.../rf_detr/ImageUtils.kt` | Asset/gallery decode, EXIF orientation, square resize, RGB conversion |
| `android/.../rf_detr/view/DetectionScreen.kt` | Compose UI: status header, image picker, results list |
| `android/.../rf_detr/view/DetectionOverlay.kt` | Draws the image and detection boxes on a Compose `Canvas` |
| `android/.../rf_detr/view/Theme.kt`, `view/Color.kt` | App theme and the per-class box palette |
| `android/.../assets/coco_labels.txt` | 91-line COCO label table (index = COCO category id) |
| `conversion/` | litert-torch conversion (2-graph split + SafeLayerNorm) + notes |

**Original model**: [roboflow/rf-detr](https://github.com/roboflow/rf-detr) (RF-DETR Nano) ·
[Apache-2.0](https://github.com/roboflow/rf-detr/blob/main/LICENSE)

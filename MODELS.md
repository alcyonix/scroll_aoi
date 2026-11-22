# Model files (download manually, once)

aoi_tool **never downloads anything**. Put these files in `models/` yourself,
then disconnect. Each file's SHA-256 is printed by `python -m aoi_tool selftest`
and written to `run_metadata.json` on every run: record them once so you can
prove later that the same weights were used for every participant.

| AOI | Model | Licence | Put it at |
|---|---|---|---|
| 1 Faces | YuNet 2023mar (OpenCV Zoo) | MIT | `models/face_detection_yunet_2023mar.onnx` |
| 4 Subtitles | PP-OCRv3 English text detector (PaddleOCR, via OpenCV Zoo) | Apache-2.0 | `models/text_detection_en_ppocrv3_2023may.onnx` |
| 2 Bodily actions, 3 Interactions | RTMO-s body7 (OpenMMLab MMPose) | Apache-2.0 | `models/rtmo-s/end2end.onnx` |
| 5 Salient objects, 6 Emotional items (+ interaction prompts) | OWLv2 base patch16 ensemble (Google), ONNX export | Apache-2.0 | `models/owlv2-base-patch16-ensemble/` (see below) |

## 1. YuNet (about 230 KB)
- https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
- or the same path on GitHub: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet

## 2. PP-OCRv3 text detection (about 2.4 MB)
- https://huggingface.co/opencv/opencv_zoo/resolve/main/models/text_detection_ppocr/text_detection_en_ppocrv3_2023may.onnx

If a GitHub "raw" link gives you a tiny text file instead of a model, you got a
Git-LFS pointer. Use the Hugging Face link instead.

## 3. RTMO-s (zip, about 40 MB)
- https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip
- Unzip it, find `end2end.onnx` inside, and copy it to `models/rtmo-s/end2end.onnx`.
- Mirror if that server is down: https://huggingface.co/Tau-J/RTMPose
- Larger and more accurate (slower): `rtmo-m` / `rtmo-l` (same page). Change
  `detectors.pose.model` in your config.

## 4. OWLv2 (about 600 MB)
Repository: https://huggingface.co/Xenova/owlv2-base-patch16-ensemble
(ONNX export of https://huggingface.co/google/owlv2-base-patch16-ensemble)

Download these into `models/owlv2-base-patch16-ensemble/`, keeping the `onnx/` subfolder:
```
models/owlv2-base-patch16-ensemble/onnx/model.onnx
models/owlv2-base-patch16-ensemble/vocab.json
models/owlv2-base-patch16-ensemble/merges.txt
models/owlv2-base-patch16-ensemble/special_tokens_map.json
```
`onnx/model_fp16.onnx` (half the size) also works on GPU. Set
`detectors.openvocab.onnx_file: onnx/model_fp16.onnx`.

Downloading in a browser is fine. You do NOT need `huggingface_hub`,
`transformers`, `torch`, `mmpose` or `rtmlib`: aoi_tool does its own
pre- and post-processing.

## After downloading
```
python -m aoi_tool selftest
```
This must show `[OK]` for each model, the CLIP tokenizer check
`a photo of a cat -> [49406, 320, 1125, 539, 320, 2368, 49407]`, and every
network test `[BLOCKED]`.

## Files are data, not code
`.onnx` files are loaded with ONNX Runtime / OpenCV DNN. These parse a
computation graph and cannot run arbitrary Python (unlike PyTorch `.pt`/`.pth`
pickles, which this project deliberately avoids). The JSON/TXT tokenizer files
are read with `json.load` / plain text reading.

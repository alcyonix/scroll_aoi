# Notes for your code review

## Where third-party code is (and is not)
No GitHub repository code is copied, vendored or imported. Every line under
`aoi_tool/` was written for this project. Reference implementations were *read*
only to match each model's expected input/output format. The relevant lines
are re-implemented and cited in each module's docstring:

| Module | Reference consulted | What was taken |
|---|---|---|
| `detectors/pose.py` | Tau-J/rtmlib `rtmo.py` | 640x640 canvas, pad value 114, top-left placement, no mean/std, divide outputs by resize ratio |
| `detectors/text.py` | opencv/opencv_zoo `ppocr_det.py` | DB thresholds 0.3 / 0.5, unclip 2.0, 200 candidates, ImageNet mean/scale |
| `detectors/faces.py` | OpenCV `FaceDetectorYN` API | N x 15 output layout |
| `detectors/openvocab.py` | huggingface/transformers `image_processing_owlv2.py` | rescale → pad bottom/right to square → resize 960 → CLIP normalise; sigmoid scores; boxes scaled by max(h, w) |
| `clip_tokenizer.py` | OpenAI CLIP byte-level BPE algorithm | algorithm only; reads `vocab.json` / `merges.txt` |
| `camera.py` | dcnieho/glassesTools `importing/tobii_G3.py` | JSON key names in `recording.g3` |
| `saliency.py` | Hou & Zhang (2007), CVPR paper | spectral-residual equations |

Some of these pages were read through a summarising web reader, not as raw
text. Where a detail could not be confirmed verbatim, the code checks it at run
time instead of assuming it: RTMO output tensors are identified by shape, OWLv2
inputs by name, the CLIP tokenizer is verified against known ids by `selftest`,
and the OWLv2 grey pad value (0.5) is configurable.

## Suggested review order
1. `aoi_tool/__main__.py` and `aoi_tool/netguard.py`: the guard is installed before anything else
2. `aoi_tool/onnx_utils.py` (provider allowlist, telemetry off) and `aoi_tool/video_io.py` (local files only)
3. `aoi_tool/audit.py`, then run `python -m aoi_tool audit` and `audit --deps`
4. Everything else is numerical code that performs no I/O, except for:
   - reading: the video, config YAML, model files, the gaze file, the seeds CSV
   - writing: CSV/JSON/MP4 into `--out`, and the seeds CSV in `annotate-screen`

## Every file the program opens
- read: `config/*.yaml` (`yaml.safe_load`), model files in `models/`, `--video`,
  `recording.g3` next to the video, `--gaze`, `--seeds`
- write: `--out/detections.csv`, `screen_track.csv`, `gaze_aoi_hits.csv`,
  `aoi_summary.csv`, `run_metadata.json`, `qa_overlay.mp4`; the `annotate-screen` seeds CSV

## Dependency scan result (this machine, `audit --deps`)
- numpy: `lib/_datasource.py` imports urllib. It backs `np.load`/`np.genfromtxt`
  when given a URL; aoi_tool never calls these. The other hits are numpy's own test files.
- onnxruntime: `transformers/convert_tf_models_to_pytorch.py` imports requests.
  This is a standalone conversion script that is never imported.
- cv2, yaml: no Python-level network imports.
In any case, the runtime guard blocks an actual attempt from any of them.

## Things to double-check on your machine
- `selftest` shows every network test `[BLOCKED]` on your OS (tested here on Linux).
- If a process-spawn block ever triggers on Windows or macOS during normal use,
  the error names the call. `--allow-subprocess` (put it *before* the command)
  logs instead of blocking; network stays blocked.
- `run_metadata.json` → `models.*.sha256`: keep these with your data.

# aoi_tool: offline AOI detection on phone screens in Tobii scene video

Finds the phone screen in each frame of a Tobii Pro Glasses scene video,
corrects its perspective, and draws a box around every AOI **on the screen
only**, frame by frame. This keeps working when the participant opens the
TikTok comments sheet. Optionally, it maps each gaze sample onto those AOIs.

| # | AOI class (`aoi_class`) | How it is detected |
|---|---|---|
| 1 | `face` | YuNet face detector |
| 2 | `bodily_action` | RTMO pose + explicit limb-motion rule (dancing / gesturing / leg movement) |
| 3 | `physical_interaction` | RTMO pose + explicit contact rule (hug / handshake / face contact / touch), plus optional OWLv2 prompts |
| 4 | `subtitle` | PP-OCRv3 text detector + UI-zone filter (text in the video area only) |
| 5 | `salient_object` | OWLv2 product prompts + explicit salience score (saliency, size, centrality, focus, held in hand) |
| 6 | `emotional_item` | OWLv2 with **every item listed explicitly** in `config/emotional_items.yaml` (valence + category) |

Everything runs locally. No line of code downloads, uploads or phones home.
See **Network safety** below and `REVIEW_NOTES.md`.

---

## 1. Install (once, with network)
```
python -m pip install -r requirements.txt
```
Then download the four model files listed in `MODELS.md` into `models/`.
After that you can (and should) work fully offline.

## 2. Check everything
```
python -m aoi_tool selftest      # network guard, models, tokenizer
python -m aoi_tool audit         # static scan of this project's source
python -m aoi_tool audit --deps  # also list network-capable files in dependencies
```

## 3. Run
Always start with a 30-60 s test clip and look at the QA video before
processing a whole session:
```
python -m aoi_tool run --video P01/scenevideo.mp4 --out results/P01 \
       --gaze P01/gazedata.gz --start 0 --end 1500
```
Useful options:
- `--providers CUDAExecutionProvider,CPUExecutionProvider` uses an NVIDIA GPU (needs `onnxruntime-gpu`)
- `--config my_study.yaml` overrides any value in `config/default.yaml`
- `--seeds P01_screen.csv` adds manual screen corners (see step 4)
- `--skip-missing` runs with whichever models are present
- `--no-qa` skips the overlay video (faster)

If `recording.g3` (Glasses 3) sits next to the video, the lens is undistorted
automatically using Tobii's own scene-camera calibration.

## 4. Fix hard frames (optional)
If the QA video shows the screen outline (green or yellow) drifting or lost,
click the four screen corners on a few frames:
```
python -m aoi_tool annotate-screen --video P01/scenevideo.mp4 --seeds P01_screen.csv
```
Keys: left-click the corners TL, TR, BR, BL; `space` saves; `a`/`d` step ±1 frame;
`s`/`w` ±25; `q`/`e` ±250; `u` accepts the automatic outline; `esc` saves and quits.
Annotated frames override detection and re-anchor tracking.

---

## Outputs (all in `--out`)
| File | Content |
|---|---|
| `detections.csv` | One row per AOI per frame (see columns below) |
| `screen_track.csv` | Per frame: screen visible, method (`detected` / `tracked` / `manual` / `lost`), confidence, 4 corners in raw video px, `ui_state` (`feed` / `comments_open`), comments-panel top, `content_change` (new video / scroll) |
| `gaze_aoi_hits.csv` | Per gaze sample: frame, gaze px, on screen?, screen coordinates, list of AOIs hit |
| `aoi_summary.csv` | Per AOI label: frames present, seconds present, gaze samples, approximate dwell seconds |
| `qa_overlay.mp4` | Scene view with screen outline, AOIs and gaze (left) + rectified screen with labelled boxes (right) |
| `run_metadata.json` | Software versions, model SHA-256 hashes, full config used, warnings |

`detections.csv` columns:
- `aoi_class`, `label`, `category`, `valence`: what was found (e.g. `emotional_item`, `puppy`, `Young & baby beings`, `positive`)
- `score`: model confidence
- `source`: `detected` (model ran this frame), `carried` (detector skipped this frame, so the box is reused from its last run), `rule` (pose rule), `openvocab` (interaction prompt)
- `track_id`: stable id for the same AOI across frames (resets on a new video / scroll)
- `zone`: `video_content`, `comments_panel`, `right_rail`, `caption_area`, `top_bar`, `bottom_nav`
- `screen_x1..y2`: box in screen coordinates, 0-1 (top-left = 0,0)
- `scene_x1..y4`: the box's 4 corners in **raw scene-video pixels** (a tilted quadrilateral, because the phone is seen in perspective)
- `scene_bbox_*`: axis-aligned bounding box of that quadrilateral
- `extra`: JSON with rule details (speeds, person ids, salience features, landmarks)

**Which coordinates to trust:** the `scene_*` coordinates are exact for any
detected box. The 0-1 `screen_*` coordinates depend on where the screen
outline was placed. At the bottom edge that can be off by a few percent when
TikTok's black nav bar merges with a black bezel (see Limitations).

---

## Network safety (what guarantees "0 external connections")
1. **No network code.** `python -m aoi_tool audit` parses every source file
   and flags any import of socket/urllib/http/requests/subprocess etc., any URL,
   and any eval/exec/pickle. Result: 0 unexplained findings. The only
   exceptions are in `netguard.py`, which uses them to *test* that they are blocked.
2. **Runtime kill switch** (`aoi_tool/netguard.py`). Before numpy, OpenCV or
   ONNX Runtime are imported, a CPython audit hook (PEP 578) is installed. It
   aborts any socket creation, DNS lookup, connect, URL open, browser launch,
   or process spawn in this process, including through the raw `_socket` C
   module. Audit hooks cannot be removed once installed. `selftest` proves it.
3. **Library hardening.** ONNX Runtime's Windows telemetry is switched off
   (`disable_telemetry_events`). Execution providers are restricted to a
   local-only allowlist. This matters because the stock ONNX Runtime wheel ships
   `AzureExecutionProvider`, a cloud-inference backend, which is refused. OpenCV
   only ever receives paths to existing local files, never URLs (its FFmpeg
   backend could otherwise open http/rtsp streams from native code). YAML uses
   `safe_load`. Models are ONNX graphs, not pickles.
4. **No ML frameworks with auto-download behaviour.** No PyTorch, transformers,
   huggingface_hub, ultralytics, mmpose or rtmlib is used. All pre- and
   post-processing is written in this project, so there is no third-party repo
   code to trust.

### Running fully air-gapped (the absolute guarantee)
The Python guard cannot see native C/C++ code that might open sockets on its
own. Neither OpenCV nor ONNX Runtime does this for local files, but the only
absolute guarantee is the operating system:
- **Simplest:** turn off Wi-Fi / unplug Ethernet while processing.
- **Linux:** `unshare -rn python -m aoi_tool run ...` (no network namespace) or `firejail --net=none ...`
- **Windows:** Windows Defender Firewall → Outbound Rules → New Rule → Program → your `python.exe` → Block.
- **macOS:** use an application firewall that blocks outbound connections (e.g. LuLu) for python, or disconnect.

---

## Configuration you will probably tune
All thresholds live in `config/default.yaml`. Put your changes in your own YAML and pass `--config`.
- `screen.expected_aspect`: your phone's screen height/width (iPhone 15: 2556/1179 = 2.17)
- `detectors.*.every_n_frames`: run expensive detectors less often (boxes are carried in between)
- `detectors.openvocab.default_threshold` and per-item `threshold:` in the item YAMLs
- `aoi.bodily_action.*_speed_threshold`, `aoi.interaction.*_distance`, `aoi.salient_object.threshold` / `weights`
- `layout.zones_feed`: positions of TikTok UI regions (they differ slightly between app versions and phones)
- `gaze.hit_margin_deg`: pad AOIs by the tracker's accuracy (Glasses 3: 0.6°) when testing gaze hits

## Validating against human coding (strongly recommended)
These detectors are pretrained on general photos, not on 230-pixel-wide phone
screens filmed by a head camera. Before analysing results:
1. Hand-code a stratified sample of frames (for example 300 frames across participants, including comments-open frames).
2. Compare with `detections.csv` per AOI class: precision, recall, and IoU of boxes.
3. Adjust thresholds, then re-check on a *different* sample.
Report the final agreement figures in your methods section.

## Performance (rough)
- Screen tracking + layout: about 20 frames/s on 2 CPU cores.
- Faces (YuNet): a few ms per frame. Text (PP-OCRv3 at 416x896): about 0.1-0.3 s per frame on CPU.
- Pose (RTMO-s): about 50-100 ms per frame on CPU.
- OWLv2: about 1-4 s per call on CPU, about 0.1 s on a modern GPU. It runs every 5th frame by default
  (and immediately on every new video or scroll).
A 20-minute session (30,000 frames) is many hours on CPU and well under an hour on a GPU.

## Limitations (please read)
- **Screen outline at the bottom edge.** TikTok's black bottom bar and a black bezel look identical, so the
  outline may stop at the top of the bar or extend to the bezel. In synthetic tests the top and side edges
  were within 1-3 px; the bottom edge was off by about 20-40 px on a 500 px tall screen when the bar was
  black, and in dark-content frames. AOIs mapped to scene pixels are unaffected. Use `annotate-screen` or
  `screen.extend_bottom_to_aspect` if needed.
- **Resolution.** At a typical 30-40 cm viewing distance, the phone screen is only about 230 x 500 px in the
  1920 x 1080 Glasses 3 frame. Small items (comment avatars, emoji, small text) are near the limit of every detector.
- **Bodily action / interaction / salience are explicit heuristics** on top of pose and saliency. They are
  transparent and tunable, but they are not trained action classifiers. Validate them.
- **"Subtitle" = any text in the video area,** including creator text overlays and TikTok auto-captions. The
  creator caption/username, comments and UI text are classified `other_text` (hidden unless
  `keep_non_subtitle_text: true`).
- **Emotional items are object-presence flags** with a normative valence. They are not a measure of the
  participant's emotion, and context can change the valence.
- The comments-sheet detector expects TikTok's flat light or dark sheet background. Other apps or unusual
  themes need the `layout.*` thresholds adjusted.
- Phones held in landscape are not handled (the canvas is portrait).

## Tests
```
python tests/test_units.py
python tests/make_synthetic.py synth.mp4 synth_gt.csv
python tests/test_e2e.py synth.mp4 synth_gt.csv out_e2e
python -m aoi_tool run --video synth.mp4 --out out_s --skip-missing --no-qa
python tests/eval_screen.py synth_gt.csv out_s/screen_track.csv
```

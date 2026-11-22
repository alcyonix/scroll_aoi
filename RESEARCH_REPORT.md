# Research notes: building a local AOI detector for TikTok viewing in Tobii scene video

## 1. How Tobii Pro Glasses record, and what that means here

| | Glasses 3 | Glasses 2 | Glasses X (2025) |
|---|---|---|---|
| Scene video | 1920 x 1080 @ 25 fps, H.264 MP4 | 1920 x 1080 @ 25 fps | 2560 x 1440 @ 30 fps, H.264 MP4 |
| Scene FOV | 106° diag / 95° H / 63° V (16:9) | 90° diag (82° H / 52° V) | 132° diag / 118° H / 67° V |
| Gaze rate | 50 or 100 Hz | 50 or 100 Hz | 50-100 Hz |
| Accuracy (spec) | 0.6° | | |
| Recording folder | `scenevideo.mp4`, `gazedata.gz`, `recording.g3` (contains scene-camera calibration) | `livedata.json.gz` | cloud workflow (Glasses Explore) |

Sources: Tobii product page and comparison page; Tobii Pro Glasses 3 Developer
Guide; the GlassesViewer / glassesTools projects (Niehorster et al.).

**Gaze format (Glasses 3).** `gazedata.gz` is newline-delimited JSON. `gaze2d`
is in normalised scene-video coordinates ((0,0) is the top-left of the frame,
(1,1) the bottom-right). Timestamps are seconds from recording start. Samples
without a gaze estimate carry an empty `data` object. aoi_tool reads this
directly.

**Implications for AOI detection (Glasses 3, worked out from the specs):**
- About 20 px per degree near the image centre (1920 px / 95°). A 6.1-6.7" phone
  held at 30-40 cm subtends roughly 10-12° x 22-25°, so **the whole screen is
  only about 230 x 500 px** in the scene video. That is about 4% of the phone's
  native pixel count. A subtitle line is about 15 px tall; a comment avatar's
  face about 10-15 px. These sizes drove the choices below: detectors run on a
  rectified, upscaled screen; YuNet gets 1.5x upscaling; text uses a
  416 x 896 input.
- 0.6° accuracy ≈ 12 px in the scene ≈ 3-4 mm on the screen, about 5% of
  screen width. That is fine for a large face; it is borderline for a single
  subtitle word or emoji. Hence the optional `gaze.hit_margin_deg`.
- The wide-angle lens bends straight phone edges. `recording.g3` stores the
  intrinsics (focal length, principal point, radial and tangential distortion),
  so aoi_tool undistorts first and the screen becomes a true quadrilateral.
- 25 fps with head and hand motion produces motion blur and screen glare. The
  screen also changes exposure relative to the room (often overexposed). The
  screen locator is therefore multi-threshold and tracking-based, not a single
  fixed threshold.

## 2. How image-detection models work (short primer)
- **Backbone and head.** A convolutional network or Vision Transformer turns
  the image into feature maps. A detection head predicts, for many candidate
  locations, a box and class scores. Non-maximum suppression (NMS) removes
  duplicate boxes.
- **Closed-set detectors** (YOLO family, RT-DETR, RF-DETR, YOLOX) can only
  output the fixed classes they were trained on (COCO: 80 classes, including
  person and cell phone). They are fast and accurate within those classes, but
  they cannot detect "subtitle", "snake", "wedding ring" or "lipstick" unless
  retrained on labelled data.
- **Specialist detectors** are trained for one object type: faces (YuNet,
  RetinaFace, SCRFD), text (DB / PP-OCR, EAST, CRAFT), human pose (RTMPose,
  RTMO, ViTPose; these output 17 body keypoints per person).
- **Open-vocabulary detectors** (OWL-ViT/OWLv2, Grounding DINO, YOLO-World,
  YOLOE, SAM 3) pair an image encoder with a text encoder trained on image-text
  data. You type the class names; the model scores every image region against
  every text query. This is the only practical way to detect a long, explicit,
  researcher-defined list (200 emotional items) without collecting training data.
- **Action and interaction recognition** either classifies short clips (video
  transformers such as VideoMAE) or reasons over pose keypoints over time.
  Pose-based rules are less accurate than a trained classifier but transparent:
  every threshold can be reported and tuned.
- **Saliency** models predict where attention is drawn. Classic model-free
  methods such as spectral residual (Hou & Zhang, 2007) need no training data or weights.

## 3. Model choice per AOI (and why)

| AOI | Chosen | Alternatives considered | Reason |
|---|---|---|---|
| Phone screen | Classical CV (multi-threshold contours + line-fit corners + KLT/RANSAC homography tracking) + manual seeds | COCO "cell phone" detector; segmentation (SAM) | A bounding box is not enough: perspective correction needs the four corners. Classical CV is fast, fully transparent and needs no weights. |
| 1 Faces | **YuNet** (MIT) | RetinaFace, SCRFD (InsightFace) | Tiny (about 230 KB), built into OpenCV, rated for faces of 10-300 px. InsightFace models auto-download and have non-commercial weight licences. |
| 2 & 3 Pose | **RTMO-s** (Apache-2.0) | YOLOv8/11-pose (AGPL; the Ultralytics package has analytics, auto-download and auto-pip-install), RTMPose (two-stage) | One-stage multi-person, a direct ONNX download, permissive licence, no framework needed. |
| 4 Subtitles | **PP-OCRv3 DB** text detector (Apache-2.0) via OpenCV | EAST, CRAFT, EasyOCR (auto-downloads) | Robust on overlaid text, 2.4 MB, runs in OpenCV. "Subtitle versus UI text" is decided by screen zone. |
| 5 & 6 | **OWLv2** base ensemble (Apache-2.0) | Grounding DINO (prompt-length limit; collapses nearby instances), YOLO-World/YOLOE (fast but needs Ultralytics to export with fixed classes), SAM 3 (heavy, custom licence), Florence-2 | Independent benchmarks on art and aerial imagery found OWLv2 the most effective zero-shot detector, with high recall. It scores hundreds of queries in one pass. An ONNX export exists, so no PyTorch or transformers is needed. |
| 5 salience | Spectral residual + focus + size + centrality + in-hand | U²-Net, DeepGaze | Model-free and explainable. Each feature is logged in `extra`. |

**Speed trade-off.** OWLv2 is the bottleneck (about 1-4 s per frame on CPU).
If you have a GPU, set `every_n_frames: 1` for true per-frame open-vocabulary
detection. If speed matters more than flexibility, a YOLOE/YOLO-World model
exported once to ONNX with this exact item list would be about 20-50x faster.
That export step needs the Ultralytics package in a separate, networked
environment, which is why it is not the default here.

## 4. TikTok-specific handling
- **Comments open.** The comment sheet slides up and the video shrinks into the
  top part of the screen. aoi_tool detects the sheet from its flat light or
  dark background and its top edge, with 3-frame hysteresis. Detection keeps
  running on the whole screen. Every AOI is tagged with its `zone`
  (`video_content` vs `comments_panel`), and the video region used for
  salience shrinks accordingly.
- **New video / scroll.** A large change in the rectified screen resets
  trackers, motion histories and carried boxes, and forces every detector to
  run on that frame.
- **UI zones** (right action rail, caption/username area, nav bars) are
  configurable rectangles. They differ slightly across app versions and phones.

## 5. Emotional-item taxonomy (AOI 6): 200 explicit items
Every item has a fixed valence and category in `config/emotional_items.yaml`.
The "basis" field records whether a category follows a normed affective picture
database (GAPED: snakes, spiders, human-rights violations and animal
mistreatment as negative; human and animal babies and nature scenery as
positive; inanimate objects as neutral), the wider threat/disgust stimulus
literature (e.g. traffic accidents, dirty toilets), or researcher judgement
(to be validated).

**Positive (69 items)**
- Young & baby beings (7)
- Companion & pleasant animals (8)
- Appetizing food & treats (14)
- Nature & scenic beauty (12)
- Celebration & festivity (10)
- Affection & romance symbols (5)
- Leisure, fun & luxury (7)
- Positive emoji & stickers (6)

**Neutral (70 items)**
- Household items & furniture (12)
- Kitchenware & tableware (12)
- Electronics & devices (10)
- Clothing & accessories (7)
- Stationery & office objects (7)
- Vehicles & transport (8)
- Buildings & built environment (9)
- Common plants & natural elements (5)

**Negative (61 items)**
- Weapons & violence (6)
- Injury, blood & medical distress (9)
- Threatening animals & pests (9)
- Disgust & contamination (7)
- Danger, disaster & destruction (9)
- Death & mortality (6)
- Human distress & rights violations (6)
- Animal mistreatment (3)
- Health-risk substances (2)
- Negative emoji & warning signs (4)

Open questions for you to decide:
- Should alcohol (wine glass, beer bottle; currently neutral) and cigarettes
  or vapes (currently negative) keep these valences for your population?
- Should emoji in the comments count? They are detected and tagged
  `zone = comments_panel`, so you can filter them either way.
- "Crying person" and "screaming person" overlap with facial-expression
  analysis. Keep them as items, or move them to a separate expression coder?

## 6. Salient-object candidates (AOI 5): 30 product types
Listed in `config/salient_object_prompts.yaml`. An object is reported only
when its salience score is at least 0.55. The score is a weighted mean of
saliency 0.30, size 0.20, centrality 0.20, focus contrast 0.15 and held in
hand 0.15. These are the compositional cues of advertising product shots.

## 7. Verification done in this build
- The network guard blocks socket creation (including the raw `_socket` C
  module), DNS, connect, urllib, http.client and process spawns. Static audit:
  0 unexplained findings.
- Synthetic test video (hand-held phone, feed → comments → dark content →
  thumb occlusion, 200 frames):
  - Screen corners: 1.2 px mean error with the comments sheet open. In the
    feed, top corners were within 3.3 px and bottom corners within 23 px (black
    nav bar). Dark content: about 18 px. With thumb occlusion: top 8 px, bottom 38 px.
  - Comments-state detection correct on 95-100% of frames; the misses are the
    3-frame hysteresis at transitions.
  - End to end, with colour stand-in detectors: AOI centres mapped back to
    scene pixels within 1.1 px mean (3.9 px max). Subtitle and UI text
    separated correctly. 100% of gaze samples aimed at the face hit the face AOI.
- Model adapters were tested against simulated ONNX sessions: input shape,
  padding value, coordinate rescaling, and OWLv2 padded-square box scaling.
- **Not yet tested:** the real pretrained models on real Tobii footage (no
  network in the build environment to download weights). Run `selftest`, then
  a short clip, and inspect `qa_overlay.mp4` before scaling up.

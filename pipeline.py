"""Frame-by-frame orchestration.

For every scene-video frame:
  raw frame -> (optional) undistort -> locate phone screen -> rectify screen
  -> layout (feed vs. comments open, UI zones) -> content-change test
  -> detectors (faces, text, pose, open-vocab), each on its own schedule;
     frames in between reuse the last result in SCREEN coordinates
     (source = "carried") unless the screen content changed
  -> rules (bodily action, interaction, salience) -> zone filtering
  -> track ids -> map boxes back to raw scene-video pixels
  -> optional gaze-to-AOI hit test -> CSV rows -> optional QA video frame.

Only the phone screen is ever analysed: detectors never see pixels outside the
rectified screen image.
"""

import copy
import os
import platform
import sys
import time

import cv2
import numpy as np

from . import __version__
from .aoi_rules import BodilyActionRule, InteractionRule, PersonTracker, SalientObjectRule
from .camera import find_calibration
from .config import load_query_items, resolve
from .detectors import Detection
from .geom import box_corners, transform_points
from .layout import LayoutAnalyzer
from .output import OutputWriter
from .qa import render as qa_render
from .saliency import focus_map, spectral_residual
from .screen import ScreenLocator, load_seeds
from .tracking import IoUTracker
from .video_io import VideoReader, VideoWriter, assert_local_file

AOI_CLASSES = ["face", "bodily_action", "physical_interaction", "subtitle",
               "salient_object", "emotional_item", "other_text"]


class Pipeline:
    def __init__(self, cfg, video_path, out_dir, gaze_path=None, seeds_path=None,
                 skip_missing=False, qa_video=None, log=print):
        self.cfg = cfg
        self.log = log
        self.reader = VideoReader(video_path)
        self.cam = find_calibration(self.reader.path, cfg)
        seeds_path = seeds_path or resolve(cfg["screen"].get("seeds_csv"))
        seeds = load_seeds(assert_local_file(seeds_path)) if seeds_path else {}
        if self.cam is not None and seeds:
            seeds = {k: self.cam.undistort_points(v).astype(np.float32) for k, v in seeds.items()}
        self.locator = ScreenLocator(cfg, seeds)
        self.layout = LayoutAnalyzer(cfg)
        self.out_dir = out_dir
        self.warnings = []
        self.models, self.model_files = {}, {}
        self._load_detectors(skip_missing)

        self.kp_thr = float(cfg["detectors"]["pose"]["keypoint_threshold"])
        self.ptracker = PersonTracker()
        self.body_rule = BodilyActionRule(cfg, self.kp_thr)
        self.inter_rule = InteractionRule(cfg, self.kp_thr)
        self.sal_rule = SalientObjectRule(cfg, self.kp_thr)
        self.trackers = {c: IoUTracker(0.3, 5) for c in AOI_CLASSES}
        self.cache = {}
        self.prev_small = None

        self.gaze = None
        if gaze_path:
            from .gaze import GazeStream
            self.gaze = GazeStream(assert_local_file(gaze_path), cfg["gaze"].get("time_offset_s", 0.0))
        self.out = OutputWriter(out_dir, with_gaze=self.gaze is not None)
        self.qa_enabled = cfg["output"].get("qa_video", True) if qa_video is None else qa_video
        self.qa = None

    # ---------------------------------------------------------------- setup
    def _load_detectors(self, skip_missing):
        from .detectors.faces import FaceDetector
        from .detectors.openvocab import OwlV2Detector
        from .detectors.pose import PoseDetector
        from .detectors.text import TextDetector
        ctors = {
            "faces": FaceDetector,
            "text": TextDetector,
            "pose": PoseDetector,
            "openvocab": lambda c: OwlV2Detector(c, load_query_items(c)),
        }
        for name, ctor in ctors.items():
            if not self.cfg["detectors"][name].get("enabled", True):
                continue
            try:
                m = ctor(self.cfg)
                self.models[name] = m
                self.model_files[name] = m.path
                if name == "openvocab" and m.truncated_prompts:
                    self.warnings.append(f"prompts truncated to 16 tokens: {m.truncated_prompts}")
            except FileNotFoundError as exc:
                if not skip_missing:
                    raise
                msg = f"detector '{name}' disabled: {exc}"
                self.warnings.append(msg)
                self.log("WARNING " + msg)

    def _due(self, name, idx, changed):
        c = self.cache.get(name)
        every = int(self.cfg["detectors"][name].get("every_n_frames", 1))
        return changed or c is None or idx - c["frame"] >= every

    def _reset_temporal(self):
        self.cache.clear()
        self.ptracker.reset()
        self.body_rule.reset()
        self.inter_rule.reset()
        for t in self.trackers.values():
            t.reset()

    @staticmethod
    def _carry(dets):
        out = []
        for d in dets:
            d2 = copy.deepcopy(d)
            d2.source = "carried"
            out.append(d2)
        return out

    # -------------------------------------------------------------- detection
    def _detect(self, idx, t, rect, lay, changed):
        rh, rw = rect.shape[:2]
        vr = self.layout.video_region(lay)
        region_px = [vr[0] * rw, vr[1] * rh, vr[2] * rw, vr[3] * rh]
        dets = []

        for name in ("faces", "text"):
            if name not in self.models:
                continue
            if self._due(name, idx, changed):
                res = self.models[name](rect)
                self.cache[name] = {"frame": idx, "out": res}
                dets += copy.deepcopy(res)
            else:
                dets += self._carry(self.cache[name]["out"])

        persons = []
        if "pose" in self.models:
            if self._due("pose", idx, changed):
                persons = self.ptracker.assign(self.models["pose"](rect))
                res = self.body_rule.update(t, persons) + self.inter_rule.update(persons)
                self.cache["pose"] = {"frame": idx, "out": res, "persons": persons}
                dets += copy.deepcopy(res)
            else:
                persons = self.cache["pose"]["persons"]
                dets += self._carry(self.cache["pose"]["out"])

        if "openvocab" in self.models:
            if self._due("openvocab", idx, changed):
                res = self._openvocab(rect, region_px, persons)
                self.cache["openvocab"] = {"frame": idx, "out": res}
                dets += copy.deepcopy(res)
            else:
                dets += self._carry(self.cache["openvocab"]["out"])
        return dets

    def _openvocab(self, rect, region_px, persons):
        owl = self.models["openvocab"]
        raw = owl(rect)
        out = []
        sal_cfg = self.cfg["aoi"]["salient_object"]
        sal = foc = None
        wrists = SalientObjectRule.wrists_from(persons, self.kp_thr)
        for r in raw:
            it = owl.items[r["item"]]
            g = it["group"]
            if g == "emotional_item":
                out.append(Detection(r["box"], r["score"], "emotional_item", it["name"],
                                     category=it["category"], valence=it["valence"],
                                     source="detected"))
            elif g == "interaction_prompt":
                out.append(Detection(r["box"], r["score"], "physical_interaction", it["name"],
                                     source="openvocab"))
            is_candidate = g == "salient_candidate" or (
                g == "emotional_item" and sal_cfg.get("include_emotional_items_as_candidates"))
            if is_candidate and sal_cfg.get("enabled", True):
                if sal is None:
                    sal, foc = spectral_residual(rect), focus_map(rect)
                feats = self.sal_rule.features(r["box"], region_px, sal, foc, wrists)
                if feats is None:
                    continue
                s = self.sal_rule.score(feats)
                if s >= sal_cfg["threshold"]:
                    out.append(Detection(r["box"], r["score"], "salient_object", it["name"],
                                         category=it["category"] or "", source="detected",
                                         extra={"salience": round(s, 3),
                                                **{k: round(v, 3) for k, v in feats.items()}}))
        return out

    # ------------------------------------------------------------ coordinates
    def _to_raw(self, pts_work):
        pts_work = np.asarray(pts_work, dtype=np.float64).reshape(-1, 2)
        return self.cam.distort_points(pts_work) if self.cam is not None else pts_work

    def _gaze_to_screen(self, gx, gy, state, W, H):
        p = np.array([[gx * W, gy * H]], dtype=np.float64)
        pw = self.cam.undistort_points(p) if self.cam is not None else p
        s = transform_points(state.H, pw)[0]
        return p[0], s

    # --------------------------------------------------------------- main loop
    def run(self, start=None, end=None):
        cfg = self.cfg
        start = int(cfg["video"].get("start_frame", 0) if start is None else start)
        end = cfg["video"].get("end_frame") if end is None else end
        fps, W, H = self.reader.fps, self.reader.width, self.reader.height
        rw, rh = self.locator.rw, self.locator.rh
        thr_change = float(cfg["carry_forward"]["content_change_threshold"])
        keep_other = cfg["detectors"]["text"].get("keep_non_subtitle_text", False)
        margin_deg = float(cfg["gaze"].get("hit_margin_deg", 0.0))
        px_per_deg = W / float(cfg["gaze"].get("scene_hfov_deg", 95))
        t0 = time.time()
        n = 0
        first = True
        for idx, t, raw in self.reader.frames(start, end):
            n += 1
            work = self.cam.undistort_image(raw) if self.cam is not None else raw
            st = self.locator.update(idx, work)

            gaze_samples = ([], [], [])
            if self.gaze is not None:
                if first:
                    self.gaze.skip_until(t - 0.5 / fps)
                gaze_samples = self.gaze.take_until(t + 0.5 / fps)
            first = False

            rect, lay, dets, changed = None, {"ui_state": "unknown", "panel_top": None}, [], False
            scene_polys = []
            if st.visible:
                rect = self.locator.rectify(work, st)
                lay = self.layout.update(rect)
                small = cv2.resize(cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY), (32, 69),
                                   interpolation=cv2.INTER_AREA).astype(np.float32)
                changed = self.prev_small is not None and \
                    float(np.mean(np.abs(small - self.prev_small))) > thr_change
                self.prev_small = small
                if changed:
                    self._reset_temporal()
                dets = self._detect(idx, t, rect, lay, changed)

                kept = []
                for d in dets:
                    bn = [d.box[0] / rw, d.box[1] / rh, d.box[2] / rw, d.box[3] / rh]
                    d.zone = self.layout.zone_of(bn, lay)
                    if d.aoi_class == "subtitle" and d.zone != "video_content":
                        d.aoi_class, d.label = "other_text", f"text ({d.zone})"
                        if not keep_other:
                            continue
                    kept.append(d)
                dets = kept
                for cls in AOI_CLASSES:
                    group = [d for d in dets if d.aoi_class == cls]
                    if not group:
                        continue
                    keys = [d.label for d in group] if cls in ("emotional_item", "salient_object") else None
                    for d, tid in zip(group, self.trackers[cls].update([d.box for d in group], keys)):
                        d.track_id = tid

                for d in dets:
                    poly = self._to_raw(transform_points(st.Hinv, box_corners(d.box)))
                    scene_polys.append((d.aoi_class, poly))
                    self.out.detection({
                        "frame": idx, "time_s": round(t, 4), "ui_state": lay["ui_state"],
                        "aoi_class": d.aoi_class, "label": d.label, "category": d.category,
                        "valence": d.valence, "score": round(d.score, 4), "source": d.source,
                        "track_id": d.track_id, "zone": d.zone,
                        "screen_x1": round(d.box[0] / rw, 4), "screen_y1": round(d.box[1] / rh, 4),
                        "screen_x2": round(d.box[2] / rw, 4), "screen_y2": round(d.box[3] / rh, 4),
                        **{f"scene_{a}{i + 1}": round(float(poly[i][j]), 1)
                           for i in range(4) for j, a in enumerate("xy")},
                        "scene_bbox_x1": round(float(poly[:, 0].min()), 1),
                        "scene_bbox_y1": round(float(poly[:, 1].min()), 1),
                        "scene_bbox_x2": round(float(poly[:, 0].max()), 1),
                        "scene_bbox_y2": round(float(poly[:, 1].max()), 1),
                        "extra": d.extra,
                    })
            else:
                self._reset_temporal()
                self.prev_small = None

            quad_raw = self._to_raw(st.quad) if st.visible else None
            self.out.screen_row({
                "frame": idx, "time_s": round(t, 4), "visible": int(st.visible),
                "method": st.method, "confidence": round(st.confidence, 3),
                **({f"{a}{i + 1}": round(float(quad_raw[i][j]), 1)
                    for i in range(4) for j, a in enumerate("xy")} if st.visible else {}),
                "ui_state": lay["ui_state"],
                "panel_top": "" if lay["panel_top"] is None else round(lay["panel_top"], 4),
                "content_change": int(changed),
            })

            gaze_px = []
            if self.gaze is not None:
                q_top_w = float(np.linalg.norm(st.quad[1] - st.quad[0])) if st.visible else 1.0
                margin = margin_deg * px_per_deg * (rw / max(q_top_w, 1e-6))
                for gt, gx, gy in zip(*gaze_samples):
                    valid = bool(np.isfinite(gx) and np.isfinite(gy))
                    row = {"gaze_time_s": round(float(gt), 4), "frame": idx, "valid": int(valid),
                           "gaze_x_px": "", "gaze_y_px": "", "on_screen": 0,
                           "screen_x": "", "screen_y": ""}
                    hits = []
                    if valid:
                        p_raw, s = (self._gaze_to_screen(gx, gy, st, W, H) if st.visible
                                    else (np.array([gx * W, gy * H]), None))
                        gaze_px.append(p_raw)
                        row.update(gaze_x_px=round(float(p_raw[0]), 1), gaze_y_px=round(float(p_raw[1]), 1))
                        if s is not None and 0 <= s[0] < rw and 0 <= s[1] < rh:
                            row.update(on_screen=1, screen_x=round(s[0] / rw, 4), screen_y=round(s[1] / rh, 4))
                            for d in dets:
                                x1, y1, x2, y2 = d.box
                                if x1 - margin <= s[0] <= x2 + margin and y1 - margin <= s[1] <= y2 + margin:
                                    hits.append({"aoi_class": d.aoi_class, "label": d.label,
                                                 "track_id": d.track_id})
                    self.out.gaze_row(row, hits)

            if self.qa_enabled:
                state = {"quad_raw": quad_raw, "method": st.method}
                info = f"f{idx} t={t:.2f}s screen={st.method} ui={lay['ui_state']} aois={len(dets)}"
                frame_qa = qa_render(
                    raw, scene_polys, rect, dets, state, lay, info, gaze_px,
                    float(cfg["output"].get("qa_scale", 0.5)))
                if self.qa is None:
                    self.qa = VideoWriter(os.path.join(self.out_dir, "qa_overlay.mp4"), fps,
                                          (frame_qa.shape[1], frame_qa.shape[0]))
                self.qa.write(frame_qa)

            if n % 100 == 0:
                el = time.time() - t0
                self.log(f"frame {idx}  ({n / el:.2f} frames/s, {self.out.n_rows} AOI rows)")

        self._finish(n, time.time() - t0)
        return n

    def _finish(self, n, elapsed):
        from .onnx_utils import sha256_file
        gaze_rate = None
        if self.gaze is not None and len(self.gaze.t) > 2:
            gaze_rate = 1.0 / float(np.median(np.diff(self.gaze.t)))
        self.out.write_summary(self.reader.fps, gaze_rate)
        hashes = {}
        if self.cfg["output"].get("hash_models", True):
            for k, p in self.model_files.items():
                hashes[k] = {"path": p, "sha256": sha256_file(p)}
        import onnxruntime
        self.out.write_json("run_metadata.json", {
            "aoi_tool_version": __version__,
            "python": sys.version, "platform": platform.platform(),
            "opencv": cv2.__version__, "onnxruntime": onnxruntime.__version__,
            "numpy": np.__version__,
            "video": {"path": self.reader.path, "fps": self.reader.fps,
                      "width": self.reader.width, "height": self.reader.height,
                      "frame_count": self.reader.frame_count},
            "undistorted": self.cam is not None,
            "frames_processed": n, "seconds": round(elapsed, 1),
            "models": hashes, "warnings": self.warnings, "config": self.cfg,
        })
        self.out.close()
        if self.qa is not None:
            self.qa.release()
        self.reader.release()

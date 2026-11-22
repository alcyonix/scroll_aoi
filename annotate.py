"""Interactive screen-corner annotation (OpenCV window, fully local).

Use it to fix frames where automatic screen detection fails (very dark
content, heavy occlusion by the hand). Saved corners become 'manual' anchors
that override detection and re-seed tracking.

Controls
  left click     add a corner (click TL, TR, BR, BL - order is fixed afterwards)
  right click    remove last corner
  d / a          next / previous frame          w / s   +25 / -25 frames
  e / q          +250 / -250 frames
  space / enter  save the 4 clicked corners for this frame
  u              accept the automatic detection (yellow) for this frame
  x              delete the annotation of this frame
  esc            save file and quit
"""

import csv
import os

import cv2
import numpy as np

from .config import load_config
from .geom import order_quad
from .screen import ScreenLocator, load_seeds
from .video_io import VideoReader


def run_annotator(video_path, seeds_csv, config_path=None, max_width=1280):
    cfg = load_config(config_path)
    reader = VideoReader(video_path)
    seeds = load_seeds(seeds_csv) if os.path.isfile(seeds_csv) else {}
    loc = ScreenLocator(cfg)
    scale = min(1.0, max_width / reader.width)
    clicks = []
    idx = 0
    win = "annotate-screen (esc = save & quit)"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append([x / scale, y / scale])
        elif event == cv2.EVENT_RBUTTONDOWN and clicks:
            clicks.pop()

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    frame, auto, cached_idx = None, None, -1
    while True:
        if cached_idx != idx:
            frame = reader.read_at(idx)
            if frame is None:
                idx = max(0, idx - 1)
                continue
            small_scale = min(1.0, float(cfg["screen"]["detect_width"]) / reader.width)
            g = cv2.cvtColor(cv2.resize(frame, None, fx=small_scale, fy=small_scale), cv2.COLOR_BGR2GRAY)
            q, _ = loc.detect(g)
            auto = None if q is None else q / small_scale
            cached_idx = idx
        view = cv2.resize(frame, None, fx=scale, fy=scale)
        if auto is not None:
            cv2.polylines(view, [(auto * scale).astype(np.int32)], True, (0, 255, 255), 1)
        if idx in seeds:
            cv2.polylines(view, [(seeds[idx] * scale).astype(np.int32)], True, (255, 0, 255), 2)
        for p in clicks:
            cv2.circle(view, (int(p[0] * scale), int(p[1] * scale)), 4, (0, 0, 255), -1)
        cv2.putText(view, f"frame {idx}  seeds={len(seeds)}  clicks={len(clicks)}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(win, view)
        k = cv2.waitKey(30) & 0xFF
        step = {ord("d"): 1, ord("a"): -1, ord("w"): 25, ord("s"): -25, ord("e"): 250, ord("q"): -250}
        if k == 27:
            break
        if k in step:
            idx = max(0, idx + step[k])
            clicks.clear()
        elif k in (13, 32) and len(clicks) == 4:
            seeds[idx] = order_quad(clicks)
            clicks.clear()
        elif k == ord("u") and auto is not None:
            seeds[idx] = order_quad(auto)
        elif k == ord("x"):
            seeds.pop(idx, None)
    cv2.destroyAllWindows()
    reader.release()
    with open(seeds_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame"] + [f"{a}{i}" for i in range(1, 5) for a in "xy"])
        for fr in sorted(seeds):
            w.writerow([fr] + [round(float(v), 1) for v in np.asarray(seeds[fr]).reshape(-1)])
    print(f"saved {len(seeds)} annotated frames to {seeds_csv}")

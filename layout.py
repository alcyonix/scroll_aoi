"""TikTok screen-layout analysis on the rectified screen image.

* Detects whether the comments sheet is open (and where its top edge is), so
  that AOI detection keeps working when the video shrinks to the top of the
  screen and the lower part shows comments.
* Assigns every detection to a UI zone (video_content, comments_panel,
  right_rail, caption_area, top_bar, bottom_nav) so analysts can filter.

Comments-sheet test (all explicit, tune in default.yaml -> layout):
  For each candidate boundary row y, the region below y (down to the input
  bar) must be dominated by one flat background grey level (light mode
  >= bright_bg_min, dark mode <= dark_bg_max), and there must be a horizontal
  edge at y. The state only flips after `open_frames` / `close_frames`
  consecutive agreeing frames (hysteresis).
"""

import cv2
import numpy as np


def _panel_candidate(rect_bgr, c):
    g = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(g, (g.shape[1] // 4, g.shape[0] // 4), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape
    y_lo, y_hi = int(c["panel_search_top"] * sh), int(c["panel_search_bottom"] * sh)
    y_end = int(c["panel_bottom_exclude"] * sh)
    tol = int(c["uniform_tolerance"])
    sm = small.astype(np.int16)
    best_y, best_edge = None, 0.0
    for y in range(max(y_lo, 3), y_hi):
        below = small[y + 1:y_end]
        if below.size == 0:
            continue
        mode = int(np.bincount(below.reshape(-1), minlength=256).argmax())
        if not (mode >= c["bright_bg_min"] or mode <= c["dark_bg_max"]):
            continue
        frac = float(np.mean(np.abs(below.astype(np.int16) - mode) <= tol))
        if frac < c["min_uniform_frac"]:
            continue
        edge = float(np.mean(np.abs(sm[y + 1] - sm[y - 2])))
        above = small[:y]
        # The video area above must not itself be flat panel background.
        above_frac = float(np.mean(np.abs(above.astype(np.int16) - mode) <= tol))
        if above_frac > 0.85:
            continue
        if edge > best_edge:
            best_y, best_edge = y, edge
    if best_y is None or best_edge < c["min_edge_strength"]:
        return None
    return (best_y + 1) / float(sh)


class LayoutAnalyzer:
    def __init__(self, cfg):
        self.c = cfg["layout"]
        self.is_open = False
        self.panel_top = None
        self._streak = 0
        self._recent_tops = []

    def update(self, rect_bgr):
        if not self.c.get("comments_detection", True):
            return {"ui_state": "feed", "panel_top": None}
        cand = _panel_candidate(rect_bgr, self.c)
        agrees = (cand is not None) == self.is_open
        if agrees:
            self._streak = 0
        else:
            self._streak += 1
            need = self.c["open_frames"] if not self.is_open else self.c["close_frames"]
            if self._streak >= need:
                self.is_open = not self.is_open
                self._streak = 0
                self._recent_tops = []
        if self.is_open and cand is not None:
            self._recent_tops = (self._recent_tops + [cand])[-5:]
            self.panel_top = float(np.median(self._recent_tops))
        if not self.is_open:
            self.panel_top = None
        return {"ui_state": "comments_open" if self.is_open else "feed",
                "panel_top": self.panel_top}

    def zone_of(self, box_norm, layout):
        """box_norm: [x1,y1,x2,y2] in 0..1 screen coordinates."""
        cx = (box_norm[0] + box_norm[2]) / 2.0
        cy = (box_norm[1] + box_norm[3]) / 2.0
        if layout["ui_state"] == "comments_open" and layout["panel_top"] is not None:
            if cy >= layout["panel_top"]:
                return "comments_panel"
            zones = self.c.get("zones_comments", {})
        else:
            zones = self.c.get("zones_feed", {})
        for name, (x1, y1, x2, y2) in zones.items():
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return name
        return "video_content"

    def video_region(self, layout):
        """Normalised [x1,y1,x2,y2] of the area currently showing the video."""
        if layout["ui_state"] == "comments_open" and layout["panel_top"] is not None:
            return [0.0, 0.0, 1.0, layout["panel_top"]]
        return [0.0, 0.0, 1.0, 1.0]

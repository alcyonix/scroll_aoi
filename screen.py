"""Locate, track and rectify the phone screen in egocentric scene video.

Pipeline per frame
------------------
1. DETECT  - search a downscaled frame for bright, rectangular, phone-shaped
             regions (multi-threshold luminance masks + an edge mask), fit a
             quadrilateral to each, refine corners with line fits (phone
             screens have rounded corners), and score each candidate by
             rectangularity, aspect ratio, corner angles, size, brightness
             contrast to its surround, and agreement with the previous frame.
2. TRACK   - propagate last frame's quad with sparse optical flow on features
             in a band around the screen border (bezel / edges), using a
             RANSAC homography. Features *inside* the screen are deliberately
             excluded because video content moves independently of the phone.
3. FUSE    - blend detection and tracking, re-acquire on strong detections,
             fall back to tracking for up to `max_track_frames`.
4. MANUAL  - frames listed in a seeds CSV (from `annotate-screen`) override
             everything and re-anchor tracking.
5. RECTIFY - homography from the quad to a fixed portrait canvas
             (rect_width x rect_height). All AOI detectors run on this canvas,
             so AOIs are naturally restricted to the phone screen.
"""

import csv
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .geom import order_quad, polygon_iou, quad_angles_deg, quad_aspect


@dataclass
class ScreenState:
    frame_idx: int
    visible: bool
    quad: Optional[np.ndarray]     # (4,2) TL,TR,BR,BL in working-frame px
    confidence: float
    method: str                    # manual | detected | tracked | lost
    H: Optional[np.ndarray] = None     # working frame -> rectified screen
    Hinv: Optional[np.ndarray] = None  # rectified screen -> working frame


def load_seeds(path):
    """CSV with columns frame,x1,y1,x2,y2,x3,y3,x4,y4 (TL,TR,BR,BL, raw video px)."""
    seeds = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pts = [[float(row[f"x{i}"]), float(row[f"y{i}"])] for i in range(1, 5)]
            seeds[int(row["frame"])] = order_quad(pts)
    return seeds


def _fit_quad(hull):
    peri = cv2.arcLength(hull, True)
    for eps in (0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return order_quad(approx.reshape(4, 2))
    return order_quad(cv2.boxPoints(cv2.minAreaRect(hull)))


def _refine_quad_lines(quad, contour):
    """Fit a line to the contour points along each side and intersect adjacent
    lines. Recovers the 'virtual' sharp corners of a rounded-corner screen."""
    pts = contour.reshape(-1, 2).astype(np.float32)
    lines = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        ab = b - a
        L = float(np.linalg.norm(ab))
        if L < 5:
            return quad
        u = ab / L
        rel = pts - a
        t = rel @ u
        d = np.abs(rel[:, 0] * u[1] - rel[:, 1] * u[0])
        sel = pts[(t > 0.15 * L) & (t < 0.85 * L) & (d < max(3.0, 0.04 * L))]
        if len(sel) < 5:
            return quad
        vx, vy, x0, y0 = cv2.fitLine(sel, cv2.DIST_HUBER, 0, 0.01, 0.01).reshape(-1)
        lines.append((np.array([x0, y0]), np.array([vx, vy])))
    out = []
    for i in range(4):
        (p1, d1), (p2, d2) = lines[i - 1], lines[i]
        A = np.array([d1, -d2]).T
        if abs(np.linalg.det(A)) < 1e-6:
            return quad
        s = np.linalg.solve(A, p2 - p1)
        out.append(p1 + s[0] * d1)
    out = np.array(out, dtype=np.float32)
    # Reject refinements that move a corner implausibly far.
    if np.max(np.linalg.norm(out - quad, axis=1)) > 0.15 * max(np.ptp(quad[:, 0]), np.ptp(quad[:, 1])):
        return quad
    return out


def _snap_inner_edges(gray, quad, learned=None, band_frac=0.08, min_step=25.0, dark_max=70.0):
    """If the quad sits on the phone's OUTER outline, move each side inwards to
    the bezel->screen boundary: the strongest dark-to-bright step within
    `band_frac` of the longest side, measured on an intensity profile averaged
    along the middle 70% of the side.

    Sides with no visible step but sitting on dark pixels (a black TikTok nav
    bar merging with a black bezel, or dark video content) use the inset
    `learned` from earlier frames where that side's step WAS visible.
    `learned` is a list of 4 floats or None (top, right, bottom, left), as a
    fraction of the longest side. Returns (quad, observed_insets)."""
    h, w = gray.shape
    q = quad.astype(np.float64)
    c = q.mean(axis=0)
    lmax = max(np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4))
    band = max(3, int(band_frac * lmax))
    lines, observed = [], [None] * 4
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        d = b - a
        L = np.linalg.norm(d)
        if L < 20:
            return quad, observed
        u = d / L
        n = np.array([-u[1], u[0]])
        if np.dot(c - a, n) < 0:
            n = -n                                   # inward normal
        ts = np.linspace(0.15, 0.85, 30)
        base = a[None, :] + ts[:, None] * d[None, :]
        prof = []
        for off in range(0, band + 2):
            p = base + off * n
            xs = np.clip(np.round(p[:, 0]).astype(int), 0, w - 1)
            ys = np.clip(np.round(p[:, 1]).astype(int), 0, h - 1)
            prof.append(float(gray[ys, xs].mean()))
        prof = np.array(prof)
        grad = prof[2:] - prof[:-2]                   # grad[k] ~ step at offset k+1
        k = int(np.argmax(grad))
        shift = 0.0
        if grad[k] >= min_step and prof[k] <= dark_max:
            shift = float(k + 1)
            observed[i] = shift / lmax
        elif learned is not None and learned[i] is not None and prof[:3].mean() <= dark_max:
            shift = learned[i] * lmax
        lines.append((a + shift * n, u))
    out = []
    for i in range(4):
        (p1, d1), (p2, d2) = lines[i - 1], lines[i]
        A = np.array([d1, -d2]).T
        if abs(np.linalg.det(A)) < 1e-6:
            return quad, observed
        s_ = np.linalg.solve(A, p2 - p1)
        out.append(p1 + s_[0] * d1)
    return np.array(out, dtype=np.float32), observed


class ScreenLocator:
    def __init__(self, cfg, seeds=None):
        s = cfg["screen"]
        self.cfg = s
        self.rw, self.rh = int(s["rect_width"]), int(s["rect_height"])
        self.dst = np.array([[0, 0], [self.rw - 1, 0], [self.rw - 1, self.rh - 1],
                             [0, self.rh - 1]], dtype=np.float32)
        self.seeds = seeds or {}
        self.prev_small = None
        self.prev_quad = None          # full-res working coords
        self.prev_conf = 0.0
        self.frames_since_detect = 10 ** 9
        self.inset_hist = [deque(maxlen=60) for _ in range(4)]   # per side
        self.aspect_hist = deque(maxlen=100)

    # ------------------------------------------------------------------ public
    def update(self, frame_idx, frame_bgr):
        h, w = frame_bgr.shape[:2]
        scale = min(1.0, float(self.cfg.get("detect_width", 960)) / w)
        small = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        tracked = None
        if self.prev_small is not None and self.prev_quad is not None:
            tq = self._track(self.prev_small, gray, self.prev_quad * scale)
            tracked = None if tq is None else tq / scale

        if frame_idx in self.seeds:
            quad, conf, method = self.seeds[frame_idx].astype(np.float32), 1.0, "manual"
            self.frames_since_detect = 0
        else:
            prior = tracked if tracked is not None else (
                self.prev_quad if self.frames_since_detect < self.cfg["max_track_frames"] else None)
            det, det_score = self.detect(gray, None if prior is None else prior * scale)
            if det is not None:
                det = det / scale
            t_score = 0.0
            if tracked is not None:
                tq = tracked * scale
                t_score = self._score(tq, cv2.contourArea(tq.astype(np.float32)), gray, None)
            quad, conf, method = self._fuse(det, det_score, tracked, t_score)

        self.prev_small = gray
        if quad is None:
            self.prev_quad, self.prev_conf = None, 0.0
            return ScreenState(frame_idx, False, None, 0.0, "lost")
        quad = order_quad(quad)
        self.prev_quad, self.prev_conf = quad, conf
        H = cv2.getPerspectiveTransform(quad.astype(np.float32), self.dst)
        return ScreenState(frame_idx, True, quad, float(conf), method, H, np.linalg.inv(H))

    def rectify(self, frame_bgr, state):
        return cv2.warpPerspective(frame_bgr, state.H, (self.rw, self.rh), flags=cv2.INTER_LINEAR)

    # --------------------------------------------------------------- internals
    def _fuse(self, det, det_score, tracked, tracked_score):
        """Combine a fresh detection with the optically-tracked quad.
        A detection that DISAGREES with tracking only wins if it scores clearly
        better than the tracked quad does under the same scoring function."""
        c = self.cfg
        if det is not None and det_score >= c["min_detect_score"]:
            if tracked is not None:
                iou = polygon_iou(det, tracked)
                if iou > 0.75:
                    a = float(c["smoothing_alpha"])
                    self.frames_since_detect = 0
                    return a * det + (1 - a) * tracked, det_score, "detected"
                if det_score >= c["reacquire_score"] and det_score > tracked_score + 0.1:
                    self.frames_since_detect = 0
                    return det, det_score, "detected"
                self.frames_since_detect += 1
                return tracked, max(tracked_score, self.prev_conf * 0.97), "tracked"
            self.frames_since_detect = 0
            return det, det_score, "detected"
        if tracked is not None and self.frames_since_detect < c["max_track_frames"]:
            self.frames_since_detect += 1
            return tracked, self.prev_conf * 0.97, "tracked"
        self.frames_since_detect += 1
        return None, 0.0, "lost"

    def detect(self, gray, prior=None):
        """Return (quad, score) in the coordinates of `gray`, or (None, 0)."""
        h, w = gray.shape
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        k = max(3, (w // 90) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        thresholds = set()
        otsu, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresholds.add(int(otsu))
        for p in (55, 70, 85, 93):
            thresholds.add(int(np.percentile(blur, p)))
        masks = []
        for t in sorted(thresholds):
            m = (blur > t).astype(np.uint8) * 255
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
            masks.append(m)
        edges = cv2.Canny(blur, 30, 90)
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)
        masks.append(edges)

        best, best_score = None, 0.0
        for m in masks:
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                frac = area / float(h * w)
                if frac < self.cfg["min_area_frac"] or frac > self.cfg["max_area_frac"]:
                    continue
                hull = cv2.convexHull(cnt)
                quad = _fit_quad(hull)
                quad = _refine_quad_lines(quad, hull)
                score = self._score(quad, cv2.contourArea(hull), gray, prior)
                if score > best_score:
                    best, best_score = quad, score
        if best is not None:
            learned = [float(np.median(hq)) if len(hq) >= 5 else None for hq in self.inset_hist]
            best, observed = _snap_inner_edges(gray, best, learned)
            for hq, v in zip(self.inset_hist, observed):
                if v is not None:
                    hq.append(v)
            if all(v is not None for v in observed):
                self.aspect_hist.append(quad_aspect(best))
            if self.cfg.get("extend_bottom_to_aspect", False):
                best = self._extend_bottom(gray, best)
        return best, best_score

    def _extend_bottom(self, gray, quad, dark_max=70.0):
        """TikTok's black bottom bar (and black video letterboxing) is
        indistinguishable from a black bezel, so a quad found from the lit area
        can stop short. If the quad is clearly shorter than the screen's aspect
        ratio (learned from frames where all four screen edges were visible,
        else screen.expected_aspect) and the strip below it is dark, extend the
        two bottom corners along the side edges to restore the aspect."""
        target = float(np.median(self.aspect_hist)) if len(self.aspect_hist) >= 5 \
            else float(self.cfg["expected_aspect"])
        a = quad_aspect(quad)
        if a >= target * 0.96:
            return quad
        f = target / a
        q = quad.astype(np.float64).copy()
        new_br = q[1] + (q[2] - q[1]) * f
        new_bl = q[0] + (q[3] - q[0]) * f
        strip = np.array([q[3], q[2], new_br, new_bl], dtype=np.float32)
        mask = np.zeros_like(gray)
        cv2.fillConvexPoly(mask, strip.astype(np.int32), 255)
        if mask.any() and cv2.mean(gray, mask)[0] > dark_max:
            return quad
        q[2], q[3] = new_br, new_bl
        return q.astype(np.float32)

    def _score(self, quad, hull_area, gray, prior):
        h, w = gray.shape
        qa = cv2.contourArea(quad.astype(np.float32))
        if qa <= 1:
            return 0.0
        rect = min(hull_area, qa) / max(hull_area, qa)
        asp = quad_aspect(quad)
        exp_a, tol = float(self.cfg["expected_aspect"]), float(self.cfg["aspect_tolerance"])
        asp_s = float(np.exp(-((asp - exp_a) / tol) ** 2))
        ang_dev = max(abs(a - 90.0) for a in quad_angles_deg(quad))
        ang_s = float(np.exp(-(ang_dev / 20.0) ** 2))
        if rect < 0.6 or asp_s < 0.1 or ang_s < 0.2:
            return 0.0
        frac = qa / float(h * w)
        lo, hi = self.cfg["min_area_frac"], self.cfg["max_area_frac"]
        area_s = 1.0 if 2 * lo <= frac <= 0.7 * hi else 0.5

        mask = np.zeros_like(gray)
        cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)
        ring_k = max(5, int(0.08 * np.sqrt(qa)) | 1)
        ring = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (ring_k, ring_k))) - mask
        m_in = cv2.mean(gray, mask)[0]
        m_out = cv2.mean(gray, ring)[0] if ring.any() else m_in
        con_s = float(np.clip((m_in - m_out) / 50.0, 0.0, 1.0))

        score = 0.25 * rect + 0.20 * asp_s + 0.15 * ang_s + 0.10 * area_s + 0.30 * con_s
        if prior is not None:
            score = 0.75 * score + 0.25 * polygon_iou(quad, prior)
        return float(score)

    def _track(self, prev_gray, gray, prev_quad):
        qa = cv2.contourArea(prev_quad.astype(np.float32))
        if qa < 50:
            return None
        band = max(4, int(0.08 * np.sqrt(qa)))
        mask = np.zeros_like(prev_gray)
        cv2.polylines(mask, [prev_quad.astype(np.int32)], True, 255, thickness=2 * band)
        p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=250, qualityLevel=0.01,
                                     minDistance=4, mask=mask)
        if p0 is None or len(p0) < 8:
            return None
        lk = dict(winSize=(21, 21), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk)
        p0r, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, **lk)
        fb = np.linalg.norm((p0 - p0r).reshape(-1, 2), axis=1)
        good = (st.reshape(-1) == 1) & (st2.reshape(-1) == 1) & (fb < 1.0)
        if good.sum() < 8:
            return None
        Hm, inl = cv2.findHomography(p0[good], p1[good], cv2.RANSAC, 3.0)
        if Hm is None or inl is None or inl.sum() < 8:
            return None
        q = cv2.perspectiveTransform(prev_quad.reshape(-1, 1, 2).astype(np.float64), Hm).reshape(4, 2)
        ratio = cv2.contourArea(q.astype(np.float32)) / qa
        if not (0.6 < ratio < 1.6):
            return None
        return q.astype(np.float32)

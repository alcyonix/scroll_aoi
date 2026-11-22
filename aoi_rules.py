"""Explicit, inspectable rules that turn raw detections into AOIs 2, 3 and 5.

Every threshold lives in config/default.yaml -> aoi.*  These are heuristics:
validate them against a hand-coded subset of your footage before relying on
them (README, "Validating against human coding").
"""

from collections import deque

import numpy as np

from .detectors import Detection
from .geom import box_overlap_min, union_box
from .tracking import IoUTracker

# COCO keypoint indices
NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = \
    0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
UPPER = [L_EL, R_EL, L_WR, R_WR]
LOWER = [L_KN, R_KN, L_AN, R_AN]
TORSO = [L_SH, R_SH, L_HIP, R_HIP]
WRISTS = [L_WR, R_WR]
BODY_TARGETS = [NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP]


class PersonTracker:
    """Shared person ids for the motion and interaction rules."""

    def __init__(self):
        self.t = IoUTracker(iou_thr=0.3, max_missed=3)

    def reset(self):
        self.t.reset()

    def assign(self, persons):
        ids = self.t.update([p["box"] for p in persons])
        for p, i in zip(persons, ids):
            p["pid"] = i
        return persons


# --------------------------------------------------------------------- AOI 2
class BodilyActionRule:
    """A person is performing a bodily action when their limbs move relative to
    their own torso faster than a threshold, measured over a short window.

      * Keypoints are expressed relative to the torso centre and divided by the
        person's box height -> camera pans/zooms in the TikTok video and the
        person walking across frame do NOT count as bodily action.
      * speed(joint) = median over the window of |delta position| / delta t
        (body-heights per second); jumps > 0.5 body-heights in one step are
        ignored as edit cuts / keypoint swaps.
      * label 'whole-body movement (e.g. dancing)' if both legs and arms move,
        'gesturing (arms/hands)' if only arms move, 'leg movement' if only legs.
    """

    def __init__(self, cfg, kp_thr):
        self.c = cfg["aoi"]["bodily_action"]
        self.kp_thr = kp_thr
        self.hist = {}       # pid -> deque[(t, kp_rel (17,2), conf (17,))]
        self.last = {}       # pid -> (t_active, label, speeds)

    def reset(self):
        self.hist.clear()
        self.last.clear()

    def _rel(self, p):
        kp, kc = p["kpts"], p["kconf"]
        x1, y1, x2, y2 = p["box"]
        scale = max(y2 - y1, 1.0)
        ok = [i for i in TORSO if kc[i] >= self.kp_thr]
        center = kp[ok].mean(axis=0) if ok else np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        return (kp - center) / scale

    def _speed(self, hist, joints):
        best = 0.0
        for j in joints:
            v = []
            for (t0, k0, c0), (t1, k1, c1) in zip(list(hist)[:-1], list(hist)[1:]):
                if c0[j] < self.kp_thr or c1[j] < self.kp_thr or t1 <= t0:
                    continue
                d = float(np.linalg.norm(k1[j] - k0[j]))
                if d > 0.5:
                    continue
                v.append(d / (t1 - t0))
            if len(v) >= 2:
                best = max(best, float(np.median(v)))
        return best

    def update(self, t, persons):
        if not self.c.get("enabled", True):
            return []
        out, alive = [], set()
        for p in persons:
            pid = p["pid"]
            alive.add(pid)
            h = self.hist.setdefault(pid, deque())
            h.append((t, self._rel(p), p["kconf"]))
            while h and t - h[0][0] > self.c["window_s"]:
                h.popleft()
            label, up, lo = None, 0.0, 0.0
            if len(h) >= self.c["min_frames"]:
                up, lo = self._speed(h, UPPER), self._speed(h, LOWER)
                up_on = up >= self.c["upper_speed_threshold"]
                lo_on = lo >= self.c["lower_speed_threshold"]
                if up_on and lo_on:
                    label = "whole-body movement (e.g. dancing)"
                elif up_on:
                    label = "gesturing (arms/hands)"
                elif lo_on:
                    label = "leg movement"
            if label:
                self.last[pid] = (t, label, (up, lo))
            prev = self.last.get(pid)
            if prev and t - prev[0] <= self.c["hold_s"]:
                lbl, (u, l) = prev[1], prev[2]
                out.append(Detection(list(p["box"]), float(p["score"]), "bodily_action", lbl,
                                     source="rule",
                                     extra={"person_id": pid, "upper_speed": round(u, 3),
                                            "lower_speed": round(l, 3)}))
        for pid in list(self.hist):
            if pid not in alive:
                del self.hist[pid]
                self.last.pop(pid, None)
        return out


# --------------------------------------------------------------------- AOI 3
class InteractionRule:
    """Physical contact between two detected people, from pose keypoints.

    Distances are divided by the mean box height of the pair. Checked in order:
      face contact (e.g. kiss) : nose-to-nose < face_contact_distance
      handshake                : a wrist of A near a wrist of B (< handshake_
                                 distance) with both wrists between shoulder and
                                 hip height
      hug                      : boxes overlap >= hug_box_overlap (of the smaller
                                 box) AND a wrist of one is near the other's
                                 shoulders/upper body (< 2 x touch_distance)
      touch                    : any wrist of one within touch_distance of the
                                 other's body keypoints
    Reported only after `min_persist_frames` consecutive frames. The AOI box is
    the union of both people's boxes.
    """

    def __init__(self, cfg, kp_thr):
        self.c = cfg["aoi"]["interaction"]
        self.kp_thr = kp_thr
        self.streak = {}

    def reset(self):
        self.streak.clear()

    def _pts(self, p, idx):
        return [(i, p["kpts"][i]) for i in idx if p["kconf"][i] >= self.kp_thr]

    def _min_dist(self, A, B):
        if not A or not B:
            return np.inf
        return min(float(np.linalg.norm(a - b)) for _, a in A for _, b in B)

    def _between_sh_hip(self, p, y):
        kp, kc = p["kpts"], p["kconf"]
        sh = [kp[i][1] for i in (L_SH, R_SH) if kc[i] >= self.kp_thr]
        hp = [kp[i][1] for i in (L_HIP, R_HIP) if kc[i] >= self.kp_thr]
        if not sh or not hp:
            return True
        return min(sh) - 0.1 * (max(hp) - min(sh)) <= y <= max(hp) + 0.2 * (max(hp) - min(sh))

    def classify(self, a, b):
        s = (a["box"][3] - a["box"][1] + b["box"][3] - b["box"][1]) / 2.0
        s = max(s, 1.0)
        c = self.c
        nose_d = self._min_dist(self._pts(a, [NOSE]), self._pts(b, [NOSE])) / s
        if nose_d < c["face_contact_distance"]:
            return "face contact (e.g. kiss)"
        wa, wb = self._pts(a, WRISTS), self._pts(b, WRISTS)
        if wa and wb:
            pairs = [(np.linalg.norm(x - y) / s, x, y) for _, x in wa for _, y in wb]
            d, x, y = min(pairs, key=lambda q: q[0])
            if d < c["handshake_distance"] and self._between_sh_hip(a, x[1]) and self._between_sh_hip(b, y[1]):
                return "handshake"
        upper_a = self._pts(a, [L_SH, R_SH, L_EL, R_EL])
        upper_b = self._pts(b, [L_SH, R_SH, L_EL, R_EL])
        overlap = box_overlap_min(a["box"], b["box"])
        near_upper = min(self._min_dist(wa, upper_b), self._min_dist(wb, upper_a)) / s
        if overlap >= c["hug_box_overlap"] and near_upper < 2 * c["touch_distance"]:
            return "hug"
        touch = min(self._min_dist(wa, self._pts(b, BODY_TARGETS)),
                    self._min_dist(wb, self._pts(a, BODY_TARGETS))) / s
        if touch < c["touch_distance"]:
            return "touch"
        return None

    def update(self, persons):
        if not self.c.get("enabled", True):
            return []
        out, seen = [], set()
        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                a, b = persons[i], persons[j]
                key = tuple(sorted((a["pid"], b["pid"])))
                label = self.classify(a, b)
                if label is None:
                    continue
                seen.add(key)
                n = self.streak.get(key, (0, None))[0] + 1
                self.streak[key] = (n, label)
                if n >= self.c["min_persist_frames"]:
                    out.append(Detection(union_box(a["box"], b["box"]),
                                         float(min(a["score"], b["score"])),
                                         "physical_interaction", label, source="rule",
                                         extra={"person_ids": list(key)}))
        for k in list(self.streak):
            if k not in seen:
                del self.streak[k]
        return out


# --------------------------------------------------------------------- AOI 5
class SalientObjectRule:
    """Score how strongly a candidate object is made salient. Features (0..1):
      saliency   : mean spectral-residual saliency in box / mean in video area
                   (ratio 1 -> 0, ratio >= 2.5 -> 1)
      size       : box area / video area / size_full_score_frac (capped at 1)
      centrality : 1 - distance(box centre, video centre) / half-diagonal
      focus      : mean Laplacian in box / mean outside box in the video area
                   (ratio 1 -> 0, ratio >= 3 -> 1)
      in_hand    : 1 if a detected wrist keypoint lies in the box (+10% margin)
    score = weighted mean (weights in config). Reported if score >= threshold.
    """

    def __init__(self, cfg, kp_thr):
        self.c = cfg["aoi"]["salient_object"]
        self.kp_thr = kp_thr

    def features(self, box, region, sal, foc, wrists):
        rx1, ry1, rx2, ry2 = [int(round(v)) for v in region]
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, y1 = max(x1, rx1), max(y1, ry1)
        x2, y2 = min(x2, rx2), min(y2, ry2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None
        reg_area = float((rx2 - rx1) * (ry2 - ry1))
        area = float((x2 - x1) * (y2 - y1))
        s_in = float(sal[y1:y2, x1:x2].mean())
        s_all = float(sal[ry1:ry2, rx1:rx2].mean()) + 1e-6
        f_in = float(foc[y1:y2, x1:x2].mean())
        f_sum = float(foc[ry1:ry2, rx1:rx2].sum())
        out_area = max(reg_area - area, 1.0)
        f_out = max((f_sum - foc[y1:y2, x1:x2].sum()) / out_area, 1e-6)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        rcx, rcy = (rx1 + rx2) / 2.0, (ry1 + ry2) / 2.0
        half_diag = 0.5 * np.hypot(rx2 - rx1, ry2 - ry1)
        mx, my = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
        in_hand = any(x1 - mx <= wx <= x2 + mx and y1 - my <= wy <= y2 + my for wx, wy in wrists)
        return {
            "saliency": float(np.clip((s_in / s_all - 1.0) / 1.5, 0, 1)),
            "size": float(np.clip(area / reg_area / self.c["size_full_score_frac"], 0, 1)),
            "centrality": float(np.clip(1.0 - np.hypot(cx - rcx, cy - rcy) / half_diag, 0, 1)),
            "focus": float(np.clip((f_in / f_out - 1.0) / 2.0, 0, 1)),
            "in_hand": 1.0 if in_hand else 0.0,
        }

    def score(self, feats):
        w = self.c["weights"]
        tot = sum(w.values())
        return sum(w[k] * feats[k] for k in w) / tot if tot else 0.0

    @staticmethod
    def wrists_from(persons, kp_thr):
        pts = []
        for p in persons:
            for i in WRISTS:
                if p["kconf"][i] >= kp_thr:
                    pts.append((float(p["kpts"][i][0]), float(p["kpts"][i][1])))
        return pts

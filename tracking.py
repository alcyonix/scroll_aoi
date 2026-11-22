"""Greedy IoU tracker giving AOIs stable ids across frames (in screen coords)."""

import itertools

from .geom import box_iou

_ID_SOURCE = itertools.count(1)


class IoUTracker:
    def __init__(self, iou_thr=0.3, max_missed=5):
        self.iou_thr = iou_thr
        self.max_missed = max_missed
        self.tracks = {}   # id -> {"box":..., "key":..., "missed": int}

    def reset(self):
        self.tracks = {}

    def update(self, boxes, keys=None):
        keys = keys or [None] * len(boxes)
        pairs = []
        for tid, t in self.tracks.items():
            for j, b in enumerate(boxes):
                if t["key"] != keys[j]:
                    continue
                iou = box_iou(t["box"], b)
                if iou >= self.iou_thr:
                    pairs.append((iou, tid, j))
        pairs.sort(reverse=True)
        ids = [None] * len(boxes)
        used = set()
        for _, tid, j in pairs:
            if tid in used or ids[j] is not None:
                continue
            ids[j] = tid
            used.add(tid)
        for j, b in enumerate(boxes):
            if ids[j] is None:
                ids[j] = next(_ID_SOURCE)
                self.tracks[ids[j]] = {"box": b, "key": keys[j], "missed": 0}
            else:
                self.tracks[ids[j]].update(box=b, missed=0)
        for tid in list(self.tracks):
            if tid not in ids:
                self.tracks[tid]["missed"] += 1
                if self.tracks[tid]["missed"] > self.max_missed:
                    del self.tracks[tid]
        return ids

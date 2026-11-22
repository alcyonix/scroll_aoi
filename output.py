"""CSV / JSON outputs, all written to the local output folder only."""

import csv
import json
import os

DET_FIELDS = [
    "frame", "time_s", "ui_state", "aoi_class", "label", "category", "valence",
    "score", "source", "track_id", "zone",
    "screen_x1", "screen_y1", "screen_x2", "screen_y2",
    "scene_x1", "scene_y1", "scene_x2", "scene_y2",
    "scene_x3", "scene_y3", "scene_x4", "scene_y4",
    "scene_bbox_x1", "scene_bbox_y1", "scene_bbox_x2", "scene_bbox_y2",
    "extra",
]
SCREEN_FIELDS = [
    "frame", "time_s", "visible", "method", "confidence",
    "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4",
    "ui_state", "panel_top", "content_change",
]
GAZE_FIELDS = [
    "gaze_time_s", "frame", "gaze_x_px", "gaze_y_px", "valid", "on_screen",
    "screen_x", "screen_y", "aoi_hits",
]


class OutputWriter:
    def __init__(self, out_dir, with_gaze):
        os.makedirs(out_dir, exist_ok=True)
        self.dir = out_dir
        self._files = []
        self.det = self._open("detections.csv", DET_FIELDS)
        self.screen = self._open("screen_track.csv", SCREEN_FIELDS)
        self.gaze = self._open("gaze_aoi_hits.csv", GAZE_FIELDS) if with_gaze else None
        self.summary = {}      # (aoi_class, label) -> {"frames": set, "gaze": int}
        self.n_rows = 0

    def _open(self, name, fields):
        f = open(os.path.join(self.dir, name), "w", newline="", encoding="utf-8")
        self._files.append(f)
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        return w

    def detection(self, row):
        row = dict(row)
        row["extra"] = json.dumps(row.get("extra") or {}, separators=(",", ":"))
        self.det.writerow(row)
        key = (row["aoi_class"], row["label"])
        self.summary.setdefault(key, {"frames": set(), "gaze": 0})["frames"].add(row["frame"])
        self.n_rows += 1

    def screen_row(self, row):
        self.screen.writerow(row)

    def gaze_row(self, row, hits):
        row = dict(row)
        row["aoi_hits"] = json.dumps(hits, separators=(",", ":"))
        self.gaze.writerow(row)
        for h in hits:
            key = (h["aoi_class"], h["label"])
            self.summary.setdefault(key, {"frames": set(), "gaze": 0})["gaze"] += 1

    def write_summary(self, fps, gaze_rate_hz=None):
        path = os.path.join(self.dir, "aoi_summary.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["aoi_class", "label", "frames_present", "seconds_present",
                        "gaze_samples_on_aoi", "approx_gaze_seconds"])
            for (cls, label), v in sorted(self.summary.items()):
                n = len(v["frames"])
                gs = v["gaze"]
                w.writerow([cls, label, n, round(n / fps, 3), gs,
                            round(gs / gaze_rate_hz, 3) if gaze_rate_hz else ""])

    def write_json(self, name, obj):
        with open(os.path.join(self.dir, name), "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str)

    def close(self):
        for f in self._files:
            f.close()

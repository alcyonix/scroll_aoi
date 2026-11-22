"""Optional gaze input, so each gaze sample can be mapped onto the screen AOIs.

Supported inputs (local files only):
  * Tobii Pro Glasses 3 `gazedata.gz` (or the decompressed `gazedata`):
    newline-delimited JSON, one object per sample, e.g.
      {"type":"gaze","timestamp":681.751,"data":{"gaze2d":[0.535,0.477], ...}}
    gaze2d is in normalised scene-video coordinates ((0,0) top-left,
    (1,1) bottom-right); timestamp is seconds from recording start. Samples
    without a gaze estimate have an empty "data" object.
  * A simple CSV with header `timestamp_s,x_norm,y_norm` (use this for Glasses 2,
    Glasses X or Tobii Pro Lab exports after converting columns).
"""

import csv
import gzip
import json

import numpy as np


def load_gaze(path):
    t, x, y = [], [], []
    low = path.lower()
    if low.endswith(".csv") or low.endswith(".tsv"):
        delim = "\t" if low.endswith(".tsv") else ","
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter=delim):
                t.append(float(row["timestamp_s"]))
                x.append(float(row["x_norm"]) if row["x_norm"] not in ("", "nan") else np.nan)
                y.append(float(row["y_norm"]) if row["y_norm"] not in ("", "nan") else np.nan)
    else:
        opener = gzip.open if low.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("type", "gaze") != "gaze":
                    continue
                g2 = (obj.get("data") or {}).get("gaze2d")
                t.append(float(obj["timestamp"]))
                if g2 and len(g2) == 2:
                    x.append(float(g2[0]))
                    y.append(float(g2[1]))
                else:
                    x.append(np.nan)
                    y.append(np.nan)
    order = np.argsort(t)
    return np.asarray(t)[order], np.asarray(x)[order], np.asarray(y)[order]


class GazeStream:
    """Hands out the gaze samples belonging to each video frame, in order.
    A sample belongs to the frame whose timestamp is nearest to it."""

    def __init__(self, path, offset_s=0.0):
        self.t, self.x, self.y = load_gaze(path)
        self.t = self.t + float(offset_s)
        self.i = 0

    def take_until(self, t_limit):
        j = int(np.searchsorted(self.t, t_limit, side="right"))
        j = max(j, self.i)
        sl = slice(self.i, j)
        self.i = j
        return self.t[sl], self.x[sl], self.y[sl]

    def skip_until(self, t_limit):
        self.take_until(t_limit)

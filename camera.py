"""Optional lens undistortion using the scene-camera calibration that Tobii Pro
Glasses 3 store in `recording.g3` (JSON):

    recording.g3 -> "scenecamera" -> "camera-calibration" ->
        "focal-length" [fx, fy], "principal-point" [cx, cy], "skew",
        "radial-distortion" [k1, k2, k3], "tangential-distortion" [p1, p2],
        "resolution" [w, h]

(Key names cross-checked against dcnieho/glassesTools importing/tobii_G3.py.)
The Glasses 3 scene camera is wide-angle (106 deg diagonal), so straight
phone edges are slightly curved in the raw video; undistorting first makes the
screen a true quadrilateral and the homography exact.
"""

import json
import os

import cv2
import numpy as np


class Camera:
    def __init__(self, K, dist, size):
        self.K = np.asarray(K, dtype=np.float64)
        self.dist = np.asarray(dist, dtype=np.float64).reshape(-1)
        self.size = tuple(int(v) for v in size)
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.K, self.dist, None, self.K, self.size, cv2.CV_16SC2)

    @classmethod
    def from_recording_g3(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            info = json.load(f)
        cal = info["scenecamera"]["camera-calibration"]
        fx, fy = cal["focal-length"]
        cx, cy = cal["principal-point"]
        skew = float(cal.get("skew", 0.0))
        k = list(cal.get("radial-distortion", [0, 0, 0])) + [0, 0, 0]
        p = list(cal.get("tangential-distortion", [0, 0])) + [0, 0]
        K = [[fx, skew, cx], [0, fy, cy], [0, 0, 1]]
        dist = [k[0], k[1], p[0], p[1], k[2]]   # OpenCV order
        return cls(K, dist, cal["resolution"])

    def undistort_image(self, img):
        return cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR)

    def undistort_points(self, pts):
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.undistortPoints(pts, self.K, self.dist, P=self.K).reshape(-1, 2)

    def distort_points(self, pts):
        """Inverse of undistort_points: undistorted pixels -> raw video pixels."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        ones = np.ones((len(pts), 1))
        norm = (np.linalg.inv(self.K) @ np.hstack([pts, ones]).T).T
        obj = np.hstack([norm[:, :2], np.ones((len(pts), 1))]).reshape(-1, 1, 3)
        out, _ = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), self.K, self.dist)
        return out.reshape(-1, 2)


def find_calibration(video_path, cfg):
    mode = str(cfg["video"].get("undistort", "auto")).lower()
    if mode == "off":
        return None
    explicit = cfg["video"].get("calibration_file")
    candidate = explicit or os.path.join(os.path.dirname(os.path.abspath(video_path)), "recording.g3")
    if os.path.isfile(candidate):
        try:
            return Camera.from_recording_g3(candidate)
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            if mode == "on" or explicit:
                raise ValueError(f"Could not read calibration from {candidate}: {exc}")
            return None
    if mode == "on":
        raise FileNotFoundError(f"undistort=on but no recording.g3 found at {candidate}")
    return None

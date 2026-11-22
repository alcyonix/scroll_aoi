"""Local video reading/writing. Refuses anything that is not a local file."""

import os
import re

import cv2

_URL_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")


def assert_local_file(path):
    """OpenCV's FFmpeg backend can open network URLs (http://, rtsp://, ...)
    from native code, which the Python network guard cannot see. So we only
    ever hand OpenCV a path to an existing local file."""
    p = str(path)
    if _URL_RE.match(p) or p.startswith("\\\\"):
        raise ValueError(f"Refusing non-local path: {p}")
    if not os.path.isfile(p):
        raise FileNotFoundError(p)
    return os.path.abspath(p)


class VideoReader:
    def __init__(self, path):
        self.path = assert_local_file(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise IOError(f"Could not open video: {self.path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 25.0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def seek(self, frame_idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))

    def frames(self, start=0, end=None):
        """Yield (frame_idx, time_s, frame_bgr). time_s comes from the container
        timestamps when available, else frame_idx / fps."""
        if start:
            self.seek(start)
        idx = int(start)
        while end is None or idx < end:
            ok, frame = self.cap.read()
            if not ok:
                break
            t_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            t = (t_ms / 1000.0) if t_ms and t_ms > 0 else idx / self.fps
            if idx == 0:
                t = 0.0
            yield idx, t, frame
            idx += 1

    def read_at(self, frame_idx):
        self.seek(frame_idx)
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self):
        self.cap.release()


class VideoWriter:
    def __init__(self, path, fps, size):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not self.writer.isOpened():
            raise IOError(f"Could not open video writer: {path}")
        self.size = size

    def write(self, frame):
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size)
        self.writer.write(frame)

    def release(self):
        self.writer.release()

"""Small geometry helpers (pure numpy / OpenCV, no I/O)."""

import cv2
import numpy as np


def order_quad(pts):
    """Order 4 points as TL, TR, BR, BL (clockwise in image coordinates,
    starting from the point nearest the top-left). Robust to moderate rotation."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]                 # clockwise (y axis points down)
    start = int(np.argmin(pts.sum(axis=1)))    # smallest x+y = top-left
    return np.roll(pts, -start, axis=0)


def quad_side_lengths(q):
    q = np.asarray(q, dtype=np.float32)
    return [float(np.linalg.norm(q[(i + 1) % 4] - q[i])) for i in range(4)]


def quad_aspect(q):
    """height / width of a TL,TR,BR,BL quad (averaging opposite sides)."""
    top, right, bottom, left = quad_side_lengths(q)
    w = (top + bottom) / 2.0
    h = (left + right) / 2.0
    return h / max(w, 1e-6)


def quad_angles_deg(q):
    q = np.asarray(q, dtype=np.float32)
    out = []
    for i in range(4):
        a, b, c = q[i - 1], q[i], q[(i + 1) % 4]
        v1, v2 = a - b, c - b
        cosv = np.dot(v1, v2) / max(np.linalg.norm(v1) * np.linalg.norm(v2), 1e-6)
        out.append(float(np.degrees(np.arccos(np.clip(cosv, -1, 1)))))
    return out


def polygon_iou(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1, 1, 2)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 1, 2)
    inter, _ = cv2.intersectConvexConvex(a, b)
    union = cv2.contourArea(a) + cv2.contourArea(b) - inter
    return float(inter / union) if union > 0 else 0.0


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def box_overlap_min(a, b):
    """Intersection / area of the smaller box."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / smaller if smaller > 0 else 0.0


def nms(boxes, scores, iou_thr):
    """Greedy non-maximum suppression. boxes: (N,4) xyxy. Returns kept indices."""
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / np.maximum(areas[i] + areas[order[1:]] - inter, 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


def transform_points(H, pts):
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def box_corners(box):
    x1, y1, x2, y2 = box
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)


def clip_box(box, w, h):
    x1, y1, x2, y2 = box
    return [float(np.clip(x1, 0, w)), float(np.clip(y1, 0, h)),
            float(np.clip(x2, 0, w)), float(np.clip(y2, 0, h))]


def union_box(a, b):
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]

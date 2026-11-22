"""Quality-assurance overlay video: scene view (left) + rectified screen (right)."""

import cv2
import numpy as np

COLORS = {  # BGR
    "face": (0, 200, 255),
    "bodily_action": (255, 120, 0),
    "physical_interaction": (255, 0, 200),
    "subtitle": (0, 255, 0),
    "salient_object": (0, 0, 255),
    "emotional_item": (200, 200, 0),
    "other_text": (128, 128, 128),
}
SCREEN_COLORS = {"manual": (255, 0, 255), "detected": (0, 255, 0),
                 "tracked": (0, 255, 255), "lost": (0, 0, 255)}


def render(raw_frame, scene_polys, rect_img, dets, state, layout, info, gaze_pts, scale):
    h, w = raw_frame.shape[:2]
    left = cv2.resize(raw_frame, (int(w * scale), int(h * scale)))
    if state is not None and state.get("quad_raw") is not None:
        q = (np.asarray(state["quad_raw"]) * scale).astype(np.int32)
        cv2.polylines(left, [q], True, SCREEN_COLORS.get(state["method"], (255, 255, 255)), 2)
    for cls, poly in scene_polys:
        cv2.polylines(left, [(np.asarray(poly) * scale).astype(np.int32)], True,
                      COLORS.get(cls, (255, 255, 255)), 1)
    for gx, gy in gaze_pts:
        if np.isfinite(gx) and np.isfinite(gy):
            cv2.circle(left, (int(gx * scale), int(gy * scale)), 8, (0, 0, 255), 2)

    lh = left.shape[0]
    if rect_img is not None:
        r = rect_img.copy()
        for d in dets:
            x1, y1, x2, y2 = [int(v) for v in d.box]
            col = COLORS.get(d.aoi_class, (255, 255, 255))
            cv2.rectangle(r, (x1, y1), (x2, y2), col, 2 if d.source != "carried" else 1)
            txt = f"{d.aoi_class[:4]}:{d.label[:18]}"
            cv2.putText(r, txt, (x1 + 2, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
        if layout and layout.get("panel_top"):
            y = int(layout["panel_top"] * r.shape[0])
            cv2.line(r, (0, y), (r.shape[1], y), (255, 0, 255), 2)
        right = cv2.resize(r, (int(r.shape[1] * lh / r.shape[0]), lh))
    else:
        right = np.zeros((lh, int(lh * 0.46), 3), dtype=np.uint8)
        cv2.putText(right, "screen not found", (5, lh // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 1)
    out = np.hstack([left, right])
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, info, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out

"""Model-free visual-saliency and focus maps used to decide whether an object is
'made salient'. No learned weights, so nothing to download or audit.

* Spectral residual saliency - Hou & Zhang (2007), "Saliency detection: a
  spectral residual approach", CVPR. Implemented from the paper's equations:
    A = log|F(I)|,  P = angle(F(I)),  R = A - mean_filter_3x3(A),
    S = gaussian( |F^-1(exp(R + iP))|^2 )
* Focus map - absolute Laplacian response (sharp, in-focus regions respond
  strongly; blurred backgrounds of product shots respond weakly).
"""

import cv2
import numpy as np


def spectral_residual(img_bgr, width=64):
    h, w = img_bgr.shape[:2]
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sh = max(8, int(round(width * h / float(w))))
    g = cv2.resize(g, (width, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
    F = np.fft.fft2(g)
    A = np.log(np.abs(F) + 1e-8)
    P = np.angle(F)
    R = A - cv2.blur(A, (3, 3))
    S = np.abs(np.fft.ifft2(np.exp(R + 1j * P))) ** 2
    S = cv2.GaussianBlur(S.astype(np.float32), (0, 0), 2.5)
    S -= S.min()
    S /= max(float(S.max()), 1e-8)
    return cv2.resize(S, (w, h), interpolation=cv2.INTER_LINEAR)


def focus_map(img_bgr):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap = np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3))
    return cv2.GaussianBlur(lap, (0, 0), 3)

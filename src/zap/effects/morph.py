"""Two-image face morph: the classic film "morph" between two faces.

Both pictures get the same mesh of control points: face landmarks (jaw line,
brows, eyes, nose, lips), two rings around the head so hair and neck travel
with the face, and fixed points on the frame border. For a morph amount ``t``
every point moves to ``(1 - t) * a + t * b``; each picture is warped
piecewise-affinely (one affine map per Delaunay triangle) onto that in-between
shape, and the two warped pictures are cross-dissolved. Eyes slide onto eyes
and mouths onto mouths, so the face itself transforms instead of one picture
fading over the other.

The per-pixel source maps are evaluated on a 1/4-size grid (the warp is
piecewise linear, so bilinear upsampling of the maps is nearly exact) and
applied with one ``cv2.remap`` per picture: ~8-12 ms at 1280x720.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial import Delaunay, QhullError

# Face landmarks used as control points (MediaPipe 478-point mesh indices).
FACE_OVAL = (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
             152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109)
INNER = (
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300,  # brows
    33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,  # eyes
    168, 6, 197, 1, 98, 327, 2,  # nose
    61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91, 13, 14,  # lips
)
RINGS = (1.3, 1.75)  # head rings, as multiples of the face oval about its centre
GRID = 8  # the warp maps are computed on a grid this many times smaller (exact inside triangles)
BORDER_N = 5  # border points per edge (corners included)
MARGIN = 2.0  # control points must stay this far inside the frame


def control_points(lm: np.ndarray) -> np.ndarray:
    """Face control points for one face, (N, 2): landmarks then the head rings."""
    p = np.asarray(lm, np.float32)[:, :2]
    oval = p[list(FACE_OVAL)]
    c = oval.mean(axis=0)
    rings = [c + (oval[::2] - c) * k for k in RINGS]
    return np.concatenate([oval, p[list(INNER)], *rings]).astype(np.float32)


def _border(W: int, H: int) -> np.ndarray:
    xs = np.linspace(0, W - 1, BORDER_N, dtype=np.float32)
    ys = np.linspace(0, H - 1, BORDER_N, dtype=np.float32)[1:-1]
    pts = [(x, 0) for x in xs] + [(x, H - 1) for x in xs] + [(0, y) for y in ys] + [(W - 1, y) for y in ys]
    return np.float32(pts)


class Morph:
    """Reusable morph renderer for one output size."""

    def __init__(self, W: int, H: int) -> None:
        self.W, self.H = W, H
        self.border = _border(W, H)
        gw, gh = -(-W // GRID), -(-H // GRID)
        sx, sy = W / gw, H / gh
        gx = (np.arange(gw, dtype=np.float32) + 0.5) * sx - 0.5
        gy = (np.arange(gh, dtype=np.float32) + 0.5) * sy - 0.5
        self.gx, self.gy = np.meshgrid(gx, gy)
        self.gsize = (gw, gh)
        self.label = np.zeros((gh, gw), np.int32)
        self.wa = np.empty((H, W, 3), np.uint8)
        self.wb = np.empty((H, W, 3), np.uint8)

    def render(self, a: np.ndarray, pa: np.ndarray, b: np.ndarray, pb: np.ndarray, t: float,
               out: np.ndarray, alpha: float | None = None) -> None:
        """Morph picture ``a`` (control points ``pa``) toward ``b`` (``pb``) by ``t`` into ``out``.

        ``pa``/``pb`` come from :func:`control_points`; ``alpha`` is the colour
        cross-dissolve (defaults to ``t``).
        """
        W, H = self.W, self.H
        inside = ((pa[:, 0] > MARGIN) & (pa[:, 0] < W - 1 - MARGIN) & (pa[:, 1] > MARGIN) & (pa[:, 1] < H - 1 - MARGIN)
                  & (pb[:, 0] > MARGIN) & (pb[:, 0] < W - 1 - MARGIN) & (pb[:, 1] > MARGIN)
                  & (pb[:, 1] < H - 1 - MARGIN))
        A = np.concatenate([pa[inside], self.border])
        B = np.concatenate([pb[inside], self.border])
        P = (1.0 - t) * A + t * B
        try:
            tris = Delaunay(P).simplices
        except (QhullError, ValueError):  # degenerate (all points collinear): plain cross-dissolve
            cv2.addWeighted(a, 1.0 - (t if alpha is None else alpha), b, t if alpha is None else alpha, 0, dst=out)
            return
        self._rasterize(P, tris)
        ma, mb = self._maps(np.concatenate([_affines(P, A, tris), _affines(P, B, tris)], axis=1))
        cv2.remap(a, ma, None, cv2.INTER_LINEAR, dst=self.wa, borderMode=cv2.BORDER_REPLICATE)
        cv2.remap(b, mb, None, cv2.INTER_LINEAR, dst=self.wb, borderMode=cv2.BORDER_REPLICATE)
        k = t if alpha is None else alpha
        cv2.addWeighted(self.wa, 1.0 - k, self.wb, k, 0, dst=out)

    def _rasterize(self, P: np.ndarray, tris: np.ndarray) -> None:
        """Triangle index of every grid cell (in-between shape)."""
        self.label.fill(0)
        g = (P * np.float32([self.gsize[0] / self.W, self.gsize[1] / self.H]) * 16).astype(np.int32)
        for i, tri in enumerate(tris):
            cv2.fillConvexPoly(self.label, g[tri], i, lineType=cv2.LINE_8, shift=4)

    def _maps(self, m: np.ndarray) -> list[np.ndarray]:
        """Full-size 2-channel source maps, one per 2x3 affine stacked in ``m`` (n, k*2, 3), per grid cell."""
        c_all = m.reshape(len(m), -1)[self.label.astype(np.intp)]  # (gh, gw, k*6): one gather
        gx, gy, size = self.gx, self.gy, (self.W, self.H)
        out = []
        for r in range(0, c_all.shape[-1], 6):
            c = [c_all[..., r + i] for i in range(6)]
            xy = cv2.merge([c[0] * gx + c[1] * gy + c[2], c[3] * gx + c[4] * gy + c[5]])
            out.append(cv2.resize(xy, size, interpolation=cv2.INTER_LINEAR))
        return out


def _affines(dst: np.ndarray, src: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """(n, 2, 3) affine maps taking each triangle of ``dst`` onto the same triangle of ``src``."""
    d = dst[tris].astype(np.float64)  # (n, 3, 2)
    s = src[tris].astype(np.float64)
    dh = np.concatenate([d, np.ones((len(tris), 3, 1))], axis=2)  # (n, 3, 3)
    det = np.linalg.det(dh)
    ok = np.abs(det) > 1e-6
    out = np.zeros((len(tris), 2, 3))
    out[:, 0, 0] = out[:, 1, 1] = 1.0  # degenerate triangles: identity (they cover no cells anyway)
    if ok.any():
        # M @ dh.T = s.T  ->  M = s.T @ inv(dh.T)
        out[ok] = np.einsum("nij,njk->nik", s[ok].transpose(0, 2, 1), np.linalg.inv(dh[ok].transpose(0, 2, 1)))
    return out.astype(np.float32)

import numpy as np

from kukkiri import vr
from kukkiri.detector import Detection
from kukkiri.geometry import clip_window, frame_window


def test_fisheye_round_trip():
    n = 2048
    rng = np.random.default_rng(0)
    x, y = rng.uniform(0, n, 500), rng.uniform(0, n, 500)
    u, v = vr.equirect_to_fisheye(x, y, n)
    x2, y2 = vr.fisheye_to_equirect(u, v, n)
    assert np.abs(x2 - x).max() < 1e-6 and np.abs(y2 - y).max() < 1e-6
    assert np.allclose(vr.equirect_to_fisheye(np.array([n / 2]), np.array([n / 2]), n), [[n / 2], [n / 2]])
    assert np.allclose(vr.equirect_to_fisheye(np.array([0.0]), np.array([n / 2]), n), [[0.0], [n / 2]])


class _Eyes:

    def detect_frames(self, frames, batch_size=32):
        out = []
        for f in frames:
            h, w = f.shape[:2]
            m = np.zeros((4, 4), bool)
            m[1:3, 1:3] = True
            out.append([Detection((w // 4, h // 4, 3 * w // 4, 3 * h // 4), 0.5, 0, m)])
        return out


def test_detect_eyes_books_right_eye_in_frame_coordinates():
    frames = [np.zeros((100, 200, 3), np.uint8)]
    (dets,) = vr.detect_eyes(_Eyes(), frames)
    left, right = sorted(dets, key=lambda d: d.box[0])
    assert left.box == (25, 25, 75, 75) and right.box == (125, 25, 175, 75)
    assert left.mask.shape == right.mask.shape == (4, 8)
    assert left.mask[:, :4].any() and not left.mask[:, 4:].any()
    assert right.mask[:, 4:].any() and not right.mask[:, :4].any()


class _Track:
    def __init__(self, boxes, bounds=None):
        self.boxes, self.start = boxes, min(boxes)
        if bounds:
            self.bounds = bounds


def test_windows_stay_in_their_eye():
    hw = (2048, 4096)
    near_seam = {0: (1950, 1000, 2040, 1100)}
    for tr in (_Track(near_seam, (0, 2048)),):
        assert clip_window(tr, [0], hw)[2] <= 2048
        assert frame_window(tr, 0, hw, 512)[2] <= 2048
    right = {0: (2060, 1000, 2150, 1100)}
    tr = _Track(right, vr.eye_bounds(right[0], 4096))
    assert tr.bounds == (2048, 4096)
    assert clip_window(tr, [0], hw)[0] >= 2048 and frame_window(tr, 0, hw, 512)[0] >= 2048


def test_the_more_confident_view_keeps_a_mosaic():
    m = np.ones((4, 8), bool)
    eye = Detection((100, 100, 200, 200), 0.3, 0, m)
    dets = [eye]
    vr._keep_stronger(dets, Detection((110, 105, 205, 210), 0.6, 0, None))
    assert len(dets) == 1 and dets[0].mask is None
    dets = [Detection((100, 100, 200, 200), 0.7, 0, m)]
    vr._keep_stronger(dets, Detection((110, 105, 205, 210), 0.6, 0, None))
    assert len(dets) == 1 and dets[0].mask is not None
    vr._keep_stronger(dets, Detection((500, 500, 600, 600), 0.2, 0, None))
    assert len(dets) == 2

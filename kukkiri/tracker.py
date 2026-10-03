from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from kukkiri.detector import Detection

Box = tuple[int, int, int, int]

BOX_MERGE_IOU = 0.5
FACE_LABEL = 2
FACE_TRACK_SHARE = 0.5


def is_face_track(tr: "Track") -> bool:
    return tr.face_frames > 0 and tr.face_score >= FACE_TRACK_SHARE * tr.score


def iou(a: Box, b: Box) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if inter == 0:
        return 0.0
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def _boxes_touch(a: Box, b: Box) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def _same_region(a: Detection, b: Detection) -> bool:
    if a.mask is not None and b.mask is not None and a.mask.shape == b.mask.shape:
        return _boxes_touch(a.box, b.box) and bool(np.logical_and(a.mask, b.mask).any())
    return iou(a.box, b.box) > BOX_MERGE_IOU


def merge_regions(dets: list[Detection]) -> list[Detection]:
    n = len(dets)
    if n < 2:
        return list(dets)
    parent = list(range(n))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if _same_region(dets[i], dets[j]):
                parent[root(i)] = root(j)
    groups: dict[int, list[Detection]] = {}
    for i, d in enumerate(dets):
        groups.setdefault(root(i), []).append(d)
    out = []
    for g in groups.values():
        if len(g) == 1:
            out.append(g[0])
            continue
        box = (min(d.box[0] for d in g), min(d.box[1] for d in g), max(d.box[2] for d in g), max(d.box[3] for d in g))
        masks = [d.mask for d in g if d.mask is not None]
        mask = np.logical_or.reduce(masks) if masks and all(m.shape == masks[0].shape for m in masks) else None
        face = max((d.score for d in g if d.label == FACE_LABEL), default=-1.0)
        plain = max((d.score for d in g if d.label != FACE_LABEL), default=-1.0)
        label = FACE_LABEL if face >= plain else next(d.label for d in g if d.label != FACE_LABEL)
        out.append(Detection(box, max(d.score for d in g), label, mask))
    return out


@dataclass
class Track:
    track_id: int
    boxes: dict[int, Box] = field(default_factory=dict)
    masks: dict[int, np.ndarray] = field(default_factory=dict)
    seen: list[int] = field(default_factory=list)
    face_frames: int = 0
    face_score: float = 0.0
    score: float = 0.0

    @property
    def start(self) -> int:
        return min(self.boxes)

    @property
    def end(self) -> int:
        return max(self.boxes)

    def predict(self, fi: int) -> Box:
        last = self.boxes[self.seen[-1]]
        if len(self.seen) < 2:
            return last
        p, c = self.seen[-2], self.seen[-1]
        v = (np.array(last, float) - np.array(self.boxes[p], float)) / (c - p)
        return tuple(int(round(x)) for x in np.array(last, float) + v * (fi - c))

    def add(self, fi: int, det: Detection) -> None:
        if self.seen:
            a = self.seen[-1]
            ba, bb = np.array(self.boxes[a], float), np.array(det.box, float)
            for f in range(a + 1, fi):
                t = (f - a) / (fi - a)
                self.boxes[f] = tuple(int(round(x)) for x in ba + t * (bb - ba))
                m = self.masks.get(a) if t < 0.5 else det.mask
                if m is not None:
                    self.masks[f] = m
        self.boxes[fi] = det.box
        self.face_frames += det.label == FACE_LABEL
        self.face_score += det.score if det.label == FACE_LABEL else 0.0
        self.score += det.score
        if det.mask is not None:
            self.masks[fi] = det.mask
        self.seen.append(fi)


class GappedTracker:

    def __init__(self, iou_threshold: float = 0.3, max_gap: int = 6, min_length: int = 2):
        self.iou_threshold, self.max_gap, self.min_length = iou_threshold, max_gap, min_length
        self.active: list[Track] = []
        self.confirmed: dict[int, int] = {}
        self.next_id = 0

    def push(self, fi: int, dets: list[Detection]) -> list[Track]:
        regions = merge_regions(list(dets))
        pairs = sorted(
            ((iou(tr.predict(fi), r.box), ti, ri) for ti, tr in enumerate(self.active) for ri, r in enumerate(regions)),
            reverse=True,
        )
        used_t, used_r = set(), set()
        for overlap, ti, ri in pairs:
            if overlap <= self.iou_threshold:
                break
            if ti in used_t or ri in used_r:
                continue
            used_t.add(ti)
            used_r.add(ri)
            self.active[ti].add(fi, regions[ri])
            self.confirmed[self.active[ti].track_id] = fi
        for ri, r in enumerate(regions):
            if ri not in used_r:
                tr = Track(self.next_id)
                tr.add(fi, r)
                self.confirmed[tr.track_id] = fi
                self.active.append(tr)
                self.next_id += 1
        still, closed = [], []
        for tr in self.active:
            (closed if fi - self.confirmed[tr.track_id] > self.max_gap else still).append(tr)
        self.active = still
        return [t for t in map(self._close, closed) if t is not None]

    def finish(self) -> list[Track]:
        closed, self.active = self.active, []
        return [t for t in map(self._close, closed) if t is not None]

    def _close(self, tr: Track) -> Track | None:
        self.confirmed.pop(tr.track_id, None)
        if tr.end - tr.start + 1 < self.min_length:
            return None
        return tr

import numpy as np

from kukkiri.add_mosaic import block_grid, pixelate
from kukkiri.detector import Detection


def test_blocks_cover_the_region_on_a_fixed_grid_and_take_their_mean():
    h, w, block = 72, 96, 8
    det = Detection(box=(20, 20, 40, 36), score=0.9, label=1)
    grid = block_grid([det, Detection(box=(0, 0, 90, 70), score=0.9, label=0)], (h, w), block)
    assert grid.shape == (9, 12)
    ys, xs = np.nonzero(grid)
    assert ys.min() * block <= 20 and (ys.max() + 1) * block >= 36 and xs.min() * block <= 20 and (xs.max() + 1) * block >= 40
    assert not grid[0].any() and not grid[:, -1].any()
    frame = np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)
    out = pixelate(frame, grid, block)
    by, bx = ys[0], xs[0]
    cell = out[by * block:(by + 1) * block, bx * block:(bx + 1) * block]
    assert (cell == cell[0, 0]).all()
    assert (out[:block] == frame[:block]).all()


def test_nothing_found_gives_no_grid():
    assert block_grid([], (72, 96), 8) is None
    assert block_grid([Detection(box=(0, 0, 10, 10), score=0.9, label=2)], (72, 96), 8) is None

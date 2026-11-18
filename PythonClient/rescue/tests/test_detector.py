import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from detector import Detection, decode, letterbox, merge_fragments, tile_origins  # noqa: E402


@pytest.mark.parametrize("length,tiles,overlap", [(1280, 1, 0.2), (1280, 2, 0.2), (720, 3, 0.1), (1281, 2, 0.25)])
def test_tiles_cover_image_with_overlap(length, tiles, overlap):
    starts, size = tile_origins(length, tiles, overlap)
    assert starts[0] == 0
    assert starts[-1] + size >= length
    for a, b in zip(starts, starts[1:]):
        assert b - a <= size * (1 - overlap) + 1  # neighbours overlap by at least `overlap`


def test_letterbox_keeps_aspect_and_centres():
    image = np.zeros((720, 900, 3), np.uint8)
    boxed, scale, (pad_x, pad_y) = letterbox(image, 640)
    assert boxed.shape == (640, 640, 3)
    assert scale == pytest.approx(640 / 900)
    assert pad_x == 0 and pad_y == (640 - round(720 * scale)) // 2


def test_decode_maps_letterbox_box_back_to_full_image():
    # Tile of 900x720 starting at x=380 in the full image; a person at tile pixels (100..140, 200..300).
    tile_w, tile_h, x0, y0 = 900, 720, 380, 0
    _, scale, (pad_x, pad_y) = letterbox(np.zeros((tile_h, tile_w, 3), np.uint8), 640)
    cx, cy = (120 * scale + pad_x), (250 * scale + pad_y)
    output = np.zeros((6, 3), np.float32)
    output[:, 0] = [cx, cy, 40 * scale, 100 * scale, 0.9, 0.1]   # person
    output[:, 1] = [cx, cy, 40 * scale, 100 * scale, 0.1, 0.8]   # vehicle: wrong class
    output[:, 2] = [cx, cy, 40 * scale, 100 * scale, 0.1, 0.05]  # below threshold

    dets = decode(output, 0.25, [0], scale, pad_x, pad_y, x0, y0, tile_w, tile_h)
    assert len(dets) == 1
    d = dets[0]
    assert (d.x1, d.y1, d.x2, d.y2) == pytest.approx((x0 + 100, 200, x0 + 140, 300), abs=0.01)
    assert d.confidence == pytest.approx(0.9)


def test_partial_box_from_neighbouring_tile_is_merged():
    full = Detection(500, 200, 540, 300, 0.7, 0)
    fragment = Detection(500, 200, 520, 300, 0.9, 0)   # same person, cut by the seam at x=520
    other = Detection(700, 200, 740, 300, 0.8, 0)
    kept = merge_fragments([fragment, full, other])
    assert len(kept) == 2
    merged = next(k for k in kept if k.x1 == 500)
    assert merged.x2 == 540 and merged.confidence == pytest.approx(0.9)

import os
import sys
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from vision.screen import grab_screenshot
from config import Config


# Checkerboard sign matrix: +1 on one color, -1 on the other.
_SIGN = np.ones((8, 8), dtype=np.float32)
_SIGN[1::2, ::2] = -1
_SIGN[::2, 1::2] = -1

# The score is the sum of 64 square means, so this corresponds to only six
# gray levels of average light/dark-square contrast. Anything weaker is much
# more likely to be repeated UI decoration than a chess board.
_MIN_BOARD_SCORE = 64.0 * 6.0


def _score_region(region: np.ndarray) -> float:
    """How strongly `region` looks like an 8x8 checkerboard.

    Resize to 8x8 (which averages each square) and correlate with the
    checkerboard sign pattern. A real board gives a large magnitude; almost
    any other image gives ~0."""
    r8 = cv2.resize(region, (8, 8), interpolation=cv2.INTER_AREA)
    if r8.ndim == 3:
        g = cv2.cvtColor(r8, cv2.COLOR_BGR2GRAY)
    else:
        g = r8
    return float(abs(np.sum(g.astype(np.float32) * _SIGN)))


def _roi_from_inner_corners(gray: np.ndarray, coarse_roi):
    """Recover the outer board from its 7x7 internal square corners.

    OpenCV's chessboard detector checks the two-dimensional corner lattice as
    one object. That is considerably safer than combining unrelated horizontal
    and vertical line repetitions from the surrounding UI.
    """
    x, y, size = coarse_roi
    pad = max(12, int(size * 0.18))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1 = min(gray.shape[1], x + size + pad)
    y1 = min(gray.shape[0], y + size + pad)
    search = gray[y0:y1, x0:x1]

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(search, (7, 7), flags)
    if not found:
        return None

    corners = corners.reshape(7, 7, 2).astype(np.float32)
    corners[:, :, 0] += x0
    corners[:, :, 1] += y0

    # Fit x/y as functions of lattice column/row. This averages all 49 corners
    # and is stable even when pieces obscure several boundary pixels.
    rows, cols = np.mgrid[0:7, 0:7]
    design = np.column_stack((np.ones(49), cols.ravel(), rows.ravel()))
    xfit = np.linalg.lstsq(design, corners[:, :, 0].ravel(), rcond=None)[0]
    yfit = np.linalg.lstsq(design, corners[:, :, 1].ravel(), rcond=None)[0]
    px = float(np.hypot(xfit[1], yfit[1]))
    py = float(np.hypot(xfit[2], yfit[2]))
    pitch = (px + py) / 2.0
    if pitch <= 0 or abs(px - py) > 0.08 * pitch:
        return None

    center = corners.mean(axis=(0, 1))
    board_size = 8.0 * pitch
    roi = [int(round(center[0] - board_size / 2.0)),
           int(round(center[1] - board_size / 2.0)),
           int(round(board_size)), int(round(board_size))]
    return _clip_square_roi(roi, gray.shape[1], gray.shape[0])


def _roi_from_square_colors(img: np.ndarray, coarse_roi):
    """Refine a coarse hit to the connected region formed by the board colors.

    Chess.com draws rank/file labels in a gutter beside the squares. At some
    board sizes that gutter itself preserves enough of the alternating pattern
    to win the coarse scan. The two square colors, however, form one large
    8-connected component whose bounds are the actual playable board.
    """
    x, y, size = coarse_roi
    # Coarse checkerboard correlation is phase-ambiguous by one complete square
    # when a header/footer happens to resemble the missing rank. Search beyond
    # one square pitch on every side so the true 8x8 color component is visible.
    pad = max(12, int(size * 0.18))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1 = min(img.shape[1], x + size + pad)
    y1 = min(img.shape[0], y + size + pad)
    search = img[y0:y1, x0:x1]

    colors, counts = np.unique(search.reshape(-1, 3), axis=0,
                               return_counts=True)
    if len(colors) < 2:
        return None
    top = np.argsort(counts)[-min(6, len(colors)):][::-1]
    palette = colors[top].astype(np.int32)
    pixels = search.astype(np.int32)
    color_masks = []
    for color in palette:
        dist2 = np.sum((pixels - color) ** 2, axis=2)
        color_masks.append(dist2 <= 18 ** 2)

    best = None
    for i in range(len(palette)):
        for j in range(i + 1, len(palette)):
            if np.linalg.norm(palette[i] - palette[j]) < 25:
                continue
            mask = (color_masks[i] | color_masks[j]).astype(np.uint8)
            count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
                mask, connectivity=8)
            for label in range(1, count):
                bx, by, bw, bh, area = stats[label]
                candidate_size = (bw + bh) / 2.0
                if not (0.72 * size <= candidate_size <= 1.08 * size):
                    continue
                if abs(bw - bh) > 0.025 * candidate_size:
                    continue
                fill = area / float(bw * bh)
                if fill < 0.55:
                    continue

                cx = x0 + bx + bw / 2.0
                cy = y0 + by + bh / 2.0
                coarse_cx, coarse_cy = x + size / 2.0, y + size / 2.0
                center_error = np.hypot(cx - coarse_cx, cy - coarse_cy)
                if center_error > 0.18 * size:
                    continue

                board_size = int(round(candidate_size))
                roi = _clip_square_roi([
                    int(round(cx - board_size / 2.0)),
                    int(round(cy - board_size / 2.0)),
                    board_size, board_size,
                ], img.shape[1], img.shape[0])
                checker = checkerboard_score(roi, img)
                if checker < _MIN_BOARD_SCORE:
                    continue
                rank = fill + checker / 10000.0 - center_error / size
                if best is None or rank > best[0]:
                    best = (rank, roi, checker)
    return None if best is None else (best[1], best[2])


def _clip_square_roi(roi, image_w, image_h):
    """Clip a square ROI without silently turning it into a rectangle."""
    x, y, w, h = [int(v) for v in roi]
    size = min(w, h, image_w, image_h)
    x = min(max(0, x), image_w - size)
    y = min(max(0, y), image_h - size)
    return [x, y, size, size]


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix, iy = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix2 - ix), max(0, iy2 - iy)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def _scan(img, bx, by, w, h, smin, smax, step):
    """Scans the search space for the best square region.
    bx, by: top-left corner of the search box.
    w, h: width and height of the search box.
    smin, smax: minimum and maximum square to search.
    step: how much to skip between sizes.
    """
    best = None
    best_score = -1.0

    # This loop tries all square sizes from smin to smax to find the best square region.
    # min(w, h) = 1080 on a 1920x1080 screen = the largest square that fits.
    # min(smax, 1080) takes the smaller of the two, so if smax=512, the cap becomes 512.
    # So e.g. range(smin, 512 + 1, step).
    for size in range(smin, min(smax, min(w, h)) + 1, step):
        for y in range(by, by + h - size + 1, step):
            for x in range(bx, bx + w - size + 1, step):
                region = img[y:y + size, x:x + size]
                s = _score_region(region)
                if s > best_score:
                    best_score = s
                    best = (x, y, size)
    return best, best_score


def _smooth(arr, k):
    k = max(1, int(k))
    kernel = np.ones(2 * k + 1) / (2 * k + 1)
    return np.convolve(arr, kernel, mode="same")


def _cluster(positions, gap):
    positions = sorted(positions)
    if not positions:
        return []
    groups, cur = [], [positions[0]]
    for p in positions[1:]:
        if p - cur[-1] <= gap:
            cur.append(p)
        else:
            groups.append(cur)
            cur = [p]
    groups.append(cur)
    return [int(np.median(g)) for g in groups]


def _peaks(sig, min_dist, thr):
    """Strict local maxima of `sig` above `thr`, merging any closer than
    `min_dist` (keeping the stronger). Avoids the doubled boundary pixels that
    central-difference gradients create on each square edge."""
    cand = [i for i in range(1, len(sig) - 1)
            if sig[i] > thr and sig[i] > sig[i - 1] and sig[i] >= sig[i + 1]]
    merged = []
    for p in cand:
        if merged and p - merged[-1] < min_dist:
            if sig[p] > sig[merged[-1]]:
                merged[-1] = p
        else:
            merged.append(p)
    return merged


def _find_edges(line_strength, p_approx, center_approx):
    """Find the board's 9 grid lines along one axis from gradient peaks.

    The board's square boundaries are the strongest edges (square-vs-square
    steps), so they form a set of 9 evenly-spaced gradient peaks. We pick the
    evenly-spaced 9-peak set with the *highest total peak strength*; weaker,
    sparse spikes from a surrounding frame or coordinate-label text lose. With
    flat-shaded squares there are no edges at square centers, so a whole-square
    shift cannot produce a competing evenly-spaced set -- this is immune to the
    integer-shift ambiguity that plagues checkerboard-score methods."""
    if line_strength.max() < 1e-3:
        return None
    # Light smoothing only: just enough to kill single-pixel noise without
    # smearing the sharp spikes into broad plateaus (which would register every
    # pixel as a local maximum).
    s = _smooth(line_strength, 1.0)
    thr = 0.15 * s.max()
    min_dist = max(2, int(p_approx * 0.3))
    peaks = _peaks(s, min_dist, thr)
    if len(peaks) < 9:
        return None
    best = None
    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            span = peaks[j] - peaks[i]
            p = span / 8.0
            if not (0.82 * p_approx <= p <= 1.18 * p_approx):
                continue
            a = peaks[i]
            # A valid board must remain close to the coarse checkerboard hit.
            # Without this constraint a distant toolbar/sidebar can supply one
            # of the axes and create an enormous, unrelated rectangle.
            candidate_center = a + 4.0 * p
            if abs(candidate_center - center_approx) > 0.65 * p_approx:
                continue
            # Exactly one closest peak per grid line (k = 0..8); using *all*
            # peaks near a line would double-count a weak frame/label spike that
            # sits next to a true boundary and let a bogus set win on strength.
            total, cnt = 0.0, 0
            for k in range(9):
                target = a + k * p
                cand = [pk for pk in peaks if abs(pk - target) < p * 0.5]
                if cand:
                    c = min(cand, key=lambda pk: abs(pk - target))
                    total += s[c]
                    cnt += 1
            if cnt == 9 and (best is None or total > best[0]):
                best = (total, int(round(a)), int(round(a + 8.0 * p)), p)
    if best is None:
        return None
    return best[1], best[2], best[3]


def detect_board_roi(img: np.ndarray):
    """Find the 8x8 chess board in `img`.

    Coarse checkerboard scan locates the board's *size and rough center*, then a
    grid-line projection pins the exact *position* (the 9 evenly spaced square
    lines), which is robust to the board being shifted by whole squares.

    Returns (roi, score)."""

    if img is None or img.size == 0:
        raise ValueError("Cannot detect a board in an empty image")

    # we downscale the image so _scan() runs faster while keeping the board detectable.
    h, w = img.shape[:2] # img.shape is (height, width, channels). [:2] grabs (height, width).
    scale = 560.0 / w # scale factor to resize the image to a width of 560 pixels while maintaining aspect ratio.
    small_img = cv2.resize(img, (560, int(h * scale))) # downscale the image.
    sh, sw = small_img.shape[:2]

    result, score = _scan(small_img, 0, 0, sw, sh, 120, min(sw, sh), 8)

    if result is None or score < _MIN_BOARD_SCORE:
        raise RuntimeError(
            "No clear 8x8 chess board found. Keep the entire board visible "
            "and unobstructed, then try again."
        )
    x, y, size = result

    # Rough center + pitch in original pixels (size is reliable even when the
    # position is ambiguous by whole squares).
    f = w / sw
    cx = (x + size / 2.0) * f
    cy = (y + size / 2.0) * f
    p_approx = size * f / 8.0

    coarse_roi = (int(round(x * f)), int(round(y * f)),
                  int(round(size * f)))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert img to grayscale
    corner_roi = _roi_from_inner_corners(gray, coarse_roi)
    if corner_roi is not None:
        corner_score = checkerboard_score(corner_roi, img)
        if corner_score >= _MIN_BOARD_SCORE:
            return corner_roi, corner_score

    color_result = _roi_from_square_colors(img, coarse_roi)
    if color_result is not None:
        return color_result

    # Search region: expand well past the board so the true edges are included.
    margin = int(p_approx * 9)
    sxb = max(0, int(cx - margin))
    syb = max(0, int(cy - margin))
    sxe = min(w, int(cx + margin))
    sye = min(h, int(cy + margin))
    sub = gray[syb:sye, sxb:sxe]

    # Grid-line strength: sum the perpendicular gradient. Vertical grid lines
    # show up as peaks in the column-wise horizontal-gradient sum (and vice
    # versa). This is robust to constant frame/background strips that would
    # wash out a plain intensity projection.
    gx = np.gradient(sub, axis=1)
    gy = np.gradient(sub, axis=0)
    vlines = np.abs(gx).sum(axis=0)
    hlines = np.abs(gy).sum(axis=1)

    xs = _find_edges(vlines, p_approx, cx - sxb)
    ys = _find_edges(hlines, p_approx, cy - syb)
    if xs is None or ys is None:
        # A coarse checkerboard hit is still safer than mixing unrelated line
        # sets. Keep it square and within the image.
        return _clip_square_roi([coarse_roi[0], coarse_roi[1],
                                 coarse_roi[2], coarse_roi[2]], w, h), score

    L, R, px = xs
    T, B, py = ys
    if abs(px - py) > 0.08 * ((px + py) / 2.0):
        return _clip_square_roi([coarse_roi[0], coarse_roi[1],
                                 coarse_roi[2], coarse_roi[2]], w, h), score

    board_size = int(round(8.0 * (px + py) / 2.0))
    center_x = sxb + (L + R) / 2.0
    center_y = syb + (T + B) / 2.0
    roi = _clip_square_roi([int(round(center_x - board_size / 2.0)),
                            int(round(center_y - board_size / 2.0)),
                            board_size, board_size], w, h)
    refined_score = checkerboard_score(roi, img)
    if refined_score < _MIN_BOARD_SCORE:
        return _clip_square_roi([coarse_roi[0], coarse_roi[1],
                                 coarse_roi[2], coarse_roi[2]], w, h), score
    return roi, refined_score


def checkerboard_score(roi, img) -> float:
    """Score of a specific ROI on `img` (used to validate a cached ROI)."""
    x, y, w, h = [int(v) for v in roi]
    if w <= 0 or h <= 0 or x < 0 or y < 0 \
            or x + w > img.shape[1] or y + h > img.shape[0]:
        return 0.0
    region = img[y:y + h, x:x + w]
    return _score_region(region)


def find_board_roi():
    """Grab a screenshot and return the detected board ROI."""
    img = grab_screenshot()
    return detect_board_roi(img)[0]


def save_debug_image(img, roi, path=None):
    """Draw the detected ROI (red box) and the 8x8 grid (green) on `img`."""
    import cv2
    path = path or os.path.join(BASE_DIR, "vision", "debug_roi.png")
    vis = img.copy()
    x, y, w, h = [int(v) for v in roi]
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 4)
    for i in range(1, 8):
        gx = x + int(w * i / 8)
        gy = y + int(h * i / 8)
        cv2.line(vis, (gx, y), (gx, y + h), (0, 255, 0), 1)
        cv2.line(vis, (x, gy), (x + w, gy), (0, 255, 0), 1)
    cv2.imwrite(path, vis)
    return path


def show_detection(roi_hint=None):
    """Detect the board and open an annotated screenshot so you can see exactly
    which region was found."""
    img = grab_screenshot()
    roi, score = detect_board_roi(img, roi_hint)
    path = save_debug_image(img, roi)
    print("Debug image saved to:", path)
    print("Detected ROI:", roi, "checkerboard score:", round(score, 1))
    try:
        os.startfile(path)
    except Exception as e:
        print("Could not auto-open the image:", e)
    return roi


if __name__ == "__main__":
    if "--show" in sys.argv:
        roi = show_detection()
    else:
        roi = find_board_roi()
    cfg = Config.load()
    cfg.board_roi = roi
    cfg.save()
    print("Detected ROI:", roi)
    if "--show" not in sys.argv:
        img = grab_screenshot()
        print("checkerboard score:", checkerboard_score(roi, img))
    print("Saved to config.json. Verify with: python -m vision.board_inspect")

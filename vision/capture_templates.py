import os
import sys
import chess
import numpy as np

from config import Config
from vision.screen import grab_screenshot
from vision.detect_roi import find_board_roi
from vision.template_classifier import TemplateClassifier
from vision import dataset as ds


def _verify_starting_structure(img, roi):
    """Independent (template-free) check that the on-screen board is the starting
    position: the four central ranks must be empty and the two pawn ranks plus
    both back ranks must be fully occupied. This is NOT circular like the
    self-test (which labels crops by the FEN) -- it uses raw pixels, so it can
    actually reject a capture taken from the wrong board."""
    from vision.template_classifier import TemplateClassifier
    squares = ds.crop_squares(img, roi)
    occ = []
    for sq in squares:
        bg = TemplateClassifier._corner_bg(sq)
        fg = sq.astype(np.float32) - bg
        occ.append(float(np.linalg.norm(fg)))
    thr = (max(occ) + min(occ)) / 2.0 * 0.25
    grid = [[occ[r * 8 + c] > thr for c in range(8)] for r in range(8)]
    for r in (2, 3, 4, 5):
        if any(grid[r]):
            return False
    for r in (0, 1, 6, 7):
        if not all(grid[r]):
            return False
    return True


def _save_board_debug(img, roi, orientation, clf, fen, path="vision/debug_board.png"):
    """Annotate the screenshot with the ROI box and, for each square, the
    expected symbol (from FEN) vs what the classifier detected. Opens it so you
    can see exactly where/why the capture failed."""
    import cv2
    from vision import dataset as ds
    vis = img.copy()
    x, y, w, h = [int(v) for v in roi]
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 4)
    squares = ds.crop_squares(img, roi)
    board = chess.Board(fen)
    font = max(0.3, w / 640.0 * 0.5)
    for i, sq in enumerate(squares):
        r, c = divmod(i, 8)
        name = ds.square_name_for_crop(r, c, orientation)
        piece = board.piece_at(chess.parse_square(name))
        exp = piece.symbol() if piece else "."
        sc = "light" if (chess.square_file(chess.parse_square(name)) +
                         chess.square_rank(chess.parse_square(name))) % 2 == 1 else "dark"
        got = clf.predict(sq, sc)[0]
        got_s = "." if got is None else got
        cx = x + int((c + 0.5) * w / 8)
        cy = y + int((r + 0.5) * h / 8)
        cv2.putText(vis, f"{exp}/{got_s}", (cx - 22, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font, (0, 255, 0), 1)
    cv2.imwrite(path, vis)
    try:
        os.startfile(path)
    except Exception:
        pass
    return path

# Usage:
#   python -m vision.capture_templates
#       -> capture from the STARTING position (board must show the opening setup)
#   python -m vision.capture_templates "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
#       -> capture from any FEN you have set on screen
#
# The board must match the FEN exactly. The starting position is best because it
# contains every piece type on both light and dark squares, plus empty squares.

if __name__ == "__main__":
    fen = sys.argv[1] if len(sys.argv) > 1 else chess.STARTING_FEN
    cfg = Config.load()

    # Always re-detect the ROI precisely (alignment must be exact so every
    # square crop lines up with the templates).
    print("Detecting board ROI precisely...")
    roi = find_board_roi()
    cfg.board_roi = roi
    cfg.save()
    print("ROI:", roi)

    try:
        chess.Board(fen)
    except Exception as e:
        print("Invalid FEN:", e)
        sys.exit(1)

    print(f"Capturing templates from FEN:\n  {fen}")
    print("Make sure the on-screen board matches this position exactly.\n")
    input("Press Enter when the board is set and visible...")

    img = grab_screenshot()
    clf = TemplateClassifier()
    clf.build_from_screenshot(img, roi, fen, cfg.board_orientation)

    # Self-test: re-classify the same board and compare to the FEN we captured
    # from. This confirms the ROI alignment and templates are consistent. Only
    # save when the self-check passes -- a misaligned or wrong board must never
    # poison templates.pkl.
    squares = ds.crop_squares(img, roi)
    board = chess.Board(fen)
    correct = total = 0
    mism = []
    for i, sq in enumerate(squares):
        r, c = divmod(i, 8)
        name = ds.square_name_for_crop(r, c, cfg.board_orientation)
        piece = board.piece_at(chess.parse_square(name))
        exp = None if piece is None else (
            ("w" if piece.color else "b") + piece.symbol().upper())
        sc = "light" if (chess.square_file(chess.parse_square(name)) +
                         chess.square_rank(chess.parse_square(name))) % 2 == 1 else "dark"
        got = clf.predict(sq, sc)[0]
        total += 1
        if exp == got:
            correct += 1
        else:
            mism.append((name, exp, got))
    acc = correct / total if total else 0.0
    print(f"\nBuilt {len(clf.templates)} templates.")
    print(f"Self-check accuracy vs FEN: {acc*100:.1f}%")
    if acc < 0.9:
        dbg = _save_board_debug(img, roi, cfg.board_orientation, clf, fen)
        # Do not replace a known-good model with a bad capture. The temporary
        # classifier is intentionally never saved until all checks pass.
        print("WARNING: the on-screen board did NOT match the FEN you gave.")
        print("Templates were NOT saved. Use the exact position's FEN and retry.")
        print("Mismatches (square, expected, detected):", mism[:12])
        print("Wrote a debug overlay (expected/detected per square):", dbg)
        sys.exit(1)
    # The raw-pixel structure check only applies to the default starting FEN.
    # Custom FEN captures are valid too, provided the classifier self-test passed.
    if fen == chess.STARTING_FEN and not _verify_starting_structure(img, roi):
        print("WARNING: the board on screen is NOT the starting position")
        print("(the central ranks are not empty). Templates were NOT saved.")
        print("Set up a fresh game so the opening position is shown, then re-run.")
        sys.exit(1)
    clf.save()
    print(f"Saved to {clf.path}")
    print("Looks good. Now run: python -m vision.board_inspect")

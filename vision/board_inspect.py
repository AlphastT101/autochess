import os
import sys
import chess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config import Config
from vision.template_classifier import TemplateClassifier
from vision import dataset as ds
from vision.detector import BoardTracker
from vision.screen import grab_screenshot
from vision.detect_roi import detect_board_roi

cfg = Config.load()
clf = TemplateClassifier()
if not clf.loaded:
    print("No piece templates captured yet. Run first:")
    print("  python -m vision.capture_templates   (set the board to the "
          "starting position)")
    sys.exit(1)

# Revalidate the cached ROI against the live screen. The board can be resized
# between runs, and classifying with a stale ROI misaligns every square crop,
# which looks like a piece-recognition bug rather than a geometry problem.
try:
    roi, _score = detect_board_roi(grab_screenshot(), roi_hint=cfg.board_roi)
    if roi != cfg.board_roi:
        print(f"ROI changed on screen: {cfg.board_roi} -> {roi} (updated)")
        cfg.board_roi = roi
        cfg.save()
except Exception as e:
    print("ROI revalidation failed, using cached ROI:", e)

tracker = BoardTracker(cfg.board_orientation, cfg.board_roi, clf)

print(f"color={cfg.color} orientation={cfg.board_orientation} "
      f"roi={cfg.board_roi} classifier={type(clf).__name__}")
print("Classifying current screen...\n")

curr = tracker.snapshot_classified()

# Print detected board as a grid (rank 8 at top -> rank 1 at bottom)
print("Detected board (from screen):")
for r in range(8):
    row = []
    for c in range(8):
        name = ds.square_name_for_crop(r, c, cfg.board_orientation)
        sym = curr.get(name) or "."
        row.append(sym)
    print("  " + " ".join(f"{s:>2}" for s in row))

# Build placement + show validity
placement = ds.build_placement(curr)
print("\nplacement string:", placement)
try:
    # Side to move is not observable from a screenshot, so try both. Reporting
    # only "w" made legal positions look invalid whenever the side that just
    # moved had given check.
    results = {}
    for turn in ("w", "b"):
        b = chess.Board(placement + f" {turn} - - 0 1")
        results[turn] = b
    valid_turns = [t for t, b in results.items() if b.is_valid()]
    b = results[valid_turns[0]] if valid_turns else results["w"]
    wk = len(b.pieces(chess.KING, chess.WHITE))
    bk = len(b.pieces(chess.KING, chess.BLACK))
    if valid_turns:
        print(f"kings W/B = {wk}/{bk}  is_valid = True "
              f"(with {'/'.join(valid_turns)} to move)")
    else:
        print(f"kings W/B = {wk}/{bk}  is_valid = False "
              f"(status w={results['w'].status()}, b={results['b'].status()})")
except Exception as e:
    print("FEN build error:", e)

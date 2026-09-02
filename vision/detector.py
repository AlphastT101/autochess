import chess
import time
import numpy as np
import cv2
from vision.template_classifier import TemplateClassifier
from vision import dataset as ds
from vision.screen import grab_screenshot


class BoardTracker:
    """Owns the live chess.Board and updates it from screenshots.

    The board is maintained INCREMENTALLY from detected moves (the visual
    diff between frames is reliable) rather than re-classifying the whole
    board every frame. This keeps the position fed to Stockfish clean even
    when the per-square classifier is noisy.
    """

    def __init__(self, orientation: str, roi: list, classifier: TemplateClassifier,
                  start_fen: str = None, our_color: int = None):
        self.orientation = orientation
        self.roi = roi
        self.clf = classifier
        self.start_fen = start_fen
        self.our_color = our_color
        self.board = chess.Board(start_fen) if start_fen else chess.Board()
        self.last_classified = None

    def snapshot_classified(self):
        img = grab_screenshot()
        squares = ds.crop_squares(img, self.roi)
        return ds.classify_board(squares, self.clf, self.orientation)

    def _log_diff(self, prev, curr):
        """Log per-square differences between two classification dicts."""
        for name in sorted(set(prev) | set(curr)):
            pv, cv = prev.get(name), curr.get(name)
            if pv != cv:
                print(f"  [diff] {name}: {pv} -> {cv}", flush=True)

    def _dump_all_squares(self):
        """Dump per-square raw classification details for diagnosis."""
        import numpy as np
        img = grab_screenshot()
        squares = ds.crop_squares(img, self.roi)
        for i, sq_img in enumerate(squares):
            r, c = divmod(i, 8)
            name = ds.square_name_for_crop(r, c, self.orientation)
            sc = self.clf._sq_color(name)
            local_fg = self.clf._foreground(sq_img)
            fg_norm = float(np.linalg.norm(local_fg))
            flat = sq_img.reshape(-1, 3).astype(float)
            n_bright = int(np.sum(np.max(flat, axis=1) > 240))
            n_dark = int(np.sum(np.min(flat, axis=1) < 70))
            sym, score = self.clf.predict(sq_img, sc)
            tracked_piece = self.board.piece_at(chess.parse_square(name))
            tracked_sym = None
            if tracked_piece:
                tracked_sym = ("w" if tracked_piece.color else "b") + tracked_piece.symbol().upper()
            print(f"  {name} sc={sc} fg_norm={fg_norm:.1f} bright={n_bright} dark={n_dark} "
                  f"classified={sym}(s={score:.3f}) tracked={tracked_sym}", flush=True)

    def is_board_valid(self):
        try:
            b = self.board
            wk = b.pieces(chess.KING, chess.WHITE)
            bk = b.pieces(chess.KING, chess.BLACK)
            if len(wk) != 1 or len(bk) != 1:
                return False
            return b.is_valid()
        except Exception:
            return False

    @staticmethod
    def _count_matching_squares(fen1, fen2):
        """Count how many of the 64 squares have the same piece in both positions."""
        try:
            b1 = chess.Board(fen1 + " w - - 0 1")
            b2 = chess.Board(fen2 + " w - - 0 1")
        except Exception:
            return 0
        count = 0
        for sq in chess.SQUARES:
            if b1.piece_at(sq) == b2.piece_at(sq):
                count += 1
        return count

    def _find_best_move(self, target, min_match=62, require_margin=True):
        """Find the legal move whose resulting position best matches the screen read.

        Requires at least min_match squares to match. When require_margin is
        True, the best move must also beat the second-best by at least 2 squares
        (to avoid accepting a move just because noise happened to align with it).
        Returns the best matching move and its match count, or (None, 0) if
        nothing qualifies."""
        best_move, best_match, second_match = None, 0, 0
        for m in self.board.legal_moves:
            b2 = self.board.copy()
            b2.push(m)
            expected = b2.board_fen()
            match = self._count_matching_squares(expected, target)
            if match > best_match:
                second_match = best_match
                best_match = match
                best_move = m
            elif match > second_match:
                second_match = match
        margin_ok = (not require_margin) or (best_match - second_match) >= 2
        if best_move and best_match >= min_match and margin_ok:
            return best_move, best_match
        return None, best_match

    def update(self):
        """Detect a move from a new screenshot. Returns UCI, 'INIT', or None.

        Uses fuzzy matching: for each legal move, counts how many squares match
        the screen read, tolerating up to 2 misclassified squares.  This is
        robust against single-square classification errors (e.g. a black pawn
        read as a rook due to last-move highlight) that would make the full FEN
        invalid and block move detection.
        """
        curr = self.snapshot_classified()
        if self.last_classified is None:
            self.last_classified = curr
            if self.start_fen is None:
                placement = ds.build_placement(curr)
                try:
                    turn = "w" if self.board.turn == chess.WHITE else "b"
                    self.board = ds.board_from_placement(placement, turn)
                except Exception:
                    pass  # keep startpos
            return "INIT", ""
        # No change vs the last accepted read -> nothing happened.
        if ds.build_placement(curr) == ds.build_placement(self.last_classified):
            return None, "no change between frames"
        # Board changed. Find the legal move that best matches the screen read.
        # Do not accept a 62/64 match here: two classifier errors can make an
        # unrelated legal move look plausible and permanently desynchronise the
        # incremental board. A later stable frame can still be accepted.
        target = ds.build_placement(curr)
        best_move, best_match = self._find_best_move(target)
        if best_move:
            self.last_classified = curr
            self.board.push(best_move)
            if best_match < 60:
                print(f"  [tracker] accepted move {best_move.uci()} "
                      f"with {best_match}/64 squares matching", flush=True)
            return best_move.uci(), ""
        # No good match (noise / animation / desync). Keep the last accepted
        # baseline. Replacing it with a noisy frame would hide the real move on
        # the next poll and leave the tracked board behind the screen.
        return None, f"best match was only {best_match}/64"

    def screen_match_count(self):
        """Return how many squares in one fresh screen read match our board."""
        curr = self.snapshot_classified()
        target = ds.build_placement(curr)
        return self._count_matching_squares(self.board.board_fen(), target), curr

    def screen_matches_board(self, min_match=62):
        """Check that the live screen still represents the tracked position."""
        matches, curr = self.screen_match_count()
        return matches >= min_match, matches, curr

    def resync_after_our_move(self, delay: float = 1.2):
        """Re-screenshot after we played, so the next diff only sees the
        opponent's move (not our own). Waits for the click animation."""
        import time
        time.sleep(delay)
        _placement, curr = self._stable_placement(tries=4)
        if curr is not None:
            self.last_classified = curr
        else:
            # A transient animation frame must not become the reference for
            # detecting the opponent's next move.
            self.last_classified = None

    def _stable_placement(self, tries: int = 5):
        """Take several screenshots and return a placement only if it repeats
        (the board is static), so a single noisy frame can't be trusted."""
        prev, prev_count = None, 0
        for _ in range(tries):
            curr = self.snapshot_classified()
            p = ds.build_placement(curr)
            if p == prev:
                prev_count += 1
                if prev_count >= 2:
                    return p, curr
            else:
                prev, prev_count = p, 1
            time.sleep(0.25)
        return None, None

    def force_sync(self, turn_color: int = None):
        """Re-classify the current screen and replace the tracked board with it.
        Tries progressively more lenient matching:
        1. Exact valid position from screen read
        2. Fuzzy match (56/64) from tracked board
        3. Fuzzy match (48/64) from tracked board (handles more drift)
        4. Fuzzy match (48/64) from a starting-position hint
        Returns the new FEN, or None on failure."""
        placement, curr = self._stable_placement()
        if placement is None:
            print("  [force_sync] could not get a stable board read.", flush=True)
            return None
        self.last_classified = curr
        # First try: exact valid position
        try:
            b = ds.board_from_placement(placement, "w")
            if b.is_valid():
                if turn_color is not None:
                    b.turn = turn_color
                self.board = b
                return b.fen()
        except Exception:
            pass
        # Second try: fuzzy match from tracked board
        best_move, best_match = self._find_best_move(placement)
        if best_move:
            self.board.push(best_move)
            if turn_color is not None:
                self.board.turn = turn_color
            print(f"  [force_sync] accepted via fuzzy match ({best_match}/64): "
                  f"{best_move.uci()}", flush=True)
            return self.board.fen()
        # Third try: more lenient fuzzy match (tracked board may have drifted)
        best_move, best_match = self._find_best_move(placement, min_match=63,
                                                     require_margin=False)
        if best_move:
            self.board.push(best_move)
            if turn_color is not None:
                self.board.turn = turn_color
            print(f"  [force_sync] accepted via lenient match ({best_match}/64): "
                  f"{best_move.uci()}", flush=True)
            return self.board.fen()
        # Fourth try: try from a fresh starting board as baseline
        saved_board = self.board.copy()
        self.board = chess.Board()
        best_move, best_match = self._find_best_move(placement, min_match=56,
                                                     require_margin=False)
        if best_move:
            self.board.push(best_move)
            if turn_color is not None:
                self.board.turn = turn_color
            print(f"  [force_sync] accepted from startpos ({best_match}/64): "
                  f"{best_move.uci()}", flush=True)
            return self.board.fen()
        self.board = saved_board
        print("  [force_sync] read invalid, placement:", placement, flush=True)
        return None

    def resync_to_screen(self):
        """Re-classify the screen and advance the tracked board if it changed.

        Takes up to 3 screenshots and tries fuzzy matching on each, to handle
        intermittent noise (animation artifacts, highlights).  Uses the same
        strict threshold as update() (62/64 + margin) to avoid accepting
        phantom moves that corrupt the tracked position.  Returns the UCI
        move applied, "UNMATCHED" if nothing matches well enough, or None if
        the screen matches the tracked board (nothing happened).
        """
        for attempt in range(3):
            curr = self.snapshot_classified()
            target = ds.build_placement(curr)
            if target == self.board.board_fen():
                return None  # screen matches the tracked board; nothing happened
            best_move, best_match = self._find_best_move(target)
            if best_move:
                self.last_classified = curr
                self.board.push(best_move)
                if best_match < 62:
                    print(f"  [tracker] resync accepted {best_move.uci()} "
                          f"({best_match}/64, attempt {attempt+1})", flush=True)
                return best_move.uci()
            time.sleep(0.3)
        # All attempts failed. Keep the accepted baseline so a transient noisy
        # read cannot become the new reference position.
        return "UNMATCHED"


if __name__ == "__main__":
    # Quick test: classify the current screen's board.
    from config import Config
    cfg = Config.load()
    if not cfg.board_roi:
        print("Detect the board first (python -m vision.detect_roi).")
    else:
        clf = TemplateClassifier()
        if not clf.loaded:
            print("No templates yet. Run: python -m vision.capture_templates")
        else:
            t = BoardTracker(cfg.board_orientation, cfg.board_roi, clf)
            print("Detected move:", t.update())
            print(t.board.fen())

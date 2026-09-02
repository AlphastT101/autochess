"""Helper mode: analyze the current board (as white or black), find the best
move with Stockfish, explain why it's good, and — optionally — show opponent
threats. It NEVER clicks/moves pieces; it just reads and helps.

Drawing:
  - green arrow: the recommended best move.
  - red arrows: opponent threats / attacks on our pieces (common-sense filtered).
Keyboard:
  - 'r' in the console clears the red (threat) arrows.
  - 'h' toggles the green best-move arrow.
Overlay arrows are drawn on a transparent click-through always-on-top window.
"""

import time
import random
import chess

from overlay import Overlay, PIECE_VALUES


PIECE_NAMES = {None: "", "p": "pawn", "n": "knight", "b": "bishop",
               "r": "rook", "q": "queen", "k": "king"}


COLOR_NAMES = {chess.WHITE: "white", chess.BLACK: "black"}


class HelperMode:
    def __init__(self, tracker, engine_fn, cfg, overlay_enabled=True):
        # tracker: vision.detector.BoardTracker (owns the live board)
        # engine_fn: callable(fen, smartness, movetime) -> best move uci
        self.tracker = tracker
        self.engine_fn = engine_fn
        self.cfg = cfg
        self.board = tracker.board
        self.overlay = None
        self.overlay_enabled = overlay_enabled
        self.show_attacks = bool(getattr(cfg, "show_attacks", False))
        self.show_green = True

    # ---------------- overlay helpers ----------------

    def _ensure_overlay(self):
        if self.overlay is None and self.overlay_enabled and self.cfg.board_roi:
            self.overlay = Overlay(self.cfg.board_roi)
            self.overlay.open()

    def _hide_for_capture(self):
        if self.overlay is not None:
            self.overlay.hide()
            self.overlay.update()

    def _show_after_capture(self):
        if self.overlay is not None:
            self.overlay.show()
            self.overlay.update()

    def redraw(self):
        if not self.overlay_enabled or self.overlay is None:
            return
        orientation = self.cfg.board_orientation
        self.overlay.clear()
        # green best move
        green = getattr(self, "_green", None)
        if green and self.show_green:
            self.overlay.arrow(green[0], green[1], "green", orientation)
        # red threats
        if self.show_attacks:
            for frm, to in getattr(self, "_reds", []):
                self.overlay.arrow(frm, to, "red", orientation)
        self.overlay.update()

    def clear_reds(self):
        self._reds = []
        self.redraw()

    def close(self):
        if self.overlay is not None:
            self.overlay.close()
            self.overlay = None

    # ---------------- current position ----------------

    def read_position(self):
        """Re-sync the tracker's board to the live screen. Hides the overlay
        during capture so the arrows don't pollute the screenshot used for
        classification. Returns (board, detected_move_uci_or_None, mover_is_ours_or_None)."""
        self._hide_for_capture()
        try:
            prev_board = self.tracker.board.copy()
            mv, reason = self.tracker.update()
        finally:
            self._show_after_capture()
        uci = mv if isinstance(mv, str) and mv not in ("INIT", "") else None
        mover_is_ours = None
        if uci:
            try:
                from_sq = chess.parse_square(uci[0:2])
                pc = prev_board.piece_at(from_sq)
                if pc:
                    mover_is_ours = pc.color == self.tracker.our_color
                else:
                    mover_is_ours = (not self.tracker.board.turn) == self.tracker.our_color
            except Exception:
                mover_is_ours = (not self.tracker.board.turn) == self.tracker.our_color
        return self.tracker.board, uci, mover_is_ours

    # ---------------- analysis ----------------

    def explain(self, mv: chess.Move):
        """Return a short, human 'why this is good' description for a move,
        using python-chess heuristics (deterministic, no extra Stockfish)."""
        board = self.board
        tags = []
        promo = board.piece_at(mv.from_square)
        promo_name = promo.symbol().lower() if promo else None
        captured = board.piece_at(mv.to_square)
        # 1. Recapture / material gain
        if captured:
            tags.append(f"captures the {COLOR_NAMES[captured.color]} "
                        f"{PIECE_NAMES[captured.symbol().lower()]}")
        # 2. Check / mate
        nb = board.copy()
        nb.push(mv)
        if nb.is_checkmate():
            tags.append("CHECKMATE!")
        elif nb.is_check():
            tags.append("check")
        elif nb.is_stalemate():
            tags.append("stalemate")
        # 3. Promotion
        if mv.promotion:
            tags.append(f"promotes to {chess.piece_symbol(mv.promotion)}")
        # 4. Central control (for non-capture quiet piece moves)
        if not captured:
            to_sq = mv.to_square
            if chess.square_file(to_sq) in (3, 4) and chess.square_rank(to_sq) in (3, 4):
                tags.append("controls the center")
        # 5. Development of a minor piece
        if not captured and promo_name in ("n", "b"):
            if board.fullmove_number <= 8 or board.ply() <= 16:
                if chess.square_rank(mv.from_square) in (0, 7):
                    tags.append("develops a piece")
        # 5b. Queen to the center (activate it)
        if not captured and promo_name == "q":
            if chess.square_rank(mv.to_square) in (3, 4) and \
               chess.square_file(mv.to_square) in (3, 4):
                tags.append("centralizes the queen")
            if chess.square_rank(mv.from_square) in (0, 7) and \
               not chess.square_rank(mv.to_square) in (0, 7):
                tags.append("activates the queen")
        # 6. Castling = king safety
        if promo_name == "k":
            if abs(chess.square_file(mv.to_square) - chess.square_file(mv.from_square)) == 2:
                tags.append("castles for king safety")
        # 7. Responds to an immediate threat (evades capture)
        if not captured:
            attacked = board.is_attacked_by(not board.turn, mv.from_square)
            still_attacked = nb.is_attacked_by(not nb.turn, mv.to_square) \
                and board.piece_at(mv.to_square) is None
            if attacked and not still_attacked:
                tags.append("moves out of danger")
        # 8. Opening space gain (pawn push two squares)
        if not captured and promo_name == "p":
            if abs(chess.square_rank(mv.to_square) - chess.square_rank(mv.from_square)) == 2:
                tags.append("gains space")
        # 9. Capture is safe (recapture doesn't just lose the attacker back)
        if captured and promo_name in ("n", "b", "q"):
            from_sq = mv.from_square
            to_sq = mv.to_square
            # opponent can just recapture on to_sq
            recapture_by = None
            for s2, p2 in board.piece_map().items():
                if p2 and p2.color != board.turn:
                    if to_sq in board.attacks(s2):
                        recapture_by = p2.symbol().lower()
                        break
            if recapture_by is None:
                tags.append("a safe, undefended capture")
        if not tags:
            tags.append("quiet developing move")
        return tags

    def engine_eval(self, fen, smartness):
        """Best move + rough centipawn eval change via a shallow extra search."""
        mv = self.engine_fn(fen, smartness, movetime=max(200, self.cfg.think_time_ms))
        if not mv:
            return None, 0
        return mv, 0

    def analyze(self, fen, smartness):
        """Top-level: compute best move, green arrow, red threats, explanation."""
        self.board = self.tracker.board
        try:
            best_uci = self.engine_fn(fen, smartness, movetime=self.cfg.think_time_ms)
        except Exception as e:
            print(f"[helper] engine error: {type(e).__name__} {e}", flush=True)
            return None
        if not best_uci:
            return None
        try:
            mv = chess.Move.from_uci(best_uci)
        except Exception:
            return None
        # Never trust a non-legal engine move (desynced board can produce one).
        if mv not in self.board.legal_moves:
            print(f"[helper] engine move {best_uci} not legal (board desync?)", flush=True)
            # Try to recover: rebuild board from a stable screen read.
            try:
                placement, curr = self.tracker._stable_placement(tries=4)
                if placement:
                    try:
                        # Try both turn senses; keep the valid one that makes best_uci legal.
                        for turn in ("w", "b"):
                            b2 = chess.Board(placement + f" {turn} - - 0 1")
                            if b2.is_valid() and mv in b2.legal_moves:
                                self.tracker.board = b2
                                self.tracker.last_classified = curr
                                self.board = b2
                                print(f"[helper] resynced to screen (now {turn} to move) -> retrying", flush=True)
                                return self.analyze(b2.fen(), smartness)
                        # If still not legal, just adopt the screen position.
                        b2 = chess.Board(placement + f" {'w' if self.tracker.our_color==chess.WHITE else 'b'} - - 0 1")
                        if b2.is_valid():
                            self.tracker.board = b2
                            self.tracker.last_classified = curr
                            self.board = b2
                            print(f"[helper] resynced to screen (forced) -> {placement}", flush=True)
                    except Exception:
                        pass
            except Exception:
                pass
            return None
        reason = self.explain(mv)
        self._green = (best_uci[0:2], best_uci[2:4])
        if self.show_attacks:
            self._reds = self.find_threats()
        else:
            self._reds = []
        return {
            "move": best_uci,
            "mv": mv,
            "reason": reason,
            "fen_after": None,
        }

    # ---------------- threats (red arrows) ----------------

    def find_threats(self, max_arrows=6):
        """Return a list of (from_square, to_square) red arrows = opponent's
        meaningful threats, using common sense to avoid visual noise."""
        board = self.board
        our_color = board.turn  # opponent is about to move against us
        opp = not our_color
        threats = []
        seen = set()

        def add(frm, to):
            # convert square indices to names (e.g. 28 -> "e4") for the overlay
            frm, to = chess.square_name(frm), chess.square_name(to)
            key = (frm, to)
            if key not in seen and len(threats) < max_arrows:
                threats.append(key)
                seen.add(key)

        pmap = board.piece_map()  # {square: Piece}
        opp_pieces = {sq: p for sq, p in pmap.items() if p.color == opp}
        our_pieces = {sq: p for sq, p in pmap.items() if p.color == our_color}

        # 1. Checks on our king -> always show (opponent giving check).
        opp_board = board.copy()
        opp_board.turn = opp
        for mv in opp_board.legal_moves:
            nb = opp_board.copy(); nb.push(mv)
            if nb.is_check() or nb.is_checkmate():
                add(mv.from_square, mv.to_square)

        # 2. Attacks that capture our pieces (weighted by value).
        capture_candidates = []
        for sq, piece in opp_pieces.items():
            for to in board.attacks(sq):
                target = board.piece_at(to)
                if target and target.color == our_color:
                    # value gained vs value risked (the attacker may be recaptured)
                    val = PIECE_VALUES[target.symbol().lower()]
                    if to == our_king(board):
                        val += 20  # king capture is huge (though illegal, a threat)
                    capture_candidates.append((val, sq, to))
        capture_candidates.sort(reverse=True, key=lambda t: t[0])
        added = 0
        for val, sq, to in capture_candidates:
            if len(threats) >= max_arrows:
                break
            if added < 3:  # show up to 3 captures, prioritize attacks on our king/major
                add(sq, to)
                added += 1

        # 3. Opponent's hanging / aggressive pieces pushing forward (optional,
        #    only if we still have budget and attacker is a minor/major advancing)
        return threats

    def run(self):
        """Main helper loop: read position, analyze, draw, and react to keys.
        Blocking until Ctrl+C."""
        import keyboard
        self._ensure_overlay()
        try:
            while True:
                try:
                    _board, moved_uci, mover_is_ours = self.read_position()
                    self.board = _board
                    fen = self.board.fen()
                    changed = fen != getattr(self, "_last_analyzed_fen", None)
                    # Report the actual move that just happened on screen.
                    # De-duplicate: tracker.update returns each move once, but
                    # a noisy frame can re-trigger the same UCI.
                    if moved_uci and moved_uci != getattr(self, "_last_moved_uci", None):
                        if mover_is_ours is None:
                            mover_is_ours = (not self.board.turn) == self.tracker.our_color
                        mover = "You" if mover_is_ours else "Opponent"
                        print(f"{mover} moved: {moved_uci[0:2]} -> {moved_uci[2:4]}", flush=True)
                        self._last_moved_uci = moved_uci
                    # Only recommend a move when it's actually our turn. On the
                    # opponent's turn there is no best move for us to play.
                    our_turn = self.board.turn == self.tracker.our_color
                    if our_turn and changed:
                        self._green = None  # clear stale arrows while recomputing
                        self._reds = []
                        self._last_analyzed_fen = fen
                        mv = self.analyze(fen, self.cfg.smartness)
                        if mv:
                            uci = mv["move"]
                            color_name = COLOR_NAMES[self.board.turn]
                            print(f"Best move as {color_name}: {uci[0:2]} -> {uci[2:4]}", flush=True)
                            for reason in mv["reason"]:
                                print(f"  * {reason}", flush=True)
                            if self.show_attacks and len(self._reds) > 0:
                                print(f"  {len(self._reds)} opponent threat(s)", flush=True)
                            print()
                    elif not our_turn and changed:
                        # Opponent is still moving - no recommendation to show
                        self._green = None
                        self._reds = []
                    self.redraw()
                except Exception as e:
                    print("helper analyze error:", type(e).__name__, e, flush=True)

                # handle keys
                if keyboard.is_pressed("s"):
                    print("[helper] s pressed - resyncing board from screen...", flush=True)
                    self._hide_for_capture()
                    try:
                        placement, curr = self.tracker._stable_placement(tries=4)
                        if placement is None:
                            print("  resync failed: no stable board read.", flush=True)
                        else:
                            from vision import dataset as ds
                            new_board = None
                            for turn in ("w", "b"):
                                try:
                                    cand = ds.board_from_placement(placement, turn)
                                    if cand.is_valid():
                                        new_board = cand
                                        break
                                except Exception:
                                    continue
                            if new_board is None:
                                print("  resync failed: invalid placement.", flush=True)
                            else:
                                self.tracker.board = new_board
                                self.tracker.last_classified = curr
                                self.board = new_board
                                self._last_analyzed_fen = None
                                self._last_moved_uci = None
                                self._green = None
                                self._reds = []
                                print(f"  resynced: {new_board.fen()} ({'white' if new_board.turn==chess.WHITE else 'black'} to move)", flush=True)
                    finally:
                        self._show_after_capture()
                    self.redraw()
                    time.sleep(0.3)
                if keyboard.is_pressed("r"):
                    self.clear_reds()
                    print("[helper] cleared red threat arrows.", flush=True)
                    time.sleep(0.3)
                if keyboard.is_pressed("h"):
                    self.show_green = not self.show_green
                    self.redraw()
                    print(f"[helper] green best-move arrow {'on' if self.show_green else 'off'}.",
                          flush=True)
                    time.sleep(0.3)
                if self.overlay:
                    self.overlay.pump(20)
                time.sleep(self.cfg.poll_interval)
        except KeyboardInterrupt:
            print()
        finally:
            self.close()
            print()


def our_king(board):
    kings = board.pieces(chess.KING, board.turn)
    return next(iter(kings)) if kings else None

import sys
import time
import chess
import random
import keyboard


sys.path.insert(0, ".")
from config import Config
from vision.detector import BoardTracker
from vision.screen import grab_screenshot
from engine import best_move_once, shutdown
from input.mouse import play_move, square_to_pixel
from vision.detect_roi import find_board_roi, detect_board_roi, checkerboard_score, iou

HOTKEY = "s"            # manual trigger: detect opponent move and play
ENTER_HOTKEY = "enter"  # manual sync: "opponent just moved, it's my turn now"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AutoChess bot")
    parser.add_argument("-c", "--color", choices=["b", "w", "black", "white"], help="side to play and board orientation (b/black or w/white), overrides config.json")
    args, _ = parser.parse_known_args()
    cfg = Config.load()
    if args.color:
        col = "black" if args.color.lower().startswith("b") else "white"
        cfg.color = col
        cfg.board_orientation = col
    if not cfg.board_roi:
        print("No cached ROI - detecting board automatically...")
        try:
            cfg.board_roi = find_board_roi()
            cfg.save()
            print("Detected + saved ROI:", cfg.board_roi)
        except Exception as e:
            print("Auto board detection failed:", e)
            print("Run manually: python -m vision.detect_roi")
            return
    else:
        # Validate the cached ROI: a board should still be a strong checkerboard
        # right where the cache says. If it drifted/moved, re-detect.
        try:
            img = grab_screenshot()
            local_roi, local_score = detect_board_roi(img)
            if iou(cfg.board_roi, local_roi) < 0.6 or local_score <= 0:
                print("Cached ROI no longer valid - re-detecting board...")
                cfg.board_roi = find_board_roi()
                cfg.save()
                print("Re-detected + saved ROI:", cfg.board_roi)
            else:
                print("Cached ROI valid.")
        except Exception as e:
            print("ROI validation failed; re-detecting board:", e)
            try:
                cfg.board_roi = find_board_roi()
                cfg.save()
                print("Re-detected + saved ROI:", cfg.board_roi)
            except Exception as redetect_error:
                print("Board re-detection failed:", redetect_error)
                return

    from vision.template_classifier import TemplateClassifier
    clf = TemplateClassifier()
    if not clf.loaded:
        print("No piece templates captured. Run: python -m vision.capture_templates "
              "(set the board to the starting position first).")
    if not clf.loaded:
        return

    our_color = chess.BLACK if cfg.color == "black" else chess.WHITE
    tracker = BoardTracker(cfg.board_orientation, cfg.board_roi, clf,
                           cfg.start_fen, our_color)

    print(f"Mode={cfg.mode} Color={cfg.color} Orientation={cfg.board_orientation} "
          f"Smartness={cfg.smartness} "
          f"Auto-trigger={cfg.auto_trigger}")
    print("Press '%s' or Enter to force-sync "
          "(opponent just moved / start mid-game)." % HOTKEY)

    def our_turn():
        return tracker.board.turn == our_color

    def _print_banner(title, msg):
        bar = "=" * 54
        print("\n" + bar)
        print(f"||  {title}")
        print(f"||  {msg}")
        print(bar + "\n", flush=True)

    def do_our_move():
        if cfg.mode != "auto":
            print("[manual] our turn; not playing automatically.")
            return
        # Game already over on our turn -> we were checkmated / drawn.
        if tracker.board.is_checkmate():
            _print_banner("CHECKMATE", "You have been checkmated. Game over.")
            sys.exit(0)
        if tracker.board.is_stalemate() or tracker.board.is_insufficient_material():
            _print_banner("DRAW", "Game drawn.")
            sys.exit(0)
        try:
            # The tracker is incremental. Before asking Stockfish for a move,
            # confirm that its internal position still matches the live board.
            # Otherwise a previously accepted fuzzy opponent move could result
            # in a legal engine move that is impossible on the actual screen.
            synced, matches, _curr = tracker.screen_matches_board(min_match=63)
            if not synced:
                print(f"Board is out of sync ({matches}/64 squares match); "
                      "skipping move and re-baselining.", flush=True)
                tracker.last_classified = None
                return
            fen = tracker.board.fen()
            if not tracker.is_board_valid():
                # A single noisy frame (e.g. mid-animation, or the last-move
                # highlight) can misread a square. Re-classify a few times
                # before bailing so we don't poison the whole game state.
                ok = False
                for _ in range(3):
                    tracker.last_classified = None
                    tracker.update()
                    if tracker.is_board_valid():
                        ok = True
                        break
                    time.sleep(0.3)
                if not ok:
                    print("Board looks invalid (misclassified); skipping move and "
                          "re-baselining on next frame.", flush=True)
                    tracker.last_classified = None
                    return
            print("thinking...", flush=True)
            mv = best_move_once(fen, cfg.smartness, movetime=cfg.think_time_ms)
            engine_move = chess.Move.from_uci(mv)
            if engine_move not in tracker.board.legal_moves:
                print(f"Engine returned illegal move {mv} for tracked FEN; "
                      "skipping click and re-baselining.", flush=True)
                tracker.last_classified = None
                return
            print("Engine ->", mv, flush=True)
            # Human-like random delay before playing (configurable via move_delay [min, max])
            try:
                lo, hi = float(cfg.move_delay[0]), float(cfg.move_delay[1])
                if hi >= lo and hi > 0:
                    delay = random.uniform(lo, hi)
                    time.sleep(delay)
            except Exception:
                pass
            play_move(mv, cfg.board_roi, cfg.board_orientation)
            tracker.board.push(engine_move)
            # We just moved: did we deliver checkmate / draw?
            if tracker.board.is_checkmate():
                _print_banner("CHECKMATE", "You win! Opponent has been checkmated.")
                sys.exit(0)
            if tracker.board.is_stalemate():
                _print_banner("STALEMATE", "Draw by stalemate.")
                sys.exit(0)
            if tracker.board.is_insufficient_material() \
               or tracker.board.is_seventyfive_moves() \
               or tracker.board.is_fivefold_repetition():
                _print_banner("DRAW", "Game drawn.")
                sys.exit(0)
            tracker.resync_after_our_move()
        except Exception as e:
            print("Move FAILED:", type(e).__name__, e, flush=True)

    # State shared between the hotkey handler and the loop.
    last_played_fen = None      # position for which we already played our move
    last_move_time = time.time()  # last time any move (ours/opponent's) happened
    last_check_log = 0.0
    STALL = 5.0                 # seconds of watching before auto-recovering
    invalid_streak = 0          # consecutive polls with an invalid board read
    REDETECT_STREAK = 4         # -> auto re-detect ROI after this many

    def manual_sync():
        nonlocal last_played_fen, last_move_time
        """User says: opponent just moved, it's my turn now. Re-read the screen
        and play. The user pressing Enter is an explicit assertion that it IS our
        move, so we play even if the screen read is invalid/desynced."""
        print("[manual] syncing...", flush=True)
        mv, reason = tracker.update()
        print("  auto-detected:", mv, "reason:", reason, flush=True)
        fen = None
        for _ in range(4):
            fen = tracker.force_sync(turn_color=our_color)
            if fen:
                break
            time.sleep(0.4)
        print("  force-sync FEN:", fen, flush=True)
        if not our_turn():
            # Screen read failed/invalid or it's still our opponent's turn on the
            # tracked board. The user explicitly said it's our move -> trust them.
            if fen is None:
                print("  WARNING: screen board invalid (ROI likely wrong) - "
                      "playing on last known position. Run: python -m vision.detect_roi "
                      "to re-detect.", flush=True)
            tracker.board.turn = our_color
            print("  (forcing our turn per manual trigger)", flush=True)
        if our_turn():
            do_our_move()
            last_played_fen = tracker.board.fen()
        else:
            print("  still not our turn; check the board is visible.", flush=True)
        last_move_time = time.time()

    def redetect_roi():
        """Re-detect the board ROI on screen and adopt it live (mid-game)."""
        try:
            new_roi = find_board_roi()
            cfg.board_roi = new_roi
            cfg.save()
            tracker.roi = new_roi
            # Re-classify a few times so a transient frame doesn't poison the
            # re-baseline.
            tracker.last_classified = None
            for _ in range(3):
                tracker.last_classified = None
                tracker.update()
                if tracker.is_board_valid():
                    break
                time.sleep(0.3)
            print("Re-detected board ROI (was invalid):", new_roi, flush=True)
        except Exception as e:
            print("Mid-game re-detection failed:", e, flush=True)

    turn_name = "white" if tracker.board.turn == chess.WHITE else "black"
    side = "OUR turn -> bot will move" if our_turn() else "waiting for OPPONENT"
    print(f"Bot plays {cfg.color} ({turn_name} to move). {side}.")

    try:
        while True:
            if keyboard.is_pressed(HOTKEY) or keyboard.is_pressed(ENTER_HOTKEY):
                manual_sync()
                time.sleep(0.3)
                continue
            if cfg.auto_trigger:
                mv, reason = tracker.update()
                # If the board read is invalid for several polls in a row, the
                # ROI may have drifted (board moved / popup). Auto re-detect it.
                if tracker.is_board_valid():
                    invalid_streak = 0
                else:
                    invalid_streak += 1
                    if invalid_streak >= REDETECT_STREAK:
                        redetect_roi()
                        invalid_streak = 0
                if mv == "INIT":
                    print(f"Baseline set. Turn: {turn_name}.",
                          "Our turn -> playing." if our_turn() else "Waiting for opponent.",
                          flush=True)
                elif mv is not None:  # opponent move detected
                    print("Opponent moved:", mv, flush=True)
                    last_move_time = time.time()
                    last_played_fen = None

                # Play whenever it is actually our turn and we haven't played
                # for this position yet (decoupled from detecting their move).
                if our_turn() and tracker.board.fen() != last_played_fen:
                    do_our_move()
                    last_played_fen = tracker.board.fen()
                    last_move_time = time.time()
                else:
                    # Watching. If we've been idle past STALL and it's NOT our
                    # turn, the opponent likely moved but we missed it -> recover.
                    now = time.time()
                    if now - last_move_time > STALL and not our_turn():
                        rec = tracker.resync_to_screen()
                        if rec and rec != "UNMATCHED":
                            # A single real legal opponent move reproduces the new
                            # screen -> safe to advance and play our reply.
                            print("Opponent moved (auto-recovered):", rec, flush=True)
                            last_move_time = time.time()
                            last_played_fen = None
                            if our_turn():
                                do_our_move()
                                last_played_fen = tracker.board.fen()
                        elif rec == "UNMATCHED" and now - last_check_log > 5:
                            # Screen changed but not a clean single move: don't
                            # guess. This usually means the board read is noisy or
                            # the ROI is wrong. Wait for you to press Enter.
                            print("[watching] board changed but move unclear; "
                                  "press Enter if it's your turn.", flush=True)
                            last_check_log = now
                    # Throttled chatter so it isn't overwhelming.
                    if now - last_check_log > 5:
                        turn_name = "white" if tracker.board.turn == chess.WHITE else "black"
                        print(f"[watching] {turn_name} to move ({reason})", flush=True)
                        last_check_log = now
                time.sleep(cfg.poll_interval)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        shutdown()


if __name__ == "__main__":
    main()

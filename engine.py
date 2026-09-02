import os
import queue
import subprocess
import threading
import time

from config import STOCKFISH_PATH, SMARTNESS_PRESETS


class Engine:
    """UCI wrapper around Stockfish with a dedicated stdout-draining thread.
    Continuously reading stdout prevents the pipe-buffer deadlock that occurs
    when the main thread is busy (screenshotting/classifying) while the engine
    tries to write."""

    def __init__(self, path: str = STOCKFISH_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Stockfish not found at {path}")
        self.proc = subprocess.Popen(
            path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            bufsize=1,
            universal_newlines=True,
        )
        self.line_q: "queue.Queue[str]" = queue.Queue()
        self._pump = threading.Thread(target=self._reader, daemon=True)
        self._pump.start()
        self._send("uci")
        self._wait_for("uciok")

    def _reader(self):
        try:
            for line in iter(self.proc.stdout.readline, ""):
                self.line_q.put(line.strip())
            self.line_q.put(None)  # EOF
        except Exception:
            self.line_q.put(None)

    def _send(self, cmd: str):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _wait_for(self, token: str, timeout: float = 10.0):
        start = time.time()
        while time.time() - start < timeout:
            try:
                line = self.line_q.get(timeout=timeout)
            except queue.Empty:
                raise TimeoutError(f"Stockfish timed out waiting for '{token}'")
            if line is None:
                raise RuntimeError("Stockfish engine closed unexpectedly")
            if token in line:
                return line
        raise TimeoutError(f"Stockfish timed out waiting for '{token}'")

    def set_smartness(self, smartness: int):
        preset = SMARTNESS_PRESETS.get(int(smartness))
        if preset is not None:
            skill = preset["skill"]
            elo = preset["elo"]
            self._send(f"setoption name Skill Level value {skill}")
            if elo is None:
                self._send("setoption name UCI_LimitStrength value false")
            else:
                elo = max(1320, min(3190, int(elo)))
                self._send("setoption name UCI_LimitStrength value true")
                self._send(f"setoption name UCI_Elo value {elo}")
            return
        # fallback for arbitrary values not in presets
        if smartness >= 3200:
            self._send("setoption name Skill Level value 20")
            self._send("setoption name UCI_LimitStrength value false")
        elif smartness < 1320:
            # weak: use Skill Level only
            ratio = (smartness - 250) / (1320 - 250) if smartness > 250 else 0
            skill = int(ratio * 10)
            self._send(f"setoption name Skill Level value {max(0, min(10, skill))}")
            self._send("setoption name UCI_LimitStrength value false")
        else:
            elo = max(1320, min(3190, int(smartness)))
            self._send("setoption name Skill Level value 20")
            self._send("setoption name UCI_LimitStrength value true")
            self._send(f"setoption name UCI_Elo value {elo}")

    def set_position(self, fen: str):
        self._send(f"position fen {fen}")

    def best_move(self, fen: str, smartness: int, movetime: int = 800) -> str:
        self.set_smartness(smartness)
        self.set_position(fen)
        self._send("isready")
        self._wait_for("readyok")
        # depth from presets, with fallback via Config for arbitrary values
        if smartness in SMARTNESS_PRESETS:
            p = SMARTNESS_PRESETS[smartness]
        else:
            # use Config's mapping for non-preset values (e.g. 1500, 2700)
            from config import Config as _Cfg
            p = _Cfg(smartness=smartness).engine_params()
        # Bound thinking time with `movetime` (ms) so each move is fast and
        # predictable; `depth` is a backstop so it can't run away. A value of 0
        # means no time cap (use depth only).
        if movetime and movetime > 0:
            self._send(f"go movetime {movetime} depth {p['depth']}")
        else:
            self._send(f"go depth {p['depth']}")
        start = time.time()
        while time.time() - start < 30:
            try:
                line = self.line_q.get(timeout=30)
            except queue.Empty:
                raise TimeoutError("Stockfish timed out producing a move")
            if line is None:
                raise RuntimeError("Stockfish engine closed unexpectedly")
            if line.startswith("bestmove"):
                return line.split()[1]
        raise TimeoutError("Stockfish timed out producing a move")

    def alive(self) -> bool:
        """True if the Stockfish child process is still running."""
        return self.proc.poll() is None

    def quit(self):
        try:
            self._send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass


# A single engine is kept alive for the whole session and reused across moves
# (spawning a fresh process per move added ~0.2s of overhead). It self-heals: if
# the child ever dies (e.g. the intermittent segfault that used to occur when
# torch/OpenCV changed CPU state), the next call recreates it.
_engine = None


def _get_engine() -> "Engine":
    global _engine
    if _engine is not None and _engine.alive():
        return _engine
    if _engine is not None:
        try:
            _engine.quit()
        except Exception:
            pass
    _engine = Engine()
    return _engine


def shutdown():
    """Release the persistent engine (call on exit)."""
    global _engine
    if _engine is not None:
        try:
            _engine.quit()
        except Exception:
            pass
        _engine = None


def best_move_once(fen: str, smartness: int, retries: int = 3, movetime: int = 800) -> str:
    """Return Stockfish's best move for `fen`. Reuses a persistent engine and
    recreates it automatically if it dies. Retries on failure."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            eng = _get_engine()
            return eng.best_move(fen, smartness, movetime)
        except Exception as e:  # engine died / timed out
            last_err = e
            print(f"[engine] attempt {attempt} failed: {type(e).__name__} {e}", flush=True)
            global _engine
            if _engine is not None:
                # Kill the stale engine process so it can't keep searching in the
                # background (otherwise repeated retries spawn orphaned Stockfish
                # processes that hog the CPU). Recreate fresh on the next attempt.
                try:
                    _engine.quit()
                except Exception:
                    try:
                        _engine.proc.kill()
                    except Exception:
                        pass
            _engine = None
    raise RuntimeError(f"Stockfish failed after {retries} attempts: {last_err}")


if __name__ == "__main__":
    e = Engine()
    mv = e.best_move(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 900
    )
    print("Best move @900:", mv)
    e.quit()

import os
import pickle
import cv2
import numpy as np
from collections import defaultdict

import chess
from vision import dataset as ds

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATH = os.path.join(BASE_DIR, "vision", "templates.pkl")
_FOREGROUND_SIZE = 64


def _cos(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class TemplateClassifier:
    """Appearance-based piece classifier that is robust to the last-move
    highlight Chess.com paints on squares.

    For every square we estimate the *local* background from the four corner
    patches and subtract it, leaving just the piece shape (the "foreground").
    Because the background is taken from the square itself, a translucent
    highlight tint is cancelled out automatically -- a highlighted empty square
    yields ~zero foreground (read as empty) and a highlighted occupied square
    yields the same piece shape as an unhighlighted one. Templates are stored per
    (symbol, square-color) so low-contrast cases (e.g. a white piece on a light
    square) still match correctly.
    """

    def __init__(self, path=DEFAULT_PATH):
        self.path = path
        self.templates = {}          # (symbol, color) -> foreground (h, w, 3)
        self.bg_ref = {}             # color -> mean corner background (b, g, r)
        self.empty_thresh = None     # foreground norm below which a square is empty
        self.center_thresh = None    # centre-region norm gate (see _center_norm)
        self.color_ref = {}          # 'w'/'b' -> mean sprite brightness
        self.loaded = False
        if os.path.exists(path):
            self.load()

    # ---- background-normalized foreground -------------------------------
    @staticmethod
    def _corner_bg(sq: np.ndarray):
        h, w = sq.shape[:2]
        m = max(2, w // 5)
        corners = [sq[0:m, 0:m], sq[0:m, -m:], sq[-m:, 0:m], sq[-m:, -m:]]
        # Median over all corner pixels (not mean): a large piece that bleeds
        # into a corner is an outlier and is ignored, so the estimate stays the
        # true square background even when a piece reaches the corners.
        vals = np.concatenate([c.reshape(-1, 3) for c in corners], axis=0)
        return np.median(vals, axis=0)

    @staticmethod
    def _foreground(sq: np.ndarray):
        return sq.astype(np.float32) - TemplateClassifier._corner_bg(sq)

    @staticmethod
    def _normalized_foreground(sq: np.ndarray):
        """Normalize a square foreground so templates work at any board size."""
        fg = TemplateClassifier._foreground(sq)
        return cv2.resize(fg, (_FOREGROUND_SIZE, _FOREGROUND_SIZE),
                          interpolation=cv2.INTER_AREA).astype(np.float32)

    @staticmethod
    def _silhouette(fg: np.ndarray):
        """Return a soft, normalized piece mask from a foreground image."""
        magnitude = np.max(np.abs(fg), axis=2).astype(np.float32)
        # Keep antialiased edges, while removing tiny board/highlight noise.
        mask = np.clip((magnitude - 10.0) / 35.0, 0.0, 1.0)
        return cv2.GaussianBlur(mask, (3, 3), 0)

    @staticmethod
    def _center_norm(fg: np.ndarray):
        """Foreground energy in the centre of the square only.

        Chess.com draws rank/file coordinate labels and last-move/check markers
        along the square's edges and corners. Those decorations survive
        background subtraction and can push the whole-square norm past the
        occupancy threshold, inventing a piece on an empty square. A real sprite
        always fills the centre, so gating on the centre removes that entire
        class of false positives.
        """
        margin = _FOREGROUND_SIZE // 6
        return float(np.linalg.norm(fg[margin:-margin, margin:-margin]))

    def _sprite_brightness(self, sq_img, fg):
        """Mean brightness of the piece pixels only (background excluded)."""
        small = cv2.resize(sq_img, (_FOREGROUND_SIZE, _FOREGROUND_SIZE),
                           interpolation=cv2.INTER_AREA).astype(np.float32)
        mask = self._silhouette(fg) > 0.5
        if int(mask.sum()) < 20:
            return None
        return float(small[mask].mean())

    def _calibrate(self):
        """Derive occupancy and colour references from the captured templates."""
        center_norms = [self._center_norm(t) for t in self.templates.values()]
        self.center_thresh = 0.4 * float(min(center_norms)) if center_norms else 0.0
        acc = defaultdict(list)
        for (sym, col), tmpl in self.templates.items():
            bg = self.bg_ref.get(col)
            if bg is None:
                continue
            mask = self._silhouette(tmpl) > 0.5
            if int(mask.sum()) < 20:
                continue
            acc[sym[0]].append(float((tmpl + bg)[mask].mean()))
        self.color_ref = {k: float(np.mean(v)) for k, v in acc.items() if v}

    @staticmethod
    def _color_of(bg) -> str:
        return "light" if float(np.mean(bg)) > 128 else "dark"

    @staticmethod
    def _sq_color(name) -> str:
        """Chess-square colour (light/dark) for a square name -- exact, derived
        from board geometry rather than from pixel brightness."""
        sqi = chess.parse_square(name)
        # a1 is a dark square; a square is light when file+rank is odd.
        return "light" if (chess.square_file(sqi) + chess.square_rank(sqi)) % 2 == 1 else "dark"

    # ---- building templates from a known position -----------------------
    def build_from_screenshot(self, img, roi, fen, orientation):
        squares = ds.crop_squares(img, roi)
        board = chess.Board(fen)
        # Pass 1: estimate the per-square-colour background. We ONLY sample
        # EMPTY squares -- a large piece bleeds into the corner patches, so the
        # corners of an occupied square are NOT pure background. In the starting
        # position ranks 3-6 are empty, giving a clean reference for the whole
        # game (the board colour is constant). We key by the EXACT chess-square
        # colour (not corner brightness) so geometry, not pixels, decides it.
        names = [ds.square_name_for_crop(i // 8, i % 8, orientation)
                 for i in range(64)]
        bg_acc = defaultdict(list)
        sqcolors = []
        for i, sq in enumerate(squares):
            sc = self._sq_color(names[i])
            sqcolors.append(sc)
            if board.piece_at(chess.parse_square(names[i])) is None:
                bg_acc[sc].append(self._corner_bg(sq))
        # Fallback: if a colour has no empty square to sample, average all of it.
        for i, sq in enumerate(squares):
            sc = sqcolors[i]
            if sc not in bg_acc or not bg_acc[sc]:
                bg_acc[sc].append(self._corner_bg(sq))
        bg_ref = {col: np.mean(bg_acc[col], axis=0) for col in bg_acc}
        # Pass 2: build templates from the LOCAL corner foreground -- this is
        # robust to any board lighting gradient and matched the same way at
        # prediction time. The global bg_ref above is only used for colour.
        acc = defaultdict(list)
        for i, sq in enumerate(squares):
            name = ds.square_name_for_crop(i // 8, i % 8, orientation)
            sc = sqcolors[i]
            piece = board.piece_at(chess.parse_square(name))
            if piece is not None:
                sym = ("w" if piece.color else "b") + piece.symbol().upper()
                acc[(sym, sc)].append(self._normalized_foreground(sq))
        if not acc:
            raise ValueError("No pieces found in the given position to learn from.")
        self.bg_ref = bg_ref
        self.templates = {k: np.mean(v, axis=0).astype(np.float32)
                          for k, v in acc.items()}
        self._fill_missing()
        occ_norms = [float(np.linalg.norm(t)) for t in self.templates.values()]
        self.empty_thresh = 0.5 * float(np.min(occ_norms))
        self._calibrate()
        self.loaded = True
        return self

    def _fill_missing(self):
        """Synthesize a (symbol, color) template when a piece only ever appears
        on one square color in the captured position (e.g. queens/kings at home).
        Since the foreground is `sprite - bg`, we can shift an existing color's
        template by the difference of the two background references."""
        seen = set(self.templates)
        for (sym, col), tmpl in list(self.templates.items()):
            missing = "dark" if col == "light" else "light"
            if (sym, missing) not in seen:
                # The foreground already subtracts the local square background
                # at capture and prediction time. Do not apply a second global
                # light/dark shift: it corrupts pieces after they move to the
                # opposite square colour.
                self.templates[(sym, missing)] = tmpl.copy().astype(np.float32)
                seen.add((sym, missing))

    def _repair_legacy_templates(self):
        """Repair models made by the old opposite-square colour synthesis.

        The original starting-position capture has one real square-colour
        sample for each non-pawn piece type. Older files do not record which
        entries were synthesized, but their source colour is fixed by the
        standard starting layout. Copying the real foreground to the missing
        colour is correct because foregrounds already have the local square
        background removed.
        """
        source_colors = {
            ("b", "B"): "light", ("b", "N"): "dark",
            ("b", "R"): "light", ("b", "Q"): "dark",
            ("b", "K"): "light", ("w", "B"): "dark",
            ("w", "N"): "light", ("w", "R"): "dark",
            ("w", "Q"): "light", ("w", "K"): "dark",
        }
        for (piece_color, piece_type), source_color in source_colors.items():
            source = (piece_color + piece_type, source_color)
            if source not in self.templates:
                continue
            for color in ("light", "dark"):
                self.templates[(piece_color + piece_type, color)] = \
                    self.templates[source].copy().astype(np.float32)

    # ---- prediction -----------------------------------------------------
    def predict(self, sq_img, sq_color=None):
        if not self.templates or self.empty_thresh is None:
            return None, 0.0
        # Emptiness and identity use the LOCAL corner foreground. Using the
        # square's own (median) corner background keeps this robust to any board
        # lighting gradient, and the median makes it robust to a piece that
        # reaches the corners. A white piece is always brighter than its own
        # square, so the foreground's mean sign gives the colour directly.
        local_fg = self._normalized_foreground(sq_img)
        fg_norm = float(np.linalg.norm(local_fg))
        if fg_norm < self.empty_thresh:
            return None, 0.0
        # Second, stricter occupancy test on the centre of the square. Board
        # coordinate labels and move/check markers live near the edges, and
        # would otherwise be classified as pieces on empty squares.
        if self.center_thresh and self._center_norm(local_fg) < self.center_thresh:
            return None, 0.0
        # Match the silhouette independently of polarity. A moved queen can be
        # on a different square colour from its capture square, and its signed
        # foreground then matches a synthesized/incorrect colour template poorly
        # even though its shape is unmistakably a queen.
        shape_fg = self._silhouette(local_fg)
        best_type, best_shape_score = None, -2.0
        for (sym, scol), tmpl in self.templates.items():
            if sq_color is not None and scol != sq_color:
                continue
            score = _cos(shape_fg, self._silhouette(tmpl))
            if score > best_shape_score:
                best_shape_score, best_type = score, sym[1]
        # Relax the square-colour constraint only when a template set is
        # incomplete, while retaining the polarity-independent shape match.
        if best_type is None:
            for (sym, _scol), tmpl in self.templates.items():
                score = _cos(shape_fg, self._silhouette(tmpl))
                if score > best_shape_score:
                    best_shape_score, best_type = score, sym[1]
        if best_type is None:
            return None, 0.0

        # Colour comes from the brightness of the sprite pixels themselves,
        # measured against references calibrated at capture time. Signed
        # template correlation cannot decide colour reliably: the five piece
        # types that start on a single square colour have a *copy* as their
        # opposite-colour template, so its sign carries no information about the
        # piece actually on screen. That is what turned white rooks/bishops into
        # black ones after they moved to the other square colour.
        brightness = self._sprite_brightness(sq_img, local_fg)
        piece_color = None
        if brightness is not None and len(self.color_ref) == 2:
            piece_color = min(self.color_ref,
                              key=lambda k: abs(brightness - self.color_ref[k]))
        if piece_color is None:
            # Mask too small to measure (heavily occluded square) -> fall back
            # to signed correlation.
            color_scores = {}
            for (sym, scol), tmpl in self.templates.items():
                if sym[1] != best_type:
                    continue
                if sq_color is not None and scol != sq_color:
                    continue
                color_scores[sym[0]] = max(
                    color_scores.get(sym[0], -2.0), _cos(local_fg, tmpl))
            if not color_scores:
                return None, 0.0
            piece_color = max(color_scores, key=color_scores.get)
        return piece_color + best_type, best_shape_score

    # ---- persistence ----------------------------------------------------
    def save(self, path=None):
        path = path or self.path
        with open(path, "wb") as f:
            pickle.dump({"templates": self.templates,
                         "bg_ref": self.bg_ref,
                         "empty_thresh": self.empty_thresh,
                         "center_thresh": self.center_thresh,
                         "color_ref": self.color_ref,
                         "foreground_size": _FOREGROUND_SIZE,
                         "template_version": 3}, f)

    def load(self, path=None):
        path = path or self.path
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            raw_templates = data["templates"]
            self.bg_ref = data.get("bg_ref", {})
            old_size = data.get("foreground_size")
            if old_size is None and raw_templates:
                old_size = next(iter(raw_templates.values())).shape[0]
            old_size = int(old_size or _FOREGROUND_SIZE)
            self.templates = {
                key: np.asarray(value, dtype=np.float32)
                if np.asarray(value).shape[:2] == (_FOREGROUND_SIZE,
                                                   _FOREGROUND_SIZE)
                else cv2.resize(np.asarray(value, dtype=np.float32),
                                (_FOREGROUND_SIZE, _FOREGROUND_SIZE),
                                interpolation=cv2.INTER_AREA)
                for key, value in raw_templates.items()
            }
            self.empty_thresh = data["empty_thresh"]
            if old_size != _FOREGROUND_SIZE:
                self.empty_thresh *= _FOREGROUND_SIZE / float(old_size)
            if data.get("template_version", 0) < 2:
                self._repair_legacy_templates()
            self.center_thresh = data.get("center_thresh")
            self.color_ref = data.get("color_ref", {})
            if not self.center_thresh or len(self.color_ref) != 2:
                # Older model file: derive the new references from its templates
                # so an existing capture keeps working without a recapture.
                self._calibrate()
            self.loaded = bool(self.templates)
        except Exception:
            self.loaded = False
        return self.loaded

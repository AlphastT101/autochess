import os
import cv2
import numpy as np
import chess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "vision", "data")


def crop_squares(img: np.ndarray, roi: list):
    """Crop the board ROI into 64 squares in image order (row 0 = top).
    Returns list of 64 BGR uint8 images."""
    x, y, w, h = [int(v) for v in roi]
    board = img[y:y + h, x:x + w]
    sq = w / 8.0
    squares = []
    for r in range(8):
        for c in range(8):
            sx = int(c * sq)
            sy = int(r * sq)
            crop = board[sy:int((r + 1) * sq), sx:int((c + 1) * sq)]
            squares.append(crop)
    return squares


def square_name_for_crop(r, c, orientation):
    """Map crop (row r from top, col c from left) to a square name."""
    if orientation == "white":
        file_idx = c
        rank = 8 - r
    else:  # black -> board rotated 180
        file_idx = 7 - c
        rank = r + 1
    return f"{chr(ord('a') + file_idx)}{rank}"


def classify_board(squares, classifier, orientation):
    """Return dict {square_name: symbol_or_None}."""
    result = {}
    for i, sq_img in enumerate(squares):
        r, c = divmod(i, 8)
        name = square_name_for_crop(r, c, orientation)
        sqi = chess.parse_square(name)
        sc = "light" if (chess.square_file(sqi) + chess.square_rank(sqi)) % 2 == 1 else "dark"
        symbol, _ = classifier.predict(sq_img, sc)
        result[name] = symbol
    return result


def build_placement(classified):
    """Build FEN piece-placement string from a {square: symbol} dict."""
    board = chess.Board(None)  # empty board
    for name, symbol in classified.items():
        if symbol:
            # symbol is like 'wN'/'bP'. Keep the color: lowercase = black.
            letter = symbol[1]
            if symbol[0] == "b":
                letter = letter.lower()
            board.set_piece_at(chess.parse_square(name),
                               chess.Piece.from_symbol(letter))
    return board.board_fen()


def board_from_placement(placement, turn):
    """Build a chess.Board from a placement string, granting castling rights
    when the king and the relevant rook are still on their home squares. The
    tracked board is otherwise maintained incrementally (pushes update rights
    correctly), but rebuilds (force_sync / INIT) would otherwise lose all rights
    and never be able to detect a castle."""
    b = chess.Board(placement + " w - - 0 1")
    rights = ""
    if b.piece_at(chess.E1) == chess.Piece(chess.KING, chess.WHITE):
        if b.piece_at(chess.A1) == chess.Piece(chess.ROOK, chess.WHITE):
            rights += "Q"
        if b.piece_at(chess.H1) == chess.Piece(chess.ROOK, chess.WHITE):
            rights += "K"
    if b.piece_at(chess.E8) == chess.Piece(chess.KING, chess.BLACK):
        if b.piece_at(chess.A8) == chess.Piece(chess.ROOK, chess.BLACK):
            rights += "q"
        if b.piece_at(chess.H8) == chess.Piece(chess.ROOK, chess.BLACK):
            rights += "k"
    return chess.Board(placement + f" {turn} {rights or '-'} - 0 1")


def detect_move(prev, curr, board: chess.Board):
    """Diff two classified boards and apply the opponent's move to `board`.

    Robust design: the moved piece type is taken from the tracked `board`
    (reliable), not from the (noisy) image classification. We look at *all*
    squares that changed between frames and find a single legal move on the
    tracked board whose from-square and to-square are among the changed squares.

    This is deliberately tolerant of artifacts like Chess.com's last-move
    highlight, which repaints the (now empty) source square so the classifier
    still "sees a piece" there - i.e. the source may appear as a changed square
    rather than an emptied one. By matching against the legal-move set we still
    recover the move.

    Returns a tuple (uci_or_None, reason). `reason` explains why a move was not
    returned (empty string when a move is found) so callers can log it."""
    changed = []
    for s in set(prev) | set(curr):
        if prev.get(s) != curr.get(s):
            changed.append(s)
    if not changed:
        return None, "no change between frames"
    if len(changed) > 8:
        # Too many changes to be a single move (desync / big error). Ignore.
        return None, f"too many changed squares ({len(changed)}) - ignoring"

    # Candidate sources: were occupied, now empty OR a different piece (the
    # highlight can make an emptied square look occupied-but-changed).
    sources = [s for s in changed
               if prev.get(s) is not None
               and (curr.get(s) is None or curr.get(s) != prev.get(s))]
    # Candidate dests: now occupied, and either was empty or a different piece.
    dests = [s for s in changed
             if curr.get(s) is not None
             and (prev.get(s) is None or prev.get(s) != curr.get(s))]
    if not sources or not dests:
        return None, f"ambiguous diff: {len(sources)} sources, {len(dests)} dests"

    # Find legal moves whose from/to are within the changed squares, then
    # disambiguate by actually playing each and comparing the resulting piece
    # placement to the screen. This correctly handles multi-square moves like
    # castling (king + rook) and en passant, which would otherwise look ambiguous.
    target = build_placement(curr)
    legal_pairs = []
    for m in board.legal_moves:
        fs, ts = chess.square_name(m.from_square), chess.square_name(m.to_square)
        if fs in sources and ts in dests and board.piece_at(m.from_square) is not None:
            legal_pairs.append(m)
    verified = []
    for m in legal_pairs:
        b2 = board.copy()
        b2.push(m)
        if b2.board_fen() == target:
            verified.append(m)
    if not verified:
        return None, f"no legal move reproduces board for sources {sources} dests {dests}"
    if len(verified) > 1:
        return None, f"ambiguous: {len(verified)} candidate moves {[m.uci() for m in verified]}"
    m = verified[0]
    board.push(m)
    return m.uci(), ""


if __name__ == "__main__":
    print("dataset utils ready. DATA_DIR =", DATA_DIR)

import time
import pyautogui


def square_to_pixel(square: str, roi: list, orientation: str) -> tuple:
    """Map a square like 'e4' to screen (x, y) center.

    orientation: 'white' -> white pieces at bottom (a1 bottom-left).
                 'black' -> board flipped, black pieces at bottom.
    roi: [x, y, w, h] bounding the board on screen.
    """
    x, y, w, h = roi
    sq = w / 8.0
    file_idx = ord(square[0]) - ord("a")      # 0..7
    rank_idx = int(square[1]) - 1             # 0..7 (0 == rank '1')

    if orientation == "white":
        col = file_idx
        row_from_top = 7 - rank_idx
    else:  # black -> board rotated 180
        col = 7 - file_idx
        row_from_top = rank_idx

    px = x + (col + 0.5) * sq
    py = y + (row_from_top + 0.5) * sq
    return int(px), int(py)


def click_square(square: str, roi: list, orientation: str, delay: float = 0.15):
    px, py = square_to_pixel(square, roi, orientation)
    pyautogui.click(px, py)
    time.sleep(delay)


def promotion_click_square(move_uci: str, roi: list, orientation: str):
    """After clicking the destination square of a promotion, click the
    chosen piece in Chess.com's promotion chooser. The chooser stacks the
    four options (Q,R,B,N) in the destination file, toward the player."""
    if len(move_uci) < 5:
        return
    promo = move_uci[4].lower()
    order = {"q": 0, "r": 1, "b": 2, "n": 3}
    k = order.get(promo, 0)
    file_ch = move_uci[2]
    rank = int(move_uci[3])
    # chooser extends toward the player's side:
    # white player is at bottom -> downward (decreasing rank number)
    # black player is at top   -> upward   (increasing rank number)
    rank2 = rank - k if orientation == "white" else rank + k
    target = f"{file_ch}{rank2}"
    click_square(target, roi, orientation, delay=0.1)


def play_move(move_uci: str, roi: list, orientation: str):
    """Click source square, then destination square, then (if promoting)
    the desired promotion piece in the chooser."""
    src = move_uci[0:2]
    dst = move_uci[2:4]
    click_square(src, roi, orientation)
    click_square(dst, roi, orientation)
    if len(move_uci) >= 5:  # promotion
        promotion_click_square(move_uci, roi, orientation)


if __name__ == "__main__":
    # Demo: show where a1..h8 land for a fake 800x800 board at (100,100).
    roi = [100, 100, 800, 800]
    for s in ["a1", "h1", "a8", "h8", "e4"]:
        print(s, "white->", square_to_pixel(s, roi, "white"),
              "black->", square_to_pixel(s, roi, "black"))

"""Always-on-top, click-through overlay for drawing arrows over the chess board.
Used by helper mode to show the green best-move arrow and red opponent-threat arrows.

The overlay window is:
  - always-on-top (topmost)
  - transparent (only the drawn arrows are opaque)
  - click-through (mouse events pass through to the window below)
so it does not block clicks. It is drawn over the board region on screen.

The transparency is achieved via Tkinter's -transparentcolor attribute: any pixel
that is pure black (0,0,0) becomes fully transparent, while non-black pixels
(arrow colors) are fully opaque. This is the simplest, most reliable way to get
a transparent overlay on Windows without fighting GDI alpha-blending edge cases.
"""

import math
import ctypes
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw

PIECE_VALUES = {None: 0, "p": 1, "n": 3, "b": 3, "r": 5, "q": 9, "k": 0}


class Overlay:
    """A click-through, always-on-top transparent window that draws arrows."""

    def __init__(self, roi: list):
        self.roi = roi
        self.x, self.y, self.w, self.h = [int(v) for v in roi]
        self.root = None
        self._base = None
        self._tkimg = None
        self._img_id = None

    def open(self):
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        # Make pure black transparent at the OS level
        try:
            self.root.attributes("-transparentcolor", "black")
        except Exception:
            pass
        self.root.geometry(f"{self.w}x{self.h}+{self.x}+{self.y}")
        self.root.configure(bg="black")
        # Canvas with black bg - black becomes transparent via -transparentcolor
        self.canvas = tk.Canvas(self.root, width=self.w, height=self.h,
                                highlightthickness=0, bd=0, bg="black")
        self.canvas.pack()
        # Create the image item once
        self._base = Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0))
        self._tkimg = ImageTk.PhotoImage(self._base)
        self._img_id = self.canvas.create_image(0, 0, anchor="nw", image=self._tkimg)
        self.root.update()
        self.show()

    def show(self):
        self.root.deiconify()
        self.root.attributes("-topmost", True)

    def hide(self):
        self.root.withdraw()

    def close(self):
        if self.root:
            try:
                self.root.destroy()
            except Exception:
                pass
            self.root = None

    def clear(self):
        """Reset the overlay to fully transparent (no arrows)."""
        self._base = Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0))
        self._redraw()

    def square_px(self, square: str, orientation: str):
        """Center pixel (within overlay coords) of a chess square."""
        file_idx = ord(square[0]) - ord("a")
        rank_idx = int(square[1]) - 1
        sq = self.w / 8.0
        if orientation == "white":
            col = file_idx
            row_from_top = 7 - rank_idx
        else:
            col = 7 - file_idx
            row_from_top = rank_idx
        return (col + 0.5) * sq, (row_from_top + 0.5) * sq

    def arrow(self, frm: str, to: str, color: str, orientation: str = "white",
              width: float = 12.0, head: float = 18.0):
        """Draw an arrow from square `frm` to square `to` on the overlay.
        color: 'green' or 'red'. Returns True if drawn."""
        try:
            (fx, fy), (tx, ty) = self.square_px(frm, orientation), self.square_px(to, orientation)
        except Exception:
            return False
        sq = self.w / 8.0
        dx, dy = tx - fx, ty - fy
        dist = math.hypot(dx, dy) or 1.0
        ux, uy = dx / dist, dy / dist
        # shorten the arrow so it doesn't fully cover the target square
        tx = fx + ux * (dist - 0.18 * sq)
        ty = fy + uy * (dist - 0.18 * sq)
        fill = (0, 200, 0, 220) if color == "green" else (230, 40, 40, 220)
        draw = ImageDraw.Draw(self._base)
        arrow_half = width / 2.0
        px, py = -uy * arrow_half, ux * arrow_half
        draw.line([fx + px, fy + py, tx - ux * head + px, ty - uy * head + py,
                   tx - ux * head - px, ty - uy * head - py,
                   fx - px, fy - py], fill=fill, joint="curve")
        hn = head * 1.2
        hw = width * 0.9
        ax, ay = ux * hn, uy * hn
        bx, by = -uy * hw, ux * hw
        draw.polygon([(tx, ty), (tx - ax + bx, ty - ay + by),
                      (tx - ax - bx, ty - ay - by)], fill=fill)
        self._redraw()
        return True

    def _redraw(self):
        if self.root is None or self._base is None:
            return
        # Update the PhotoImage - PIL RGBA alpha is respected by Tkinter,
        # and -transparentcolor "black" makes black pixels OS-transparent.
        self._tkimg = ImageTk.PhotoImage(self._base)
        self.canvas.itemconfig(self._img_id, image=self._tkimg)
        self.root.update_idletasks()
        self.root.update()

    def update(self):
        if self.root:
            self.root.update()

    def pump(self, delay_ms=50):
        if self.root:
            try:
                self.root.update()
            except Exception:
                pass
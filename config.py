from dataclasses import dataclass, field, asdict
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STOCKFISH_PATH = os.path.join(
    BASE_DIR, "stockfish", "stockfish-windows-x86-64-avx2.exe"
)

# Smartness presets 250-4000. Below ~1320 Stockfish's UCI_Elo floor,
# strength is achieved via Skill Level + shallow depth. Above 3200 is full
# strength (Skill 20, no Elo cap) with increasing depth.
SMARTNESS_PRESETS = {
    250:  {"skill": 0,  "depth": 1,  "elo": None},
    400:  {"skill": 1,  "depth": 1,  "elo": None},
    600:  {"skill": 2,  "depth": 2,  "elo": None},
    800:  {"skill": 3,  "depth": 3,  "elo": None},
    1000: {"skill": 5,  "depth": 4,  "elo": None},
    1200: {"skill": 7,  "depth": 5,  "elo": None},
    1400: {"skill": 9,  "depth": 6,  "elo": None},
    1600: {"skill": 11, "depth": 8,  "elo": None},
    1800: {"skill": 13, "depth": 10, "elo": None},
    2000: {"skill": 15, "depth": 12, "elo": 2000},
    2200: {"skill": 16, "depth": 13, "elo": 2200},
    2400: {"skill": 17, "depth": 14, "elo": 2400},
    2600: {"skill": 18, "depth": 15, "elo": 2600},
    2800: {"skill": 19, "depth": 16, "elo": 2800},
    3000: {"skill": 20, "depth": 18, "elo": 3000},
    3200: {"skill": 20, "depth": 20, "elo": 3190},
    3400: {"skill": 20, "depth": 22, "elo": None},
    3600: {"skill": 20, "depth": 24, "elo": None},
    3800: {"skill": 20, "depth": 26, "elo": None},
    4000: {"skill": 20, "depth": 30, "elo": None},
}


@dataclass
class Config:
    mode: str = "auto"          # "manual" | "auto" | "helper"
    color: str = "black"        # side played by the bot: "white" | "black"
    board_orientation: str = "white"  # side displayed at the bottom
    smartness: int = 900        # target Elo: 250-4000 (250=weakest, 4000=full strength)
    board_roi: list = field(default_factory=list)  # [x, y, w, h] of board on screen
    start_fen: str = None       # set to current position to start mid-game
    auto_trigger: bool = True   # poll screenshots to detect opponent move
    poll_interval: float = 1.0  # seconds between polls
    think_time_ms: int = 800    # max time Stockfish spends per move (0 = no cap)
    move_delay: list = field(default_factory=lambda: [0.3, 0.5])  # random delay [min, max] seconds before playing a confirmed move
    show_attacks: bool = False  # helper mode: draw red arrows for opponent threats
    overlay_enabled: bool = True  # helper mode: draw green best-move arrow + red threats

    def engine_params(self) -> dict:
        if self.smartness in SMARTNESS_PRESETS:
            return SMARTNESS_PRESETS[self.smartness]
        s = self.smartness
        if s >= 3200:
            depth = 20 + int((s - 3200) / 200) * 2
            return {"skill": 20, "depth": min(30, depth), "elo": None}
        if s < 1320:
            # map 250-1320 -> skill 0-10, depth 1-6
            ratio = (s - 250) / (1320 - 250) if s > 250 else 0
            skill = int(ratio * 10)
            depth = 1 + int(ratio * 5)
            return {"skill": max(0, min(10, skill)), "depth": depth, "elo": None}
        return {"skill": 20, "depth": 12, "elo": max(1320, min(3190, s))}

    def save(self, path: str = None):
        path = path or os.path.join(BASE_DIR, "config.json")
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str = None) -> "Config":
        path = path or os.path.join(BASE_DIR, "config.json")
        if os.path.exists(path):
            with open(path) as f:
                return cls(**json.load(f))
        return cls()


if __name__ == "__main__":
    c = Config()
    print("Stockfish exists:", os.path.exists(STOCKFISH_PATH))
    print(c)

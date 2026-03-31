"""Playfield tiles: normal, bonus, penalty, or blocked (wall)."""

from __future__ import annotations

from enum import Enum
from pathlib import Path


class TileKind(str, Enum):
    NORMAL = "."
    BONUS = "+"
    PENALTY = "-"
    BLOCKED = "#"


def tile_kind_index(k: TileKind) -> int:
    """Stable 0..3 index for one-hot state features (NORMAL, BONUS, PENALTY, BLOCKED)."""
    return {
        TileKind.NORMAL: 0,
        TileKind.BONUS: 1,
        TileKind.PENALTY: 2,
        TileKind.BLOCKED: 3,
    }[k]


# Score change when the snake *head enters* the cell (once per visit)
BONUS_SCORE = 3
PENALTY_SCORE = -2


class TileGrid:
    """
    Grid aligned to the same 20px cells as the snake.
    Valid head coordinates are x,y in [20, game_width-40] step 20 (same as engine).
    """

    def __init__(self, rows: list[str], *, cell: int = 20, margin: int = 20):
        if not rows:
            raise ValueError("rows must be non-empty")
        w = len(rows[0])
        if any(len(r) != w for r in rows):
            raise ValueError("All rows must have the same length")
        self._rows = [list(r) for r in rows]
        self.cols = w
        self.rows = len(rows)
        self.cell = cell
        self.margin = margin

    @classmethod
    def all_normal(cls, cols: int, rows: int) -> TileGrid:
        line = TileKind.NORMAL.value * cols
        return cls([line for _ in range(rows)])

    @classmethod
    def from_text(cls, text: str) -> TileGrid:
        lines = [ln.rstrip() for ln in text.strip().splitlines() if ln.strip()]
        return cls(lines)

    @classmethod
    def from_file(cls, path: str | Path) -> TileGrid:
        p = Path(path)
        return cls.from_text(p.read_text(encoding="utf-8"))

    def kind_at_pixel(self, x: float, y: float) -> TileKind:
        col = int((x - self.margin) // self.cell)
        row = int((y - self.margin) // self.cell)
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return TileKind.NORMAL
        ch = self._rows[row][col]
        return _CHAR_TO_KIND.get(ch, TileKind.NORMAL)

    def is_blocked_pixel(self, x: float, y: float) -> bool:
        return self.kind_at_pixel(x, y) == TileKind.BLOCKED

    def score_delta_on_enter(self, x: float, y: float) -> int:
        k = self.kind_at_pixel(x, y)
        if k == TileKind.BONUS:
            return BONUS_SCORE
        if k == TileKind.PENALTY:
            return PENALTY_SCORE
        return 0

    def kind_at_cell(self, row: int, col: int) -> TileKind:
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return TileKind.NORMAL
        ch = self._rows[row][col]
        return _CHAR_TO_KIND.get(ch, TileKind.NORMAL)


_CHAR_TO_KIND = {
    ".": TileKind.NORMAL,
    "+": TileKind.BONUS,
    "-": TileKind.PENALTY,
    "#": TileKind.BLOCKED,
}

from dataclasses import dataclass
from typing import Optional, Type


@dataclass
class SportConfig:
    name: str                       # display name e.g. "Soccer"
    key: str                        # slug e.g. "soccer"
    emoji: str                      # e.g. "⚽"
    court_w: float                  # metres (width)
    court_l: float                  # metres (length)
    players_per_team: int           # max players per half tracked
    ball_coco_class: Optional[int]  # 32=sports_ball, None=no ball (fencing)
    use_tracknet: bool              # True for small fast balls (tennis, table_tennis)
    scoring_cls: type               # class with award(winner), snapshot(), reset()
    court_shape: str                # "rect", "diamond" (baseball), "oval" (cricket)


SPORTS: dict[str, SportConfig] = {}  # filled by each sport module registering itself


class BaseScoring:
    def award(self, winner: str): raise NotImplementedError
    def snapshot(self) -> dict: raise NotImplementedError
    def reset(self): raise NotImplementedError

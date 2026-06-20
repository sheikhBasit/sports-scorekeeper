from .base import SportConfig, SPORTS, BaseScoring


class BasketballScoring(BaseScoring):
    def __init__(self, **kw):
        self._pts = {'A': 0, 'B': 0}
        self._quarter = 1

    def award(self, winner: str, pts: int = 2):
        self._pts[winner] += pts

    def snapshot(self) -> dict:
        return {
            'a': self._pts['A'],
            'b': self._pts['B'],
            'quarter': self._quarter,
            'type': 'points',
        }

    def next_quarter(self):
        self._quarter = min(self._quarter + 1, 4)

    def reset(self):
        self._pts = {'A': 0, 'B': 0}
        self._quarter = 1


SPORTS['basketball'] = SportConfig(
    name='Basketball',
    key='basketball',
    emoji='🏀',
    court_w=15.24,
    court_l=28,
    players_per_team=5,
    ball_coco_class=32,
    use_tracknet=False,
    scoring_cls=BasketballScoring,
    court_shape='rect',
)

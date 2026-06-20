from .base import SportConfig, SPORTS, BaseScoring


class FencingScoring(BaseScoring):
    def __init__(self, target: int = 5, **kw):
        self._touches = {'A': 0, 'B': 0}
        self._target = target
        self._match_over = False
        self._winner = None

    def award(self, winner: str):
        if self._match_over:
            return
        self._touches[winner] += 1
        if self._touches[winner] >= self._target:
            self._match_over = True
            self._winner = winner

    def snapshot(self) -> dict:
        return {
            'a': self._touches['A'],
            'b': self._touches['B'],
            'target': self._target,
            'match_over': self._match_over,
            'winner': self._winner,
            'type': 'touches',
        }

    def reset(self):
        self.__init__(self._target)


SPORTS['fencing'] = SportConfig(
    name='Fencing',
    key='fencing',
    emoji='🤺',
    court_w=2,
    court_l=14,
    players_per_team=1,
    ball_coco_class=None,
    use_tracknet=False,
    scoring_cls=FencingScoring,
    court_shape='rect',
)

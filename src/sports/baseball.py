from .base import SportConfig, SPORTS, BaseScoring


class BaseballScoring(BaseScoring):
    def __init__(self, **kw):
        self._runs = {'A': 0, 'B': 0}
        self._inning = 1
        self._half = 'top'   # top = A batting, bottom = B batting
        self._outs = 0
        self._game_over = False

    def award(self, winner: str):
        """Record a run scored for the given team."""
        if self._game_over:
            return
        self._runs[winner] += 1

    def next_half(self):
        self._outs = 0
        if self._half == 'top':
            self._half = 'bottom'
        else:
            self._half = 'top'
            self._inning += 1
            if self._inning > 9:
                self._game_over = True

    def add_out(self):
        self._outs += 1
        if self._outs >= 3:
            self.next_half()

    def snapshot(self) -> dict:
        return {
            'a': self._runs['A'],
            'b': self._runs['B'],
            'inning': self._inning,
            'half': self._half,
            'outs': self._outs,
            'game_over': self._game_over,
            'type': 'runs',
        }

    def reset(self):
        self.__init__()


SPORTS['baseball'] = SportConfig(
    name='Baseball',
    key='baseball',
    emoji='⚾',
    court_w=27.43,
    court_l=27.43,
    players_per_team=9,
    ball_coco_class=32,
    use_tracknet=False,
    scoring_cls=BaseballScoring,
    court_shape='diamond',
)

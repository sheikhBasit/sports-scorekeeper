from .base import SportConfig, SPORTS, BaseScoring


class VolleyballScoring(BaseScoring):
    def __init__(self, **kw):
        self._pts = {'A': 0, 'B': 0}
        self._sets = {'A': 0, 'B': 0}
        self._set_no = 1
        self._match_over = False
        self._winner = None

    def award(self, winner: str):
        if self._match_over:
            return
        loser = 'B' if winner == 'A' else 'A'
        self._pts[winner] += 1
        target = 15 if self._set_no == 5 else 25
        pw, pl = self._pts[winner], self._pts[loser]
        if pw >= target and pw - pl >= 2:
            self._sets[winner] += 1
            self._pts = {'A': 0, 'B': 0}
            self._set_no += 1
            if self._sets[winner] >= 3:
                self._match_over = True
                self._winner = winner

    def snapshot(self) -> dict:
        return {
            'a': self._sets['A'],
            'b': self._sets['B'],
            'pts_a': self._pts['A'],
            'pts_b': self._pts['B'],
            'set_no': self._set_no,
            'match_over': self._match_over,
            'winner': self._winner,
            'type': 'sets',
        }

    def reset(self):
        self.__init__()


SPORTS['volleyball'] = SportConfig(
    name='Volleyball',
    key='volleyball',
    emoji='🏐',
    court_w=9,
    court_l=18,
    players_per_team=6,
    ball_coco_class=32,
    use_tracknet=False,
    scoring_cls=VolleyballScoring,
    court_shape='rect',

    rf_ball_model='roboflow-100/volleyball-players-detection/2',
)
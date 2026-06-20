from .base import SportConfig, SPORTS, BaseScoring


class TableTennisScoring(BaseScoring):
    def __init__(self, **kw):
        self._pts = {'A': 0, 'B': 0}
        self._games = {'A': 0, 'B': 0}
        self._game_no = 1
        self._match_over = False
        self._winner = None

    def award(self, winner: str):
        if self._match_over:
            return
        loser = 'B' if winner == 'A' else 'A'
        self._pts[winner] += 1
        pw, pl = self._pts[winner], self._pts[loser]
        # Deuce: keep going until one player leads by 2 (both >= 10)
        if pw >= 11 and pw - pl >= 2:
            self._games[winner] += 1
            self._pts = {'A': 0, 'B': 0}
            self._game_no += 1
            if self._games[winner] >= 4:
                self._match_over = True
                self._winner = winner

    def snapshot(self) -> dict:
        return {
            'a': self._games['A'],
            'b': self._games['B'],
            'pts_a': self._pts['A'],
            'pts_b': self._pts['B'],
            'game_no': self._game_no,
            'match_over': self._match_over,
            'winner': self._winner,
            'type': 'games',
        }

    def reset(self):
        self.__init__()


SPORTS['table_tennis'] = SportConfig(
    name='Table Tennis',
    key='table_tennis',
    emoji='🏓',
    court_w=1.525,
    court_l=2.74,
    players_per_team=2,
    ball_coco_class=32,
    use_tracknet=True,
    scoring_cls=TableTennisScoring,
    court_shape='rect',
)

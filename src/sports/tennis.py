from .base import SportConfig, SPORTS, BaseScoring


class TennisScoring(BaseScoring):
    """
    Full tennis scoring — 15/30/40/deuce/advantage, games, sets, match (best of 3).
    Adapted from PadelMatch in padel-scorekeeper/src/scoring_padel.py.
    golden_point is always False for tennis.
    """

    _DISP = ['0', '15', '30', '40']

    def __init__(self, first_server: str = 'A', **kw):
        self.golden_point = False
        self._server = first_server

        # Raw point counts in current game (0-4, where 4 = advantage)
        self._pts = {'A': 0, 'B': 0}
        # Games in current set
        self._games = {'A': 0, 'B': 0}
        # Sets won
        self._sets = {'A': 0, 'B': 0}

        self._in_tiebreak = False
        self._tb_pts = {'A': 0, 'B': 0}
        self._match_over = False
        self._winner = None

    # ── public ────────────────────────────────────────────────────────────────

    def award(self, winner: str):
        if self._match_over:
            return
        if self._in_tiebreak:
            self._tb_pts[winner] += 1
            a, b = self._tb_pts['A'], self._tb_pts['B']
            if max(a, b) >= 7 and abs(a - b) >= 2:
                self._win_set(winner)
        else:
            self._score_point(winner)

    def snapshot(self) -> dict:
        pa, pb = self._pts['A'], self._pts['B']
        if self._in_tiebreak:
            pts_a = str(self._tb_pts['A'])
            pts_b = str(self._tb_pts['B'])
        else:
            pts_a = self._disp(pa, pb)
            pts_b = self._disp(pb, pa)

        return {
            'a': self._games['A'],
            'b': self._games['B'],
            'pts_a': pts_a,
            'pts_b': pts_b,
            'sets_a': self._sets['A'],
            'sets_b': self._sets['B'],
            'server': self._server,
            'in_tiebreak': self._in_tiebreak,
            'match_over': self._match_over,
            'winner': self._winner,
            'type': 'games',
        }

    def reset(self):
        self.__init__(first_server=self._server)

    # ── internals ─────────────────────────────────────────────────────────────

    def _score_point(self, w: str):
        l = 'B' if w == 'A' else 'A'
        pw, pl = self._pts[w], self._pts[l]

        if pw == 4:
            # had advantage → game
            self._win_game(w)
        elif pl == 4:
            # opponent had advantage → back to deuce
            self._pts = {'A': 3, 'B': 3}
        elif pw == 3 and pl == 3:
            # deuce — tennis always uses advantage (golden_point=False)
            self._pts[w] = 4
        elif pw == 3:
            # 40 vs <40 → game
            self._win_game(w)
        else:
            self._pts[w] += 1

    def _win_game(self, w: str):
        self._pts = {'A': 0, 'B': 0}
        self._games[w] += 1
        self._server = 'B' if self._server == 'A' else 'A'

        ga, gb = self._games['A'], self._games['B']
        if ga == 6 and gb == 6:
            self._in_tiebreak = True
        elif max(ga, gb) >= 6 and abs(ga - gb) >= 2:
            self._win_set(w)
        elif max(ga, gb) >= 7:
            self._win_set(w)

    def _win_set(self, w: str):
        self._in_tiebreak = False
        self._tb_pts = {'A': 0, 'B': 0}
        self._games = {'A': 0, 'B': 0}
        self._sets[w] += 1
        if self._sets[w] >= 2:
            self._match_over = True
            self._winner = w

    def _disp(self, p: int, opp: int) -> str:
        if p <= 2:
            return self._DISP[p]
        if p == 4:
            return 'AD'
        # p == 3
        return '40'


SPORTS['tennis'] = SportConfig(
    name='Tennis',
    key='tennis',
    emoji='🎾',
    court_w=10.97,
    court_l=23.77,
    players_per_team=2,
    ball_coco_class=32,
    use_tracknet=True,
    scoring_cls=TennisScoring,
    court_shape='rect',

    rf_ball_model=None,  # TrackNetV3 handles ball
)
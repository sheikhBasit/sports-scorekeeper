from .base import SportConfig, SPORTS, BaseScoring


class CricketScoring(BaseScoring):
    def __init__(self, overs: int = 20, **kw):
        self._runs = {'A': 0, 'B': 0}
        self._wickets = {'A': 0, 'B': 0}
        self._balls = {'A': 0, 'B': 0}
        self._batting = 'A'
        self._innings = 1
        self._total_overs = overs
        self._match_over = False

    def award(self, winner: str, runs: int = 1):
        """Record runs scored by the batting team (winner = batting team)."""
        if self._match_over:
            return
        self._runs[winner] += runs
        self._balls[winner] += 1
        # Check if second innings team has chased successfully
        if self._innings == 2:
            batting = self._batting
            fielding = 'B' if batting == 'A' else 'A'
            if self._runs[batting] > self._runs[fielding]:
                self._match_over = True

    def add_wicket(self):
        if self._match_over:
            return
        self._wickets[self._batting] += 1
        self._balls[self._batting] += 1
        if self._wickets[self._batting] >= 10:
            self._next_innings()

    def _next_innings(self):
        if self._innings == 1:
            self._innings = 2
            self._batting = 'B' if self._batting == 'A' else 'A'
        else:
            self._match_over = True

    def snapshot(self) -> dict:
        bat = self._batting
        return {
            'a': self._runs['A'],
            'b': self._runs['B'],
            'wickets_a': self._wickets['A'],
            'wickets_b': self._wickets['B'],
            'balls_a': self._balls['A'],
            'balls_b': self._balls['B'],
            'batting': bat,
            'innings': self._innings,
            'overs_a': self._balls['A'] // 6,
            'overs_b': self._balls['B'] // 6,
            'match_over': self._match_over,
            'type': 'runs',
        }

    def reset(self):
        self.__init__(self._total_overs)


SPORTS['cricket'] = SportConfig(
    name='Cricket',
    key='cricket',
    emoji='🏏',
    court_w=20,
    court_l=20,
    players_per_team=11,
    ball_coco_class=32,
    use_tracknet=False,
    scoring_cls=CricketScoring,
    court_shape='rect',
)

from .base import SportConfig, SPORTS, BaseScoring


class SoccerScoring(BaseScoring):
    def __init__(self, **kw):
        self._goals = {'A': 0, 'B': 0}

    def award(self, winner: str):
        self._goals[winner] += 1

    def snapshot(self) -> dict:
        return {'a': self._goals['A'], 'b': self._goals['B'], 'type': 'goals'}

    def reset(self):
        self._goals = {'A': 0, 'B': 0}


SPORTS['soccer'] = SportConfig(
    name='Soccer',
    key='soccer',
    emoji='⚽',
    court_w=68,
    court_l=105,
    players_per_team=11,
    ball_coco_class=32,
    use_tracknet=False,
    scoring_cls=SoccerScoring,
    court_shape='rect',
    rf_ball_model='roboflow-100/football-players-detection-3zvbc/2',
)

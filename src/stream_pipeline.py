"""
Sport-agnostic real-time streaming pipeline.

Uses a SportConfig to determine court dimensions, ball detection strategy,
players per team, scoring class, and TrackNet usage.
"""
import cv2
import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO
from collections import deque

from calibrate_court import CourtMapper
from pipeline import filter_to_court, PERSON_CLASS_ID
from sports.base import SportConfig

SPEED_WINDOW       = 9
SPEED_MIN_KMH      = 5
SPEED_MAX_KMH      = 500
SPEED_DECAY_FRAMES = 60
RALLY_GAP_FRAMES   = 45
MIN_RALLY_FRAMES   = 15


class StreamPipeline:
    def __init__(self, sport: SportConfig, mapper: CourtMapper,
                 model_path='yolo11n.pt', conf=0.25, court_margin=0.4,
                 tracknet_repo=None, tracknet_ckpt=None, inpaint_ckpt=None,
                 fps=10.0, scoring_kwargs=None, names=None):

        self.sport = sport
        self.mapper = mapper
        self.fps = fps
        self.names = names or {'A': 'A', 'B': 'B'}

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.model = YOLO(model_path)
        self.conf = conf
        self.court_margin = court_margin
        self.tracker = sv.ByteTrack()

        # ball tracking
        self._tracknet = None
        self._frame_buf: deque = deque(maxlen=3)
        if tracknet_repo and sport.use_tracknet:
            self._tracknet = _TrackNetInference(
                tracknet_repo, tracknet_ckpt, inpaint_ckpt, device)

        # scoring
        self._match = sport.scoring_cls(**(scoring_kwargs or {}))

        # speed + rally state
        self._ball_history: deque = deque(maxlen=SPEED_WINDOW)
        self._last_speed: float = None
        self._last_ball_idx: int = -1
        self._rally_frames: int = 0
        self._invisible_frames: int = 0
        self._rally_active: bool = False

    def process(self, frame: np.ndarray, frame_idx: int) -> dict:
        result = {
            'frame': frame_idx,
            'players': [],
            'ball': None,
            'score': self._score_snap(),
            'speed_kmh': None,
            'rally_active': self._rally_active,
            'sport': self.sport.key,
        }

        # player detection — use supervision annotators for consistency
        classes = [PERSON_CLASS_ID]
        if self.sport.ball_coco_class is not None and not self._tracknet:
            classes.append(self.sport.ball_coco_class)

        preds = self.model(frame, conf=self.conf, device=self.device,
                           verbose=False, classes=classes)[0]
        det_all = sv.Detections.from_ultralytics(preds)

        # players
        det_p = det_all[det_all.class_id == PERSON_CLASS_ID]
        det_p = filter_to_court(det_p, self.mapper, self.court_margin)
        if len(det_p):
            feet_m = self.mapper.to_metres(
                np.column_stack([(det_p.xyxy[:, 0] + det_p.xyxy[:, 2]) / 2,
                                  det_p.xyxy[:, 3]]))
            areas = ((det_p.xyxy[:, 2] - det_p.xyxy[:, 0]) *
                     (det_p.xyxy[:, 3] - det_p.xyxy[:, 1]))
            keep = []
            n = self.sport.players_per_team
            for mask in (feet_m[:, 1] < self.mapper.court_l / 2,
                         feet_m[:, 1] >= self.mapper.court_l / 2):
                idxs = np.where(mask)[0]
                if len(idxs):
                    top = idxs[np.argsort(areas[idxs])[-n:]]
                    keep.extend(top.tolist())
            if keep:
                det_p = det_p[np.array(keep)]
        det_p = self.tracker.update_with_detections(det_p)
        if len(det_p):
            feet = np.column_stack([(det_p.xyxy[:, 0] + det_p.xyxy[:, 2]) / 2,
                                     det_p.xyxy[:, 3]])
            mets = self.mapper.to_metres(feet)
            for tid, m in zip(det_p.tracker_id, mets):
                result['players'].append({
                    'tid': int(tid),
                    'x_m': round(float(m[0]), 3),
                    'y_m': round(float(m[1]), 3),
                    'half': 'far' if m[1] < self.mapper.court_l / 2 else 'near',
                })

        # ball via TrackNetV3
        ball_visible = False
        self._frame_buf.append((frame_idx, frame))
        if self._tracknet and len(self._frame_buf) == 3:
            bx, by, vis = self._tracknet.predict(list(self._frame_buf))
            if vis:
                ball_visible = True
                bm = self.mapper.to_metres([[bx, by]])[0]
                result['ball'] = {
                    'x': round(bx, 1), 'y': round(by, 1),
                    'x_m': round(float(bm[0]), 3),
                    'y_m': round(float(bm[1]), 3),
                }
                self._ball_history.append((frame_idx, bx, by))
                self._last_ball_idx = frame_idx
                self._update_speed()

        # ball via YOLO class (non-tracknet sports)
        elif self.sport.ball_coco_class is not None and not self._tracknet:
            det_b = det_all[det_all.class_id == self.sport.ball_coco_class]
            if len(det_b):
                best = int(np.argmax(det_b.confidence))
                bx1, by1, bx2, by2 = det_b.xyxy[best]
                bx, by = (bx1 + bx2) / 2, (by1 + by2) / 2
                bm = self.mapper.to_metres([[bx, by]])[0]
                result['ball'] = {
                    'x': round(float(bx), 1), 'y': round(float(by), 1),
                    'x_m': round(float(bm[0]), 3),
                    'y_m': round(float(bm[1]), 3),
                }
                ball_visible = True
                self._ball_history.append((frame_idx, bx, by))
                self._last_ball_idx = frame_idx
                self._update_speed()

        if (self._last_speed is not None and
                frame_idx - self._last_ball_idx < SPEED_DECAY_FRAMES):
            result['speed_kmh'] = round(self._last_speed, 1)

        # rally state machine
        if ball_visible:
            self._rally_frames += 1
            self._invisible_frames = 0
            if self._rally_frames >= MIN_RALLY_FRAMES and not self._rally_active:
                self._rally_active = True
        else:
            self._invisible_frames += 1
            if self._rally_active and self._invisible_frames >= RALLY_GAP_FRAMES:
                self._end_rally()

        result['rally_active'] = self._rally_active
        result['score'] = self._score_snap()
        return result

    def _update_speed(self):
        if len(self._ball_history) >= 3:
            fi, fx, fy = self._ball_history[0]
            li, lx, ly = self._ball_history[-1]
            dt = (li - fi) / self.fps
            if dt > 0:
                kmh = self.mapper.speed_kmh((fx, fy), (lx, ly), dt)
                if SPEED_MIN_KMH <= kmh < SPEED_MAX_KMH:
                    self._last_speed = kmh

    def _end_rally(self):
        if self._ball_history:
            _, lx, ly = self._ball_history[-1]
            lm = self.mapper.to_metres([[lx, ly]])[0]
            winner = 'B' if lm[1] < self.mapper.court_l / 2 else 'A'
        else:
            winner = 'A'
        self._match.award(winner)
        self._reset_rally()

    def award_point(self, winner: str):
        self._match.award(winner)
        self._reset_rally()

    def _reset_rally(self):
        self._rally_active = False
        self._rally_frames = 0
        self._invisible_frames = 0
        self._ball_history.clear()
        self._last_speed = None

    def _score_snap(self) -> dict:
        s = self._match.snapshot()
        return {**s, 'name_a': self.names['A'], 'name_b': self.names['B']}


class _TrackNetInference:
    def __init__(self, repo: str, tracknet_ckpt: str, inpaint_ckpt: str, device: str):
        import sys as _sys
        _sys.path.insert(0, repo)
        try:
            from Model import TrackNet, InpaintNet
        except ModuleNotFoundError:
            from model import TrackNet, InpaintNet
        import torch as _torch

        self.device = device
        self.H, self.W = 288, 512
        self.sigma = 2.5

        self.tracknet = TrackNet(in_dim=9, out_dim=3)
        self.tracknet.load_state_dict(
            _torch.load(tracknet_ckpt, map_location=device)['model'])
        self.tracknet.to(device).eval()

        self.inpaintnet = InpaintNet()
        self.inpaintnet.load_state_dict(
            _torch.load(inpaint_ckpt, map_location=device)['model'])
        self.inpaintnet.to(device).eval()

    def predict(self, frame_buf: list) -> tuple:
        import torch as _torch
        frames = [cv2.resize(f, (self.W, self.H)) for _, f in frame_buf]
        imgs = np.concatenate(
            [cv2.cvtColor(f, cv2.COLOR_BGR2RGB).transpose(2, 0, 1) / 255.0
             for f in frames], axis=0).astype(np.float32)
        inp = _torch.from_numpy(imgs).unsqueeze(0).to(self.device)
        with _torch.no_grad():
            heatmap = self.tracknet(inp)[0, -1].cpu().numpy()
        y, x = np.unravel_index(heatmap.argmax(), heatmap.shape)
        if float(heatmap[y, x]) < 0.5:
            return 0.0, 0.0, False
        _, orig = frame_buf[-1]
        oh, ow = orig.shape[:2]
        return float(x / self.W * ow), float(y / self.H * oh), True

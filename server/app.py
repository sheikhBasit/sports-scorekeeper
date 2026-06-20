"""
FastAPI server for sports-scorekeeper (sport-agnostic).

POST /calibrate   — init CourtMapper + StreamPipeline from 4 corners + sport
POST /frame       — JPEG frame → inference → broadcast to display WS clients
POST /award       — manually award a point to team A or B
WS   /ws          — display clients
GET  /status      — health check
GET  /history     — game records
POST /history     — save game record
GET  /            — serves webapp/index.html
"""
import asyncio
import base64
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import supervision as sv
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from calibrate_court import CourtMapper, make_court_pts
from stream_pipeline import StreamPipeline
from sports.base import SPORTS

app = FastAPI(title='Sports Scorekeeper')
app.add_middleware(CORSMiddleware, allow_origins=['*'],
                   allow_methods=['*'], allow_headers=['*'])

WEBAPP       = Path(__file__).parent.parent / 'webapp' / 'index.html'
HISTORY_FILE = Path(__file__).parent.parent / 'data' / 'history.json'
HISTORY_FILE.parent.mkdir(exist_ok=True)

_pipeline: Optional[StreamPipeline] = None
_frame_idx: int = 0
_last_result: Optional[dict] = None
_display_clients: list[WebSocket] = []
_executor = ThreadPoolExecutor(max_workers=1)

# Player colours — BGR for cv2; also used via sv.Color
_NEAR_BGR = (38, 149, 255)   # orange in BGR
_FAR_BGR  = (60, 220, 60)    # green  in BGR
_BALL_BGR = (255, 240, 60)   # yellow in BGR

_COLOR_NEAR = sv.Color(r=255, g=149, b=38)
_COLOR_FAR  = sv.Color(r=60,  g=220, b=60)


# ---------------------------------------------------------------------------
# Score string helpers
# ---------------------------------------------------------------------------

def _score_string(result: dict) -> str:
    sc = result.get('score', {})
    score_type = sc.get('type', 'goals')
    na = sc.get('name_a', 'A')
    nb = sc.get('name_b', 'B')

    if score_type == 'goals':
        ga, gb = sc.get('a', 0), sc.get('b', 0)
        return f"{na} {ga} – {gb} {nb}"

    elif score_type == 'points':
        # basketball, etc.
        pa, pb = sc.get('a', 0), sc.get('b', 0)
        quarter = sc.get('quarter', sc.get('period', ''))
        suffix = f"  Q{quarter}" if quarter else ''
        return f"{na} {pa} – {pb} {nb}{suffix}"

    elif score_type == 'sets':
        # volleyball: sets + per-set game scores
        sets_a = sc.get('sets_a', 0)
        sets_b = sc.get('sets_b', 0)
        set_scores = sc.get('set_scores', [])   # list of (a_pts, b_pts) per finished set
        current_a = sc.get('current_a', 0)
        current_b = sc.get('current_b', 0)
        current_set = sc.get('current_set', len(set_scores) + 1)
        # summarise finished sets as "25-18" pairs
        if set_scores:
            history = ' / '.join(f"{a}-{b}" for a, b in set_scores)
            return f"{na} {sets_a} | {history} – {current_a}-{current_b} | {sets_b} {nb}  Set {current_set}"
        return f"{na} {sets_a} | {current_a}-{current_b} | {sets_b} {nb}  Set {current_set}"

    elif score_type == 'games':
        # table tennis / badminton
        games_a = sc.get('games_a', 0)
        games_b = sc.get('games_b', 0)
        game_scores = sc.get('game_scores', [])
        current_a = sc.get('current_a', 0)
        current_b = sc.get('current_b', 0)
        current_game = sc.get('current_game', len(game_scores) + 1)
        if game_scores:
            history = ' / '.join(f"{a}-{b}" for a, b in game_scores)
            return f"{na} {games_a} | {history} – {current_a}-{current_b} | {games_b} {nb}  G{current_game}"
        return f"{na} {games_a} | {current_a}-{current_b} | {games_b} {nb}  G{current_game}"

    elif score_type == 'tennis':
        sets_a  = sc.get('sets_a', 0)
        sets_b  = sc.get('sets_b', 0)
        games_a = sc.get('games_a', 0)
        games_b = sc.get('games_b', 0)
        pts_a   = sc.get('pts_a', '0')
        pts_b   = sc.get('pts_b', '0')
        return f"{na} {sets_a}/{games_a}/{pts_a} – {pts_b}/{games_b}/{sets_b} {nb}"

    elif score_type == 'touches':
        # fencing
        ta, tb = sc.get('a', 0), sc.get('b', 0)
        target = sc.get('target', 5)
        return f"{na} {ta} – {tb} {nb}  (target {target})"

    elif score_type == 'runs':
        sport_key = sc.get('sport_key', '')
        if sport_key == 'cricket':
            # "A 142/3 (18.2) – B 120/5 (20)"
            runs_a   = sc.get('runs_a', 0)
            wkts_a   = sc.get('wickets_a', 0)
            overs_a  = sc.get('overs_a', '')
            runs_b   = sc.get('runs_b', 0)
            wkts_b   = sc.get('wickets_b', 0)
            overs_b  = sc.get('overs_b', '')
            a_str = f"{runs_a}/{wkts_a}"
            b_str = f"{runs_b}/{wkts_b}"
            if overs_a:
                a_str += f" ({overs_a})"
            if overs_b:
                b_str += f" ({overs_b})"
            return f"{na} {a_str} – {b_str} {nb}"
        else:
            # baseball
            ra, rb = sc.get('a', 0), sc.get('b', 0)
            inning  = sc.get('inning', '')
            half    = sc.get('half', '')    # 'top' / 'bot'
            outs    = sc.get('outs', '')
            suffix_parts = []
            if inning:
                suffix_parts.append(f"Inn{inning}")
            if half:
                suffix_parts.append(half)
            if outs != '':
                suffix_parts.append(f"{outs} out")
            suffix = '  ' + ' '.join(suffix_parts) if suffix_parts else ''
            return f"{na} {ra} – {rb} {nb}{suffix}"

    # fallback
    return f"{na} ? – ? {nb}"


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

def _annotate(frame: np.ndarray, result: dict) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    H_inv = None
    if _pipeline is not None:
        try:
            H_inv = np.linalg.inv(_pipeline.mapper.H)
        except Exception:
            pass

    players = result.get('players', [])
    near_count = far_count = 0
    labels = []
    xyxy_list = []
    colors = []

    for p in players:
        is_near = p['half'] == 'near'
        if is_near:
            near_count += 1
            label = f"A{near_count}"
            color = _COLOR_NEAR
        else:
            far_count += 1
            label = f"B{far_count}"
            color = _COLOR_FAR
        labels.append(label)
        colors.append(color)

        # Recover pixel position from metres via inverse homography, or use
        # stored pixel coords if the pipeline put them in the result dict.
        if 'px' in p and 'py' in p:
            px, py = int(p['px']), int(p['py'])
        elif H_inv is not None:
            pt = cv2.perspectiveTransform(
                np.array([[[p['x_m'], p['y_m']]]], dtype=np.float32), H_inv)[0][0]
            px, py = int(pt[0]), int(pt[1])
        else:
            continue

        # Synthesise a 40×80 bounding box centred on the feet position.
        x1 = px - 20
        y1 = py - 80
        x2 = px + 20
        y2 = py
        xyxy_list.append([x1, y1, x2, y2])

    if xyxy_list:
        xyxy = np.array(xyxy_list, dtype=np.float32)
        detections = sv.Detections(xyxy=xyxy)

        # Draw bounding boxes — one colour per team.
        for i, (bbox, color) in enumerate(zip(xyxy_list, colors)):
            single = sv.Detections(xyxy=np.array([bbox], dtype=np.float32))
            sv.BoundingBoxAnnotator(color=color, thickness=2).annotate(
                scene=out, detections=single)

        # Draw labels.
        sv.LabelAnnotator(
            text_scale=0.5,
            text_thickness=1,
            text_padding=4,
        ).annotate(scene=out, detections=detections, labels=labels)

    # Ball — sv has no single-point annotator; use cv2.
    ball = result.get('ball')
    if ball:
        bx, by = int(ball['x']), int(ball['y'])
        cv2.circle(out, (bx, by), 10, _BALL_BGR, -1)
        cv2.circle(out, (bx, by), 12, (255, 255, 255), 1)

    # Ball / shuttle speed.
    spd = result.get('speed_kmh')
    if spd:
        txt = f'{spd:.0f} km/h'
        cv2.putText(out, txt, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 4)
        cv2.putText(out, txt, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, _BALL_BGR, 2)

    # Score banner.
    txt = _score_string(result)
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(out, (0, 0), (tw + 16, th + 16), (0, 0, 0), -1)
    cv2.putText(out, txt, (8, th + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)

    return out


# ---------------------------------------------------------------------------
# Frame encode + broadcast
# ---------------------------------------------------------------------------

def _encode_frame(frame: np.ndarray) -> str:
    h, w = frame.shape[:2]
    scale = min(1.0, 640 / w)
    if scale < 1.0:
        frame = cv2.resize(frame, (640, int(h * scale)))
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
    return base64.b64encode(buf).decode()


async def _broadcast(result: dict, raw_frame: Optional[np.ndarray] = None):
    payload = dict(result)
    if raw_frame is not None and _display_clients:
        payload['frame_b64'] = _encode_frame(_annotate(raw_frame, result))
    msg = json.dumps(payload)
    dead = []
    for ws in _display_clients:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _display_clients.remove(ws)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class CalibrateRequest(BaseModel):
    corners: list[list[float]]
    sport: str = 'soccer'
    frame_w: int = 854
    frame_h: int = 480
    model: str = 'yolo11n.pt'
    conf: float = 0.25
    court_margin: float = 0.5
    init_score: Optional[str] = None
    first_server: str = 'A'
    name_a: str = 'A'
    name_b: str = 'B'
    fps: float = 10.0
    golden_point: bool = False
    # TrackNetV3 — if not sent, fall back to env vars set by kernel.py
    tracknet_repo: Optional[str] = None
    tracknet_ckpt: Optional[str] = None
    inpaint_ckpt:  Optional[str] = None


class AwardRequest(BaseModel):
    winner: str   # 'A' or 'B'


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post('/calibrate')
async def calibrate(req: CalibrateRequest):
    global _pipeline, _frame_idx

    if req.sport not in SPORTS:
        raise HTTPException(400, f"Unknown sport '{req.sport}'. Available: {list(SPORTS)}")

    sport_config = SPORTS[req.sport]
    court_pts = make_court_pts(sport_config.court_w, sport_config.court_l)

    corners = np.float32(req.corners)
    H, _ = cv2.findHomography(corners, court_pts)
    if H is None:
        raise HTTPException(400, 'Homography failed — corners likely collinear')

    mapper = CourtMapper(H, corners, (req.frame_w, req.frame_h))

    init = None
    if req.init_score:
        a, b = (int(x) for x in req.init_score.split(','))
        init = {'A': a, 'B': b}

    tn_repo = req.tracknet_repo or os.environ.get('TRACKNET_REPO')
    tn_ckpt = req.tracknet_ckpt or os.environ.get('TRACKNET_CKPT')
    ip_ckpt = req.inpaint_ckpt  or os.environ.get('INPAINT_CKPT')

    if tn_repo:
        print(f'[calibrate] TrackNetV3 enabled — repo={tn_repo}', flush=True)
    else:
        print('[calibrate] TrackNetV3 not configured — ball tracking disabled', flush=True)

    _pipeline = StreamPipeline(
        sport=sport_config,
        mapper=mapper,
        model_path=req.model,
        conf=req.conf,
        court_margin=req.court_margin,
        fps=req.fps,
        tracknet_repo=tn_repo,
        tracknet_ckpt=tn_ckpt,
        inpaint_ckpt=ip_ckpt,
        initial_score=init,
        first_server=req.first_server,
        names={'A': req.name_a, 'B': req.name_b},
        golden_point=req.golden_point,
    )
    _frame_idx = 0
    print(f'[calibrate] ready — sport={req.sport}  corners={req.corners}', flush=True)
    return {'status': 'ok', 'sport': req.sport}


@app.post('/frame')
async def process_frame(file: UploadFile = File(...)):
    global _frame_idx, _last_result
    if _pipeline is None:
        raise HTTPException(400, 'Not calibrated')
    data = await file.read()
    arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, 'Could not decode image')
    idx = _frame_idx
    _frame_idx += 1
    if _executor._work_queue.qsize() > 0:
        return {'status': 'dropped', 'frame': idx}
    h, w = frame.shape[:2]
    if w > 640:
        frame = cv2.resize(frame, (640, int(h * 640 / w)))
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(_executor, _pipeline.process, frame, idx)
    _last_result = result
    await _broadcast(result, frame)
    return result


@app.post('/award')
async def award_point(req: AwardRequest):
    if _pipeline is None:
        raise HTTPException(400, 'Not calibrated')
    winner = req.winner.upper()
    if winner not in ('A', 'B'):
        raise HTTPException(400, "winner must be 'A' or 'B'")
    result = _pipeline.award_point(winner)
    _last_result = result
    await _broadcast(result)
    return result


@app.websocket('/ws')
async def display_ws(websocket: WebSocket):
    await websocket.accept()
    _display_clients.append(websocket)
    if _last_result:
        await websocket.send_text(json.dumps(_last_result))
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in _display_clients:
            _display_clients.remove(websocket)


@app.get('/status')
async def status():
    return {
        'calibrated': _pipeline is not None,
        'sport': getattr(getattr(_pipeline, 'sport', None), 'key', None),
        'frame': _frame_idx,
        'display_clients': len(_display_clients),
    }


@app.get('/history')
async def get_history():
    if HISTORY_FILE.exists():
        return json.loads(HISTORY_FILE.read_text())
    return []


@app.post('/history')
async def save_game(game: dict):
    history = json.loads(HISTORY_FILE.read_text()) if HISTORY_FILE.exists() else []
    history.insert(0, game)
    HISTORY_FILE.write_text(json.dumps(history, indent=2))
    return {'saved': True, 'total': len(history)}


@app.get('/')
async def root():
    if WEBAPP.exists():
        return FileResponse(WEBAPP, media_type='text/html')
    return JSONResponse({'status': 'webapp not found'})

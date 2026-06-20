"""
Sports-scorekeeper pipeline test on Kaggle.

Ball detection priority per sport:
  tennis / table_tennis → TrackNetV3  (skipped in this test — needs weights)
  soccer, basketball, volleyball, baseball, cricket → Roboflow Universe model
  fencing → no ball (pose-based touch detection, skipped here)
  all → YOLO class 32 fallback if Roboflow key absent or model errors

Output: annotated video + preview JPEG + stats JSON per sport.
"""
import json, os, subprocess, sys, time
import cv2
import numpy as np

WORK = "/kaggle/working"
REPO = f"{WORK}/sports-scorekeeper"
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # force CPU (Kaggle P100 sm_60)

def sh(cmd, check=True):
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=check)

# ── 1. deps + repo ────────────────────────────────────────────────────────────
sh("pip -q install ultralytics 'supervision>=0.25.0' 'inference>=0.9.0' yt-dlp")
if not os.path.isdir(REPO):
    sh(f"git clone https://github.com/sheikhBasit/sports-scorekeeper {REPO}")
else:
    sh(f"git -C {REPO} pull --ff-only")

sys.path.insert(0, f"{REPO}/src")

from ultralytics import YOLO
import supervision as sv

model = YOLO("yolo11n.pt")
PERSON = 0
BALL   = 32

# ── 2. Roboflow API key ───────────────────────────────────────────────────────
RF_KEY = None
try:
    from kaggle_secrets import UserSecretsClient
    RF_KEY = UserSecretsClient().get_secret("ROBOFLOW_API_KEY")
    print(f"[roboflow] API key loaded", flush=True)
except Exception as _e:
    print(f"[roboflow] no key ({_e}) — using YOLO class 32 fallback", flush=True)

_BALL_CLASS_NAMES = {
    'ball', 'football', 'basketball', 'volleyball', 'baseball',
    'cricket ball', 'cricket-ball', 'tennis ball', 'ping pong ball',
    'ping-pong-ball', 'shuttlecock', 'puck',
}

def load_rf_model(model_id):
    if not RF_KEY or not model_id:
        return None
    try:
        from inference import get_model
        m = get_model(model_id, api_key=RF_KEY)
        print(f"  [roboflow] {model_id} ready", flush=True)
        return m
    except Exception as e:
        print(f"  [roboflow] {model_id} failed: {e}", flush=True)
        return None

def rf_detect_ball(rf_model, frame):
    """Run Roboflow model, return (cx, cy) of highest-conf ball or None."""
    try:
        results = rf_model.infer(frame)[0]
        det = sv.Detections.from_inference(results)
        if len(det) == 0:
            return None
        # filter to ball classes if class names available
        if hasattr(det, 'data') and 'class_name' in det.data:
            names = np.array([n.lower() for n in det.data['class_name']])
            mask = np.array([n in _BALL_CLASS_NAMES for n in names])
            if mask.any():
                det = det[mask]
        if len(det) == 0:
            return None
        best = int(np.argmax(det.confidence))
        x1, y1, x2, y2 = det.xyxy[best]
        return (x1 + x2) / 2, (y1 + y2) / 2, det[best:best+1]
    except Exception as e:
        return None

# ── 3. sport definitions ──────────────────────────────────────────────────────
# (key, emoji, yt_query, max_dur_s, roboflow_model_id)
SPORTS = [
    ("soccer",       "⚽", "soccer football match broadcast full pitch 2024 highlights",  300,
     "roboflow-100/football-players-detection-3zvbc/2"),
    ("basketball",   "🏀", "NBA basketball game highlights overhead court 2024",           300,
     "roboflow-100/basketball-players-detection/1"),
    ("volleyball",   "🏐", "volleyball match highlights side view full court 2024",        300,
     "roboflow-100/volleyball-players-detection/2"),
    ("tennis",       "🎾", "tennis match highlights side court overhead 2024",             300,
     None),   # TrackNetV3 in production; YOLO fallback here
    ("table_tennis", "🏓", "table tennis ping pong match highlights side view 2024",       300,
     None),   # TrackNetV3 in production
    ("fencing",      "🤺", "fencing épée foil match competition 2024",                    300,
     None),   # pose-based; no ball model
    ("baseball",     "⚾", "baseball game highlights broadcast 2024",                      300,
     "roboflow-100/baseball-detection/2"),
    ("cricket",      "🏏", "cricket match highlights broadcast side view 2024",            300,
     "roboflow-100/cricket-ball-detection/1"),
]

N_FRAMES   = 80
FRAME_STEP = 3

# ── 4. annotators ─────────────────────────────────────────────────────────────
box_ann   = sv.BoundingBoxAnnotator(thickness=2)
label_ann = sv.LabelAnnotator(text_scale=0.5, text_thickness=1,
                               text_padding=3, text_position=sv.Position.TOP_LEFT)
trace_ann = sv.TraceAnnotator(thickness=2, trace_length=20,
                               position=sv.Position.BOTTOM_CENTER)
ball_ann  = sv.BoundingBoxAnnotator(thickness=3, color=sv.Color.from_hex('#00FFFF'))

# ── 5. per-sport test ─────────────────────────────────────────────────────────
summary = []

for sport_key, emoji, query, max_dur, rf_model_id in SPORTS:
    print(f"\n{'='*60}", flush=True)
    print(f"{emoji}  {sport_key.upper()}", flush=True)
    print(f"{'='*60}", flush=True)

    rf_model = load_rf_model(rf_model_id)
    use_rf   = rf_model is not None

    vid_path = f"{WORK}/{sport_key}.mp4"
    if not os.path.exists(vid_path):
        print(f"[download] searching: {query}", flush=True)
        dl_cmd = (
            f'yt-dlp -f "best[height<=480]" '
            f'--match-filter "duration < {max_dur}" '
            f'--no-playlist -o "{vid_path}" '
            f'"ytsearch1:{query}"'
        )
        result = subprocess.run(dl_cmd, shell=True, capture_output=True, text=True)
        if not os.path.exists(vid_path):
            print(f"[SKIP] download failed: {result.stderr[-300:]}", flush=True)
            summary.append({"sport": sport_key, "status": "download_failed"})
            continue
        print(f"[download] saved {os.path.getsize(vid_path)//1024} KB", flush=True)

    cap = cv2.VideoCapture(vid_path)
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 25.0
    TOT = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[video] {W}x{H} @ {FPS:.1f}fps  {TOT} frames ({TOT/FPS:.0f}s)", flush=True)
    print(f"[ball detector] {'Roboflow: ' + rf_model_id if use_rf else 'YOLO class 32'}", flush=True)

    start_f  = min(int(10 * FPS), TOT // 4)
    end_f    = min(start_f + N_FRAMES * FRAME_STEP, TOT)
    sample_f = list(range(start_f, end_f, FRAME_STEP))[:N_FRAMES]

    out_path = f"{WORK}/{sport_key}_annotated.mp4"
    writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                FPS / FRAME_STEP, (W, H))

    tracker      = sv.ByteTrack()
    persons_tot  = 0
    balls_tot    = 0
    rf_balls_tot = 0
    t0           = time.time()

    for i, fi in enumerate(sample_f):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            break

        # person + YOLO-ball detection
        yolo_classes = [PERSON] if use_rf else [PERSON, BALL]
        preds    = model(frame, conf=0.25, verbose=False, classes=yolo_classes)[0]
        det_all  = sv.Detections.from_ultralytics(preds)

        det_p = det_all[det_all.class_id == PERSON]
        det_p = tracker.update_with_detections(det_p)

        # ball detection
        det_b_rf   = None
        det_b_yolo = None
        if use_rf:
            hit = rf_detect_ball(rf_model, frame)
            if hit:
                _, _, det_b_rf = hit
                rf_balls_tot += 1
        else:
            det_b_yolo = det_all[det_all.class_id == BALL]

        persons_tot += len(det_p)
        balls_tot   += int(det_b_rf is not None or
                           (det_b_yolo is not None and len(det_b_yolo) > 0))

        # annotate
        annotated = frame.copy()
        labels_p = [f"P{int(tid)%99:02d}" for tid in (det_p.tracker_id or [])]
        if len(det_p):
            annotated = box_ann.annotate(annotated, det_p)
            annotated = label_ann.annotate(annotated, det_p, labels=labels_p)
            annotated = trace_ann.annotate(annotated, det_p)

        if det_b_rf is not None:
            annotated = ball_ann.annotate(annotated, det_b_rf)
            bx, by, _ = rf_detect_ball(rf_model, frame) or (0, 0, None)
            src = "RF"
            cv2.putText(annotated, f"Ball[{src}]", (int(bx)+6, int(by)-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        elif det_b_yolo is not None and len(det_b_yolo):
            annotated = ball_ann.annotate(annotated, det_b_yolo)
            annotated = label_ann.annotate(annotated, det_b_yolo,
                                           labels=["Ball[YOLO]"]*len(det_b_yolo))

        cv2.putText(annotated, f"{emoji} {sport_key}  f{fi}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
        cv2.putText(annotated, f"{emoji} {sport_key}  f{fi}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        writer.write(annotated)
        if i == 0:
            cv2.imwrite(f"{WORK}/{sport_key}_preview.jpg", annotated)

        if (i+1) % 20 == 0:
            b_info = (f"rf_ball={rf_balls_tot}" if use_rf
                      else f"yolo_ball={balls_tot}")
            print(f"  [{i+1:3d}/{len(sample_f)}] persons={len(det_p)}  {b_info}", flush=True)

    cap.release()
    writer.release()

    n = len(sample_f)
    elapsed = time.time() - t0
    avg_p = persons_tot / n if n else 0
    avg_b = balls_tot   / n if n else 0

    print(f"\n  avg persons/frame : {avg_p:.1f}", flush=True)
    print(f"  avg balls/frame   : {avg_b:.2f}  "
          f"({'Roboflow' if use_rf else 'YOLO'})", flush=True)
    print(f"  time              : {elapsed:.1f}s  ({elapsed/n:.2f}s/frame)", flush=True)

    summary.append({
        "sport":                sport_key,
        "status":               "ok",
        "ball_detector":        f"roboflow:{rf_model_id}" if use_rf else "yolo:class32",
        "frames":               n,
        "avg_persons_per_frame": round(avg_p, 2),
        "avg_balls_per_frame":  round(avg_b, 3),
        "elapsed_s":            round(elapsed, 1),
    })

# ── 6. summary ────────────────────────────────────────────────────────────────
print(f"\n{'='*60}", flush=True)
print("SUMMARY", flush=True)
print(f"{'='*60}", flush=True)
for s in summary:
    if s["status"] != "ok":
        print(f"  {s['sport']:12s}  FAILED ({s['status']})", flush=True)
    else:
        print(f"  {s['sport']:12s}  persons={s['avg_persons_per_frame']:5.1f}/f  "
              f"ball={s['avg_balls_per_frame']:.3f}/f  "
              f"[{s['ball_detector']}]  ({s['elapsed_s']}s)", flush=True)

out_json = f"{WORK}/sports_test_results.json"
with open(out_json, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nResults saved to {out_json}", flush=True)
print("Preview images: /kaggle/working/<sport>_preview.jpg", flush=True)
print("Annotated videos: /kaggle/working/<sport>_annotated.mp4", flush=True)

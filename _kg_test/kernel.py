"""
Sports-scorekeeper pipeline test on Kaggle.

For each sport: downloads a short broadcast clip, runs YOLO person + ball
detection with supervision annotators, saves an annotated video + stats JSON.
All output lands in /kaggle/working/ for download.

No court calibration is done here — this tests raw detection fidelity per sport.
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
sh("pip -q install ultralytics 'supervision>=0.25.0' yt-dlp")
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

# ── 2. sport definitions ──────────────────────────────────────────────────────
# Each entry: (key, emoji, youtube_search_query, max_duration_s)
# Queries chosen for broadcast angles that show the full court + all players.
SPORTS = [
    ("soccer",       "⚽", "soccer football match broadcast full pitch 2024 highlights",  300),
    ("basketball",   "🏀", "NBA basketball game highlights overhead court 2024",           300),
    ("volleyball",   "🏐", "volleyball match highlights side view full court 2024",        300),
    ("tennis",       "🎾", "tennis match highlights side court overhead 2024",             300),
    ("table_tennis", "🏓", "table tennis ping pong match highlights side view 2024",       300),
    ("fencing",      "🤺", "fencing épée foil match competition 2024",                    300),
    ("baseball",     "⚾", "baseball game highlights broadcast 2024",                      300),
    ("cricket",      "🏏", "cricket match highlights broadcast side view 2024",            300),
]

N_FRAMES = 80      # frames to process per sport
FRAME_STEP = 3     # sample every Nth frame (reduces runtime)

# ── 3. annotators ─────────────────────────────────────────────────────────────
box_ann   = sv.BoundingBoxAnnotator(thickness=2)
label_ann = sv.LabelAnnotator(text_scale=0.5, text_thickness=1,
                               text_padding=3, text_position=sv.Position.TOP_LEFT)
trace_ann = sv.TraceAnnotator(thickness=2, trace_length=20,
                               position=sv.Position.BOTTOM_CENTER)

# ── 4. per-sport test ─────────────────────────────────────────────────────────
summary = []

for sport_key, emoji, query, max_dur in SPORTS:
    print(f"\n{'='*60}", flush=True)
    print(f"{emoji}  {sport_key.upper()}", flush=True)
    print(f"{'='*60}", flush=True)

    vid_path = f"{WORK}/{sport_key}.mp4"

    # download
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

    # skip first 10s (title cards) and pick N_FRAMES evenly spaced
    start_f  = min(int(10 * FPS), TOT // 4)
    end_f    = min(start_f + N_FRAMES * FRAME_STEP, TOT)
    sample_f = list(range(start_f, end_f, FRAME_STEP))[:N_FRAMES]

    out_path = f"{WORK}/{sport_key}_annotated.mp4"
    writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                FPS / FRAME_STEP, (W, H))

    tracker     = sv.ByteTrack()
    persons_tot = 0
    balls_tot   = 0
    t0          = time.time()

    for i, fi in enumerate(sample_f):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            break

        preds    = model(frame, conf=0.25, verbose=False,
                         classes=[PERSON, BALL])[0]
        det_all  = sv.Detections.from_ultralytics(preds)

        det_p = det_all[det_all.class_id == PERSON]
        det_b = det_all[det_all.class_id == BALL]

        det_p = tracker.update_with_detections(det_p)

        persons_tot += len(det_p)
        balls_tot   += len(det_b)

        # build labels A1 A2 ... for persons, Ball for ball
        labels_p = [f"P{int(tid)%99:02d}" for tid in (det_p.tracker_id or [])]
        labels_b = ["Ball"] * len(det_b)

        # annotate persons
        annotated = frame.copy()
        if len(det_p):
            annotated = box_ann.annotate(annotated, det_p)
            annotated = label_ann.annotate(annotated, det_p, labels=labels_p)
            annotated = trace_ann.annotate(annotated, det_p)

        # annotate ball (cyan dot)
        if len(det_b):
            annotated = box_ann.annotate(annotated, det_b)
            annotated = label_ann.annotate(annotated, det_b, labels=labels_b)

        # sport + frame counter overlay
        cv2.putText(annotated, f"{emoji} {sport_key}  f{fi}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
        cv2.putText(annotated, f"{emoji} {sport_key}  f{fi}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        writer.write(annotated)

        # save first frame as preview image
        if i == 0:
            cv2.imwrite(f"{WORK}/{sport_key}_preview.jpg", annotated)

        if (i+1) % 20 == 0:
            print(f"  [{i+1:3d}/{len(sample_f)}] persons={len(det_p)} balls={len(det_b)}", flush=True)

    cap.release()
    writer.release()

    n = len(sample_f)
    avg_p = persons_tot / n if n else 0
    avg_b = balls_tot   / n if n else 0
    elapsed = time.time() - t0

    print(f"\n  avg persons/frame : {avg_p:.1f}", flush=True)
    print(f"  avg balls/frame   : {avg_b:.2f}", flush=True)
    print(f"  time              : {elapsed:.1f}s  ({elapsed/n:.2f}s/frame)", flush=True)
    print(f"  output            : {out_path}", flush=True)

    summary.append({
        "sport": sport_key,
        "status": "ok",
        "frames": n,
        "avg_persons_per_frame": round(avg_p, 2),
        "avg_balls_per_frame": round(avg_b, 3),
        "elapsed_s": round(elapsed, 1),
    })

# ── 5. summary ────────────────────────────────────────────────────────────────
print(f"\n{'='*60}", flush=True)
print("SUMMARY", flush=True)
print(f"{'='*60}", flush=True)
for s in summary:
    if s["status"] != "ok":
        print(f"  {s['sport']:12s}  FAILED ({s['status']})", flush=True)
    else:
        print(f"  {s['sport']:12s}  persons={s['avg_persons_per_frame']:5.1f}/f  "
              f"ball={s['avg_balls_per_frame']:.3f}/f  "
              f"({s['elapsed_s']}s)", flush=True)

out_json = f"{WORK}/sports_test_results.json"
with open(out_json, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nResults saved to {out_json}", flush=True)
print("Preview images: /kaggle/working/<sport>_preview.jpg", flush=True)
print("Annotated videos: /kaggle/working/<sport>_annotated.mp4", flush=True)

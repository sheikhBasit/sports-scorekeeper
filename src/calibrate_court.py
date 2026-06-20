"""
Court calibration — homography (pixels → real-world metres).

Court dimensions come from a SportConfig rather than module-level constants.
Pick the 4 OUTER corners in order: TL → TR → BR → BL.

Usage:
    python src/calibrate_court.py --source match.mp4 --frame 0 \
        --corners "x1,y1 x2,y2 x3,y3 x4,y4" --out court.npz \
        --sport padel
"""
import argparse
import cv2
import numpy as np


def make_court_pts(w: float, l: float) -> np.ndarray:
    """Return 4 reference corners (TL, TR, BR, BL) in metres."""
    return np.float32([
        [0.0, 0.0],   # TL
        [w,   0.0],   # TR
        [w,   l  ],   # BR
        [0.0, l  ],   # BL
    ])


PREVIEW_PPM = 25   # pixels-per-metre for top-down preview image


class CourtMapper:
    """Maps image pixels to court metres and estimates real-world speed."""

    def __init__(self, H, image_corners, frame_size, court_w, court_l):
        self.H = np.asarray(H, dtype=np.float64)
        self.image_corners = np.asarray(image_corners, dtype=np.float32)
        self.frame_size = tuple(frame_size)
        self.court_w = float(court_w)
        self.court_l = float(court_l)

    def to_metres(self, pts):
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def contains(self, pts, margin_m: float = 0.0) -> np.ndarray:
        m = self.to_metres(np.asarray(pts, dtype=np.float32))
        return (
            (m[:, 0] >= -margin_m) & (m[:, 0] <= self.court_w + margin_m) &
            (m[:, 1] >= -margin_m) & (m[:, 1] <= self.court_l + margin_m)
        )

    def speed_kmh(self, a, b, dt: float) -> float:
        am = self.to_metres([a])[0]
        bm = self.to_metres([b])[0]
        dist = float(np.linalg.norm(bm - am))
        return (dist / dt) * 3.6 if dt > 0 else 0.0

    def save(self, path: str):
        np.savez(path, H=self.H,
                 image_corners=self.image_corners,
                 frame_size=np.array(self.frame_size),
                 court_w=np.array(self.court_w),
                 court_l=np.array(self.court_l))

    @classmethod
    def load(cls, path: str) -> 'CourtMapper':
        d = np.load(path)
        return cls(d['H'], d['image_corners'], tuple(d['frame_size'].tolist()),
                   float(d['court_w']), float(d['court_l']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--frame', type=int, default=0)
    ap.add_argument('--corners', help='"x1,y1 x2,y2 x3,y3 x4,y4"')
    ap.add_argument('--out', default='court.npz')
    ap.add_argument('--sport', default='padel',
                    help='sport key (used to load SportConfig for court dims)')
    args = ap.parse_args()

    # load sport config to get court dimensions
    from sports.registry import get_sport
    sport = get_sport(args.sport)

    cap = cv2.VideoCapture(args.source)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError('Could not read frame')

    h, w = frame.shape[:2]
    if args.corners:
        pts = [[float(v) for v in p.split(',')] for p in args.corners.split()]
        corners = np.float32(pts)
    else:
        print('Click TL, TR, BR, BL — press any key after each')
        corners = []
        def click(e, x, y, *_):
            if e == cv2.EVENT_LBUTTONDOWN:
                corners.append([x, y])
                cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)
                cv2.imshow('court', frame)
        cv2.imshow('court', frame)
        cv2.setMouseCallback('court', click)
        while len(corners) < 4:
            cv2.waitKey(1)
        cv2.destroyAllWindows()
        corners = np.float32(corners)

    court_pts = make_court_pts(sport.court_w, sport.court_l)
    H, _ = cv2.findHomography(corners, court_pts)
    mapper = CourtMapper(H, corners, (w, h), sport.court_w, sport.court_l)
    mapper.save(args.out)

    # top-down preview
    pw = int(sport.court_w * PREVIEW_PPM)
    ph = int(sport.court_l * PREVIEW_PPM)
    H_inv = np.linalg.inv(H)
    topdown = cv2.warpPerspective(frame, H_inv * PREVIEW_PPM, (pw, ph))
    cv2.imwrite('court_topdown.jpg', topdown)
    print(f'Saved {args.out} and court_topdown.jpg')


if __name__ == '__main__':
    main()

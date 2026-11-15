"""Ground-truth bench for the reader's fit: a known motion of bench/proj's creature, readings simulated from it (the
true keypoints through the start cameras, plus reader-like noise and hidden stretches), the fit as `run` does it
(road.fit_reading), and how far the fit lands from the truth.

    python bench/fit_bench.py --clip walk --kind 30-5 [--seed 0] [--set KEY=VALUE]

Prints one JSON line: kp_err_px (3D keypoints vs truth, px of a 720-row frame), angle_err_deg (joint channels),
vel_err (keypoint speed vs truth by real time, px/s; all readings and only the hidden ones, where the fit's
smoothness decides), shake_px of truth and fit on an even time grid.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--repo', default='.')
ap.add_argument('--project', default='bench/proj')
ap.add_argument('--clip', default='walk')
ap.add_argument('--kind', default='24', choices=['24', '30-5', '25-25'])
ap.add_argument('--motion', default='walk', choices=['walk', 'idle', 'cycle'],
                help='walk: waves and a strike; idle: the legs and the body still (feet planted), the rest waving; '
                     'cycle: every wave a multiple of 0.8 Hz (it repeats every 1.25 s)')
ap.add_argument('--frames', type=int, default=120)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
ap.add_argument('--log', action='store_true', help='print the fit lines')
a = ap.parse_args()
repo = Path(a.repo).resolve()
sys.path.insert(0, str(repo))

import cv2  # noqa: E402
import torch  # noqa: E402

from video2rig.common import Project  # noqa: E402
from video2rig.creature import Creature  # noqa: E402
from video2rig.reader import KeypointRig, keypoints, side_of  # noqa: E402
from video2rig.reader.fit import SETTINGS, projector  # noqa: E402
from video2rig.reader.road import fit_reading  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from truth import content_times, truth  # noqa: E402



def along_floor(X, up):
    """X (..., 3) with the up-down part taken out (the bench's own: every checkout is measured the same way)."""
    up = np.asarray(up, float) / np.linalg.norm(up)
    return X - (X @ up)[..., None] * up


def find_loop(X, times, up, shortest=.5):
    """(i, j) the window whose end looks most like its start (as reader/fit.py's find_loop on the game branch)."""
    R = X - along_floor(X.mean(1, keepdims=True), up)
    step = np.gradient(R, axis=0)
    gap = int(np.ceil(shortest / float(np.median(np.diff(times)))))
    best, out = np.inf, (0, len(R) - 1)
    for i in range(len(R) - gap):
        d = np.linalg.norm(R[i + gap:] - R[i], axis=-1).mean(1) + 4 * np.linalg.norm(step[i + gap:] - step[i],
                                                                                     axis=-1).mean(1)
        j = int(np.argmin(d))
        if d[j] < best:
            best, out = float(d[j]), (i, i + gap + j)
    return out


root = Path(a.project).resolve()
clip = f'{a.clip}-{a.kind}'
T = a.frames
i = np.arange(T)
times = content_times(T, a.kind)
# the project: a clip of start.png copies (the fit uses only the cameras, aligned on frame 0), the timing set
cfg_path = root / 'creature.json'
cfg_all = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
cfg_all.setdefault('clips', {})[clip] = {'fps': a.kind}
cfg_path.write_text(json.dumps(cfg_all, indent=1))
video = root / 'clips' / f'{clip}.mp4'
if not video.exists():
    video.parent.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(root / 'start' / 'start.png'))
    vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 24, (img.shape[1], img.shape[0]))
    for _ in range(T):
        vw.write(img)
    vw.release()
pr = Project(str(root))
cfg = pr.clip_cfg(clip)
cfg.setdefault('solve', {})
cr = Creature(pr, cfg)
kps = keypoints(cr)
rig = cr.rig
lo, hi, p0 = cr.bounds(clip)
P_true = truth(cr, clip, times, a.motion, a.seed)
grid_t = np.round(np.arange(int(round(times[-1] * 120)) + 1) / 120, 6)          # 120 Hz: the truth between frames
P_fine = truth(cr, clip, grid_t, a.motion, a.seed)
kr = KeypointRig(cr, kps)
X_true = np.stack([kr.points(p) for p in P_true])                             # (T, K, 3)
X_fine = np.stack([kr.points(p) for p in P_fine])
views = cr.views(clip)
names = [v['name'] for v in views]
V, K = len(views), len(kps)
uv = np.zeros((T, V, K, 2))
conf = np.zeros((T, V, K))
hidden = np.zeros((T, V, K), bool)
noise = np.random.default_rng(1000 + a.seed)
for v, view in enumerate(views):
    cam = view['cam']
    clean = projector(cam)(torch.tensor(X_true)).numpy()
    scale = cam.h / 720.0
    ar = np.zeros((K, 2))
    for t in range(T):                                                        # a reader's error: part frame to frame,
        ar = .8 * ar + noise.normal(0, 1.2 * scale, (K, 2))                   # part slowly wandering
        uv[t, v] = clean[t] + ar + noise.normal(0, 2.2 * scale, (K, 2))
    conf[:, v] = noise.uniform(.65, 1.0, (T, K))
    for k in range(K):                                                        # hidden stretches: a far limb in the
        far = 'side' in names[v] and side_of(kps[k][0]) == 'R'                # side view more often
        for _ in range(noise.poisson(2.5 if far else .6)):
            t0 = int(noise.integers(0, T))
            span = slice(t0, min(T, t0 + int(noise.integers(6, 24))))
            hidden[span, v, k] = True
            conf[span, v, k] = noise.uniform(.02, .2)
            uv[span, v, k] += noise.normal(0, 25 * scale, 2)                  # "not here": read somewhere near
np.savez_compressed(pr.work(clip) / 'reader.npz', uv=uv, conf=conf, keypoints=json.dumps(kps),
                    views=np.array(names))                       # for python -m video2rig.reader.fit
settings = dict(SETTINGS)
for kv in a.set:
    key, val = kv.split('=', 1)
    settings[key] = json.loads(val)
t0 = time.time()
res = fit_reading(pr, clip, cfg, uv, conf, kps, names, s=settings, log=print if a.log else lambda *_: None)
secs = time.time() - t0
P_fit = res['params']
X_fit = np.stack([kr.points(p) for p in P_fit])
px = max(float(v['cam'].h) for v in views) / 720.0
kpx = float(np.mean([v['cam'].px_per_unit for v in views])) / px
body = np.array([not str(kp[1]).startswith('@') for kp in kps])
joint_ch = [3 + c for c, (j, _) in enumerate(rig.channels) if j != rig.root]
dt = np.diff(times)[:, None, None]
vel_true, vel_fit = np.diff(X_true, axis=0) / dt, np.diff(X_fit, axis=0) / dt
vel_err = np.linalg.norm(vel_fit - vel_true, axis=-1)[:, body] * kpx           # (T - 1, K)
arms = np.array([any(w in kp[1].lower() for w in ('hand', 'forearm')) for kp in kps])     # the strike's
hid = hidden.all(1)[:-1, body] | hidden.all(1)[1:, body]                     # hidden in every panel: a guess
hid_any = hidden.any(1)[:-1, body] | hidden.any(1)[1:, body]


def even(X, t):
    """X (len(t), K, 3) resampled onto an even 1/24 s grid (linear), for the shake measure."""
    g = np.arange(0, t[-1] + 1e-9, 1 / 24)
    flat = X.reshape(len(t), -1)
    return np.stack([np.interp(g, t, flat[:, c]) for c in range(flat.shape[1])], 1).reshape(len(g), *X.shape[1:])


def shake(X):
    return float(np.linalg.norm(X[3:] - 3 * X[2:-1] + 3 * X[1:-2] - X[:-3], axis=-1).mean() * kpx)


m = res.get('motion')
X_out, t_out = (np.stack([kr.points(p) for p in m['params']]), m['times']) if m else (X_fit, times)
X0 = kr.points(p0)
feet = body & ((X0 @ cr.up - cr.floor) <= .05 * cr.height)
t_out = np.asarray(t_out, float)


def slide(X, t):
    """The feet's mean speed along the floor, px/s."""
    return float((np.linalg.norm(np.diff(along_floor(X[:, feet], cr.up), axis=0), axis=-1) /
                  np.diff(t)[:, None]).mean() * kpx)


extra = dict(foot_slide_true=round(slide(X_fine, grid_t), 2), foot_slide_fit=round(slide(X_out, t_out), 2))
if a.motion == 'cycle':
    if m and 'loop' in m:
        XL, tl = np.stack([kr.points(p) for p in m['loop']['params']]), m['loop']['times']
        win = res['report']['loop']
    else:                                                                   # the same window cut from the plain fit
        i0, j0 = find_loop(X_out[:, body], t_out, cr.up)
        XL, tl = X_out[i0:j0 + 1], t_out[i0:j0 + 1] - t_out[i0]
        win = dict(start_s=round(float(t_out[i0]), 3), end_s=round(float(t_out[j0]), 3))
    R = XL[:, body] - along_floor(XL[:, body].mean(1, keepdims=True), cr.up)
    h = float(np.median(np.diff(tl)))
    extra.update(loop=win, seam_jump_px=round(float(np.linalg.norm(R[-1] - R[0], axis=-1).mean() * kpx), 3),
                 seam_speed_px_s=round(float(np.linalg.norm((R[-1] - R[-2]) - (R[1] - R[0]), axis=-1).mean() * kpx / h), 2),
                 speed_px_s=round(float(np.linalg.norm(np.diff(R, axis=0), axis=-1).mean() * kpx / h), 2))
rev = subprocess.run(['git', '-C', str(repo), 'rev-parse', '--abbrev-ref', 'HEAD'], capture_output=True,
                     text=True).stdout.strip()
print(json.dumps(dict(
    branch=rev, clip=a.clip, kind=a.kind, seed=a.seed, set=a.set, frames=T, motion_frames=len(t_out),
    kp_err_px=round(float(np.linalg.norm(X_fit - X_true, axis=-1)[:, body].mean() * kpx), 3),
    angle_err_deg=round(float(np.degrees(np.abs(P_fit[:, joint_ch] - P_true[:, joint_ch]).mean())), 3),
    vel_err=round(float(vel_err.mean()), 2),
    strike_err_px=round(float(np.linalg.norm(X_fit - X_true, axis=-1)[(times > 1.8) & (times < 2.2)][:, arms].mean()
                              * kpx), 3),
    vel_err_hidden_one_panel=round(float(vel_err[hid_any & ~hid].mean()), 2) if (hid_any & ~hid).any() else None,
    vel_err_hidden_all=round(float(vel_err[hid].mean()), 2) if hid.any() else None,
    shake_true=round(shake(even(X_fine[:, body], grid_t)), 3),
    shake_fit=round(shake(even(X_out[:, body], np.asarray(t_out, float))), 3),
    secs=round(secs, 1), motion=a.motion, **extra)), flush=True)

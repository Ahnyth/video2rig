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
ap.add_argument('--frames', type=int, default=120)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
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

root = Path(a.project).resolve()
clip = f'{a.clip}-{a.kind}'
T = a.frames
i = np.arange(T)
times = np.round({'24': i / 24, '30-5': (i + i // 4) / 30, '25-25': (i + i // 24) / 25}[a.kind], 6)
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
rng = np.random.default_rng(a.seed)


def truth(t):
    """(len(t), rig.n) the known motion: every joint channel a sum of two slow waves inside its range, the arms a fast
    strike at 2 s, the body swaying and turning a little."""
    t = np.asarray(t, float)[:, None]
    P = np.tile(p0, (len(t), 1))
    for c, (j, ax) in enumerate(rig.channels):
        k = 3 + c
        room = max(min(hi[k] - p0[k], p0[k] - lo[k]), 0.0)
        f = rng.uniform(.3, .9, 2)
        ph = rng.uniform(0, 2 * np.pi, 2)
        wave = .6 * np.sin(2 * np.pi * f[0] * t + ph[0]) + .4 * np.sin(2 * np.pi * f[1] * t + ph[1])
        P[:, k] = p0[k] + min(.15 * room, .5) * wave[:, 0]
        if 'arm' in rig.names[j].lower() and ax == 0:
            P[:, k] += min(.5 * room, .8) * np.exp(-((t[:, 0] - 2.0) / .08) ** 2)   # a strike: 0.2 s up and back
    s = cr.height
    P[:, 0] += .05 * s * np.sin(2 * np.pi * .4 * t[:, 0])
    P[:, 2] += .02 * s * np.sin(2 * np.pi * 1.1 * t[:, 0])
    return np.clip(P, lo, hi)


rng_motion = np.random.default_rng(a.seed)
rng = rng_motion
P_true = truth(times)
grid_t = np.round(np.arange(int(round(times[-1] * 120)) + 1) / 120, 6)          # 120 Hz: the truth between frames
rng = np.random.default_rng(a.seed)
P_fine = truth(grid_t)
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
settings = dict(SETTINGS)
for kv in a.set:
    key, val = kv.split('=', 1)
    settings[key] = json.loads(val)
t0 = time.time()
res = fit_reading(pr, clip, cfg, uv, conf, kps, names, s=settings, log=lambda *_: None)
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
rev = subprocess.run(['git', '-C', str(repo), 'rev-parse', '--abbrev-ref', 'HEAD'], capture_output=True,
                     text=True).stdout.strip()
print(json.dumps(dict(
    branch=rev, clip=a.clip, kind=a.kind, seed=a.seed, set=a.set, frames=T, motion_frames=len(t_out),
    kp_err_px=round(float(np.linalg.norm(X_fit - X_true, axis=-1)[:, body].mean() * kpx), 3),
    angle_err_deg=round(float(np.degrees(np.abs(P_fit[:, joint_ch] - P_true[:, joint_ch]).mean())), 3),
    vel_err=round(float(vel_err.mean()), 2),
    vel_err_hidden_one_panel=round(float(vel_err[hid_any & ~hid].mean()), 2) if (hid_any & ~hid).any() else None,
    vel_err_hidden_all=round(float(vel_err[hid].mean()), 2) if hid.any() else None,
    shake_true=round(shake(even(X_fine[:, body], grid_t)), 3),
    shake_fit=round(shake(even(X_out[:, body], np.asarray(t_out, float))), 3),
    secs=round(secs, 1))), flush=True)

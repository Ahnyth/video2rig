"""One fit of a real reading, with the code of one checkout, and what can be measured without ground truth.

    python bench/real_fit_one.py --repo <checkout> --project <folder> --clip <clip> [--set KEY=VALUE]

Prints one JSON line:
    sure_miss_px   per panel, the median miss of the fit on the readings the reader was sure of (conf >= 0.3), as the
                   fit's own report gives it: how faithful the fit stays to what the reader saw
    shake_px       the body keypoints' mean third difference on an even 24 fps grid (the motion file's own times,
                   resampled), px of a 720-row frame: how much the motion shakes
    foot_slide     the feet's mean speed along the floor where they stand (low, slow), px/s: how much standing feet slide
    loop_seam      where a loop window wraps (the fit's own -loop motion, or the same kind of window cut from the plain
                   motion): the second difference across the wrap over its mean inside the window (1: seamless)
    secs           the fit's time
The measures are computed here, the same way for every checkout (they do not use the checkout's own helpers).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--repo', required=True)
ap.add_argument('--project', required=True)
ap.add_argument('--clip', required=True)
ap.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
a = ap.parse_args()
sys.path.insert(0, str(Path(a.repo).resolve()))

from video2rig.common import Project  # noqa: E402
from video2rig.reader import KeypointRig  # noqa: E402
from video2rig.reader import fit as fitmod  # noqa: E402
from video2rig.reader.road import fit_reading  # noqa: E402

pr = Project(a.project)
cfg = pr.clip_cfg(a.clip)
cfg.setdefault('solve', {})
sets = {kv.split('=', 1)[0]: json.loads(kv.split('=', 1)[1]) for kv in a.set}
if hasattr(fitmod, 'settings_for'):
    s = fitmod.settings_for(cfg, sets)
else:
    unknown = set(sets) - set(fitmod.SETTINGS)
    if unknown:
        raise SystemExit(f'this checkout has no setting {", ".join(sorted(unknown))}')
    s = {**fitmod.SETTINGS, **sets}
d = np.load(pr.work(a.clip) / 'reader.npz')
kps = [tuple(k) for k in json.loads(str(d['keypoints']))]
t0 = time.time()
res = fit_reading(pr, a.clip, cfg, d['uv'], d['conf'], kps, [str(v) for v in d['views']], s=s, log=lambda *_: None)
secs = time.time() - t0
cr, rep = res['creature'], res['report']
motion = res.get('motion') or dict(times=cr.times(a.clip)[:len(res['params'])], params=res['params'])
kr = KeypointRig(cr, kps)
body = np.array([not str(kp[1]).startswith('@') for kp in kps])
views = cr.views(a.clip)
px = max(float(v['cam'].h) for v in views) / 720.0
kpx = float(np.mean([v['cam'].px_per_unit for v in views])) / px
up = np.asarray(cr.up, float) / np.linalg.norm(cr.up)


def points(P):
    return np.stack([kr.points(p) for p in P])


def flat(X):
    return X - (X @ up)[..., None] * up


def even(X, t, rate=24.0):
    """X resampled on an even grid from t[0] to t[-1], both ends included (as a 24 fps scene keys it)."""
    g = np.linspace(t[0], t[-1], int(round((t[-1] - t[0]) * rate)) + 1)
    f = X.reshape(len(t), -1)
    return np.stack([np.interp(g, t, f[:, c]) for c in range(f.shape[1])], 1).reshape(len(g), *X.shape[1:]), g


t = np.asarray(motion['times'], float)
X = points(motion['params'])[:, body]
Xe, te = even(X, t)
shake = float(np.linalg.norm(Xe[3:] - 3 * Xe[2:-1] + 3 * Xe[1:-2] - Xe[:-3], axis=-1).mean() * kpx)
# feet: keypoints near the floor in the start pose; standing: within 3% of the height above their start height and
# slower than 0.3 heights a second along the floor (the median of 3 frames)
X0 = kr.points(cr.p0(None))[body]
feet = (X0 @ up - cr.floor) <= .05 * cr.height
slide = None
if feet.any():
    F = Xe[:, feet]
    rate = np.linalg.norm(np.diff(flat(F), axis=0), axis=-1) * 24.0
    speed = np.concatenate([rate[:1], (rate[1:] + rate[:-1]) / 2, rate[-1:]])
    on = ((F @ up - X0[feet] @ up <= .03 * cr.height) & (speed <= .3 * cr.height)).astype(float)
    pad = np.pad(on, ((1, 1), (0, 0)), mode='edge')
    on = np.median(np.stack([pad[:-2], pad[1:-1], pad[2:]]), 0) > 0
    both = on[1:] & on[:-1]
    slide = float(rate[both].mean() * kpx) if both.any() else None


def best_window(R, rate=24.0, shortest=.5):
    """The window [i, j] whose end looks most like its start (keypoints round their middle and their steps)."""
    step = np.gradient(R, axis=0)
    gap = int(np.ceil(shortest * rate))
    best, out = np.inf, (0, len(R) - 1)
    for i in range(len(R) - gap):
        dist = np.linalg.norm(R[i + gap:] - R[i], axis=-1).mean(1) + \
            4 * np.linalg.norm(step[i + gap:] - step[i], axis=-1).mean(1)
        j = int(np.argmin(dist))
        if dist[j] < best:
            best, out = float(dist[j]), (i, i + gap + j)
    return out


def seam(XL):
    """How a loop's wrap compares with its inside, on the motion's own keys: the keypoints' second difference across
    the wrap (the key before the last, the first, the second: the last key is the first again) over its mean inside
    the window. 1: as smooth as the rest; a plain cut whose ends differ: many times that."""
    R = XL - flat(XL.mean(1, keepdims=True))
    wrap = np.linalg.norm(R[1] - 2 * R[0] + R[-2], axis=-1).mean()
    inside = np.linalg.norm(R[2:] - 2 * R[1:-1] + R[:-2], axis=-1).mean()
    return float(wrap / max(inside, 1e-12))


if 'loop' in motion:
    seam_ratio = seam(points(motion['loop']['params'])[:, body])
    window = rep.get('loop')
else:                                       # the window a loop would take, cut from the plain motion
    R = X - flat(X.mean(1, keepdims=True))
    i, j = best_window(R, rate=1 / float(np.median(np.diff(t))))
    seam_ratio = seam(X[i:j + 1])
    window = dict(start_s=round(float(t[i]), 3), end_s=round(float(t[j]), 3))
print(json.dumps(dict(
    project=Path(a.project).name, clip=a.clip, set=a.set, frames=int(rep['frames']), motion_frames=len(t),
    sure_miss_px=rep['sure_miss_px'], shake_px=round(shake, 3),
    foot_slide=None if slide is None else round(slide, 2),
    loop_seam=round(seam_ratio, 2), loop=window, secs=round(secs, 1))), flush=True)

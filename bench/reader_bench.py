"""Ground-truth bench of the whole road: a rendered clip with its truth (make_gt_clip.py) read by a reader
(road.read_clip) and fitted (road.fit_reading), each against the truth.

    python bench/reader_bench.py --clip gt-walk-24 [--repo wt/fit-real-time] [--reader reader/reader.pt] [--tta 8]
        [--tag -base] [--set KEY=VALUE]

Prints one JSON line:
    read_seen_px / read_hidden_px   the reader's miss (median, and mean) on keypoints the panel shows / hides, px of a
                                    720-row frame, against the true keypoints through the clip's aligned cameras
    read_shake_px                   the readings' frame-to-frame shake on shown keypoints (mean third difference), and
                                    the truth's
    kp_err_px, angle_err_deg        the fit's keypoints (3D, px) and joint channels against the truth
    shake_px                        the fit's motion on an even 24 fps grid (the truth's beside it)
    strike_err_px                   hands and forearms from 1.8 to 2.2 s (the walk's strike)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('--repo', default=str(HERE.parent / 'wt' / 'fit-real-time'))
ap.add_argument('--project', default=str(HERE / 'proj'))
ap.add_argument('--clip', required=True)
ap.add_argument('--reader', default='')
ap.add_argument('--tta', type=int, default=None)
ap.add_argument('--tag', default='')
ap.add_argument('--cpu', action='store_true')
ap.add_argument('--exact-cameras', action='store_true', help='the cameras as rendered (the clip was not re-framed), '
                'not as the alignment on frame 0 found them')
ap.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
a = ap.parse_args()
sys.path.insert(0, str(Path(a.repo).resolve()))

import torch  # noqa: E402

from video2rig.common import Project  # noqa: E402
from video2rig.creature import Creature  # noqa: E402
from video2rig.reader import KeypointRig  # noqa: E402
from video2rig.reader import fit as fitmod  # noqa: E402
from video2rig.reader.road import fit_reading, read_clip  # noqa: E402

pr = Project(a.project)
cfg = pr.clip_cfg(a.clip)
cfg.setdefault('solve', {})
# the cameras: as the tool's alignment on frame 0 found them (kept as cameras-aligned-tool.json), or exact
aligned, tool = pr.work(a.clip) / 'cameras-aligned.json', pr.work(a.clip) / 'cameras-aligned-tool.json'
if not tool.exists():
    if not aligned.exists():
        Creature(pr, cfg).views(a.clip)
    tool.write_text(aligned.read_text())
recs = json.loads(tool.read_text())
if a.exact_cameras:
    recs = [{**r, 'scale': 1.0, 'offset': [0.0, 0.0], 'turn_deg': [0.0, 0.0, 0.0]} for r in recs]
aligned.write_text(json.dumps(recs))
reader = Path(a.reader) if a.reader else pr.root / 'reader' / 'reader.pt'
if not reader.is_absolute():
    reader = pr.root / reader
t0 = time.time()
uv, conf, kps, names = read_clip(pr, a.clip, reader=reader, tta=a.tta, tag=a.tag, cpu=a.cpu, log=lambda *_: None)
read_secs = time.time() - t0
work = pr.work(a.clip)
gt = np.load(work / 'truth.npz')
P_true, times = gt['params'], gt['times']
vis = np.load(work / 'visible.npz')['vis']                                        # (T, V, K)
cr = Creature(pr, cfg)
views = cr.views(a.clip)
order = [names.index(v['name']) for v in views]
uv, conf = np.asarray(uv)[:, order], np.asarray(conf)[:, order]
kr = KeypointRig(cr, kps)
X_true = np.stack([kr.points(p) for p in P_true])
px = max(float(v['cam'].h) for v in views) / 720.0
kpx = float(np.mean([v['cam'].px_per_unit for v in views])) / px
uv_true = np.stack([fitmod.projector(v['cam'])(torch.tensor(X_true)).numpy() for v in views], 1)
T = min(len(uv), len(uv_true))
miss = np.linalg.norm(uv[:T] - uv_true[:T], axis=-1) / px                         # (T, V, K)
seen = vis[:T].astype(bool)
d3 = lambda Y: np.linalg.norm(Y[3:] - 3 * Y[2:-1] + 3 * Y[1:-2] - Y[:-3], axis=-1)
shown = seen[3:] & seen[2:-1] & seen[1:-2] & seen[:-3]
read_shake = float(d3(uv[:T])[shown].mean() / px)
true_shake = float(d3(uv_true[:T])[shown].mean() / px)
# the fit as `run` does it, with the checkout's defaults (and --set)
sets = {kv.split('=', 1)[0]: json.loads(kv.split('=', 1)[1]) for kv in a.set}
s = fitmod.settings_for(cfg, sets) if hasattr(fitmod, 'settings_for') else {**fitmod.SETTINGS, **sets}
t1 = time.time()
res = fit_reading(pr, a.clip, cfg, uv, conf, kps, [v['name'] for v in views], s=s, log=lambda *_: None)
fit_secs = time.time() - t1
P = res['params'][:T]
X = np.stack([kr.points(p) for p in P])
body = np.array([not str(kp[1]).startswith('@') for kp in kps])
rig = cr.rig
joint_ch = [3 + c for c, (j, _) in enumerate(rig.channels) if j != rig.root]


def even(Y, t, rate=24.0):
    g = np.linspace(t[0], t[-1], int(round((t[-1] - t[0]) * rate)) + 1)
    f = Y.reshape(len(t), -1)
    return np.stack([np.interp(g, t, f[:, c]) for c in range(f.shape[1])], 1).reshape(len(g), *Y.shape[1:])


m = res.get('motion')
Xm, tm = (np.stack([kr.points(p) for p in m['params']]), np.asarray(m['times'], float)) if m else (X, times[:T])
arms = np.array([any(w in kp[1].lower() for w in ('hand', 'forearm')) for kp in kps])
window = (times[:T] > 1.8) & (times[:T] < 2.2)
up = np.asarray(cr.up, float) / np.linalg.norm(cr.up)
flat = lambda Y: Y - (Y @ up)[..., None] * up
X0 = kr.points(cr.p0(None))
feet = body & ((X0 @ up - cr.floor) <= .05 * cr.height)
Xe = even(Xm, tm)
slide = float((np.linalg.norm(np.diff(flat(Xe[:, feet]), axis=0), axis=-1) * 24.0).mean() * kpx) if feet.any() else None
seam = loop_err = None


def wrap_ratio(XL):
    """A loop's wrap against its inside: the keypoints' second difference across the wrap (the frame before the last,
    the first, the second: the last frame is the first again) over its mean inside. 1: as smooth as the rest."""
    R = XL - flat(XL.mean(1, keepdims=True))
    inside = np.linalg.norm(R[2:] - 2 * R[1:-1] + R[:-2], axis=-1).mean()
    return float(np.linalg.norm(R[1] - 2 * R[0] + R[-2], axis=-1).mean() / max(inside, 1e-12))


if m and 'loop' in m:
    XL = np.stack([kr.points(p) for p in m['loop']['params']])[:, body]
    i0 = int(np.argmin(np.abs(tm - res['report']['loop']['start_s'])))
    plain = Xm[i0:i0 + len(XL)][:, body]                                    # the same window, cut as it is
    seam = dict(loop=round(wrap_ratio(XL), 2), plain_cut=round(wrap_ratio(plain), 2))
    j0 = int(np.argmin(np.abs(times[:T] - res['report']['loop']['start_s'])))
    XT = X_true[j0:j0 + len(XL)][:, body]
    loop_err = round(float(np.linalg.norm(XL[:len(XT)] - XT, axis=-1).mean() * kpx), 3)
print(json.dumps(dict(
    clip=a.clip, reader=reader.name, tta=a.tta, set=a.set, repo=Path(a.repo).name, frames=T,
    cameras='exact' if a.exact_cameras else 'aligned',
    seen_share=round(float(seen.mean()), 3),
    read_seen_px=[round(float(np.median(miss[seen])), 2), round(float(miss[seen].mean()), 2)],
    read_hidden_px=[round(float(np.median(miss[~seen])), 2), round(float(miss[~seen].mean()), 2)] if (~seen).any() else None,
    read_shake_px=[round(read_shake, 3), round(true_shake, 3)],
    kp_err_px=round(float(np.linalg.norm(X - X_true[:T], axis=-1)[:, body].mean() * kpx), 3),
    angle_err_deg=round(float(np.degrees(np.abs(P[:, joint_ch] - P_true[:T][:, joint_ch]).mean())), 3),
    strike_err_px=round(float(np.linalg.norm(X - X_true[:T], axis=-1)[window][:, arms].mean() * kpx), 3)
    if window.any() and arms.any() else None,
    shake_px=[round(float(d3(even(Xm[:, body], tm)).mean() * kpx), 3),
              round(float(d3(even(X_true[:T][:, body], times[:T])).mean() * kpx), 3)],
    foot_slide=None if slide is None else round(slide, 2), loop_seam=seam, loop_err_px=loop_err,
    read_secs=round(read_secs, 1), fit_secs=round(fit_secs, 1))), flush=True)

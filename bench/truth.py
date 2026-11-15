"""The bench's known motions, in the tool's own parameters (fit_bench.py simulates readings of them, make_gt_clip.py
renders them as a clip).

    walk   every joint channel two slow waves (0.3-0.9 Hz) inside its range, a 0.2 s arm strike at 2 s, the body
           swaying and turning a little
    idle   the legs and the body still (the feet planted), the rest waving
    cycle  every wave a multiple of 0.8 Hz: the motion repeats every 1.25 s
"""
import re

import numpy as np

LEG = re.compile(r'(thigh|shin|foot|toe|heel|pelvis|leg)', re.I)
MOTIONS = ('walk', 'idle', 'cycle')


def truth(cr, clip, t, motion='walk', seed=0):
    """(len(t), rig.n) the motion at times t (seconds); the same seed gives the same motion at any times."""
    rng = np.random.default_rng(seed)
    rig = cr.rig
    lo, hi, p0 = cr.bounds(clip)
    t = np.asarray(t, float)[:, None]
    P = np.tile(p0, (len(t), 1))
    for c, (j, ax) in enumerate(rig.channels):
        k = 3 + c
        room = max(min(hi[k] - p0[k], p0[k] - lo[k]), 0.0)
        f = rng.uniform(.3, .9, 2) if motion != 'cycle' else .8 * rng.integers(1, 3, 2)
        if motion == 'idle' and (j == rig.root or LEG.search(rig.names[j])):
            rng.uniform(0, 1, 2)
            continue
        ph = rng.uniform(0, 2 * np.pi, 2)
        wave = .6 * np.sin(2 * np.pi * f[0] * t + ph[0]) + .4 * np.sin(2 * np.pi * f[1] * t + ph[1])
        P[:, k] = p0[k] + min(.15 * room, .5) * wave[:, 0]
        if motion == 'walk' and 'arm' in rig.names[j].lower() and ax == 0:
            P[:, k] += min(.5 * room, .8) * np.exp(-((t[:, 0] - 2.0) / .08) ** 2)   # a strike: 0.2 s up and back
    s = cr.height
    if motion != 'idle':
        f0, f1 = (.4, 1.1) if motion == 'walk' else (.8, 1.6)
        P[:, 0] += .05 * s * np.sin(2 * np.pi * f0 * t[:, 0])
        P[:, 2] += .02 * s * np.sin(2 * np.pi * f1 * t[:, 0])
    return np.clip(P, lo, hi)


def content_times(n, kind):
    """clip.cadence's times of n video frames: '24', '30-5' (30 fps, every 5th dropped) or '25-25'."""
    i = np.arange(n)
    return np.round({'24': i / 24, '30-5': (i + i // 4) / 30, '25-25': (i + i // 24) / 25}[kind], 6)

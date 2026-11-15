"""Bench only: how Unity's import changed the motion (measure.json from Video2RigMeasure.cs, blender.json from
blender_sample.py). Every variant against Unity with compression off (joint error in mm, shake at 60 Hz, clip memory),
and Unity's 'off' against Blender itself (after one rigid fit: Blender's axes are not Unity's).

    python bench/unity/compare.py measure.json blender.json"""
import json
import sys

import numpy as np

meas = json.load(open(sys.argv[1]))
blend = json.load(open(sys.argv[2]))


def shake(X):
    return float(np.linalg.norm(X[3:] - 3 * X[2:-1] + 3 * X[1:-2] - X[:-3], axis=-1).mean())


def kabsch(A, B):
    """B ~ s R A + t (least squares, R orthogonal: Unity is left-handed, so Blender's axes reach it mirrored): the
    residual (N, 3) and s."""
    a0, b0 = A.mean(0), B.mean(0)
    U, S, Vt = np.linalg.svd((A - a0).T @ (B - b0))
    R = Vt.T @ U.T
    s = S.sum() / ((A - a0) ** 2).sum()
    return B - (s * (A - a0) @ R.T + b0), s


off = next(v for v in meas['variants'] if v['label'] == 'off')
for clip in off['clips']:
    base = np.array(clip['frames']).reshape(len(clip['frames']), -1, 3)   # (n, bones, 3) metres
    name = clip['name'].split('|')[-1]
    print(f'\n### {name}: {clip["length"]:.2f} s at {clip["frame_rate"]:g} fps\n')
    if name in blend:
        B = np.array(blend[name]['frames'])
        bones = [b for b in blend[name]['bones'] if b in off['bones']]
        A = base[:, [off['bones'].index(b) for b in bones]]
        Bb = B[:, [blend[name]['bones'].index(b) for b in bones]]
        n = min(len(A), len(Bb))
        res, s = kabsch(Bb[:n].reshape(-1, 3), A[:n].reshape(-1, 3))
        err = np.linalg.norm(res, axis=-1) * 1000
        print(f'Blender -> Unity (off): mean {err.mean():.2f} mm, max {err.max():.2f} mm after one rigid fit '
              f'(scale {s:.4f})\n')
    print('| import | root motion node | loop | joint error mean / max (mm) | shake vs off | clip memory (KB) |')
    print('| --- | --- | --- | --- | --- | --- |')
    for v in meas['variants']:
        c = next(c for c in v['clips'] if c['name'].split('|')[-1] == name)
        X = np.array(c['frames']).reshape(len(c['frames']), -1, 3)
        n = min(len(X), len(base))
        err = np.linalg.norm(X[:n] - base[:n], axis=-1) * 1000
        print(f'| {v["label"]} | {v["root_node"] or "-"} | {c["loop_time"]} | {err.mean():.2f} / {err.max():.2f} | '
              f'{shake(X[:n]) / max(shake(base[:n]), 1e-12):.3f} | {c["bytes"] / 1024:.0f} |')

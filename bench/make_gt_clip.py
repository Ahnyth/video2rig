"""Bench only: a clip with ground truth. A known motion (truth.py), starting from the start pose as a generated clip
does, rendered by Blender through the start frame's cameras in its studio (render_clip.py), damaged a little the way
a video model's frames differ from a render (re-framed, other colours, softer, noise, then mp4), written as
<project>/clips/<clip>.mp4 with its truth (work/<clip>/truth.npz: params, times) and which keypoints each panel shows
(work/<clip>/visible.npz).

    python bench/make_gt_clip.py --clip gt-walk --motion walk --kind 24 [--seed 0] [--frames 120]
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default=str(HERE.parent / 'video2rig'))
    ap.add_argument('--project', default=str(HERE / 'proj'))
    ap.add_argument('--blend', default=str(HERE / 'creature.blend'))
    ap.add_argument('--blender', default=r'C:\Tools\Blender\blender.exe')
    ap.add_argument('--clip', required=True)
    ap.add_argument('--motion', default='walk')
    ap.add_argument('--kind', default='24', choices=['24', '30-5', '25-25'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--frames', type=int, default=120)
    ap.add_argument('--clean', action='store_true', help='no damage: the render as it is')
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.repo).resolve()))
    sys.path.insert(0, str(HERE))

    from truth import content_times, truth  # noqa: E402
    from video2rig.common import Project  # noqa: E402
    from video2rig.creature import Creature  # noqa: E402
    from video2rig.reader import visible  # noqa: E402

    root = Path(a.project).resolve()
    cfg_path = root / 'creature.json'
    cfg_all = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    cfg_all.setdefault('clips', {})[a.clip] = {'fps': a.kind}
    cfg_path.write_text(json.dumps(cfg_all, indent=1))
    pr = Project(str(root))
    cfg = pr.clip_cfg(a.clip)
    cfg.setdefault('solve', {})
    cr = Creature(pr, cfg)
    rig = cr.rig
    lo, hi, p0 = cr.bounds(a.clip)
    times = content_times(a.frames, a.kind)
    P = truth(cr, a.clip, times, a.motion, a.seed)
    P = np.clip(P - P[0] + p0, lo, hi)                    # frame 0 is the start frame: the start pose, as Seedance gets it
    work = pr.work(a.clip)
    np.savez_compressed(work / 'truth.npz', params=P, times=times, motion=a.motion, seed=a.seed)
    # the motion file blender/import_motion.py reads (as Creature.export_motion writes it)
    quats, locs, props = [], [], []
    for p in P:
        q, loc, prm = rig.pose_basis(p)
        quats.append(q)
        locs.append(loc)
        props.append(prm)
    motion = work / 'truth-motion.npz'
    np.savez_compressed(motion, bones=np.array(rig.names), root=rig.names[rig.root], quaternions=np.array(quats),
                        root_location=np.array(locs), props=np.array([q['name'] for q in rig.props]),
                        prop_matrices=np.array(props).reshape(len(P), len(rig.props), 4, 4), times=times,
                        armature_world=np.array(cr.data['armature_world']))
    rate = {'24': 24, '30-5': 30, '25-25': 25}[a.kind]
    blend = work / 'truth.blend'
    tool_blender = Path(a.repo) / 'blender'


    def blender(*args):
        run = subprocess.run([a.blender, '-b', *map(str, args)], capture_output=True, text=True)
        if run.returncode != 0 or 'Error' in run.stdout and 'Traceback' in run.stdout:
            raise SystemExit(f'blender failed: {" ".join(map(str, args))}\n{run.stdout[-2000:]}\n{run.stderr[-2000:]}')
        return run.stdout


    blender(a.blend, '--python-expr', f'import bpy; bpy.context.scene.render.fps = {rate}',
            '--python', tool_blender / 'import_motion.py', '--', '--motion', motion, '--save', blend)
    frames_dir = work / 'truth-frames'
    shutil.rmtree(frames_dir, ignore_errors=True)
    keys = [1 + int(round(t * rate)) for t in times]
    blender(blend, '--python', HERE / 'render_clip.py', '--', '--cams', root / 'start' / 'cameras.json',
            '--frames', ','.join(map(str, keys)), '--out', frames_dir,
            '--render-start', tool_blender / 'render_start.py')
    # a video model's frame is not our render: re-framed a little, other colours, softer, some noise (one draw per clip)
    rng = np.random.default_rng(100 + a.seed)
    first = cv2.imread(str(frames_dir / 'frame_0000.png'))
    H, W = first.shape[:2]
    s = 1 + rng.uniform(.01, .03)
    M = np.array([[s, 0, (1 - s) * W / 2 + rng.uniform(-6, 6)], [0, s, (1 - s) * H / 2 + rng.uniform(-6, 6)]])
    gain = rng.uniform(.94, 1.06, 3)
    video = root / 'clips' / f'{a.clip}.mp4'
    video.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), rate, (W, H))
    for n in range(len(keys)):
        img = cv2.imread(str(frames_dir / f'frame_{n:04d}.png'))
        if not a.clean:
            img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            img = cv2.GaussianBlur(img, (0, 0), .7).astype(np.float32) * gain
            img = np.clip(img + rng.normal(0, 2.0, img.shape), 0, 255).astype(np.uint8)
        vw.write(img)
    vw.release()
    np.savez_compressed(work / 'truth-params.npz', params=P)
    vis = visible.compute(str(root), workers=6, log=lambda *_: None, poses=str(work / 'truth-params.npz'),
                          out=str(work / 'visible.npz'))
    print(json.dumps(dict(clip=a.clip, motion=a.motion, kind=a.kind, frames=len(keys), video=str(video),
                          seen=[round(float(vis[:, v].mean()), 3) for v in range(vis.shape[1])])), flush=True)


if __name__ == '__main__':      # visible.compute starts worker processes
    main()

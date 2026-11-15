"""Bench only (Blender): every bone's world position in every motion Action at 60 Hz, for bench/unity/compare.py.

    blender motions.blend -b --python bench/unity/blender_sample.py -- out.json"""
import json
import sys

import bpy

out = sys.argv[sys.argv.index('--') + 1]
scene = bpy.context.scene
arm = next(o for o in scene.objects if o.type == 'ARMATURE')
fps = scene.render.fps / scene.render.fps_base
res = {}
for act in bpy.data.actions:
    if not act.use_fake_user or act.name.endswith(tuple(f'_{o.name}' for o in scene.objects)):
        continue
    arm.animation_data.action = act
    if hasattr(arm.animation_data, 'action_slot') and act.slots:
        arm.animation_data.action_slot = act.slots[0]
    f0, f1 = act.frame_range
    n = int((f1 - f0) / fps * 60 + 1e-3) + 1
    frames = []
    for i in range(n):
        f = f0 + min(i / 60 * fps, f1 - f0)
        scene.frame_set(int(f), subframe=f - int(f))
        frames.append([list(arm.matrix_world @ pb.head) for pb in arm.pose.bones])
    res[act.name] = dict(bones=[pb.name for pb in arm.pose.bones], frames=frames)
with open(out, 'w') as fh:
    json.dump(res, fh)
print('BLENDER_SAMPLE', {k: len(v['frames']) for k, v in res.items()}, flush=True)

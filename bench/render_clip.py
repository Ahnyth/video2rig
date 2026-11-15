"""Bench only (Blender): render the scene's motion as a clip in render_start.py's studio through the start frame's own
cameras (start/cameras.json), one picture per given frame, laid out as the start frame (panels side by side, the
dark divider between them).

    blender gt.blend -b --python bench/render_clip.py -- --cams start/cameras.json --frames 1,2,3 --out frames/
        [--samples 8] [--engine BLENDER_EEVEE] [--render-start path/to/blender/render_start.py]
"""
import argparse
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index('--') + 1:]
ap = argparse.ArgumentParser()
ap.add_argument('--cams', required=True)
ap.add_argument('--frames', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--samples', type=int, default=8)
ap.add_argument('--engine', default='BLENDER_EEVEE')
ap.add_argument('--divider', type=int, default=4)
ap.add_argument('--render-start', default=os.path.join(os.path.dirname(__file__), '..', 'video2rig', 'blender',
                                                       'render_start.py'))
a = ap.parse_args(argv)
import json  # noqa: E402

src = open(a.render_start, encoding='utf-8').read()
ns = {'__name__': 'render_start_for_bench', '__file__': a.render_start}
exec(compile(src[:src.rindex('\nmain()')], a.render_start, 'exec'), ns)      # its studio, without rendering a start

scene = bpy.context.scene
frames = [int(f) for f in a.frames.split(',') if f]
scene.frame_set(frames[0])
objs = [o for o in scene.objects if o.type == 'MESH' and o.visible_get()]
lo, hi = ns['bounds'](objs)
cams = json.load(open(a.cams, encoding='utf-8'))
floor = lo.copy()
floor.z = lo.z - float(cams.get('hover', 0.0))
ns['studio'](floor, hi, a.samples, a.engine)
views = []
for rec in cams['views']:
    data = bpy.data.cameras.new(rec['name'])
    data.type = 'ORTHO'
    data.ortho_scale = rec['ortho_height']
    data.clip_end = 1000.0
    cam = bpy.data.objects.new(rec['name'], data)
    scene.collection.objects.link(cam)
    R = Matrix((rec['right'], rec['up'], [-v for v in rec['forward']])).transposed()
    cam.matrix_world = Matrix.Translation(Vector(rec['location'])) @ R.to_4x4()
    views.append((cam, int(rec['width_px']), int(rec['height_px'])))
os.makedirs(a.out, exist_ok=True)
tmp = os.path.join(a.out, '_panel.png')
for n, f in enumerate(frames):
    scene.frame_set(f)
    panels = []
    for cam, w, h in views:
        scene.camera = cam
        scene.render.resolution_x, scene.render.resolution_y = w, h
        scene.render.resolution_percentage = 100
        scene.render.filepath = tmp
        bpy.ops.render.render(write_still=True)
        img = bpy.data.images.load(tmp)
        panels.append(np.array(img.pixels[:], np.float32).reshape(img.size[1], img.size[0], 4))
        bpy.data.images.remove(img)
    full = np.concatenate(panels, axis=1)
    x = 0
    for k in range(len(views) - 1):
        x += views[k][1]
        if a.divider > 0:
            full[:, x - a.divider // 2:x + (a.divider + 1) // 2, :3] = .03
    out = bpy.data.images.new('v2r_frame', full.shape[1], full.shape[0], alpha=True)
    out.pixels.foreach_set(full.ravel())
    out.filepath_raw = os.path.join(a.out, f'frame_{n:04d}.png')
    out.file_format = 'PNG'
    out.save()
    bpy.data.images.remove(out)
    if n % 20 == 0:
        print(f'RENDER_CLIP {n + 1}/{len(frames)}', flush=True)
os.remove(tmp)
print(f'RENDER_CLIP done: {len(frames)} frames -> {a.out}', flush=True)

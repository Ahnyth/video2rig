"""Issue #2 check (Blender): bone heat on a body whose main pieces include clumps no bone lies in (loose fur).

    blender -b --python bench/heat_loose_pieces.py -- <path to blender/autorig_build.py>
Prints HEAT_OK with the share of vertices that got weights, or the error."""
import sys

import bpy
import numpy as np

path = sys.argv[sys.argv.index('--') + 1]
src = open(path, encoding='utf-8').read()
src = src[:src.rindex('try:\n    main()')]                     # the functions, without running the script
ns = {'__name__': 'autorig_build_under_test', '__file__': path}
exec(compile(src, path, 'exec'), ns)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.object.armature_add(location=(0, 0, 0))
arm = bpy.context.object
bpy.ops.object.mode_set(mode='EDIT')
eb = arm.data.edit_bones
b0 = eb[0]
b0.head, b0.tail = (0, 0, .2), (0, 0, 1.2)
prev = b0
for k, z in enumerate((2.2, 2.8)):
    b = eb.new(f'Bone.{k + 1}')
    b.head, b.tail, b.parent, b.use_connect = prev.tail, (0, 0, z), prev, True
    prev = b
bpy.ops.object.mode_set(mode='OBJECT')
# a body round the bones, and three fur clumps beside it (each over 2% of the vertices: main pieces) with no bone in
bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=.25, depth=3, location=(0, 0, 1.5))
body = bpy.context.object
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.subdivide(number_cuts=12)
bpy.ops.object.mode_set(mode='OBJECT')
cards = '--cards' in sys.argv
if not cards:
    for z in (.8, 1.6, 2.4):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=.12, location=(.6, 0, z))
        bpy.context.object.select_set(True)
else:
    # fur cards: thin quads starting on body vertices (the same place: loose_parts joins them to the body), tilted
    rng = np.random.default_rng(0)
    co = np.array([v.co for v in body.data.vertices]) + np.array(body.location)
    side = co[np.abs(np.hypot(co[:, 0], co[:, 1]) - .25) < 1e-3]
    verts, faces = [], []
    for base in side[rng.choice(len(side), 600, replace=False)]:
        out = np.array([base[0], base[1], 0.0]) / .25 + rng.normal(0, .6, 3)
        out /= np.linalg.norm(out)
        w = np.cross(out, rng.normal(0, 1, 3))
        w = .012 * w / np.linalg.norm(w)
        tip = base + .22 * out
        k = len(verts)
        verts += [tuple(base), tuple(base + w), tuple(tip + w), tuple(tip)]
        faces.append((k, k + 1, k + 2, k + 3))
    me = bpy.data.meshes.new('cards')
    me.from_pydata(verts, [], faces)
    ob = bpy.data.objects.new('cards', me)
    bpy.context.scene.collection.objects.link(ob)
    ob.select_set(True)
body.select_set(True)
bpy.context.view_layer.objects.active = body
bpy.ops.object.join()
if '--no-deform' in sys.argv:                                 # heat that reaches nothing: the guard
    for b in arm.data.bones:
        b.use_deform = False
print('HEAT_SCENE', len(body.data.vertices), 'vertices', flush=True)
try:
    W, names, part, main = ns['heat_on_main'](body, arm, {}, solid=cards)
    print(f'HEAT_OK {float((W.sum(1) > 1e-6).mean()) * 100:.0f}% of the vertices weighted, bones {names}', flush=True)
except BaseException as e:
    print('HEAT_FAIL', type(e).__name__, e, flush=True)

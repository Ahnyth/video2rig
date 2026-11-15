"""A test creature for the fit bench (Blender): Rigify's basic human armature, a mesh grown round its bones (Skin
modifier), automatic weights. Saved as the .blend given after --.

    blender -b --python bench/make_creature.py -- bench/creature.blend"""
import sys

import bpy
import numpy as np

out = sys.argv[sys.argv.index('--') + 1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.preferences.addon_enable(module='rigify')
bpy.ops.object.armature_basic_human_metarig_add()
arm = bpy.context.object
arm.name = 'Armature'
# one vertex per joint, one edge per bone, a radius by the bone's kind
verts, edges, radius, at = [], [], [], {}

def vid(p, r):
    key = tuple(round(c, 4) for c in p)
    if key not in at:
        at[key] = len(verts)
        verts.append(tuple(p))
        radius.append(r)
    return at[key]

for b in arm.data.bones:
    n = b.name.lower()
    r = .11 if 'spine' in n or 'pelvis' in n else .045 if any(k in n for k in ('hand', 'forearm', 'shin', 'foot', 'toe')) \
        else .06
    edges.append((vid(arm.matrix_world @ b.head_local, r), vid(arm.matrix_world @ b.tail_local, r)))
me = bpy.data.meshes.new('Body')
me.from_pydata(verts, edges, [])
body = bpy.data.objects.new('Body', me)
bpy.context.scene.collection.objects.link(body)
bpy.context.view_layer.objects.active = body
body.select_set(True)
skin = body.modifiers.new('Skin', 'SKIN')
for v, r in zip(body.data.skin_vertices[0].data, radius):
    v.radius = (r, r)
body.modifiers.new('Subdivision', 'SUBSURF').levels = 1
for m in list(body.modifiers):
    bpy.ops.object.modifier_apply(modifier=m.name)
bpy.ops.object.select_all(action='DESELECT')
body.select_set(True)
arm.select_set(True)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.parent_set(type='ARMATURE_NAME')     # the Armature modifier and an empty group per bone
# weights by hand (bone heat finds no solution on the skin mesh): the two nearest bones, inverse fourth power
co = np.array([v.co for v in body.data.vertices])
segs = [(np.array(arm.matrix_world @ b.head_local), np.array(arm.matrix_world @ b.tail_local), b.name)
        for b in arm.data.bones if b.use_deform]
D = np.zeros((len(co), len(segs)))
for k, (h, t, _) in enumerate(segs):
    d = t - h
    u = np.clip((co - h) @ d / max(d @ d, 1e-12), 0, 1)
    D[:, k] = np.linalg.norm(co - (h + u[:, None] * d), axis=1)
two = np.argsort(D, 1)[:, :2]
w = 1 / np.maximum(np.take_along_axis(D, two, 1), 1e-4) ** 4
w /= w.sum(1, keepdims=True)
for i in range(len(co)):
    for k in range(2):
        body.vertex_groups[segs[two[i, k]][2]].add([i], float(w[i, k]), 'REPLACE')
print('CREATURE', len(arm.data.bones), 'bones,', len(body.data.vertices), 'vertices', flush=True)
bpy.ops.wm.save_as_mainfile(filepath=out)

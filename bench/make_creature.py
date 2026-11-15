"""A test creature for the bench (Blender): Rigify's basic human armature and a jointed mannequin round it (a cylinder
along every bone, a ball at every joint, a bigger one for the head), each vertex weighted to its two nearest bones.
Saved as the .blend given after --.

    blender -b --python bench/make_creature.py -- bench/creature.blend"""
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Vector

out = sys.argv[sys.argv.index('--') + 1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.preferences.addon_enable(module='rigify')
bpy.ops.object.armature_basic_human_metarig_add()
arm = bpy.context.object
arm.name = 'Armature'


def radius(name):
    n = name.lower()
    if 'spine' in n or 'pelvis' in n:
        return .10
    if 'breast' in n:
        return .06
    if any(k in n for k in ('hand', 'forearm', 'shin', 'foot', 'toe', 'heel')):
        return .045
    return .06


bm = bmesh.new()
for b in arm.data.bones:
    if not b.use_deform:
        continue
    head, tail = arm.matrix_world @ b.head_local, arm.matrix_world @ b.tail_local
    r = radius(b.name)
    if b.name == 'spine.006':                                   # the head
        r = .11
    axis = tail - head
    rot = axis.to_track_quat('Z', 'Y').to_matrix().to_4x4()
    mat = rot.copy()
    mat.translation = (head + tail) / 2
    bmesh.ops.create_cone(bm, cap_ends=True, segments=12, radius1=r, radius2=r * .9, depth=axis.length, matrix=mat)
    for p in (head, tail):
        m = rot.copy()
        m.translation = p
        bmesh.ops.create_uvsphere(bm, u_segments=12, v_segments=8, radius=r * 1.05, matrix=m)
me = bpy.data.meshes.new('Body')
bm.normal_update()
bm.to_mesh(me)
bm.free()
body = bpy.data.objects.new('Body', me)
bpy.context.scene.collection.objects.link(body)
bpy.ops.object.select_all(action='DESELECT')
body.select_set(True)
arm.select_set(True)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.parent_set(type='ARMATURE_NAME')     # the Armature modifier and an empty group per bone
# weights: the two nearest bones, inverse fourth power (a joint's ball blends its two bones)
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
mat = bpy.data.materials.new('Skin')
mat.diffuse_color = (.75, .55, .42, 1.0)
body.data.materials.append(mat)
print('CREATURE', len(arm.data.bones), 'bones,', len(body.data.vertices), 'vertices,',
      f'{max(co[:, 2]) - min(co[:, 2]):.2f} m tall', flush=True)
bpy.ops.wm.save_as_mainfile(filepath=out)

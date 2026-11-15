# A ground-truth bench for the reader's fit

The fit only, without a trained reader: a test creature, a known motion, readings simulated from it (the true
keypoints through the start cameras, reader-like noise, hidden stretches), `road.fit_reading` as `run` calls it,
and how far the fit lands from the truth. From the repository's root, with Blender on PATH:

```powershell
blender -b --python bench/make_creature.py -- bench/creature.blend          # Rigify's basic human, a skin, weights
blender bench/creature.blend -b --python blender/export_rig.py -- --out bench/proj/rig
blender bench/creature.blend -b --python blender/render_start.py -- --out bench/proj/start --engine BLENDER_EEVEE --samples 8
python bench/fit_bench.py --clip walk --kind 30-5 --seed 0                  # one JSON line; --kind 24 | 30-5 | 25-25
```

`fit_bench.py` prints kp_err_px (3D keypoints against the truth, px of a 720-row frame), angle_err_deg, vel_err (the
keypoints' speed against the truth by real time, px/s; also only where the readings are hidden), and shake_px of the
truth and of the fit on an even 24 fps grid. `heat_loose_pieces.py` checks the auto-rig's bone heat on a body with
bone-free clumps or fur cards (issue #2).

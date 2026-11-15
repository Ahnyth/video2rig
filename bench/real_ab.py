"""A/B of the reader fit's variants on real readings (no ground truth): every clip that has work/<clip>/reader.npz in
the given project folders, fitted by every variant (a checkout of one branch plus --set values) in its own process.

    python bench/real_ab.py <project> [<project> ...] [--clips a,b] [--variants main,pr6,...] [--out bench/real_ab.json]

Checkouts are detached git worktrees of the clone (--clone) under --worktrees, made on first use. Prints a markdown
table per clip (bench/real_fit_one.py says what each column measures) and keeps every row in --out.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VARIANTS = {                       # label: (git ref, --set values)
    'main': ('main', []),
    'pr6': ('fit-real-time', []),
    'pr6 iters=3000': ('fit-real-time', ['iters=3000']),
    'pr6 w_vel=8': ('fit-real-time', ['w_vel=8']),
    'pr6 w_ang_acc=40': ('exp-ang-acc', ['w_ang_acc=40']),
    'pr8 contact': ('game-features', ['w_contact=1']),
    'pr8 contact+loop': ('game-features', ['w_contact=1', 'loop=true']),
}

ap = argparse.ArgumentParser()
ap.add_argument('projects', nargs='+')
ap.add_argument('--clips', default='')
ap.add_argument('--variants', default=','.join(VARIANTS))
ap.add_argument('--clone', default=str(HERE.parent / 'video2rig'))
ap.add_argument('--worktrees', default=str(HERE.parent / 'wt'))
ap.add_argument('--out', default=str(HERE / 'real_ab.json'))
a = ap.parse_args()


def checkout(ref):
    """A detached worktree of the clone at ref, made once (and moved to ref's tip each run)."""
    path = Path(a.worktrees) / ref
    if not path.exists():
        subprocess.run(['git', '-C', a.clone, 'worktree', 'add', '--detach', str(path), ref], check=True,
                       capture_output=True)
    else:
        subprocess.run(['git', '-C', str(path), 'checkout', '-q', '--detach', ref], check=True)
    return path


rows = []
labels = [v.strip() for v in a.variants.split(',') if v.strip()]
for proj in a.projects:
    proj = Path(proj).resolve()
    clips = [c for c in a.clips.split(',') if c] or sorted(p.parent.name for p in (proj / 'work').glob('*/reader.npz'))
    for clip in clips:
        for label in labels:
            ref, sets = VARIANTS[label]
            cmd = [sys.executable, str(HERE / 'real_fit_one.py'), '--repo', str(checkout(ref)), '--project', str(proj),
                   '--clip', clip] + [x for kv in sets for x in ('--set', kv)]
            run = subprocess.run(cmd, capture_output=True, text=True)
            line = next((ln for ln in run.stdout.splitlines()[::-1] if ln.startswith('{')), None)
            if line is None:
                print(f'{proj.name}/{clip} {label}: failed\n{run.stderr[-1500:]}', file=sys.stderr)
                continue
            row = dict(json.loads(line), variant=label)
            rows.append(row)
            print(f'{proj.name}/{clip} {label}: done in {row["secs"]} s', file=sys.stderr, flush=True)
Path(a.out).write_text(json.dumps(rows, indent=1))

for key in dict.fromkeys((r['project'], r['clip']) for r in rows):
    mine = [r for r in rows if (r['project'], r['clip']) == key]
    panels = list(mine[0]['sure_miss_px'])
    print(f'\n### {key[0]} / {key[1]} ({mine[0]["frames"]} frames)\n')
    print('| variant | ' + ' | '.join(f'sure miss {p} px' for p in panels) +
          ' | shake px | foot slide px/s | loop seam (1 = seamless) | s |')
    print('| --- ' * (len(panels) + 5) + '|')
    for r in mine:
        print(f'| {r["variant"]} | ' + ' | '.join(str(r['sure_miss_px'][p]) for p in panels) +
              f' | {r["shake_px"]} | {r["foot_slide"]} | {r["loop_seam"]} | {r["secs"]} |')

"""
Smoke test for parse_python_file_lsp (Phase 1 of LSP-based graph construction).
Loads one astropy instance from SWE-bench-Lite, clones the repo,
picks a representative .py file, and reports parse results.
"""
import os
import sys
import subprocess

# ── path setup so we can import repograph.utils ────────────────────────────
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "repograph"))

from utils import parse_python_file_lsp
import jedi

# ── clone helpers (inline to avoid pandas dependency in get_repo_structure) ─
repo_to_top_folder = {
    "astropy/astropy": "astropy",
    "django/django": "django",
}

def clone_repo(repo_name, repo_playground):
    top = repo_to_top_folder[repo_name]
    subprocess.run(
        ["git", "clone", f"https://github.com/{repo_name}.git",
         f"{repo_playground}/{top}"],
        check=True,
    )

def checkout_commit(repo_path, commit_id):
    subprocess.run(["git", "-C", repo_path, "checkout", commit_id], check=True)

# ---------------------------------------------------------------------------
# Step 1: Load dataset
# ---------------------------------------------------------------------------
print("=" * 60)
print("Step 1: Loading SWE-bench-Lite dataset …")
from datasets import load_dataset

swe_bench = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
astropy_instance = [x for x in swe_bench if x["instance_id"].startswith("astropy")][0]

instance_id  = astropy_instance["instance_id"]
repo_name    = astropy_instance["repo"]           # e.g. "astropy/astropy"
base_commit  = astropy_instance["base_commit"]

print(f"  instance_id : {instance_id}")
print(f"  repo        : {repo_name}")
print(f"  base_commit : {base_commit}")

# ---------------------------------------------------------------------------
# Step 2: Clone repo (skip if already present)
# ---------------------------------------------------------------------------
PLAYGROUND = os.path.join(REPO_ROOT, "playground")
os.makedirs(PLAYGROUND, exist_ok=True)

top_folder = repo_to_top_folder[repo_name]        # "astropy"
repo_path  = os.path.join(PLAYGROUND, top_folder) # .../playground/astropy

print("\nStep 2: Cloning repository …")
if os.path.isdir(repo_path):
    print(f"  Already exists at {repo_path}, skipping clone.")
else:
    clone_repo(repo_name, PLAYGROUND)

# ---------------------------------------------------------------------------
# Step 3: Checkout the base commit
# ---------------------------------------------------------------------------
print("\nStep 3: Checking out base commit …")
checkout_commit(repo_path, base_commit)

# ---------------------------------------------------------------------------
# Step 4: Pick a representative .py file
# ---------------------------------------------------------------------------
print("\nStep 4: Selecting a target .py file …")

# Walk the repo and collect .py files that are likely to have both
# classes and functions (skip __init__.py, test files, setup.py)
candidates = []
for root, dirs, files in os.walk(repo_path):
    # Skip hidden dirs, test dirs, build artefacts
    dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('build', 'doc', 'docs', '__pycache__')]
    for fn in files:
        if not fn.endswith('.py'):
            continue
        if fn.startswith('test_') or fn in ('__init__.py', 'setup.py', 'conftest.py'):
            continue
        full = os.path.join(root, fn)
        size = os.path.getsize(full)
        if 2_000 < size < 30_000:          # skip tiny stubs and huge monoliths
            candidates.append((size, full))

candidates.sort(key=lambda x: x[0], reverse=True)

# Take the largest-ish file for a rich result (index 5 to avoid edge cases)
_, target_file = candidates[min(5, len(candidates) - 1)]
rel_target = os.path.relpath(target_file, repo_path)
print(f"  Target file : {rel_target}")
print(f"  Size        : {os.path.getsize(target_file):,} bytes")

# ---------------------------------------------------------------------------
# Step 5: Run parse_python_file_lsp
# ---------------------------------------------------------------------------
print("\nStep 5: Running parse_python_file_lsp …")
project = jedi.Project(path=repo_path)
class_info, function_names, file_lines = parse_python_file_lsp(target_file, project)

print(f"\n{'=' * 60}")
print(f"RESULTS  –  {rel_target}")
print(f"{'=' * 60}")
print(f"  Total source lines  : {len(file_lines)}")
print(f"  Classes found       : {len(class_info)}")
print(f"  Top-level functions : {len(function_names)}")

# ── class_info detail ──────────────────────────────────────────────────────
print(f"\n── class_info ──────────────────────────────────────────────")
for cls in class_info:
    full = cls.get('full_name', '(none)')
    print(f"  [{cls['start_line']:4d}–{cls['end_line']:4d}]  {cls['name']}")
    print(f"             full_name : {full}")
    print(f"             methods   : {[m['name'] for m in cls['methods']]}")

# ── function_names detail ──────────────────────────────────────────────────
print(f"\n── function_names (top-level) ──────────────────────────────")
for fn in function_names:
    full = fn.get('full_name', '(none)')
    print(f"  [{fn['start_line']:4d}–{fn['end_line']:4d}]  {fn['name']}")
    print(f"             full_name : {full}")

# ── sanity checks ──────────────────────────────────────────────────────────
print(f"\n── Sanity checks ───────────────────────────────────────────")

full_name_missing = [
    c['name'] for c in class_info if not c.get('full_name')
] + [
    f['name'] for f in function_names if not f.get('full_name')
]
if full_name_missing:
    print(f"  ⚠ full_name is empty for: {full_name_missing}")
else:
    print(f"  ✓ full_name populated for all classes and functions")

method_missing_full = [
    (c['name'], m['name'])
    for c in class_info for m in c['methods']
    if not m.get('full_name')
]
if method_missing_full:
    print(f"  ⚠ full_name missing for methods: {method_missing_full}")
else:
    print(f"  ✓ full_name populated for all methods")

text_ok = all(
    isinstance(c.get('text'), list) for c in class_info
) and all(
    isinstance(f.get('text'), list) for f in function_names
)
print(f"  {'✓' if text_ok else '✗'} text field is list[str] (backward-compat)")

line_ok = all(
    c['start_line'] <= c['end_line'] for c in class_info
) and all(
    f['start_line'] <= f['end_line'] for f in function_names
)
print(f"  {'✓' if line_ok else '✗'} start_line <= end_line for all entries")

print("\nDone.")

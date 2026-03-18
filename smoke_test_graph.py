"""
Smoke test for Phase 2: jedi-based CodeGraph (construct_graph.py).

Tests with a small slice of the astropy repo (3–5 files) to keep
jedi.goto() calls manageable while still exercising cross-file resolution.

Checks:
  - Node count / full_name IDs
  - Edge type distribution (contain / invoke)
  - Cross-file invoke edges (caller and callee from different files)
"""
import os, sys
from pathlib import Path
from collections import Counter

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "repograph"))

from construct_graph import CodeGraph

ASTROPY_PATH = os.path.join(REPO_ROOT, "playground", "astropy")

# ---------------------------------------------------------------------------
# Pick a small, well-connected slice of the repo
# ---------------------------------------------------------------------------
TARGET_FILES = [
    # nddata subpackage — bitmask.py references nddata_base.py helpers
    os.path.join(ASTROPY_PATH, "astropy", "nddata", "bitmask.py"),
    os.path.join(ASTROPY_PATH, "astropy", "nddata", "nddata_base.py"),
    os.path.join(ASTROPY_PATH, "astropy", "nddata", "nddata.py"),
]
# Filter to files that actually exist
TARGET_FILES = [f for f in TARGET_FILES if os.path.isfile(f)]

print("=" * 65)
print("Phase 2 smoke test – CodeGraph (jedi-based)")
print("=" * 65)
print(f"\nTarget files ({len(TARGET_FILES)}):")
for f in TARGET_FILES:
    print(f"  {os.path.relpath(f, ASTROPY_PATH)}")

# ---------------------------------------------------------------------------
# Build CodeGraph
# ---------------------------------------------------------------------------
print("\nBuilding CodeGraph …")
cg = CodeGraph(root=ASTROPY_PATH)
tags, G = cg.get_code_graph(TARGET_FILES)

print(f"\n{'=' * 65}")
print("GRAPH RESULTS")
print(f"{'=' * 65}")
print(f"  Total tags generated : {len(tags)}")
print(f"  Nodes (def symbols)  : {len(G.nodes)}")
print(f"  Edges (total)        : {len(G.edges)}")

# ---------------------------------------------------------------------------
# Edge type distribution
# ---------------------------------------------------------------------------
edge_types = Counter()
for u, v, data in G.edges(data=True):
    edge_types[data.get("edge_type", "unknown")] += 1

print(f"\n── Edge type distribution ──────────────────────────────────")
for etype, count in sorted(edge_types.items()):
    print(f"  {etype:<12} : {count}")

# ---------------------------------------------------------------------------
# Node sample (first 10)
# ---------------------------------------------------------------------------
print(f"\n── Node sample (first 10) ──────────────────────────────────")
for node_id in list(G.nodes)[:10]:
    attrs = G.nodes[node_id]
    print(f"  [{attrs.get('category','?'):8s}]  {node_id}")

# ---------------------------------------------------------------------------
# Contain edges sample
# ---------------------------------------------------------------------------
contain_edges = [
    (u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "contain"
]
print(f"\n── contain edges sample (up to 5) ──────────────────────────")
for u, v in contain_edges[:5]:
    print(f"  {u}")
    print(f"    → {v}")

# ---------------------------------------------------------------------------
# Invoke edges sample
# ---------------------------------------------------------------------------
invoke_edges = [
    (u, v, d) for u, v, d in G.edges(data=True) if d.get("edge_type") == "invoke"
]
print(f"\n── invoke edges sample (up to 5) ───────────────────────────")
for u, v, _ in invoke_edges[:5]:
    print(f"  {u}")
    print(f"    → {v}")

# ---------------------------------------------------------------------------
# Cross-file invoke edges (the key LSP advantage over tree-sitter)
# ---------------------------------------------------------------------------
def file_of(node_id):
    """Return fname from node attributes, or guess from full_name prefix."""
    attrs = G.nodes.get(node_id, {})
    return attrs.get("fname", "")

cross_file = [
    (u, v)
    for u, v, d in G.edges(data=True)
    if d.get("edge_type") == "invoke"
    and file_of(u) and file_of(v)
    and file_of(u) != file_of(v)
]
print(f"\n── Cross-file invoke edges ──────────────────────────────────")
print(f"  Total : {len(cross_file)}")
for u, v in cross_file[:5]:
    u_rel = os.path.relpath(file_of(u), ASTROPY_PATH)
    v_rel = os.path.relpath(file_of(v), ASTROPY_PATH)
    print(f"  [{u_rel}]  {u.split('.')[-1]}")
    print(f"    → [{v_rel}]  {v.split('.')[-1]}")

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
print(f"\n── Sanity checks ───────────────────────────────────────────")

# 1. All node IDs should look like dotted qualified names
bad_ids = [n for n in G.nodes if "." not in n]
print(f"  {'✓' if not bad_ids else '✗'} All node IDs are dotted qualified names"
      + (f"  (exceptions: {bad_ids[:3]})" if bad_ids else ""))

# 2. All edges have edge_type
missing_type = [(u, v) for u, v, d in G.edges(data=True) if "edge_type" not in d]
print(f"  {'✓' if not missing_type else '✗'} All edges carry edge_type attribute"
      + (f"  ({len(missing_type)} missing)" if missing_type else ""))

# 3. contain edges only go from class nodes to function nodes
bad_contain = [
    (u, v) for u, v, d in G.edges(data=True)
    if d.get("edge_type") == "contain"
    and G.nodes[u].get("category") != "class"
]
print(f"  {'✓' if not bad_contain else '✗'} All contain-edge sources are class nodes"
      + (f"  ({len(bad_contain)} violations)" if bad_contain else ""))

# 4. No self-loop invoke edges
self_loops = [(u, v) for u, v, d in G.edges(data=True)
              if d.get("edge_type") == "invoke" and u == v]
print(f"  {'✓' if not self_loops else '✗'} No self-loop invoke edges"
      + (f"  ({len(self_loops)} found)" if self_loops else ""))

# 5. Cross-file edges exist (proves jedi goto() worked)
print(f"  {'✓' if cross_file else '⚠'} Cross-file invoke edges present"
      + (" (jedi cross-file resolution confirmed)" if cross_file
         else " (none found – goto() may not have resolved across files)"))

print("\nDone.")

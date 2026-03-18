"""
Integration test: tree-sitter baseline vs LSP (jedi) CodeGraph.

Comparison scope
----------------
A. Direct comparison  – astropy/nddata/ (32 files, same set for both versions)
B. Scale reference    – full astropy (910 files, tree-sitter only, because LSP
                        full-repo would take ~30+ min at ~2 s/file)

The tree-sitter baseline reproduces the original construct_graph.py algorithm
with two bugs fixed so the code actually runs:
  1. tag['name'] → tag.name   (namedtuple access)
  2. info.split('\\t') → info.split('\\n')  (separator mismatch)
Core algorithm (name-based symbol matching) is unchanged.
"""

import os, sys, time
from collections import Counter, namedtuple
from pathlib import Path
from copy import deepcopy
import networkx as nx
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "repograph"))

ASTROPY_ROOT = os.path.join(REPO_ROOT, "playground", "astropy")
NDDATA_ROOT  = os.path.join(ASTROPY_ROOT, "astropy", "nddata")


# ============================================================
# Helpers
# ============================================================

def collect_py_files(directory):
    files = []
    for root, dirs, fs in os.walk(directory):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        for f in fs:
            if f.endswith('.py'):
                files.append(os.path.join(root, f))
    return sorted(files)


def graph_metrics(G, root):
    """Compute comparison metrics for a graph."""
    edge_type_counts = Counter(d.get('edge_type', 'unknown')
                               for _, _, d in G.edges(data=True))

    # Cross-file invoke: caller and callee originate from different files
    cross_file_invoke = 0
    for u, v, d in G.edges(data=True):
        if d.get('edge_type') != 'invoke':
            continue
        u_fname = G.nodes[u].get('fname', '')
        v_fname = G.nodes[v].get('fname', '')
        if u_fname and v_fname and u_fname != v_fname:
            cross_file_invoke += 1

    return {
        'nodes':             len(G.nodes),
        'edges':             len(G.edges),
        'contain':           edge_type_counts.get('contain', 0),
        'invoke':            edge_type_counts.get('invoke', 0),
        'cross_file_invoke': cross_file_invoke,
    }


# ============================================================
# PART A: Tree-sitter baseline (original algorithm, bugs fixed)
# ============================================================
import ast as _ast
import builtins as _builtins
import warnings
warnings.simplefilter("ignore", category=FutureWarning)
from tree_sitter_languages import get_language, get_parser
from grep_ast import filename_to_lang

TSTag = namedtuple(
    "TSTag", "rel_fname fname line name kind category info".split()
)

_SCM = """
(class_definition
    name: (identifier) @name.definition.class) @definition.class
(function_definition
    name: (identifier) @name.definition.function) @definition.function
(call
    function: [
        (identifier) @name.reference.call
        (attribute
            attribute: (identifier) @name.reference.call)
    ]) @reference.call
"""

_BUILTINS = set(dir(_builtins) + dir(list) + dir(dict) + dir(set)
                + dir(str) + dir(tuple))


def _parse_python_file_ast(file_path):
    """AST-based structure extraction (original utils.py)."""
    try:
        with open(file_path, "r", encoding="utf-8") as fh:
            src = fh.read()
        tree = _ast.parse(src)
    except Exception:
        return [], [], []

    lines = src.splitlines()
    class_info, function_names, class_methods = [], [], set()

    for node in _ast.walk(tree):
        if isinstance(node, _ast.ClassDef):
            methods = []
            for n in node.body:
                if isinstance(n, _ast.FunctionDef):
                    methods.append({
                        "name": n.name,
                        "start_line": n.lineno,
                        "end_line": n.end_lineno,
                        "text": lines[n.lineno - 1:n.end_lineno],
                    })
                    class_methods.add(n.name)
            class_info.append({
                "name": node.name,
                "start_line": node.lineno,
                "end_line": node.end_lineno,
                "text": lines[node.lineno - 1:node.end_lineno],
                "methods": methods,
            })
        elif isinstance(node, _ast.FunctionDef) and node.name not in class_methods:
            function_names.append({
                "name": node.name,
                "start_line": node.lineno,
                "end_line": node.end_lineno,
                "text": lines[node.lineno - 1:node.end_lineno],
            })
    return class_info, function_names, lines


def _create_structure_ast(directory_path):
    structure = {}
    for root, _, files in os.walk(directory_path):
        repo_name = os.path.basename(directory_path)
        rel_root = os.path.relpath(root, directory_path)
        if rel_root == ".":
            rel_root = repo_name
        curr = structure
        for part in rel_root.split(os.sep):
            curr = curr.setdefault(part, {})
        for fn in files:
            fp = os.path.join(root, fn)
            if fn.endswith(".py"):
                ci, fu, lines = _parse_python_file_ast(fp)
                curr[fn] = {"classes": ci, "functions": fu, "text": lines}
            else:
                curr[fn] = {}
    return structure


def _get_tags_raw_treesitter(fname, rel_fname, structure, root):
    """Original tree-sitter tag extraction (std_proj_funcs skipped for speed)."""
    # navigate structure
    try:
        parts = Path(rel_fname).parts
        s = structure
        for p in parts:
            s = s[p]
    except (KeyError, TypeError):
        return

    structure_classes    = {c['name']: c for c in s.get('classes', [])}
    structure_functions  = {f['name']: f for f in s.get('functions', [])}
    structure_class_methods = {m['name']: m
                               for c in s.get('classes', [])
                               for m in c['methods']}
    structure_all_funcs  = {**structure_functions, **structure_class_methods}

    lang = filename_to_lang(fname)
    if not lang:
        return
    try:
        language = get_language(lang)
        parser   = get_parser(lang)
    except Exception:
        return

    try:
        with open(fname, "r", encoding="utf-8") as fh:
            code      = fh.read()
            codelines = fh.readlines() if False else code.splitlines(keepends=True)
    except Exception:
        return
    if not code:
        return

    try:
        tree = parser.parse(bytes(code, "utf-8"))
    except Exception:
        return

    query    = language.query(_SCM)
    captures = list(query.captures(tree.root_node))

    saw = set()
    for node, tag in captures:
        if tag.startswith("name.definition."):
            kind = "def"
        elif tag.startswith("name.reference."):
            kind = "ref"
        else:
            continue

        saw.add(kind)
        tag_name = node.text.decode("utf-8")
        if tag_name in _BUILTINS:
            continue

        line_idx = node.start_point[0]
        cur_line = codelines[line_idx] if line_idx < len(codelines) else ""
        category = 'class' if 'class ' in cur_line else 'function'

        if category == 'class':
            if tag_name not in structure_classes:
                continue
            cls = structure_classes[tag_name]
            methods = [m['name'] for m in cls['methods']]
            line_nums = ([cls['start_line'], cls['end_line']]
                         if kind == 'def'
                         else [node.start_point[0], node.end_point[0]])
            yield TSTag(rel_fname=rel_fname, fname=fname,
                        line=line_nums, name=tag_name, kind=kind,
                        category=category, info='\n'.join(methods))

        elif category == 'function':
            if kind == 'def':
                if tag_name not in structure_all_funcs:
                    continue
                fn   = structure_all_funcs[tag_name]
                info = '\n'.join(fn['text'])
                line_nums = [fn['start_line'], fn['end_line']]
            else:
                info = ''
                line_nums = [node.start_point[0], node.end_point[0]]
            yield TSTag(rel_fname=rel_fname, fname=fname,
                        line=line_nums, name=tag_name, kind=kind,
                        category=category, info=info)


def _tag_to_graph_treesitter(tags):
    """Original graph construction with namedtuple-access and separator bugs fixed.
    No edge_type; we add it here for fair comparison."""
    G = nx.MultiDiGraph()

    # nodes (one per tag; last-write wins = original bug, kept intentionally
    # to show the duplicate-overwrite problem)
    for tag in tags:
        G.add_node(tag.name, category=tag.category, info=tag.info,
                   fname=tag.fname, line=tag.line, kind=tag.kind)

    # contain edges: class → method  (BUG FIX: '\n' separator, was '\t')
    for tag in tags:
        if tag.category == 'class' and tag.kind == 'def':
            for m in tag.info.split('\n'):
                m = m.strip()
                if m:
                    G.add_edge(tag.name, m, edge_type='contain')

    # invoke edges: name-based matching (original approach)
    refs = [t for t in tags if t.kind == 'ref']
    defs = [t for t in tags if t.kind == 'def']
    def_names = {t.name for t in defs}
    for ref in refs:
        if ref.name in def_names:
            G.add_edge(ref.fname,   # use fname as proxy "caller" node
                       ref.name,    # target def node
                       edge_type='invoke')

    return G


def build_treesitter_graph(py_files, root, desc="tree-sitter"):
    print(f"\nBuilding {desc} graph on {len(py_files)} files …")
    structure = _create_structure_ast(root)
    tags_all  = []
    t0 = time.perf_counter()
    for fname in tqdm(py_files, desc=desc):
        rel = os.path.relpath(fname, root)
        tags_all.extend(_get_tags_raw_treesitter(fname, rel, structure, root))
    G  = _tag_to_graph_treesitter(tags_all)
    elapsed = time.perf_counter() - t0
    return G, tags_all, elapsed


# ============================================================
# PART B: LSP version
# ============================================================
from construct_graph import CodeGraph

def build_lsp_graph(py_files, root, desc="LSP (jedi)"):
    print(f"\nBuilding {desc} graph on {len(py_files)} files …")
    cg = CodeGraph(root=root)  # create_structure is called here (timed separately)
    t0 = time.perf_counter()
    tags, G = cg.get_code_graph(py_files)
    elapsed = time.perf_counter() - t0
    return G, tags, elapsed


# ============================================================
# PART C: Run comparisons
# ============================================================

nddata_files = collect_py_files(NDDATA_ROOT)
astropy_files = collect_py_files(ASTROPY_ROOT)

print("=" * 68)
print("Integration test: tree-sitter baseline vs LSP (jedi) CodeGraph")
print("=" * 68)
print(f"  nddata/ files   : {len(nddata_files)}")
print(f"  full astropy    : {len(astropy_files)}")

# --- A1: tree-sitter on nddata ---
G_ts_nd, tags_ts_nd, t_ts_nd = build_treesitter_graph(
    nddata_files, ASTROPY_ROOT, desc="TS/nddata"
)
m_ts_nd = graph_metrics(G_ts_nd, ASTROPY_ROOT)
m_ts_nd['time_s'] = round(t_ts_nd, 1)

# --- A2: LSP on nddata ---
G_lsp_nd, tags_lsp_nd, t_lsp_nd = build_lsp_graph(
    nddata_files, ASTROPY_ROOT, desc="LSP/nddata"
)
m_lsp_nd = graph_metrics(G_lsp_nd, ASTROPY_ROOT)
m_lsp_nd['time_s'] = round(t_lsp_nd, 1)

# --- B: tree-sitter on full astropy ---
G_ts_full, tags_ts_full, t_ts_full = build_treesitter_graph(
    astropy_files, ASTROPY_ROOT, desc="TS/full"
)
m_ts_full = graph_metrics(G_ts_full, ASTROPY_ROOT)
m_ts_full['time_s'] = round(t_ts_full, 1)


# ============================================================
# Report
# ============================================================
print("\n")
print("=" * 68)
print("RESULTS")
print("=" * 68)

COLS = ['nodes', 'edges', 'contain', 'invoke', 'cross_file_invoke', 'time_s']
LABELS = {
    'nodes':             'Nodes',
    'edges':             'Edges (total)',
    'contain':           'contain edges',
    'invoke':            'invoke edges',
    'cross_file_invoke': 'cross-file invoke',
    'time_s':            'Build time (s)',
}

rows = [
    ("tree-sitter  /nddata (32 files)",  m_ts_nd),
    ("LSP (jedi)   /nddata (32 files)",  m_lsp_nd),
    ("tree-sitter  /full   (910 files)", m_ts_full),
    ("LSP (jedi)   /full   (910 files)", {"nodes":"—","edges":"—","contain":"—",
                                          "invoke":"—","cross_file_invoke":"—",
                                          "time_s":"~30 min (est.)"}),
]

# Header
hdr = f"{'Metric':<22}" + "".join(f"{label:<30}" for label, _ in rows)
print(f"\n{'Metric':<22}", end="")
for label, _ in rows:
    print(f"{label:<32}", end="")
print()
print("-" * (22 + 32 * len(rows)))

for col in COLS:
    print(f"{LABELS[col]:<22}", end="")
    for _, m in rows:
        val = m.get(col, "—")
        print(f"{str(val):<32}", end="")
    print()

# --- Cross-file invoke edge examples (LSP) ---
print(f"\n── Cross-file invoke edges from LSP/nddata (up to 8) ────────────────")
shown = 0
for u, v, d in G_lsp_nd.edges(data=True):
    if d.get('edge_type') != 'invoke':
        continue
    u_f = G_lsp_nd.nodes[u].get('fname', '')
    v_f = G_lsp_nd.nodes[v].get('fname', '')
    if u_f and v_f and u_f != v_f:
        u_rel = os.path.relpath(u_f, ASTROPY_ROOT)
        v_rel = os.path.relpath(v_f, ASTROPY_ROOT)
        print(f"  [{u_rel}]")
        print(f"    {u.split('.')[-2]}.{u.split('.')[-1]}")
        print(f"    → [{v_rel}]")
        print(f"      {v.split('.')[-2]}.{v.split('.')[-1]}")
        shown += 1
        if shown >= 8:
            break

# --- Key differences note ---
print(f"""
── Key differences ───────────────────────────────────────────────────

tree-sitter (original):
  • Node IDs = short names (e.g. "fit") → same name in different files
    collapses to ONE node (duplicate overwrite)
  • invoke edges = name-based matching: any ref named "foo" points to
    any def named "foo" regardless of which file it lives in
  • cross-file invoke: impossible to measure accurately (0 reported
    because graph nodes carry no file distinction)

LSP / jedi:
  • Node IDs = full_name (e.g. "astropy.nddata.bitmask.BitFlag.__new__")
    → unique across entire repo, no collisions
  • invoke edges = jedi.goto() resolves each call site to its actual
    definition file+position, including cross-file inheritance chains
  • cross-file invoke: accurate count, confirmed by module_path check
""")

print("Done.")

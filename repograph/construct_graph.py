# This file is adapted from the following sources:
# RepoMap: https://github.com/paul-gauthier/aider/blob/main/aider/repomap.py
# Agentless: https://github.com/OpenAutoCoder/Agentless/blob/main/get_repo_structure/get_repo_structure.py
# grep-ast: https://github.com/paul-gauthier/grep-ast

import colorsys
import os
import random
import sys
from collections import defaultdict, namedtuple
from pathlib import Path
import networkx as nx
from grep_ast import TreeContext
from tqdm import tqdm
import jedi
import pickle
import json
from copy import deepcopy
from utils import create_structure

# ---------------------------------------------------------------------------
# Tag: the unit of information passed between get_tags_raw → tag_to_graph.
#
# Fields added vs the original:
#   full_name        – fully-qualified name resolved by jedi
#                      e.g. "astropy.nddata.bitmask.BitFlagNameMeta.__new__"
#   caller_full_name – full_name of the enclosing def that makes the call
#                      (populated for kind='ref' tags only; None for def tags)
# ---------------------------------------------------------------------------
Tag = namedtuple(
    "Tag",
    "rel_fname fname line name kind category info full_name caller_full_name".split(),
)


class CodeGraph:

    warned_files = set()

    def __init__(
        self,
        map_tokens=1024,
        root=None,
        main_model=None,
        io=None,
        repo_content_prefix=None,
        verbose=False,
        max_context_window=None,
    ):
        self.io = io
        self.verbose = verbose

        if not root:
            root = os.getcwd()
        self.root = root

        self.max_map_tokens = map_tokens
        self.max_context_window = max_context_window

        self.repo_content_prefix = repo_content_prefix

        # build structure dict (jedi-based, via Phase-1 utils.py)
        self.structure = create_structure(self.root)

        # shared jedi project so all goto() calls resolve across the whole repo
        self.project = jedi.Project(path=self.root)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def get_code_graph(self, other_files, mentioned_fnames=None):
        if self.max_map_tokens <= 0:
            return
        if not other_files:
            return
        if not mentioned_fnames:
            mentioned_fnames = set()

        tags = self.get_tag_files(other_files, mentioned_fnames)
        code_graph = self.tag_to_graph(tags)
        return tags, code_graph

    def get_tag_files(self, other_files, mentioned_fnames=None):
        try:
            tags = self.get_ranked_tags(other_files, mentioned_fnames)
            return tags
        except RecursionError:
            if self.io:
                self.io.tool_error("Disabling code graph, git repo too large?")
            self.max_map_tokens = 0
            return

    # -----------------------------------------------------------------------
    # Graph construction
    # -----------------------------------------------------------------------

    def tag_to_graph(self, tags):
        """Build a MultiDiGraph from Tag objects.

        Node IDs are full_name strings (unique across the whole repo).
        Edge types:
          'contain' – class node → method node
          'invoke'  – caller def → callee def (cross-file resolved by jedi)
        """
        G = nx.MultiDiGraph()

        # --- Nodes: one per def tag, keyed by full_name ---
        for tag in tags:
            if tag.kind != "def":
                continue
            node_id = tag.full_name or tag.name
            if node_id not in G.nodes:
                G.add_node(
                    node_id,
                    name=tag.name,
                    category=tag.category,
                    info=tag.info,
                    fname=tag.fname,
                    line=tag.line,
                    kind=tag.kind,
                    full_name=node_id,
                )

        # --- contain edges: class → method (via full_name prefix) ---
        for tag in tags:
            if tag.kind != "def" or tag.category != "class":
                continue
            cls_id = tag.full_name or tag.name
            if cls_id not in G.nodes:
                continue
            for method_name in tag.info.split("\n"):
                method_name = method_name.strip()
                if not method_name:
                    continue
                method_id = f"{cls_id}.{method_name}"
                if method_id in G.nodes:
                    G.add_edge(cls_id, method_id, edge_type="contain")

        # --- invoke edges: caller → callee (jedi cross-file resolution) ---
        for tag in tags:
            if tag.kind != "ref":
                continue
            caller_id = tag.caller_full_name
            callee_id = tag.full_name
            if not caller_id or not callee_id:
                continue
            if caller_id not in G.nodes or callee_id not in G.nodes:
                continue
            if caller_id == callee_id:
                continue  # skip self-loops
            G.add_edge(caller_id, callee_id, edge_type="invoke")

        return G

    # -----------------------------------------------------------------------
    # Tag extraction
    # -----------------------------------------------------------------------

    def get_tags(self, fname, rel_fname):
        file_mtime = self.get_mtime(fname)
        if file_mtime is None:
            return []
        return list(self.get_tags_raw(fname, rel_fname))

    def get_tags_raw(self, fname, rel_fname):
        """Yield Tag objects for fname.

        def tags  – sourced directly from self.structure (populated by jedi in
                    Phase 1 utils.py).  No tree-sitter needed.
        ref tags  – each call site resolved via jedi.Script.goto() which
                    follows imports and returns the actual definition location,
                    potentially in a different file.  This is the key advantage
                    over the previous tree-sitter approach.
        """
        # -- 1. Locate this file's entry in the pre-built structure dict ----
        s = self._get_file_structure(rel_fname)
        if s is None:
            return

        # -- 2. Yield DEF tags directly from structure ----------------------
        def_tags = []  # also kept for scope lookup during ref resolution

        for cls in s.get("classes", []):
            cls_full = cls.get("full_name") or cls["name"]
            t = Tag(
                rel_fname=rel_fname,
                fname=fname,
                line=[cls["start_line"], cls["end_line"]],
                name=cls["name"],
                kind="def",
                category="class",
                # info stores method simple names (newline-separated)
                # used by tag_to_graph to build contain edges
                info="\n".join(m["name"] for m in cls["methods"]),
                full_name=cls_full,
                caller_full_name=None,
            )
            def_tags.append(t)
            yield t

            for method in cls["methods"]:
                m_full = method.get("full_name") or method["name"]
                t = Tag(
                    rel_fname=rel_fname,
                    fname=fname,
                    line=[method["start_line"], method["end_line"]],
                    name=method["name"],
                    kind="def",
                    category="function",
                    info="\n".join(method["text"]),
                    full_name=m_full,
                    caller_full_name=None,
                )
                def_tags.append(t)
                yield t

        for fn in s.get("functions", []):
            fn_full = fn.get("full_name") or fn["name"]
            t = Tag(
                rel_fname=rel_fname,
                fname=fname,
                line=[fn["start_line"], fn["end_line"]],
                name=fn["name"],
                kind="def",
                category="function",
                info="\n".join(fn["text"]),
                full_name=fn_full,
                caller_full_name=None,
            )
            def_tags.append(t)
            yield t

        # -- 3. Yield REF tags via jedi.Script.goto() -----------------------
        #
        # For every name reference found in the file:
        #   a. script.goto(line, col, follow_imports=True) resolves the name
        #      to its definition, potentially in another file.
        #   b. We filter to project-internal definitions only (module_path
        #      must be under self.root).
        #   c. The caller scope is the narrowest def tag whose line range
        #      contains the reference line.
        #
        # This is the core LSP improvement: `goto()` can trace
        # `self.method()` → actual class method definition across files,
        # while the old tree-sitter approach only saw the bare symbol name.
        try:
            with open(fname, "r", encoding="utf-8") as fh:
                code = fh.read()
            script = jedi.Script(code=code, path=fname, project=self.project)
            # definitions=False, references=True  → usage/call sites only
            ref_names = script.get_names(
                all_scopes=True, definitions=False, references=True
            )
        except Exception as exc:
            if self.verbose:
                print(f"[get_tags_raw] jedi error in {fname}: {exc}")
            return

        seen: set = set()

        for ref in ref_names:
            key = (ref.line, ref.column)
            if key in seen:
                continue
            seen.add(key)

            try:
                # follow_imports=True  : cross-file resolution (the LSP win)
                # follow_builtin_imports=False : skip stdlib internals
                defs = script.goto(
                    ref.line,
                    ref.column,
                    follow_imports=True,
                    follow_builtin_imports=False,
                )
            except Exception:
                continue

            for defn in defs:
                if not defn.full_name or not defn.module_path:
                    continue  # unresolved or builtin

                # Keep only references that resolve to project-internal files
                try:
                    Path(str(defn.module_path)).relative_to(self.root)
                except ValueError:
                    continue  # resolved to an external library → skip

                if defn.type not in ("function", "class"):
                    continue  # we only track function/class calls

                caller = self._get_enclosing_scope(ref.line, def_tags)
                yield Tag(
                    rel_fname=rel_fname,
                    fname=fname,
                    line=[ref.line, ref.line],
                    name=defn.name,
                    kind="ref",
                    category="function" if defn.type == "function" else "class",
                    info="",
                    full_name=defn.full_name,
                    caller_full_name=caller,
                )
                break  # first resolved definition is sufficient per call site

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _get_file_structure(self, rel_fname):
        """Navigate self.structure to the dict for rel_fname.

        Uses pathlib.Path.parts for cross-platform separator handling
        (the old code split by '/' which broke on Windows).
        """
        try:
            parts = Path(rel_fname).parts
            s = self.structure
            for part in parts:
                s = s[part]
            return s
        except (KeyError, TypeError):
            return None

    def _get_enclosing_scope(self, line, def_tags):
        """Return the full_name of the narrowest def tag whose range contains line.

        Used to determine which function/method is the caller for a ref tag.
        Falls back to None if the line is at module level (no enclosing def).
        """
        best = None
        best_size = float("inf")
        for tag in def_tags:
            start, end = tag.line
            if start <= line <= end:
                size = end - start
                if size < best_size:
                    best_size = size
                    best = tag.full_name
        return best

    def get_rel_fname(self, fname):
        return os.path.relpath(fname, self.root)

    def get_mtime(self, fname):
        try:
            return os.path.getmtime(fname)
        except FileNotFoundError:
            if self.io:
                self.io.tool_error(f"File not found error: {fname}")

    def get_ranked_tags(self, other_fnames, mentioned_fnames):
        tags_of_files = []
        personalization = {}
        fnames = sorted(set(other_fnames))
        personalize = 10 / len(fnames) if fnames else 1

        for fname in tqdm(fnames):
            if not Path(fname).is_file():
                if fname not in self.warned_files:
                    msg = (
                        f"Code graph can't include {fname}, it is not a normal file"
                        if Path(fname).exists()
                        else f"Code graph can't include {fname}, it no longer exists"
                    )
                    if self.io:
                        self.io.tool_error(msg)
                    else:
                        print(msg)
                self.warned_files.add(fname)
                continue

            rel_fname = self.get_rel_fname(fname)
            if fname in mentioned_fnames:
                personalization[rel_fname] = personalize

            tags = list(self.get_tags(fname, rel_fname))
            tags_of_files.extend(tags)

        return tags_of_files

    def render_tree(self, abs_fname, rel_fname, lois):
        with open(str(abs_fname), "r", encoding="utf-8") as f:
            code = f.read() or ""
        if not code.endswith("\n"):
            code += "\n"
        context = TreeContext(
            rel_fname,
            code,
            color=False,
            line_number=False,
            child_context=False,
            last_line=False,
            margin=0,
            mark_lois=False,
            loi_pad=0,
            show_top_of_file_parent_scope=False,
        )
        context.add_lines_of_interest(lois)
        context.add_context()
        return context.format()

    def find_src_files(self, directory):
        if not os.path.isdir(directory):
            return [directory]
        src_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                src_files.append(os.path.join(root, file))
        return src_files

    def find_files(self, dir):
        chat_fnames = []
        for fname in dir:
            if Path(fname).is_dir():
                chat_fnames += self.find_src_files(fname)
            else:
                chat_fnames.append(fname)
        return [f for f in chat_fnames if f.endswith(".py")]


def get_random_color():
    hue = random.random()
    r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(hue, 1, 0.75)]
    return f"#{r:02x}{g:02x}{b:02x}"


if __name__ == "__main__":

    dir_name = sys.argv[1]
    code_graph = CodeGraph(root=dir_name)
    chat_fnames_new = code_graph.find_files([dir_name])

    tags, G = code_graph.get_code_graph(chat_fnames_new)

    print("---------------------------------")
    print(f"Successfully constructed the code graph for repo directory {dir_name}")
    print(f"   Number of nodes: {len(G.nodes)}")
    print(f"   Number of edges: {len(G.edges)}")
    print("---------------------------------")

    with open(f"{os.getcwd()}/graph.pkl", "wb") as f:
        pickle.dump(G, f)

    for tag in tags:
        with open(f"{os.getcwd()}/tags.json", "a+") as f:
            line = json.dumps(
                {
                    "fname": tag.fname,
                    "rel_fname": tag.rel_fname,
                    "line": tag.line,
                    "name": tag.name,
                    "kind": tag.kind,
                    "category": tag.category,
                    "info": tag.info,
                    "full_name": tag.full_name,
                    "caller_full_name": tag.caller_full_name,
                }
            )
            f.write(line + "\n")
    print(f"Cached code graph and tags in '{os.getcwd()}'")

import os
from pathlib import Path
import jedi


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_end_line(jedi_name, file_lines):
    """Return the end line (1-indexed) for a jedi definition Name.

    Primary path: jedi internal API via parso tree node.
      jedi_name._name.tree_name.parent.end_pos[0]

    VERSION DEPENDENCY WARNING:
      _name, tree_name, and end_pos are private/semi-private attributes of
      jedi's BaseName and parso's BaseNode.  They have been stable since
      jedi >= 0.18 (parso backend), but a jedi upgrade may rename or remove
      them.  If this raises AttributeError after a version bump, inspect:
        type(jedi_name._name)           # should be TreeNameDefinition
        jedi_name._name.tree_name       # should be a parso Name node
        jedi_name._name.tree_name.parent  # should be funcdef / classdef node

    Fallback path: indentation-based block scanning (less precise but safe).
    """
    try:
        return jedi_name._name.tree_name.parent.end_pos[0]
    except AttributeError:
        pass
    return _infer_end_line_by_indent(jedi_name.line, file_lines)


def _infer_end_line_by_indent(start_line, file_lines):
    """Infer the last line of an indented block starting at start_line (1-indexed).

    Scans forward from start_line and stops when a non-blank line is found at
    the same or lower indentation level as the opening line.
    """
    if start_line > len(file_lines):
        return start_line

    def_line = file_lines[start_line - 1]
    def_indent = len(def_line) - len(def_line.lstrip())
    last_content_line = start_line

    for i in range(start_line, len(file_lines)):  # 0-indexed scan
        line = file_lines[i]
        if not line.strip() or line.strip().startswith('#'):
            continue  # blank / comment lines do not terminate a block
        indent = len(line) - len(line.lstrip())
        if indent <= def_indent:
            break       # dedented: block has ended
        last_content_line = i + 1  # convert to 1-indexed

    return last_content_line


def _is_method_of(fn_jname, cls_full_name, cls_start, cls_end):
    """Return True if the function definition belongs to the given class.

    Primary check: semantic full_name relationship.
      e.g. fn.full_name == 'pkg.module.MyClass.my_method'
           cls_full_name == 'pkg.module.MyClass'
      → fn.full_name.startswith(cls_full_name + '.') is True

    Fallback: line-range containment (used when full_name is unavailable).
    """
    if fn_jname.full_name and cls_full_name:
        if fn_jname.full_name.startswith(cls_full_name + '.'):
            return True
    # Fallback: the def line sits inside the class body
    return cls_start < fn_jname.line <= cls_end


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_python_file_lsp(file_path, project, file_content=None):
    """Parse a Python file with jedi (LSP-based) to extract class and function definitions.

    Parameters
    ----------
    file_path : str
        Absolute path to the .py file.
    project : jedi.Project
        Shared project instance created for the repository root.
        Enables cross-file name resolution (the key advantage over AST).
    file_content : str, optional
        Pre-loaded source text.  If None the file is read from disk.

    Returns
    -------
    class_info : list[dict]
        Each entry has keys: name, full_name, start_line, end_line, text, methods.
        methods is a list of dicts with the same keys (minus nested methods).
    function_names : list[dict]
        Top-level (non-method) functions.
        Each entry: name, full_name, start_line, end_line, text.
    file_lines : list[str]
        Raw source split by line (file_content.splitlines()).

    Notes
    -----
    * 'full_name' is a new field relative to the legacy parse_python_file.
      Downstream code that only reads name/start_line/end_line/text is
      unaffected.
    * 'text' is a list[str] to preserve backward compatibility with
      construct_graph.py (which joins it with '\\n').
    """
    if file_content is None:
        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                file_content = fh.read()
        except Exception as exc:
            print(f"[utils] Error reading {file_path}: {exc}")
            return [], [], []

    file_lines = file_content.splitlines()

    # Compute the dotted module prefix for this file relative to the project root.
    # Used to filter out imported names whose full_name points to another module
    # (e.g. `from collections import OrderedDict` → full_name='collections.OrderedDict').
    try:
        rel = Path(file_path).relative_to(project.path)
        module_prefix = ".".join(rel.with_suffix("").parts)
        # __init__.py → package name only (strip trailing .__init__)
        if module_prefix.endswith(".__init__"):
            module_prefix = module_prefix[: -len(".__init__")]
    except ValueError:
        module_prefix = None  # file_path not under project root; skip filter

    try:
        # jedi >= 0.19: first positional arg renamed from 'source' to 'code'
        script = jedi.Script(
            code=file_content,
            path=file_path,
            project=project,
        )
        # definitions=True  → only definition sites (not usages)
        # references=False  → skip reference sites here; handled in construct_graph
        # all_scopes=True   → include names inside classes and functions
        names = script.get_names(all_scopes=True, definitions=True, references=False)
    except Exception as exc:
        print(f"[utils] jedi parse error in {file_path}: {exc}")
        return [], [], file_lines

    class_jnames = []
    func_jnames = []
    for name in names:
        if not name.is_definition():
            continue
        # Filter out imported names: their full_name resolves to a foreign module.
        # Local definitions have full_name starting with this file's module prefix.
        if module_prefix and name.full_name:
            if not name.full_name.startswith(module_prefix + ".") and name.full_name != module_prefix:
                continue
        # Use parso tree node type (syntactic, no type inference) instead of
        # name.type to avoid jedi parser-cache KeyError on large projects.
        # name.type triggers jedi's full inference engine which can fail with
        # a KeyError when a transitively-imported file is not yet in the cache.
        # parso node types: 'classdef' → class, 'funcdef'/'async_funcdef' → function.
        try:
            parso_type = name._name.tree_name.parent.type
        except AttributeError:
            # Fallback: name.type (may trigger inference; catch errors)
            try:
                parso_type = {"class": "classdef", "function": "funcdef"}.get(
                    name.type, ""
                )
            except Exception:
                continue
        if parso_type == "classdef":
            class_jnames.append(name)
        elif parso_type in ("funcdef", "async_funcdef"):
            func_jnames.append(name)

    # ------------------------------------------------------------------
    # Build class entries with end_line pre-computed so that the line-range
    # fallback in _is_method_of is available when full_name is absent.
    # ------------------------------------------------------------------
    class_entries = []
    for cj in class_jnames:
        cls_start = cj.line
        cls_end = _get_end_line(cj, file_lines)
        class_entries.append({
            "jname": cj,
            "name": cj.name,
            "full_name": cj.full_name or cj.name,
            "start_line": cls_start,
            "end_line": cls_end,
            "methods": [],
        })

    # Sort classes by start_line descending so that nested (inner) classes are
    # checked before their enclosing class, ensuring a method is assigned to
    # the innermost matching class.
    class_entries_by_depth = sorted(class_entries, key=lambda x: x["start_line"], reverse=True)

    # ------------------------------------------------------------------
    # Assign each function to a class (method) or mark it as top-level.
    # ------------------------------------------------------------------
    method_full_names = set()

    for fn_jname in func_jnames:
        fn_full = fn_jname.full_name or fn_jname.name
        fn_start = fn_jname.line
        fn_end = _get_end_line(fn_jname, file_lines)
        fn_entry = {
            "name": fn_jname.name,
            "full_name": fn_full,
            "start_line": fn_start,
            "end_line": fn_end,
            "text": file_lines[fn_start - 1 : fn_end],
        }

        assigned = False
        for cls_entry in class_entries_by_depth:
            if _is_method_of(
                fn_jname,
                cls_entry["full_name"],
                cls_entry["start_line"],
                cls_entry["end_line"],
            ):
                cls_entry["methods"].append(fn_entry)
                method_full_names.add(fn_full)
                assigned = True
                break  # innermost-first: stop after first match

    # ------------------------------------------------------------------
    # Assemble final class_info list (drop internal jname key).
    # ------------------------------------------------------------------
    class_info = []
    for cls_entry in class_entries:  # restore original source order
        s, e = cls_entry["start_line"], cls_entry["end_line"]
        class_info.append({
            "name": cls_entry["name"],
            "full_name": cls_entry["full_name"],
            "start_line": s,
            "end_line": e,
            "text": file_lines[s - 1 : e],
            "methods": cls_entry["methods"],
        })

    # ------------------------------------------------------------------
    # Top-level functions: everything not assigned to a class.
    # ------------------------------------------------------------------
    function_names = []
    for fn_jname in func_jnames:
        fn_full = fn_jname.full_name or fn_jname.name
        if fn_full in method_full_names:
            continue
        fn_start = fn_jname.line
        fn_end = _get_end_line(fn_jname, file_lines)
        function_names.append({
            "name": fn_jname.name,
            "full_name": fn_full,
            "start_line": fn_start,
            "end_line": fn_end,
            "text": file_lines[fn_start - 1 : fn_end],
        })

    return class_info, function_names, file_lines


def create_structure(directory_path):
    """Walk the repository and build a nested structure dict using jedi.

    A single jedi.Project is created for the entire repo root so that
    all file parses share the same cross-file resolution context.
    This is the foundation that enables accurate invoke-edge construction
    in construct_graph.py.

    Return format (unchanged from legacy version):
        structure[dir_part][...][file.py] = {
            "classes":   [class_info dicts],
            "functions": [function_info dicts],
            "text":      [source lines],
        }
    """
    project = jedi.Project(path=directory_path)
    structure = {}

    for root, _, files in os.walk(directory_path):
        repo_name = os.path.basename(directory_path)
        relative_root = os.path.relpath(root, directory_path)
        if relative_root == ".":
            relative_root = repo_name
        curr_struct = structure
        for part in relative_root.split(os.sep):
            if part not in curr_struct:
                curr_struct[part] = {}
            curr_struct = curr_struct[part]

        for file_name in files:
            if file_name.endswith(".py"):
                file_path = os.path.join(root, file_name)
                class_info, function_names, file_lines = parse_python_file_lsp(
                    file_path, project
                )
                curr_struct[file_name] = {
                    "classes": class_info,
                    "functions": function_names,
                    "text": file_lines,
                }
            else:
                curr_struct[file_name] = {}

    return structure

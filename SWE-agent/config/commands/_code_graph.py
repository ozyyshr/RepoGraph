#!/usr/bin/env python3
import colorsys
import os
import random
import sys
import re
import warnings
from collections import Counter, defaultdict, namedtuple
import importlib
from pathlib import Path
import builtins
import inspect
import networkx as nx
from diskcache import Cache
from grep_ast import TreeContext, filename_to_lang
from pygments.lexers import guess_lexer_for_filename
from pygments.token import Token
from pygments.util import ClassNotFound
from tqdm import tqdm
import ast

# tree_sitter is throwing a FutureWarning
warnings.simplefilter("ignore", category=FutureWarning)
from tree_sitter_languages import get_language, get_parser  # noqa: E402

# from aider.dump import dump  # noqa: F402,E402

Tag = namedtuple("Tag", "rel_fname fname line name kind category info".split())


class RepoMap:
    TAGS_CACHE_DIR = f".repograph.cache"

    cache_missing = False

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

        self.load_tags_cache()

        self.max_map_tokens = map_tokens
        self.max_context_window = max_context_window

        # self.token_count = main_model.token_count
        self.repo_content_prefix = repo_content_prefix

    def get_repo_map(self, chat_files, other_files, mentioned_fnames=None, mentioned_idents=None):
        if self.max_map_tokens <= 0:
            return
        if not other_files:
            return
        if not mentioned_fnames:
            mentioned_fnames = set()
        if not mentioned_idents:
            mentioned_idents = set()

        max_map_tokens = self.max_map_tokens

        # With no files in the chat, give a bigger view of the entire repo
        MUL = 16
        padding = 4096
        if max_map_tokens and self.max_context_window:
            target = min(max_map_tokens * MUL, self.max_context_window - padding)
        else:
            target = 0
        if not chat_files and self.max_context_window and target > 0:
            max_map_tokens = target

        try:
            tags, repo_graph = self.get_ranked_tags(
            chat_files, other_files, mentioned_fnames, mentioned_idents
            )
        except RecursionError:
            self.io.tool_error("Disabling repo map, git repo too large?")
            self.max_map_tokens = 0
            return

        return tags, repo_graph

    def get_rel_fname(self, fname):
        return os.path.relpath(fname, self.root)

    def split_path(self, path):
        path = os.path.relpath(path, self.root)
        return [path + ":"]

    def load_tags_cache(self):
        path = Path(self.root) / self.TAGS_CACHE_DIR
        if not path.exists():
            self.cache_missing = True
        self.TAGS_CACHE = Cache(path)

    def save_tags_cache(self):
        pass

    def get_mtime(self, fname):
        try:
            return os.path.getmtime(fname)
        except FileNotFoundError:
            self.io.tool_error(f"File not found error: {fname}")

    def get_class_functions(self, tree, class_name):
        class_functions = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        class_functions.append(item.name)

        return class_functions

    def get_func_block(self, first_line, code_block):
        first_line_escaped = re.escape(first_line)
        pattern = re.compile(rf'({first_line_escaped}.*?)(?=(^\S|\Z))', re.DOTALL | re.MULTILINE)
        match = pattern.search(code_block)

        return match.group(0) if match else None

    def std_proj_funcs(self, code, fname):
        """
        write a function to analyze the *import* part of a py file.
        Input: code for fname
        output: [standard functions]
        please note that the project_dependent libraries should have specific project names.
        """
        std_libs = []
        std_funcs = []
        tree = ast.parse(code)
        codelines = code.split('\n')

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                # identify the import statement
                import_statement = codelines[node.lineno-1]
                for alias in node.names:
                    import_name = alias.name.split('.')[0]
                    if import_name in fname:
                        continue
                    else:
                        # execute the import statement to find callable functions
                        import_statement = import_statement.strip()
                        try:
                            exec(import_statement)
                        except:
                            continue
                        std_libs.append(alias.name)
                        eval_name = alias.name if alias.asname is None else alias.asname
                        std_funcs.extend([name for name, member in inspect.getmembers(eval(eval_name)) if callable(member)])

            if isinstance(node, ast.ImportFrom):
                # execute the import statement
                import_statement = codelines[node.lineno-1]
                
                module_name = node.module.split('.')[0]
                if module_name in fname:
                    continue
                else:
                    # handle imports with parentheses
                    if "(" in import_statement:
                        for ln in range(node.lineno-1, len(codelines)):
                            if ")" in codelines[ln]:
                                code_num = ln
                                break
                        import_statement = '\n'.join(codelines[node.lineno-1:code_num+1])
                        print(import_statement)
                    import_statement = import_statement.strip()
                    try:
                        exec(import_statement)
                    except:
                        continue
                    for alias in node.names:
                        std_libs.append(alias.name)
                        eval_name = alias.name if alias.asname is None else alias.asname
                        std_funcs.extend([name for name, member in inspect.getmembers(eval(eval_name)) if callable(member)])
        return std_funcs, std_libs
                    

    def get_tags(self, fname, rel_fname):
        # Check if the file is in the cache and if the modification time has not changed
        file_mtime = self.get_mtime(fname)
        if file_mtime is None:
            return []

        cache_key = fname
        if cache_key in self.TAGS_CACHE and self.TAGS_CACHE[cache_key]["mtime"] == file_mtime:
            return self.TAGS_CACHE[cache_key]["data"]

        # miss!
        data = list(self.get_tags_raw(fname, rel_fname))

        # Update the cache
        self.TAGS_CACHE[cache_key] = {"mtime": file_mtime, "data": data}
        self.save_tags_cache()
        return data

    def get_tags_raw(self, fname, rel_fname):
        lang = filename_to_lang(fname)
        if not lang:
            return

        language = get_language(lang)

        parser = get_parser(lang)

        # Load the tags queries
        try:
            # scm_fname = resources.files(__package__).joinpath(
            #     "/shared/data3/siruo2/SWE-agent/sweagent/environment/queries", f"tree-sitter-{lang}-tags.scm")
            scm_fname = """
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
        except KeyError:
            return
        query_scm = scm_fname
        # if not query_scm.exists():
        #     return
        # query_scm = query_scm.read_text()

        with open(str(fname), "r", encoding='utf-8') as f:
            code = f.read()
        with open(str(fname), "r", encoding='utf-8') as f:    
            codelines = f.readlines()

        # code = self.io.read_text(fname)
        if not code:
            return
        tree = parser.parse(bytes(code, "utf-8"))

        tree_ast = ast.parse(code)

        # functions from third-party libs or default libs
        std_funcs, std_libs = self.std_proj_funcs(code, fname)
        
        # functions from builtins
        builtins_funs = [name for name in dir(builtins)]
        builtins_funs += dir(list)
        builtins_funs += dir(dict)
        builtins_funs += dir(set)  
        builtins_funs += dir(str)
        builtins_funs += dir(tuple)

        # Run the tags queries
        query = language.query(query_scm)
        captures = query.captures(tree.root_node)

        captures = list(captures)

        saw = set()
        for node, tag in captures:
            if tag.startswith("name.definition."):
                kind = "def"
            elif tag.startswith("name.reference."):
                kind = "ref"
            else:
                continue

            saw.add(kind)
            cur_cdl = codelines[node.start_point[0]] # codeline number start from 0
            category = 'class' if 'class' in cur_cdl else 'function'
            tag_name = node.text.decode("utf-8")
            assert tag_name in cur_cdl
            
            #  we only want to consider project-dependent functions
            if tag_name in std_funcs:
                    continue
            elif tag_name in std_libs:
                continue
            elif tag_name in builtins_funs:
                continue

            if category == 'class':
                class_functions = self.get_class_functions(tree_ast, tag_name)

                result = Tag(
                    rel_fname=rel_fname,
                    fname=fname,
                    name=tag_name,
                    kind=kind,
                    category=category,
                    info='\t'.join(class_functions), # list unhashable, use string instead
                    line=node.start_point[0],
                )

            elif category == 'function':

                if kind == 'def':
                    func_block = self.get_func_block(cur_cdl, code)
                    cur_cdl =func_block
                
                result = Tag(
                    rel_fname=rel_fname,
                    fname=fname,
                    name=tag_name,
                    kind=kind,
                    category=category,
                    info=cur_cdl,
                    line=node.start_point[0],
                )

            yield result

        if "ref" in saw:
            return
        if "def" not in saw:
            return

        # We saw defs, without any refs
        # Some tags files only provide defs (cpp, for example)
        # Use pygments to backfill refs

        try:
            lexer = guess_lexer_for_filename(fname, code)
        except ClassNotFound:
            return

        tokens = list(lexer.get_tokens(code))
        tokens = [token[1] for token in tokens if token[0] in Token.Name]

        for token in tokens:
            yield Tag(
                rel_fname=rel_fname,
                fname=fname,
                name=token,
                kind="ref",
                line=-1,
                category='function',
                info='none',
            )

    def get_ranked_tags(self, chat_fnames, other_fnames, mentioned_fnames, mentioned_idents):
        defines = defaultdict(set)
        references = defaultdict(list)
        definitions = defaultdict(set)
        
        tags_of_files = list()

        personalization = dict()

        fnames = set(chat_fnames).union(set(other_fnames))
        chat_rel_fnames = set()

        fnames = sorted(fnames)

        # Default personalization for unspecified files is 1/num_nodes
        # https://networkx.org/documentation/stable/_modules/networkx/algorithms/link_analysis/pagerank_alg.html#pagerank
        personalize = 10 / len(fnames)

        if self.cache_missing:
            fnames = tqdm(fnames)
        self.cache_missing = False

        for fname in fnames:
            if not Path(fname).is_file():
                if fname not in self.warned_files:
                    if Path(fname).exists():
                        self.io.tool_error(
                            f"Repo-map can't include {fname}, it is not a normal file"
                        )
                    else:
                        self.io.tool_error(f"Repo-map can't include {fname}, it no longer exists")

                self.warned_files.add(fname)
                continue

            # dump(fname)
            rel_fname = self.get_rel_fname(fname)

            if fname in chat_fnames:
                personalization[rel_fname] = personalize
                chat_rel_fnames.add(rel_fname)

            if fname in mentioned_fnames:
                personalization[rel_fname] = personalize
            
            tags = list(self.get_tags(fname, rel_fname))

            tags_of_files.extend(tags)

            if tags is None:
                continue

            for tag in tags:
                if tag.kind == "def":
                    defines[tag.name].add(rel_fname)
                    key = (rel_fname, tag.name)
                    definitions[key].add(tag)

                if tag.kind == "ref":
                    references[tag.name].append(rel_fname)
        
        G = nx.MultiDiGraph()
        
        tag_names = set(list([(tag.name, tag.category, tag.fname) for tag in tags_of_files if tag.kind=='def']))
        node_ids = {
            tag.name: {"name": tag.name, "category": tag.category, "info": tag.info, "fname": tag.fname, "line": tag.line, "kind":tag.kind}
            for tag in tags_of_files
            }
        for tag in tags_of_files:
            # G.add_node(node_ids[tag.name])
            G.add_node(tag.name, category=tag.category, info=tag.info, fname=tag.fname, line=tag.line, kind=tag.kind)
            if tag.kind == 'ref': continue
            if tag.category == 'function':
                # for k,v in node_ids.items():
                for (n, m, f) in tag_names:
                    if n in tag.info and tag.name.strip() != n.strip():
                        if m == 'function':
                            G.add_edge(tag.name, n, relation="func-func", refname=tag.fname, defname=f)
                        elif m == 'class':
                            G.add_edge(tag.name, n, relation="func-class")
            if tag.category == 'class':
                class_funcs = tag.info.split('\t')
                for f in class_funcs:
                    if f.strip() != tag.name.strip():
                        G.add_edge(tag.name, f.strip(), relation="class-func")
        
        return tags_of_files, G
        # print(G.nodes(data=True))
        # print(G.adj['Agent'])
        ##
        # dump(defines)
        # dump(references)
        # dump(personalization)
    

    def render_tree(self, abs_fname, rel_fname, lois):
        key = (rel_fname, tuple(sorted(lois)))

        if key in self.tree_cache:
            return self.tree_cache[key]

        # code = self.io.read_text(abs_fname) or ""
        with open(str(abs_fname), "r", encoding='utf-8') as f:
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
            # header_max=30,
            show_top_of_file_parent_scope=False,
        )

        context.add_lines_of_interest(lois)
        context.add_context()
        res = context.format()
        self.tree_cache[key] = res
        return res

    def to_tree(self, tags, chat_rel_fnames):
        if not tags:
            return ""

        tags = [tag for tag in tags if tag[0] not in chat_rel_fnames]
        tags = sorted(tags)

        cur_fname = None
        cur_abs_fname = None
        lois = None
        output = ""

        # add a bogus tag at the end so we trip the this_fname != cur_fname...
        dummy_tag = (None,)
        for tag in tags + [dummy_tag]:
            this_rel_fname = tag[0]

            # ... here ... to output the final real entry in the list
            if this_rel_fname != cur_fname:
                if lois is not None:
                    output += "\n"
                    output += cur_fname + ":\n"
                    output += self.render_tree(cur_abs_fname, cur_fname, lois)
                    lois = None
                elif cur_fname:
                    output += "\n" + cur_fname + "\n"
                if type(tag) is Tag:
                    lois = []
                    cur_abs_fname = tag.fname
                cur_fname = this_rel_fname

            if lois is not None:
                lois.append(tag.line)

        # truncate long lines, in case we get minified js or something else crazy
        output = "\n".join([line[:100] for line in output.splitlines()]) + "\n"

        return output


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
        
        chat_fnames_new = []
        for item in chat_fnames:
            if ('__' in item) or (".py" not in item):
                continue
            else:
                chat_fnames_new.append(item)
    
        return chat_fnames_new
    

def get_random_color():
    hue = random.random()
    r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(hue, 1, 0.75)]
    res = f"#{r:02x}{g:02x}{b:02x}"
    return res


if __name__ == "__main__":

    dir_name, func_name = sys.argv[1], sys.argv[2]
    repo_map = RepoMap(root=dir_name)
    chat_fnames_new = repo_map.find_files([dir_name])

    tags, G = repo_map.get_repo_map([], chat_fnames_new)

    successors = list(G.successors(func_name))
    predecessors = list(G.predecessors(func_name))
    tags2names = {tag.name: tag for tag in tags}
    returned_files = []
    for item in successors+[func_name]+predecessors:
        returned_files.append({
            "fname": tags2names[item].fname,
            'rel_fname': tags2names[item].rel_fname,
            'line': tags2names[item].line,
            'name': tags2names[item].name,
            'kind': tags2names[item].kind,
            'category': tags2names[item].category,
            'info': tags2names[item].info
        })
    print(returned_files)

    # TODO: filter out node line = -1.
    # test_scripts:
    # python repomap.py /shared/data3/siruo2/pvlib-python _golden_sect_DataFrame

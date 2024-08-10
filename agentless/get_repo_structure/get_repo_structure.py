import argparse
import ast
import json
import os
import subprocess
import uuid
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import pickle
# from code_graph import RepoMap

repo_to_top_folder = {
    "django/django": "django",
    "sphinx-doc/sphinx": "sphinx",
    "scikit-learn/scikit-learn": "scikit-learn",
    "sympy/sympy": "sympy",
    "pytest-dev/pytest": "pytest",
    "matplotlib/matplotlib": "matplotlib",
    "astropy/astropy": "astropy",
    "pydata/xarray": "xarray",
    "mwaskom/seaborn": "seaborn",
    "psf/requests": "requests",
    "pylint-dev/pylint": "pylint",
    "pallets/flask": "flask",
}


def checkout_commit(repo_path, commit_id):
    """Checkout the specified commit in the given local git repository.
    :param repo_path: Path to the local git repository
    :param commit_id: Commit ID to checkout
    :return: None
    """
    try:
        # Change directory to the provided repository path and checkout the specified commit
        print(f"Checking out commit {commit_id} in repository at {repo_path}...")
        subprocess.run(["git", "-C", repo_path, "checkout", commit_id], check=True)
        print("Commit checked out successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running git command: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def clone_repo(repo_name, repo_playground):
    try:

        print(
            f"Cloning repository from https://github.com/{repo_name}.git to {repo_playground}/{repo_to_top_folder[repo_name]}..."
        )
        subprocess.run(
            [
                "git",
                "clone",
                f"https://github.com/{repo_name}.git",
                f"{repo_playground}/{repo_to_top_folder[repo_name]}",
            ],
            check=True,
        )
        print("Repository cloned successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running git command: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def get_project_structure_from_scratch(
    repo_name, commit_id, instance_id, repo_playground
):

    # Generate a temperary folder and add uuid to avoid collision
    repo_playground = os.path.join(repo_playground, str(uuid.uuid4()))

    # assert playground doesn't exist
    assert not os.path.exists(repo_playground), f"{repo_playground} already exists"

    # create playground
    os.makedirs(repo_playground)

    clone_repo(repo_name, repo_playground)
    checkout_commit(f"{repo_playground}/{repo_to_top_folder[repo_name]}", commit_id)
    structure = create_structure(f"{repo_playground}/{repo_to_top_folder[repo_name]}")
    # clean up
    subprocess.run(
        ["rm", "-rf", f"{repo_playground}/{repo_to_top_folder[repo_name]}"], check=True
    )
    d = {
        "repo": repo_name,
        "base_commit": commit_id,
        "structure": structure,
        "instance_id": instance_id,
    }
    return d

def filter_out_test_files(files):
    return_f = []
    """filter out test files from the repo"""
    for item in files:
        if "test" not in item:
            return_f.append(item)
    return return_f

def get_code_graph_from_scratch(
        repo_name, commit_id, instance_id, repo_playground
):
    # Generate a temperary folder and add uuid to avoid collision
    repo_playground = os.path.join(repo_playground, str(uuid.uuid4()))
    # repo_playground = os.path.join(instance_id)

    # assert playground doesn't exist
    assert not os.path.exists(repo_playground), f"{repo_playground} already exists"

    # create playground
    os.makedirs(repo_playground)

    clone_repo(repo_name, repo_playground)
    checkout_commit(f"{repo_playground}/{repo_to_top_folder[repo_name]}", commit_id)

    repo_map = RepoMap(root=f"{repo_playground}/{repo_to_top_folder[repo_name]}")
    chat_fnames_new = repo_map.find_files([f"{repo_playground}/{repo_to_top_folder[repo_name]}"])

    if not instance_id.startswith("pytest"):
        fnames = filter_out_test_files(chat_fnames_new)
    else:
        fnames = chat_fnames_new

    tags, G = repo_map.get_repo_map([], fnames)
    
    # clean up
    subprocess.run(
        ["rm", "-rf", f"{repo_playground}/{repo_to_top_folder[repo_name]}"], check=True
    )

    return tags, G


def parse_python_file(file_path, file_content=None):
    """Parse a Python file to extract class and function definitions with their line numbers.
    :param file_path: Path to the Python file.
    :return: Class names, function names, and file contents
    """
    if file_content is None:
        try:
            with open(file_path, "r") as file:
                file_content = file.read()
                parsed_data = ast.parse(file_content)
        except Exception as e:  # Catch all types of exceptions
            print(f"Error in file {file_path}: {e}")
            return [], [], ""
    else:
        try:
            parsed_data = ast.parse(file_content)
        except Exception as e:  # Catch all types of exceptions
            print(f"Error in file {file_path}: {e}")
            return [], [], ""

    class_info = []
    function_names = []
    class_methods = set()

    for node in ast.walk(parsed_data):
        if isinstance(node, ast.ClassDef):
            methods = []
            for n in node.body:
                if isinstance(n, ast.FunctionDef):
                    methods.append(
                        {
                            "name": n.name,
                            "start_line": n.lineno,
                            "end_line": n.end_lineno,
                            "text": file_content.splitlines()[
                                n.lineno - 1 : n.end_lineno
                            ],
                        }
                    )
                    class_methods.add(n.name)
            class_info.append(
                {
                    "name": node.name,
                    "start_line": node.lineno,
                    "end_line": node.end_lineno,
                    "text": file_content.splitlines()[
                        node.lineno - 1 : node.end_lineno
                    ],
                    "methods": methods,
                }
            )
        elif isinstance(node, ast.FunctionDef) and not isinstance(
            node, ast.AsyncFunctionDef
        ):
            if node.name not in class_methods:
                function_names.append(
                    {
                        "name": node.name,
                        "start_line": node.lineno,
                        "end_line": node.end_lineno,
                        "text": file_content.splitlines()[
                            node.lineno - 1 : node.end_lineno
                        ],
                    }
                )

    return class_info, function_names, file_content.splitlines()


def create_structure(directory_path):
    """Create the structure of the repository directory by parsing Python files.
    :param directory_path: Path to the repository directory.
    :return: A dictionary representing the structure.
    """
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
                class_info, function_names, file_lines = parse_python_file(file_path)
                curr_struct[file_name] = {
                    "classes": class_info,
                    "functions": function_names,
                    "text": file_lines,
                }
            else:
                curr_struct[file_name] = {}

    return structure

if __name__ == '__main__':

    structure = create_structure('playground/astropy')

    swe_bench_data = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")

    for bug in tqdm(swe_bench_data):

        existing_graph = os.listdir('./repo_structures/graph')
        if f"{bug['instance_id']}.pkl" in existing_graph:
            continue

        # we need to get the project structure directly
        tags, G = get_code_graph_from_scratch(
            bug["repo"], bug["base_commit"], bug['instance_id'], "playground"
        )

        instance_id = bug["instance_id"]

        with open(f'./repo_structures/graph/{instance_id}.pkl', 'wb') as f:
            pickle.dump(G, f)

        tags_json = []
        for tag in tags:
            tags_json.append({
                "fname": tag.fname,
                'rel_fname': tag.rel_fname,
                'line': tag.line,
                'name': tag.name,
                'kind': tag.kind,
                'category': tag.category,
                'info': tag.info,
            })
        with open(f'./repo_structures/graph/tags_{instance_id}.json', 'w') as f:
            json.dump(tags_json, f)
            # line = json.dumps({
            #     "fname": tag.fname,
            #     'rel_fname': tag.rel_fname,
            #     'line': tag.line,
            #     'name': tag.name,
            #     'kind': tag.kind,
            #     'category': tag.category,
            #     'info': tag.info,
            # })
            # f.write(line+'\n')
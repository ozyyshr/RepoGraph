import argparse
import json
import logging
import pickle
import os
from copy import deepcopy
from datasets import load_dataset
from tqdm import tqdm

from agentless.fl.FL import LLMFL
from agentless.util.preprocess_data import (
    filter_none_python,
    filter_out_test_files,
    get_full_file_paths_and_classes_and_functions,
    show_project_structure,
)
from agentless.util.utils import load_json, load_jsonl
from agentless.get_repo_structure.get_repo_structure import (
    clone_repo,
    get_project_structure_from_scratch,
)

# PROJECT_FILE_LOC = os.environ.get("PROJECT_FILE_LOC", None)
PROJECT_FILE_LOC = "./repo_structures"

def retrieve_graph(code_graph, graph_tags, search_term, structure, max_tags=100):
    one_hop_tags = []
    tags = []
    for tag in graph_tags:
        if tag['name'] == search_term and tag['kind'] == 'ref':
            tags.append(tag)
        if len(tags) >= max_tags:
            break
    # for tag in tags:
    for i, tag in enumerate(tags):
        # if i % 3 == 0:
        print(f"Retrieving graph for {i}/{len(tags)}")
        # find corresponding calling function/class
        path = tag['rel_fname'].split('/')
        s = deepcopy(structure)   # stuck here
        for p in path:
            s = s[p]
        for txt in s['functions']:
            if tag['line'] >= txt['start_line'] and tag['line'] <= txt['end_line']:
                one_hop_tags.append((txt, tag['rel_fname']))  
        for txt in s['classes']:
            for func in txt['methods']:
                if tag['line'] >= func['start_line'] and tag['line'] <= func['end_line']:
                    func['text'].insert(0, txt['text'][0])
                    one_hop_tags.append((func, tag['rel_fname'])) 
    return one_hop_tags

def construct_code_graph_context(found_related_locs, code_graph, graph_tags, structure):
    graph_context = ""

    graph_item_format = """
### Dependencies for {func}
{dependencies}
"""
    tag_format = """
location: {fname} lines {start_line} - {end_line}
name: {name}
contents: 
{contents}

"""
    # retrieve the code graph for dependent functions and classes
    for item in found_related_locs:
        code_graph_context = ""
        item = item[0].splitlines()
        for loc in tqdm(item):
            if loc.startswith("class: ") and "." not in loc:
                loc = loc[len("class: ") :].strip()
                tags = retrieve_graph(code_graph, graph_tags, loc, structure)
                for t, fname in tags:
                    code_graph_context += tag_format.format(
                        **t,
                        fname=fname,
                        contents="\n".join(t['text'])
                    )
            elif loc.startswith("function: ") and "." not in loc:
                loc = loc[len("function: ") :].strip()
                tags = retrieve_graph(code_graph, graph_tags, loc, structure)
                for t, fname in tags:
                    code_graph_context += tag_format.format(
                        **t,
                        fname=fname,
                        contents="\n".join(t['text'])
                    )
            elif "." in loc:
                loc = loc.split(".")[-1].strip()
                tags = retrieve_graph(code_graph, graph_tags, loc, structure)
                for t, fname in tags:
                    code_graph_context += tag_format.format(
                        **t,
                        fname=fname,
                        contents="\n".join(t['text'])
                    )
            graph_context += graph_item_format.format(func=loc, dependencies=code_graph_context)
    return graph_context

def localize(args):

    swe_bench_data = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")

    if args.start_file:
        start_file_locs = load_jsonl(args.start_file)
    count = 0
    for bug in swe_bench_data:
        if count <= 54:
            count += 1
            continue

        if args.target_id is not None:
            if args.target_id != bug["instance_id"]:
                continue

        if PROJECT_FILE_LOC is not None:
            project_file = os.path.join(PROJECT_FILE_LOC, bug["instance_id"] + ".json")
            d = load_json(project_file)
        else:
            # we need to get the project structure directly
            d = get_project_structure_from_scratch(
                bug["repo"], bug["base_commit"], bug["instance_id"], "playground"
            )

        instance_id = d["instance_id"]
        code_graph = pickle.load(
            open(f"./repo_structures/graph/{instance_id}.pkl", "rb")
        )
        graph_tags = json.load(
            open(f"./repo_structures/graph/tags_{instance_id}.json", "r")
        )

        logging.info(f"================ localize {instance_id} ================")

        bench_data = [x for x in swe_bench_data if x["instance_id"] == instance_id][0]
        problem_statement = bench_data["problem_statement"]
        structure = d["structure"]
        filter_none_python(structure)
        # some basic filtering steps
        # filter out test files (unless its pytest)
        if not d["instance_id"].startswith("pytest"):
            filter_out_test_files(structure)

        found_files = []
        found_related_locs = []
        found_edit_locs = []

        additional_artifact_loc_file = None
        additional_artifact_loc_related = None
        additional_artifact_loc_edit_location = None
        file_traj, related_loc_traj, edit_loc_traj = {}, {}, {}

        # file level localization
        if args.file_level:
            fl = LLMFL(
                d["instance_id"],
                structure,
                problem_statement,
            )
            found_files, additional_artifact_loc_file, file_traj = fl.localize(
                mock=args.mock
            )
        else:
            # assume start_file is provided
            for locs in start_file_locs:
                if locs["instance_id"] == d["instance_id"]:
                    found_files = locs["found_files"]
                    additional_artifact_loc_file = locs["additional_artifact_loc_file"]
                    file_traj = locs["file_traj"]

                    if "found_related_locs" in locs:
                        found_related_locs = locs["found_related_locs"]
                        additional_artifact_loc_related = locs[
                            "additional_artifact_loc_related"
                        ]
                        related_loc_traj = locs["related_loc_traj"]
                    break

        # related class, functions, global var localization
        if args.related_level:
            if len(found_files) != 0:
                pred_files = found_files[: args.top_n]
                fl = LLMFL(
                    d["instance_id"],
                    structure,
                    problem_statement,
                )

                additional_artifact_loc_related = []
                found_related_locs = []
                related_loc_traj = {}

                if args.compress:
                    (
                        found_related_locs,
                        additional_artifact_loc_related,
                        related_loc_traj,
                    ) = fl.localize_function_from_compressed_files(
                        pred_files,
                        mock=args.mock,
                    )
                    additional_artifact_loc_related = [additional_artifact_loc_related]
                else:
                    assert False, "Not implemented yet."

        if args.fine_grain_line_level:
            # Only supports the following args for now
            if args.repo_graph:
                code_graph_context = construct_code_graph_context(found_related_locs, code_graph, graph_tags, structure)
            else:
                code_graph_context = None

            pred_files = found_files[: args.top_n]
            fl = LLMFL(
                instance_id,
                structure,
                problem_statement,
            )
            coarse_found_locs = {}
            for i, pred_file in enumerate(pred_files):
                if len(found_related_locs) > i:
                    coarse_found_locs[pred_file] = found_related_locs[i]
            (
                found_edit_locs,
                additional_artifact_loc_edit_location,
                edit_loc_traj,
            ) = fl.localize_line_from_coarse_function_locs(
                pred_files,
                coarse_found_locs,
                context_window=args.context_window,
                add_space=args.add_space,
                code_graph=args.repo_graph,
                code_graph_context=code_graph_context,
                no_line_number=args.no_line_number,
                sticky_scroll=args.sticky_scroll,
                mock=args.mock,
                temperature=args.temperature,
                num_samples=args.num_samples,
            )

            additional_artifact_loc_edit_location = [
                additional_artifact_loc_edit_location
            ]

        with open(args.output_file, "a") as f:
            f.write(
                json.dumps(
                    {
                        "instance_id": d["instance_id"],
                        "found_files": found_files,
                        "additional_artifact_loc_file": additional_artifact_loc_file,
                        "file_traj": file_traj,
                        "found_related_locs": found_related_locs,
                        "additional_artifact_loc_related": additional_artifact_loc_related,
                        "related_loc_traj": related_loc_traj,
                        "found_edit_locs": found_edit_locs,
                        "additional_artifact_loc_edit_location": additional_artifact_loc_edit_location,
                        "edit_loc_traj": edit_loc_traj,
                    }
                )
                + "\n"
            )
        count += 1


def merge(args):
    """Merge predicted locations."""
    start_file_locs = load_jsonl(args.start_file)

    # Dump each location sample.
    for st_id in range(args.num_samples):
        en_id = st_id
        merged_locs = []
        for locs in start_file_locs:
            merged_found_locs = []
            if "found_edit_locs" in locs and len(locs["found_edit_locs"]):
                merged_found_locs = [
                    "\n".join(x) for x in locs["found_edit_locs"][st_id]
                ]
            merged_locs.append({**locs, "found_edit_locs": merged_found_locs})
        with open(
            f"{args.output_folder}/loc_merged_{st_id}-{en_id}_outputs.jsonl", "w"
        ) as f:
            for data in merged_locs:
                f.write(json.dumps(data) + "\n")

    # Pair wise merge
    for st_id in range(0, args.num_samples - 1, 2):
        en_id = st_id + 1
        print(f"Merging sample {st_id} and {en_id}...")
        merged_locs = []
        for locs in start_file_locs:
            merged_found_locs = []
            if "found_edit_locs" in locs and len(locs["found_edit_locs"]):
                merged_found_locs = [
                    "\n".join(x) for x in locs["found_edit_locs"][st_id]
                ]
                for sample_found_locs in locs["found_edit_locs"][st_id + 1 : en_id + 1]:
                    for i, file_found_locs in enumerate(sample_found_locs):
                        if isinstance(file_found_locs, str):
                            merged_found_locs[i] += "\n" + file_found_locs
                        else:
                            merged_found_locs[i] += "\n" + "\n".join(file_found_locs)
            merged_locs.append({**locs, "found_edit_locs": merged_found_locs})
        with open(
            f"{args.output_folder}/loc_merged_{st_id}-{en_id}_outputs.jsonl", "w"
        ) as f:
            for data in merged_locs:
                f.write(json.dumps(data) + "\n")

    ### Merge all
    all_merged_locs = []
    print("Merging all samples...")
    for locs in start_file_locs:
        merged_found_locs = []
        if "found_edit_locs" in locs and len(locs["found_edit_locs"]):
            merged_found_locs = ["\n".join(x) for x in locs["found_edit_locs"][0]]
            for sample_found_locs in locs["found_edit_locs"][1:]:
                for i, file_found_locs in enumerate(sample_found_locs):
                    if isinstance(file_found_locs, str):
                        merged_found_locs[i] += "\n" + file_found_locs
                    else:
                        merged_found_locs[i] += "\n" + "\n".join(file_found_locs)
        all_merged_locs.append({**locs, "found_edit_locs": merged_found_locs})
    with open(f"{args.output_folder}/loc_all_merged_outputs.jsonl", "w") as f:
        for data in all_merged_locs:
            f.write(json.dumps(data) + "\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="loc_outputs.jsonl")
    parser.add_argument(
        "--start_file",
        type=str,
        help="""previous output file to start with to reduce
        the work, should use in combination without --file_level""",
    )
    parser.add_argument("--file_level", action="store_true")
    parser.add_argument("--related_level", action="store_true")
    parser.add_argument("--fine_grain_line_level", action="store_true")
    parser.add_argument("--top_n", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--add_space", action="store_true")
    parser.add_argument("--no_line_number", action="store_true")
    parser.add_argument("--repo_graph", action="store_true")
    parser.add_argument("--sticky_scroll", action="store_true")
    parser.add_argument("--context_window", type=int, default=10)
    parser.add_argument("--target_id", type=str)
    parser.add_argument(
        "--mock", action="store_true", help="Mock run to compute prompt tokens."
    )

    args = parser.parse_args()

    import os

    args.output_file = os.path.join(args.output_folder, args.output_file)

    assert not os.path.exists(args.output_file), "Output file already exists"

    assert not (
        args.file_level and args.start_file
    ), "Cannot use both file_level and start_file"

    assert not (
        args.file_level and args.fine_grain_line_level and not args.related_level
    ), "Cannot use both file_level and fine_grain_line_level without related_level"

    assert not (
        (not args.file_level) and (not args.start_file)
    ), "Must use either file_level or start_file"

    os.makedirs(args.output_folder, exist_ok=True)

    # write the arguments
    with open(f"{args.output_folder}/args.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    logging.basicConfig(
        filename=f"{args.output_folder}/localize.log",
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    if args.merge:
        merge(args)
    else:
        localize(args)


if __name__ == "__main__":
    main()

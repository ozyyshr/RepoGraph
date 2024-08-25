# RepoGraph: Enhancing AI Software Engineering with Repository-level Code Graph

## 📜 Overview

We introduce RepoGraph, an effective plug-in repo-level module that offers the desired context and substantially boosts the LLMs' AI software engineering capability.

## 🆕 News

We released the first version RepoGraph and its integration with [SWE-bench](https://www.swebench.com/) methods!

## 🤖 Code Setup

### Foder and files

`repograph` contains the code for construct and retrieve related context from the graph. 

`agentless` and `SWE-agent` incorporates the integrated version of RepoGraph with the two methods.

Currently this version may take a little long time to run for a repo. We provide a cached version for all repos in SWE-bench, download it [here](https://drive.google.com/file/d/1-0d-OgGoOf3i54bWcf8H0egjQyTSZ8dG/view?usp=sharing) and put it under `repo_structures`.

### How to run?

To generate the repograph for a given repository, simply run:

```bash
python ./repograph/construct_graph.py <dir_to_repo>
```

This will produce two files, `tags_{instance_id}.jsonl` stores the line-level information and `{instance_id}.pkl` is the graph constructed using networkx.

## Integration with models on SWE-bench

### Procedural framework

For a procedural framework, RepoGraph could be integrated into every step of the pipeline. Refer to `--repo_graph` hyperparameter for controllability in different stages.

To run RepoGraph with Agentless, use command:

```bash
bash run_repograph_agentless.sh
```

### Agent framework

To integrate RepoGraph with agent framework such as SWE-agent, we simply add an extra action in its initial action space. Specifically, you can look up for `search_repo()` in corresponding dir. The signature is defined as:

```python
search_repo:
    docstring: searches in the current repository with a specific function or class, and returns the def and ref relations for the search term.
    signature: search_repo <search_term>
    arguments:
      - search_term (string) [required]: function or class to look for in the repository.
```

To run RepoGraph with SWE-agent, use command:

```bash
bash run_repograph_sweagent.sh
```

We are working on prepreints for details in RepoGraph and a more comprehensive/easy integration with exsiting models. Stay tuned!!
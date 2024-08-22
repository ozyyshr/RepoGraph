PYTHONPATH=".:agentless/" python agentless/fl/localize.py \
    --file_level \
    --related_level \
    --fine_grain_line_level \
    --output_folder=results/location \
    --top_n=3 \
    --compress \
    --context_window=10 \
    --repo_graph

PYTHONPATH=".:agentless/" python agentless/repair/repair.py \
    --loc_file=results/location/loc_outputs_codegraph.jsonl \
    --output_folder=results/repair \
    --loc_interval \
    --top_n=3 \
    --context_window=10 \
    --max_samples=10 \
    --cot \
    --diff_format \
    --gen_and_process \
    --repo_graph

PYTHONPATH=".:agentless/" python agentless/repair/rerank.py \
    --patch_folder=results/repair \
    --num_samples=10 \
    --deduplicate \
    --plausible

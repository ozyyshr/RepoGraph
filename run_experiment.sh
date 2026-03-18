#!/usr/bin/env bash
# ============================================================
# RepoGraph 실험: baseline (tree-sitter) vs ours (LSP)
# 사용법:
#   bash run_experiment.sh baseline [repo]      # tree-sitter 그래프 사용
#   bash run_experiment.sh lsp [repo]           # LSP 그래프 사용
#   bash run_experiment.sh lsp_limited [repo]   # LSP + max_edit_locs=2
#
# repo 생략 시 기본값: astropy
# 예: bash run_experiment.sh baseline matplotlib
# ============================================================
set -e

VARIANT="${1:-baseline}"
FILTER_REPO="${2:-astropy}"
MAX_EDIT_LOCS=""

if [[ "$VARIANT" == "baseline" ]]; then
    GRAPH_DIR="./repo_structures/graph/baseline"
    OUT_BASE="./results/${FILTER_REPO}/baseline"
elif [[ "$VARIANT" == "lsp" ]]; then
    GRAPH_DIR="./repo_structures/graph/lsp"
    OUT_BASE="./results/${FILTER_REPO}/lsp"
elif [[ "$VARIANT" == "lsp_limited" ]]; then
    GRAPH_DIR="./repo_structures/graph/lsp"
    OUT_BASE="./results/${FILTER_REPO}/lsp_limited"
    MAX_EDIT_LOCS="--max_edit_locs=2"
else
    echo "Usage: bash run_experiment.sh [baseline|lsp|lsp_limited] [repo]"
    exit 1
fi

LOC_FOLDER="${OUT_BASE}/location"
REP_FOLDER="${OUT_BASE}/repair"

echo "========================================"
echo "  Variant      : ${VARIANT}"
echo "  Repo         : ${FILTER_REPO}"
echo "  GraphDir     : ${GRAPH_DIR}"
echo "  Output       : ${OUT_BASE}"
if [[ -n "$MAX_EDIT_LOCS" ]]; then
    echo "  max_edit_locs: ${MAX_EDIT_LOCS}"
fi
echo "========================================"

# ── Step 1: Localize ──────────────────────────────────────
echo ""
echo "[1/3] Localize (file -> related -> fine-grain) ..."
mkdir -p "${LOC_FOLDER}"

PYTHONPATH=".:agentless/" python agentless/fl/localize.py \
    --file_level \
    --related_level \
    --fine_grain_line_level \
    --output_folder="${LOC_FOLDER}" \
    --top_n=3 \
    --compress \
    --context_window=10 \
    --repo_graph \
    --filter_repo="${FILTER_REPO}" \
    --graph_dir="${GRAPH_DIR}" \
    --project_file_loc="./repo_structures" \
    ${MAX_EDIT_LOCS}

echo "[1/3] Localize done."

# ── Step 2: Repair ────────────────────────────────────────
echo ""
echo "[2/3] Repair (max_samples=10) ..."

PYTHONPATH=".:agentless/" python agentless/repair/repair.py \
    --loc_file="${LOC_FOLDER}/loc_outputs.jsonl" \
    --output_folder="${REP_FOLDER}" \
    --loc_interval \
    --top_n=3 \
    --context_window=10 \
    --max_samples=10 \
    --cot \
    --diff_format \
    --gen_and_process \
    --repo_graph \
    --graph_dir="${GRAPH_DIR}"

echo "[2/3] Repair done."

# ── Step 3: Rerank ────────────────────────────────────────
echo ""
echo "[3/3] Rerank ..."

PYTHONPATH=".:agentless/" python agentless/repair/rerank.py \
    --patch_folder="${REP_FOLDER}" \
    --num_samples=10 \
    --deduplicate \
    --plausible

echo "[3/3] Rerank done."

echo ""
echo "========================================"
echo "  ${FILTER_REPO}/${VARIANT} pipeline complete."
echo "  Patches: ${REP_FOLDER}/"
echo "========================================"

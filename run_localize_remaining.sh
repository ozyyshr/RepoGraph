#!/usr/bin/env bash
# 이미 처리된 인스턴스 제외하고 나머지 5개를 순서대로 실행
# 각각 별도 임시 폴더에서 실행 후 loc_outputs.jsonl을 메인 폴더에 append

VARIANT="${1:-baseline}"
GRAPH_DIR="repo_structures/graph/${VARIANT}"
MAIN_OUT="results/${VARIANT}/location"

INSTANCES=(
    "astropy__astropy-14182"
    "astropy__astropy-14365"
    "astropy__astropy-14995"
    "astropy__astropy-6938"
    "astropy__astropy-7746"
)

for IID in "${INSTANCES[@]}"; do
    echo "========================================"
    echo "Localizing: $IID"
    TMPDIR="${MAIN_OUT}/tmp_${IID}"
    mkdir -p "$TMPDIR"

    PYTHONPATH="/c/Users/jisue/OneDrive/문서/GitHub/RepoGraph:/c/Users/jisue/OneDrive/문서/GitHub/RepoGraph/agentless" \
    python agentless/fl/localize.py \
        --file_level \
        --related_level \
        --fine_grain_line_level \
        --output_folder="$TMPDIR" \
        --top_n=3 \
        --compress \
        --context_window=10 \
        --repo_graph \
        --target_id="$IID" \
        --graph_dir="$GRAPH_DIR" \
        --project_file_loc="repo_structures" 2>&1

    # append to main output
    if [ -f "$TMPDIR/loc_outputs.jsonl" ]; then
        cat "$TMPDIR/loc_outputs.jsonl" >> "$MAIN_OUT/loc_outputs.jsonl"
        echo "  Appended $IID to $MAIN_OUT/loc_outputs.jsonl"
    else
        echo "  WARNING: No output for $IID"
    fi
done

echo "========================================"
echo "Total localized: $(wc -l < $MAIN_OUT/loc_outputs.jsonl) instances"

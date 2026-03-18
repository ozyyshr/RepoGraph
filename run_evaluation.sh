#!/usr/bin/env bash
# ============================================================
# SWE-bench Docker 평가
# 사용법:
#   bash run_evaluation.sh baseline [repo]
#   bash run_evaluation.sh lsp [repo]
#   bash run_evaluation.sh lsp_limited [repo]
#
# repo 생략 시 기본값: astropy
# 예: bash run_evaluation.sh baseline matplotlib
# ============================================================
set -e

VARIANT="${1:-baseline}"
FILTER_REPO="${2:-astropy}"
REP_FOLDER="./results/${FILTER_REPO}/${VARIANT}/repair"
EVAL_FOLDER="./results/${FILTER_REPO}/${VARIANT}/eval"

# rerank 출력 파일 찾기
PATCHES_FILE="${REP_FOLDER}/all_preds.jsonl"
if [ ! -f "$PATCHES_FILE" ]; then
    PATCHES_FILE=$(find "${REP_FOLDER}" -name "*.jsonl" -newer "${REP_FOLDER}/output.jsonl" 2>/dev/null | head -1)
fi

if [ ! -f "$PATCHES_FILE" ]; then
    echo "ERROR: patches file not found in ${REP_FOLDER}"
    echo "Available files:"
    ls "${REP_FOLDER}/"
    exit 1
fi

# 패치 파일에서 instance_id 목록 추출
INSTANCE_IDS=$(python3 -c "
import json
ids = []
with open('${PATCHES_FILE}') as f:
    for line in f:
        d = json.loads(line)
        ids.append(d['instance_id'])
print(' '.join(ids))
")

echo "========================================"
echo "  Variant      : ${VARIANT}"
echo "  Repo         : ${FILTER_REPO}"
echo "  Patches file : ${PATCHES_FILE}"
echo "  Instances    : $(echo $INSTANCE_IDS | wc -w)"
echo "  Eval output  : ${EVAL_FOLDER}"
echo "========================================"

mkdir -p "${EVAL_FOLDER}"

# SWE-bench harness 실행
# 결과 로그: logs/run_evaluation/{RUN_ID}/
# 최종 리포트: CWD에 {model}.{RUN_ID}.json 으로 저장됨
RUN_ID="${FILTER_REPO}_${VARIANT}"
python -m swebench.harness.run_evaluation \
    --dataset_name "princeton-nlp/SWE-bench_Lite" \
    --split "test" \
    --predictions_path "${PATCHES_FILE}" \
    --max_workers 4 \
    --run_id "${RUN_ID}" \
    --instance_ids ${INSTANCE_IDS}

# 결과 JSON 복사 (swebench는 CWD에 {model}.{run_id}.json 으로 저장)
mkdir -p "${EVAL_FOLDER}"
GENERATED_REPORT="./agentless.${RUN_ID}.json"
REPORT_JSON="./agentless.${FILTER_REPO}.${VARIANT}.json"

if [ -f "${GENERATED_REPORT}" ]; then
    cp "${GENERATED_REPORT}" "${EVAL_FOLDER}/report.json"
    # 기존 astropy 결과와의 일관성을 위해 루트에도 복사
    [ "${GENERATED_REPORT}" != "${REPORT_JSON}" ] && cp "${GENERATED_REPORT}" "${REPORT_JSON}"
    echo "결과 저장: ${EVAL_FOLDER}/report.json  (${REPORT_JSON})"
else
    echo "WARNING: ${GENERATED_REPORT} 를 찾지 못했습니다. CWD의 *.json 파일을 확인하세요."
fi

echo ""
echo "평가 완료. 결과: ${EVAL_FOLDER}/"

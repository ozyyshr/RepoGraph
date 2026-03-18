"""
repo_structures/ 사전 구축 스크립트.

SWE-bench-Lite에서 지정한 repo의 인스턴스를 추려 각 base_commit에 대해:
  1. AST 기반 프로젝트 구조 JSON  →  repo_structures/{instance_id}.json
  2. tree-sitter 코드 그래프 (baseline) → repo_structures/graph/baseline/
  3. LSP 코드 그래프 (ours)             → repo_structures/graph/lsp/

사용법:
  python build_repo_structures.py [--repo astropy|matplotlib|...] [--mode all|structure|baseline|lsp] [--instance_id ...]
"""

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

# ────────────────────────────────────────────────
# 경로 설정
# ────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

OUT_BASE     = os.path.join(REPO_ROOT, "repo_structures")
OUT_BASELINE = os.path.join(OUT_BASE, "graph", "baseline")
OUT_LSP      = os.path.join(OUT_BASE, "graph", "lsp")

sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "repograph"))
sys.path.insert(0, os.path.join(REPO_ROOT, "agentless"))


# ────────────────────────────────────────────────
# 디렉토리 생성
# ────────────────────────────────────────────────
for d in [OUT_BASE, OUT_BASELINE, OUT_LSP]:
    os.makedirs(d, exist_ok=True)


# ────────────────────────────────────────────────
# AST 기반 create_structure (get_repo_structure.py 버전)
# ────────────────────────────────────────────────
from agentless.get_repo_structure.get_repo_structure import (
    create_structure as _create_structure_ast,
)


def filter_non_test_py(directory):
    """Return non-test Python files under directory."""
    result = []
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
        for f in files:
            if f.endswith(".py") and "test" not in os.path.relpath(
                os.path.join(root, f), directory
            ):
                result.append(os.path.join(root, f))
    return sorted(result)


# ────────────────────────────────────────────────
# tree-sitter 함수들 (integration_test.py에서 추출)
# ────────────────────────────────────────────────
import importlib.util, types as _types

def _load_ts_helpers():
    src_path = os.path.join(REPO_ROOT, "integration_test.py")
    with open(src_path, encoding="utf-8") as f:
        src = f.read()
    # __main__ 이후 코드 잘라냄
    for marker in ["\nnddata_files = collect_py_files", "\nastropy_files = collect_py_files"]:
        cut = src.find(marker)
        if cut != -1:
            src = src[:cut]
            break
    mod = _types.ModuleType("_ts_helpers")
    mod.__file__ = src_path
    exec(compile(src, src_path, "exec"), mod.__dict__)
    return mod

_ts = _load_ts_helpers()
_create_structure_ts_internal = _ts._create_structure_ast   # tree-sitter 내부용
_get_tags_raw_treesitter       = _ts._get_tags_raw_treesitter
_tag_to_graph_treesitter       = _ts._tag_to_graph_treesitter


def build_baseline_graph(py_files, repo_dir):
    """tree-sitter 기반 그래프 빌드. (tags, G) 반환."""
    structure = _create_structure_ts_internal(repo_dir)
    tags_all = []
    for fname in tqdm(py_files, desc="  TS tags", leave=False):
        rel = os.path.relpath(fname, repo_dir).replace(os.sep, "/")
        tags_all.extend(_get_tags_raw_treesitter(fname, rel, structure, repo_dir))
    G = _tag_to_graph_treesitter(tags_all)
    return tags_all, G


# ────────────────────────────────────────────────
# LSP 기반 그래프 빌드 (repograph/construct_graph.py)
# ────────────────────────────────────────────────
from construct_graph import CodeGraph


def build_lsp_graph(py_files, repo_dir):
    """LSP(jedi) 기반 그래프 빌드. (tags, G) 반환."""
    cg = CodeGraph(root=repo_dir)
    tags, G = cg.get_code_graph(py_files)
    return tags or [], G


# ────────────────────────────────────────────────
# 태그 → JSON 직렬화 헬퍼
# ────────────────────────────────────────────────
def _tag_to_dict(tag, repo_dir):
    """tag namedtuple을 retrieve_graph 호환 JSON dict로 변환.

    핵심 규칙:
      - rel_fname: forward slash 사용 (retrieve_graph가 split('/') 사용)
      - line: 단일 정수 (ref 태그는 참조 줄 번호, def 태그는 시작 줄)
    """
    # rel_fname 정규화
    try:
        rel = os.path.relpath(tag.fname, repo_dir).replace(os.sep, "/")
    except (AttributeError, ValueError):
        rel = getattr(tag, "rel_fname", "").replace(os.sep, "/")

    # line: list[start, end] 또는 int
    line_val = tag.line
    if isinstance(line_val, (list, tuple)):
        line_int = int(line_val[0])
    else:
        line_int = int(line_val)

    d = {
        "name":     tag.name,
        "kind":     tag.kind,
        "rel_fname": rel,
        "line":     line_int,
        "category": getattr(tag, "category", "function"),
        "fname":    tag.fname,
    }
    # LSP 태그에만 있는 extra 필드
    for extra in ("full_name", "caller_full_name"):
        val = getattr(tag, extra, None)
        if val is not None:
            d[extra] = val
    return d


# ────────────────────────────────────────────────
# git checkout 헬퍼
# ────────────────────────────────────────────────
def checkout(commit_id, repo_path):
    print(f"  git checkout {commit_id[:12]} …", end=" ", flush=True)
    result = subprocess.run(
        ["git", "-C", repo_path, "checkout", commit_id],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"FAILED\n{result.stderr.strip()}")
        return False
    print("OK")
    return True


# ────────────────────────────────────────────────
# 인스턴스 처리
# ────────────────────────────────────────────────
def process_instance(instance, mode, repo_dir, force=False):
    iid         = instance["instance_id"]
    base_commit = instance["base_commit"]

    structure_path   = os.path.join(OUT_BASE, f"{iid}.json")
    baseline_pkl     = os.path.join(OUT_BASELINE, f"{iid}.pkl")
    baseline_tags    = os.path.join(OUT_BASELINE, f"tags_{iid}.json")
    lsp_pkl          = os.path.join(OUT_LSP, f"{iid}.pkl")
    lsp_tags         = os.path.join(OUT_LSP, f"tags_{iid}.json")

    # 이미 완료된 파트 스킵 여부 결정
    need_structure = mode in ("all", "structure") and (force or not os.path.exists(structure_path))
    need_baseline  = mode in ("all", "baseline")  and (force or not os.path.exists(baseline_pkl))
    need_lsp       = mode in ("all", "lsp")        and (force or not os.path.exists(lsp_pkl))

    if not (need_structure or need_baseline or need_lsp):
        print(f"[{iid}] 이미 완료됨, 건너뜀")
        return True

    print(f"\n{'='*60}")
    print(f"[{iid}]  commit={base_commit[:12]}")
    print(f"  mode={mode}  need: structure={need_structure} baseline={need_baseline} lsp={need_lsp}")

    # git checkout
    if not checkout(base_commit, repo_path=repo_dir):
        return False

    py_files = filter_non_test_py(repo_dir)
    print(f"  Python 파일 (비테스트): {len(py_files)}개")

    # 1. AST 구조 JSON
    if need_structure:
        t0 = time.perf_counter()
        print("  [1/3] AST 구조 JSON 빌드 …")
        structure = _create_structure_ast(repo_dir)
        d = {
            "repo": instance["repo"],
            "base_commit": base_commit,
            "structure": structure,
            "instance_id": iid,
        }
        with open(structure_path, "w", encoding="utf-8") as f:
            json.dump(d, f)
        print(f"       → {os.path.relpath(structure_path, REPO_ROOT)}  ({time.perf_counter()-t0:.1f}s)")
    else:
        print("  [1/3] AST 구조 JSON: 기존 파일 사용")

    # 2. tree-sitter 그래프 (baseline)
    if need_baseline:
        t0 = time.perf_counter()
        print("  [2/3] tree-sitter 그래프 빌드 …")
        try:
            tags, G = build_baseline_graph(py_files, repo_dir)
            with open(baseline_pkl, "wb") as f:
                pickle.dump(G, f)
            tag_dicts = [_tag_to_dict(t, repo_dir) for t in tags]
            with open(baseline_tags, "w", encoding="utf-8") as f:
                json.dump(tag_dicts, f)
            print(f"       → nodes={len(G.nodes)}, edges={len(G.edges)}, tags={len(tag_dicts)}  ({time.perf_counter()-t0:.1f}s)")
        except Exception as e:
            print(f"       FAILED: {e}")
            return False
    else:
        print("  [2/3] baseline 그래프: 기존 파일 사용")

    # 3. LSP 그래프 (ours)
    if need_lsp:
        t0 = time.perf_counter()
        print("  [3/3] LSP 그래프 빌드 중 (시간이 걸릴 수 있습니다) …")
        try:
            tags, G = build_lsp_graph(py_files, repo_dir)
            with open(lsp_pkl, "wb") as f:
                pickle.dump(G, f)
            tag_dicts = [_tag_to_dict(t, repo_dir) for t in tags if t is not None]
            with open(lsp_tags, "w", encoding="utf-8") as f:
                json.dump(tag_dicts, f)
            elapsed = time.perf_counter() - t0
            print(f"       → nodes={len(G.nodes)}, edges={len(G.edges)}, tags={len(tag_dicts)}  ({elapsed:.1f}s)")
        except Exception as e:
            print(f"       FAILED: {e}")
            return False
    else:
        print("  [3/3] LSP 그래프: 기존 파일 사용")

    return True


# ────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SWE-bench-Lite용 repo_structures 사전 구축")
    parser.add_argument(
        "--repo", type=str, default="astropy",
        help="처리할 repo 이름 (instance_id prefix, 예: astropy, matplotlib). "
             "playground/{repo} 디렉토리에 클론되어 있어야 함. (default: astropy)"
    )
    parser.add_argument(
        "--mode", choices=["all", "structure", "baseline", "lsp"], default="all",
        help="빌드할 항목 (default: all)"
    )
    parser.add_argument(
        "--instance_id", type=str, default=None,
        help="특정 인스턴스만 처리 (예: astropy__astropy-12907)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="기존 파일이 있어도 덮어씀"
    )
    args = parser.parse_args()

    repo_dir = os.path.join(REPO_ROOT, "playground", args.repo)

    print("SWE-bench-Lite 로드 중 …")
    swe_bench = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    instances = [x for x in swe_bench if x["instance_id"].startswith(args.repo)]
    print(f"{args.repo} 인스턴스 수: {len(instances)}")

    if args.instance_id:
        instances = [x for x in instances if x["instance_id"] == args.instance_id]
        if not instances:
            print(f"ERROR: 인스턴스 '{args.instance_id}'를 찾을 수 없습니다.")
            sys.exit(1)

    print(f"\n처리 대상: {len(instances)}개  mode={args.mode}")
    print(f"playground/{args.repo}: {repo_dir}")

    # playground/{repo} 존재 확인
    if not os.path.isdir(repo_dir):
        print(f"ERROR: {repo_dir} 디렉토리가 없습니다.")
        print(f"먼저 레포를 playground/에 클론해 주세요:")
        # SWE-bench 인스턴스에서 repo URL 추출
        if instances:
            sample_repo = instances[0]["repo"]
            print(f"  git clone https://github.com/{sample_repo} playground/{args.repo}")
        sys.exit(1)

    # git 상태 확인 (uncommitted changes 경고)
    result = subprocess.run(
        ["git", "-C", repo_dir, "status", "--porcelain"],
        capture_output=True, text=True
    )
    if result.stdout.strip():
        print(f"WARNING: playground/{args.repo}에 uncommitted 변경사항이 있습니다.")
        print("git stash 후 진행을 권장합니다.")
        answer = input("계속 진행하시겠습니까? [y/N] ").strip().lower()
        if answer != "y":
            sys.exit(0)

    success_count = 0
    fail_list = []
    total_start = time.perf_counter()

    for i, instance in enumerate(instances):
        iid = instance["instance_id"]
        print(f"\n진행: {i+1}/{len(instances)}")
        ok = process_instance(instance, mode=args.mode, repo_dir=repo_dir, force=args.force)
        if ok:
            success_count += 1
        else:
            fail_list.append(iid)

    elapsed_total = time.perf_counter() - total_start
    print(f"\n{'='*60}")
    print(f"완료: {success_count}/{len(instances)}  총 소요 {elapsed_total/60:.1f}분")
    if fail_list:
        print(f"실패한 인스턴스:")
        for x in fail_list:
            print(f"  - {x}")

    print(f"\n주의: playground/{args.repo}가 마지막 처리된 commit에 남아있습니다.")


if __name__ == "__main__":
    main()

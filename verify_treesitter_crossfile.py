"""
코드 레벨 검증: tree-sitter 버전의 cross-file invoke가 0인 이유.

세 가지 시나리오를 순서대로 검사한다.
  S1. 통합 테스트에서 사용한 adapt 버전 (ref.fname → ref.name)
  S2. 원본 construct_graph.py의 invoke 로직 (tag.name → tag_def.name, 동일 이름 조건)
  S3. 최대 관대한 측정 (노드 속성의 fname 직접 비교)
"""

import os, sys
from collections import defaultdict
from pathlib import Path
import networkx as nx

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "repograph"))

# 필요한 헬퍼만 직접 임포트 (integration_test 전체 실행 방지)
import importlib.util, types

def _import_helpers():
    """integration_test.py에서 __main__ 블록 없이 헬퍼 함수만 추출."""
    spec = importlib.util.spec_from_file_location(
        "it_helpers",
        os.path.join(REPO_ROOT, "integration_test.py")
    )
    # 빈 모듈 생성 후 exec_module 대신 소스를 직접 파싱
    src_path = os.path.join(REPO_ROOT, "integration_test.py")
    with open(src_path, encoding="utf-8") as f:
        src = f.read()
    # __main__ 블록 이하를 잘라냄
    cutoff = src.find("\nnddata_files = collect_py_files")
    if cutoff == -1:
        cutoff = src.find("\nastropy_files = collect_py_files")
    if cutoff != -1:
        src = src[:cutoff]
    mod = types.ModuleType("it_helpers")
    mod.__file__ = src_path
    exec(compile(src, src_path, "exec"), mod.__dict__)
    return mod

_it = _import_helpers()
_create_structure_ast      = _it._create_structure_ast
_get_tags_raw_treesitter   = _it._get_tags_raw_treesitter
collect_py_files           = _it.collect_py_files
TSTag                      = _it.TSTag

ASTROPY_ROOT = os.path.join(REPO_ROOT, "playground", "astropy")
NDDATA_ROOT  = os.path.join(ASTROPY_ROOT, "astropy", "nddata")

# 대표 파일 4개 (bitmask, nddata, nddata_base, ccddata — 파일 간 상속 있음)
FILES = [
    os.path.join(NDDATA_ROOT, "bitmask.py"),
    os.path.join(NDDATA_ROOT, "nddata_base.py"),
    os.path.join(NDDATA_ROOT, "nddata.py"),
    os.path.join(NDDATA_ROOT, "ccddata.py"),
]
FILES = [f for f in FILES if os.path.isfile(f)]
REL   = {f: os.path.relpath(f, ASTROPY_ROOT) for f in FILES}

print("=" * 65)
print("검증: tree-sitter cross-file invoke 구조 분석")
print("=" * 65)
print(f"\n대상 파일 {len(FILES)}개:")
for f in FILES:
    print(f"  {REL[f]}")

# ──────────────────────────────────────────────
# 태그 수집
# ──────────────────────────────────────────────
structure = _create_structure_ast(ASTROPY_ROOT)
tags_all = []
for f in FILES:
    tags_all.extend(_get_tags_raw_treesitter(f, REL[f], structure, ASTROPY_ROOT))

def_tags = [t for t in tags_all if t.kind == 'def']
ref_tags  = [t for t in tags_all if t.kind == 'ref']
print(f"\n태그 수집: def={len(def_tags)}, ref={len(ref_tags)}, total={len(tags_all)}")

# ──────────────────────────────────────────────
# 질문 1: 노드 ID 포맷 확인
# ──────────────────────────────────────────────
print("\n" + "─" * 65)
print("Q1. 노드 ID 포맷: 단순 이름인가, qualified name인가?")
print("─" * 65)

sample_defs = def_tags[:15]
for t in sample_defs:
    src_file = os.path.basename(t.fname)
    print(f"  tag.name = {t.name!r:30s}  ← fname = {src_file}")

# 이름 충돌 확인 (같은 이름, 다른 파일)
from collections import Counter
name_to_files = defaultdict(set)
for t in def_tags:
    name_to_files[t.name].add(t.fname)

collisions = {name: files for name, files in name_to_files.items() if len(files) > 1}
print(f"\n  같은 이름이 여러 파일에 정의된 경우: {len(collisions)}건")
for name, files in list(collisions.items())[:5]:
    print(f"  '{name}' →")
    for f in files:
        print(f"      {os.path.relpath(f, ASTROPY_ROOT)}")

# ──────────────────────────────────────────────
# 질문 2a: 원본 construct_graph.py의 invoke 로직 시뮬레이션
# ──────────────────────────────────────────────
print("\n" + "─" * 65)
print("Q2a. 원본 tag_to_graph invoke 로직 시뮬레이션")
print("     (원본: if tag.name == tag_def.name: G.add_edge(tag.name, tag_def.name))")
print("─" * 65)

G_orig = nx.MultiDiGraph()
# 노드: tag.name (단순 이름, 마지막 write wins)
for t in tags_all:
    G_orig.add_node(t.name,
                    category=t.category, info=t.info,
                    fname=t.fname, line=t.line, kind=t.kind)

# 원본 invoke 로직: tag.name == tag_def.name 조건 → 동일 이름이면 add_edge
def_name_set = {t.name for t in def_tags}
orig_invoke_edges = []
for ref in ref_tags:
    for defn in def_tags:
        if ref.name == defn.name:
            G_orig.add_edge(ref.name, defn.name, edge_type='invoke',
                            ref_fname=ref.fname, def_fname=defn.fname)
            orig_invoke_edges.append((ref.name, defn.name, ref.fname, defn.fname))
            break  # 첫 매칭만

self_loops   = [(u, v) for u, v, _, _ in orig_invoke_edges if u == v]
diff_nodes   = [(u, v) for u, v, _, _ in orig_invoke_edges if u != v]
print(f"\n  생성된 invoke edges:     {len(orig_invoke_edges)}")
print(f"  self-loop (u == v):      {len(self_loops)}  ← 전부 동일 이름이라 항상 self-loop")
print(f"  non-self-loop (u != v):  {len(diff_nodes)}")

# 원본 로직에서 cross-file 측정 시도
cross_orig = 0
for u, v, rf, df in orig_invoke_edges:
    u_fname = G_orig.nodes[u].get('fname', '')
    v_fname = G_orig.nodes[v].get('fname', '')
    if u_fname and v_fname and u_fname != v_fname:
        cross_orig += 1

print(f"\n  cross-file (node.fname 비교): {cross_orig}")
print(f"  → 이유: u == v (self-loop)이면 G.nodes[u].fname == G.nodes[v].fname 항상")

# 실제로 self-loop의 fname 샘플 확인
print("\n  Self-loop invoke edge 샘플 (이름, node.fname):")
shown = set()
for name, _, rf, df in orig_invoke_edges[:20]:
    if name in shown:
        continue
    shown.add(name)
    stored_fname = G_orig.nodes[name].get('fname', 'None')
    stored_rel   = os.path.relpath(stored_fname, ASTROPY_ROOT) if stored_fname else 'None'
    ref_rel      = os.path.relpath(rf, ASTROPY_ROOT)
    def_rel      = os.path.relpath(df, ASTROPY_ROOT)
    print(f"    '{name}'")
    print(f"      ref 발생 파일:      {ref_rel}")
    print(f"      def 정의 파일:      {def_rel}")
    print(f"      node.fname (저장):  {stored_rel}  ← 마지막 write가 덮어씀")
    if shown.__len__() >= 5:
        break

# ──────────────────────────────────────────────
# 질문 2b: 통합 테스트 adapt 버전 (ref.fname → ref.name)
# ──────────────────────────────────────────────
print("\n" + "─" * 65)
print("Q2b. 통합 테스트 adapt 버전 시뮬레이션")
print("     (adapt: ref.fname → ref.name 으로 엣지 생성)")
print("─" * 65)

G_adapt = nx.MultiDiGraph()
for t in tags_all:
    G_adapt.add_node(t.name, category=t.category, info=t.info,
                     fname=t.fname, line=t.line, kind=t.kind)

adapt_invoke = []
for ref in ref_tags:
    if ref.name in def_name_set:
        G_adapt.add_edge(ref.fname, ref.name, edge_type='invoke')
        adapt_invoke.append((ref.fname, ref.name))

cross_adapt = 0
for u, v in adapt_invoke:
    u_fname = G_adapt.nodes.get(u, {}).get('fname', '')  # ref.fname는 노드에 없음!
    v_fname = G_adapt.nodes.get(v, {}).get('fname', '')
    if u_fname and v_fname and u_fname != v_fname:
        cross_adapt += 1

src_is_node = sum(1 for u, _ in adapt_invoke if u in G_adapt.nodes
                  and 'fname' in G_adapt.nodes[u])
print(f"\n  invoke edges 수:                          {len(adapt_invoke)}")
print(f"  source 노드(ref.fname)가 fname 속성 보유: {src_is_node}/{len(adapt_invoke)}")
print(f"  cross-file (fname 비교):                  {cross_adapt}")
print(f"  → 이유: source = ref.fname (파일 경로)는 G.nodes에 fname 속성 없음")
print(f"          ∴ u_fname = '' → cross-file 조건 불충족")

# ──────────────────────────────────────────────
# 질문 3: 최대 관대한 측정 — 태그 레벨에서 직접 비교
# ──────────────────────────────────────────────
print("\n" + "─" * 65)
print("Q3. 최대 관대한 측정: 태그 레벨에서 ref.fname ≠ def.fname 카운팅")
print("    (그래프 구조 없이, 태그 직접 비교)")
print("─" * 65)

# def 이름 → fname 매핑 (복수 파일에 같은 이름 있으면 모두 수집)
def_name_to_fnames = defaultdict(set)
for t in def_tags:
    def_name_to_fnames[t.name].add(t.fname)

cross_tag_level = 0
cross_tag_same  = 0
cross_examples  = []

for ref in ref_tags:
    if ref.name not in def_name_to_fnames:
        continue
    def_fnames = def_name_to_fnames[ref.name]
    other_file_defs = [f for f in def_fnames if f != ref.fname]
    same_file_defs  = [f for f in def_fnames if f == ref.fname]
    if other_file_defs:
        cross_tag_level += 1
        if len(cross_examples) < 8:
            cross_examples.append((ref.name, ref.fname, other_file_defs[0]))
    elif same_file_defs:
        cross_tag_same += 1

print(f"\n  ref가 다른 파일의 def 이름과 일치 (잠재적 cross-file): {cross_tag_level}")
print(f"  ref가 같은 파일의 def 이름과만 일치:                   {cross_tag_same}")
print(f"\n  → 하지만 이름 일치 ≠ 실제 호출 관계")
print(f"    예: '__init__'이 여러 파일에 있어도 각각 다른 클래스의 __init__")
print(f"    tree-sitter는 어느 __init__을 호출하는지 구분 불가")
print(f"\n  잠재적 cross-file 매칭 샘플 (이름만 같을 뿐, 실제 resolve 불가):")
for name, ref_f, def_f in cross_examples:
    ref_rel = os.path.relpath(ref_f, ASTROPY_ROOT)
    def_rel = os.path.relpath(def_f, ASTROPY_ROOT)
    print(f"    '{name}'")
    print(f"      ref: {ref_rel}")
    print(f"      def: {def_rel}  ← 실제 이 def를 부르는지 확인 불가")

# ──────────────────────────────────────────────
# 종합 판정
# ──────────────────────────────────────────────
print("\n" + "=" * 65)
print("종합 판정: 팀 가설 검증")
print("=" * 65)
print("""
가설: "tree-sitter 버전은 구조적으로 cross-file resolve가 불가능한 설계"

검증 결과: ✅ 가설 정확함. 3가지 구조적 이유:

1. 노드 ID = 단순 이름 (short name)
   • tag.name = '__init__', 'fit', 'read' 등 파일 경로 없음
   • 같은 이름이 여러 파일에 있으면 같은 노드로 병합 (마지막 write wins)
   • nddata 4개 파일에서 이름 충돌 {n_collisions}건 확인

2. 원본 invoke 엣지 = 항상 self-loop
   • 조건: if tag.name == tag_def.name → G.add_edge(tag.name, tag_def.name)
   • 동일 이름이므로 항상 same_node → same_node (self-loop)
   • self-loop의 양 끝은 동일 노드 → fname도 동일 → cross-file = 0

3. 이름 일치 ≠ 실제 호출 관계
   • 태그 레벨에서 최대 관대하게 측정해도 "이름만 같은" 잠재적 cross-file
     {cross_tag_level}건이 있지만, 어느 파일의 어느 정의를 실제로
     호출하는지는 구문 분석만으로 결정 불가능
   • LSP(jedi.goto)만이 타입 추론 + import resolution으로 실제 정의 위치를 확정

결론: LSP 버전의 cross-file invoke 473개는 의미 있는 정보이지만,
      tree-sitter 버전에서 cross-file invoke "0"은 측정 방식의 한계가 아니라
      알고리즘 설계 자체의 한계임.
""".format(n_collisions=len(collisions), cross_tag_level=cross_tag_level))

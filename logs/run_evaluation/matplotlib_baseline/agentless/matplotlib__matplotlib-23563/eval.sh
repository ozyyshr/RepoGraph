#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff 149a0398b391cfc4eddb5e659f50b3c66f32ea65
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 149a0398b391cfc4eddb5e659f50b3c66f32ea65 lib/mpl_toolkits/tests/test_mplot3d.py
git apply -v - <<'EOF_114329324912'
diff --git a/lib/mpl_toolkits/tests/test_mplot3d.py b/lib/mpl_toolkits/tests/test_mplot3d.py
--- a/lib/mpl_toolkits/tests/test_mplot3d.py
+++ b/lib/mpl_toolkits/tests/test_mplot3d.py
@@ -1786,6 +1786,13 @@ def test_text_3d(fig_test, fig_ref):
     assert t3d.get_position_3d() == (0.5, 0.5, 1)
 
 
+def test_draw_single_lines_from_Nx1():
+    # Smoke test for GH#23459
+    fig = plt.figure()
+    ax = fig.add_subplot(projection='3d')
+    ax.plot([[0], [1]], [[0], [1]], [[0], [1]])
+
+
 @check_figures_equal(extensions=["png"])
 def test_pathpatch_3d(fig_test, fig_ref):
     ax = fig_ref.add_subplot(projection="3d")

EOF_114329324912
: '>>>>> Start Test Output'
pytest -rA lib/mpl_toolkits/tests/test_mplot3d.py
: '>>>>> End Test Output'
git checkout 149a0398b391cfc4eddb5e659f50b3c66f32ea65 lib/mpl_toolkits/tests/test_mplot3d.py

#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff 78bf53caacbb5ce0dc7aa73f07a74c99f1ed919b
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 78bf53caacbb5ce0dc7aa73f07a74c99f1ed919b lib/matplotlib/tests/test_colorbar.py
git apply -v - <<'EOF_114329324912'
diff --git a/lib/matplotlib/tests/test_colorbar.py b/lib/matplotlib/tests/test_colorbar.py
--- a/lib/matplotlib/tests/test_colorbar.py
+++ b/lib/matplotlib/tests/test_colorbar.py
@@ -657,6 +657,12 @@ def test_colorbar_scale_reset():
 
     assert cbar.outline.get_edgecolor() == mcolors.to_rgba('red')
 
+    # log scale with no vmin/vmax set should scale to the data if there
+    # is a mappable already associated with the colorbar, not (0, 1)
+    pcm.norm = LogNorm()
+    assert pcm.norm.vmin == z.min()
+    assert pcm.norm.vmax == z.max()
+
 
 def test_colorbar_get_ticks_2():
     plt.rcParams['_internal.classic_mode'] = False

EOF_114329324912
: '>>>>> Start Test Output'
pytest -rA lib/matplotlib/tests/test_colorbar.py
: '>>>>> End Test Output'
git checkout 78bf53caacbb5ce0dc7aa73f07a74c99f1ed919b lib/matplotlib/tests/test_colorbar.py

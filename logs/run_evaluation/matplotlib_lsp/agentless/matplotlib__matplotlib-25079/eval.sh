#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff 66f7956984cbfc3647e867c6e5fde889a89c64ef
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 66f7956984cbfc3647e867c6e5fde889a89c64ef lib/matplotlib/tests/test_colors.py
git apply -v - <<'EOF_114329324912'
diff --git a/lib/matplotlib/tests/test_colors.py b/lib/matplotlib/tests/test_colors.py
--- a/lib/matplotlib/tests/test_colors.py
+++ b/lib/matplotlib/tests/test_colors.py
@@ -1493,6 +1493,11 @@ def test_norm_callback():
     norm.vmax = 5
     assert increment.call_count == 2
 
+    # We only want autoscale() calls to send out one update signal
+    increment.call_count = 0
+    norm.autoscale([0, 1, 2])
+    assert increment.call_count == 1
+
 
 def test_scalarmappable_norm_update():
     norm = mcolors.Normalize()

EOF_114329324912
: '>>>>> Start Test Output'
pytest -rA lib/matplotlib/tests/test_colors.py
: '>>>>> End Test Output'
git checkout 66f7956984cbfc3647e867c6e5fde889a89c64ef lib/matplotlib/tests/test_colors.py

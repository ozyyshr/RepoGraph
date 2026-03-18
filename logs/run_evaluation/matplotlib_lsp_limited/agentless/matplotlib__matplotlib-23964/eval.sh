#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff 269c0b94b4fcf8b1135011c1556eac29dc09de15
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 269c0b94b4fcf8b1135011c1556eac29dc09de15 lib/matplotlib/tests/test_backend_ps.py
git apply -v - <<'EOF_114329324912'
diff --git a/lib/matplotlib/tests/test_backend_ps.py b/lib/matplotlib/tests/test_backend_ps.py
--- a/lib/matplotlib/tests/test_backend_ps.py
+++ b/lib/matplotlib/tests/test_backend_ps.py
@@ -256,6 +256,15 @@ def test_linedash():
     assert buf.tell() > 0
 
 
+def test_empty_line():
+    # Smoke-test for gh#23954
+    figure = Figure()
+    figure.text(0.5, 0.5, "\nfoo\n\n")
+    buf = io.BytesIO()
+    figure.savefig(buf, format='eps')
+    figure.savefig(buf, format='ps')
+
+
 def test_no_duplicate_definition():
 
     fig = Figure()

EOF_114329324912
: '>>>>> Start Test Output'
pytest -rA lib/matplotlib/tests/test_backend_ps.py
: '>>>>> End Test Output'
git checkout 269c0b94b4fcf8b1135011c1556eac29dc09de15 lib/matplotlib/tests/test_backend_ps.py

#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff 332937997d03e0c173be6d9fc1841e9186e857df
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 332937997d03e0c173be6d9fc1841e9186e857df lib/matplotlib/tests/test_axes.py
git apply -v - <<'EOF_114329324912'
diff --git a/lib/matplotlib/tests/test_axes.py b/lib/matplotlib/tests/test_axes.py
--- a/lib/matplotlib/tests/test_axes.py
+++ b/lib/matplotlib/tests/test_axes.py
@@ -5732,6 +5732,17 @@ def test_set_get_ticklabels():
     ax[1].set_yticklabels(ax[0].get_yticklabels())
 
 
+def test_set_ticks_kwargs_raise_error_without_labels():
+    """
+    When labels=None and any kwarg is passed, axis.set_ticks() raises a
+    ValueError.
+    """
+    fig, ax = plt.subplots()
+    ticks = [1, 2, 3]
+    with pytest.raises(ValueError):
+        ax.xaxis.set_ticks(ticks, alpha=0.5)
+
+
 @check_figures_equal(extensions=["png"])
 def test_set_ticks_with_labels(fig_test, fig_ref):
     """

EOF_114329324912
: '>>>>> Start Test Output'
pytest -rA lib/matplotlib/tests/test_axes.py
: '>>>>> End Test Output'
git checkout 332937997d03e0c173be6d9fc1841e9186e857df lib/matplotlib/tests/test_axes.py

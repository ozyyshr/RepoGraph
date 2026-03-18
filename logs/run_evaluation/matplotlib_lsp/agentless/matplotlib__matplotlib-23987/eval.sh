#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff e98d8d085e8f53ec0467422b326f7738a2dd695e
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout e98d8d085e8f53ec0467422b326f7738a2dd695e lib/matplotlib/tests/test_constrainedlayout.py
git apply -v - <<'EOF_114329324912'
diff --git a/lib/matplotlib/tests/test_constrainedlayout.py b/lib/matplotlib/tests/test_constrainedlayout.py
--- a/lib/matplotlib/tests/test_constrainedlayout.py
+++ b/lib/matplotlib/tests/test_constrainedlayout.py
@@ -656,3 +656,14 @@ def test_compressed1():
     pos = axs[1, 2].get_position()
     np.testing.assert_allclose(pos.x1, 0.8618, atol=1e-3)
     np.testing.assert_allclose(pos.y0, 0.1934, atol=1e-3)
+
+
+@pytest.mark.parametrize('arg, state', [
+    (True, True),
+    (False, False),
+    ({}, True),
+    ({'rect': None}, True)
+])
+def test_set_constrained_layout(arg, state):
+    fig, ax = plt.subplots(constrained_layout=arg)
+    assert fig.get_constrained_layout() is state

EOF_114329324912
: '>>>>> Start Test Output'
pytest -rA lib/matplotlib/tests/test_constrainedlayout.py
: '>>>>> End Test Output'
git checkout e98d8d085e8f53ec0467422b326f7738a2dd695e lib/matplotlib/tests/test_constrainedlayout.py

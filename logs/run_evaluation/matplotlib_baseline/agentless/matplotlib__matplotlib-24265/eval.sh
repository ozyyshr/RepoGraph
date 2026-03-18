#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff e148998d9bed9d1b53a91587ad48f9bb43c7737f
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout e148998d9bed9d1b53a91587ad48f9bb43c7737f lib/matplotlib/tests/test_style.py
git apply -v - <<'EOF_114329324912'
diff --git a/lib/matplotlib/tests/test_style.py b/lib/matplotlib/tests/test_style.py
--- a/lib/matplotlib/tests/test_style.py
+++ b/lib/matplotlib/tests/test_style.py
@@ -184,6 +184,8 @@ def test_deprecated_seaborn_styles():
     with pytest.warns(mpl._api.MatplotlibDeprecationWarning):
         mpl.style.use("seaborn-bright")
     assert mpl.rcParams == seaborn_bright
+    with pytest.warns(mpl._api.MatplotlibDeprecationWarning):
+        mpl.style.library["seaborn-bright"]
 
 
 def test_up_to_date_blacklist():

EOF_114329324912
: '>>>>> Start Test Output'
pytest -rA lib/matplotlib/tests/test_style.py
: '>>>>> End Test Output'
git checkout e148998d9bed9d1b53a91587ad48f9bb43c7737f lib/matplotlib/tests/test_style.py

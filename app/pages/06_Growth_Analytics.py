"""Page 6: Growth Visualization Dashboard (Backward-Compatible Forwarder)."""

from pathlib import Path
import sys

# Ensure project root is on sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

# Import and execute the unified Module 6 Growth Dashboard
dashboard_script = Path(__file__).resolve().parent / "06_Growth_Dashboard.py"
if dashboard_script.exists():
    with open(dashboard_script, "r", encoding="utf-8") as f:
        code = compile(f.read(), str(dashboard_script), "exec")
        exec(code, globals())

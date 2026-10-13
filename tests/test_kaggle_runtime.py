"""Ensure an unrelated Ray installation cannot attach a broken training callback."""
from pathlib import Path
import os
import subprocess
import sys


def test_kaggle_disables_ray_callback_in_a_fresh_process(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = """
import sys, types
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'kaggle/ema_dysample'))
from runtime import configure_ultralytics
configure_ultralytics()
# Simulate an installed Ray that does not expose Ultralytics' old private API.
sys.modules['ray'] = types.ModuleType('ray')
from ultralytics.utils.callbacks import raytune
from ultralytics.utils.callbacks import wb
assert raytune.callbacks == {}, 'An unused Ray integration attached callbacks'
assert raytune.tune is None
assert wb.callbacks == {}, 'An unused W&B integration attached callbacks'
print('Ray callback disabled')
"""
    environment = os.environ.copy()
    environment["YOLO_CONFIG_DIR"] = str(tmp_path / "ultralytics")
    environment["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    result = subprocess.run([sys.executable, "-c", script], cwd=root, env=environment,
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Ray callback disabled" in result.stdout

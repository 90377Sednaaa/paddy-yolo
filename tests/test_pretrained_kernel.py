"""Check the submitted notebook's checkpoint selection without ML dependencies."""
import ast
import base64
import io
import json
import csv
import gc
import hashlib
import statistics
import traceback
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock
from types import SimpleNamespace
import zipfile


class PretrainedKernelTest(unittest.TestCase):
    def test_each_fold_restarts_from_requested_initial_checkpoint(self):
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root / "run_baseline_crossval.py").read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ("training_command", "run")]
        dataset = MagicMock()
        dataset.__truediv__.return_value.iterdir.return_value = list(range(2024))
        dataset.__truediv__.return_value.glob.return_value = [SimpleNamespace(read_text=lambda: "box\n" * 2281)]
        dataset.__truediv__.return_value.read_text.return_value = "image.jpg\n"
        metrics = SimpleNamespace(box=SimpleNamespace(mp=0.5, mr=0.5, map50=0.5, map=0.3,
                                                      ap_class_index=[], class_result=lambda index: []))
        for checkpoint in ("yolov8n.pt", None):
            commands = []
            process = SimpleNamespace(stdout=[], wait=lambda: 0)
            def launch(command, **kwargs):
                commands.append(command)
                return process
            namespace = dict(sys=sys, Path=Path, csv=csv, gc=gc, hashlib=hashlib,
                             json=json, statistics=statistics, traceback=traceback,
                             torch=SimpleNamespace(__version__="test", cuda=SimpleNamespace(
                                 is_available=lambda: True, get_device_name=lambda index: "mock GPU",
                                 empty_cache=lambda: None)), ultralytics=SimpleNamespace(__version__="test"),
                             prepare_kaggle_r2000=lambda raw, out: dataset,
                             create_5fold_splits=lambda data, seed: [f"fold{i}.yaml" for i in range(5)],
                             YOLO=lambda path: SimpleNamespace(val=lambda **kwargs: metrics),
                             R2000_CLASSES=[], subprocess=SimpleNamespace(Popen=launch, PIPE=-1, STDOUT=-2),
                             check_training_history=lambda path: {"all_values_finite": True})
            exec(compile(ast.Module(body=functions, type_ignores=[]), "fold_runner", "exec"), namespace)
            with tempfile.TemporaryDirectory(prefix="paddy_folds_test_") as directory:
                namespace["run"]("raw", directory, "baseline", checkpoint)
            self.assertEqual(len(commands), 5)
            for command in commands:
                if checkpoint:
                    self.assertEqual(command[command.index("--weights") + 1], checkpoint)
                else:
                    self.assertNotIn("--weights", command)

    def test_checkpoint_reaches_each_fold_without_changing_default(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="paddy_kernel_test_") as directory:
            workspace = Path(directory)
            files = list(root.glob("*.py"))
            files += list((root / "paddy_yolo").rglob("*.py"))
            files += list((root / "configs").glob("*.yaml"))
            files += [root / "docs/replication-audit.md"]
            for source in files:
                target = workspace / source.relative_to(root)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            for checkpoint in ("yolov8n.pt", None):
                command = [sys.executable, "build_baseline_crossval_kernel.py",
                           "--owner", "test-owner", "--slug", "pretrained-test",
                           "--experiment", "rgg-dimiou"]
                if checkpoint:
                    command += ["--weights", checkpoint]
                result = subprocess.run(command, cwd=workspace, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                notebook = json.loads((workspace / "kaggle_kernel/baseline_fivefold.ipynb").read_text())
                source = "".join(notebook["cells"][1]["source"])
                tree = ast.parse(source)
                call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                            and isinstance(node.func, ast.Attribute) and node.func.attr == "run"
                            and any(isinstance(item, ast.Constant) and item.value == "run_baseline_crossval.py"
                                    for item in ast.walk(node)))
                constants = [node.value for node in ast.walk(call.args[0]) if isinstance(node, ast.Constant)]
                self.assertIn("rgg-dimiou", constants)
                if checkpoint:
                    self.assertIn("--weights", constants)
                    self.assertIn(checkpoint, constants)
                    self.assertNotIn("random initialization", "".join(notebook["cells"][0]["source"]))
                else:
                    self.assertNotIn("--weights", constants)
                archive_text = next(ast.literal_eval(node.value) for node in ast.walk(tree)
                                    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name)
                                    and t.id == "B64_ARCHIVE" for t in node.targets))
                with zipfile.ZipFile(io.BytesIO(base64.b64decode(archive_text))) as archive:
                    runner = ast.parse(archive.read("run_baseline_crossval.py"))
                # Execute the actual command builder embedded in the uploaded code.
                function = next(node for node in runner.body if isinstance(node, ast.FunctionDef)
                                and node.name == "training_command")
                namespace = {"sys": sys, "Path": Path}
                exec(compile(ast.Module(body=[function], type_ignores=[]), "submitted_runner", "exec"), namespace)
                for fold in range(5):
                    argv = namespace["training_command"]("fold.yaml", "results", "rgg_dimiou", fold,
                                                          "rgg-dimiou", checkpoint)
                    self.assertEqual(argv[argv.index("--model") + 1], "configs/paddy-yolo.yaml")
                    if checkpoint:
                        self.assertEqual(argv[argv.index("--weights") + 1], checkpoint)
                    else:
                        self.assertNotIn("--weights", argv)


if __name__ == "__main__":
    unittest.main()

"""Persist subprocess output so Kaggle's console truncation cannot hide errors."""
from pathlib import Path
import subprocess


def configure_ultralytics():
    """Use local artifacts for this standalone run rather than external trackers."""
    from ultralytics.utils import SETTINGS

    SETTINGS.update({key: False for key in
                     ("raytune", "wandb", "comet", "clearml", "mlflow", "neptune")})


def run_logged(command, log_path: Path, cwd: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True,
                                   encoding="utf-8", errors="replace", bufsize=1)
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)

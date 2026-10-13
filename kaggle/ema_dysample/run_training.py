"""Train one supplied R2000 split with EMA + DySample and YOLOv8-N transfer."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    sys.path.insert(0, str(ROOT / "kaggle/baseline"))
    from run_baseline import main as run

    defaults = ["--model", str(ROOT / "configs/paddy-yolo-ema-dysample.yaml"),
                "--hyp", str(ROOT / "configs/hyp_ema_dysample.yaml"),
                "--epochs", "300", "--batch", "32", "--imgsz", "640", "--device", "0,1",
                "--patience", "300", "--save-period", "25",
                "--weights", "yolov8n.pt", "--name", "ema_dysample_rgg_fold0",
                "--output-dir", "/kaggle/working/ema_dysample_output"]
    run(defaults + (sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    main()

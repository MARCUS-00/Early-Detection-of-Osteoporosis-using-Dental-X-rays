"""
CLI wrapper for YOLOv8 cortex-roi training.
All business logic lives in src/osteo/roi/yolo_train.py.
"""
import argparse
import yaml
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 cortex-roi detector")
    parser.add_argument("--config", default="configs/yolo.yaml", help="Path to yolo.yaml")
    parser.add_argument("--data-yaml", help="Override data.yaml path")
    parser.add_argument("--epochs", type=int, help="Override epochs")
    parser.add_argument("--imgsz", type=int, help="Override image size")
    parser.add_argument("--batch", type=int, help="Override batch size")
    parser.add_argument("--output-dir", help="Override output directory")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_yaml = args.data_yaml or cfg["data_yaml"]
    if not data_yaml:
        parser.error("data_yaml must be set in config or via --data-yaml")

    from osteo.roi.yolo_train import train_yolo

    best = train_yolo(
        data_yaml=data_yaml,
        model_weights=cfg.get("base_weights", "yolov8n.pt"),
        epochs=args.epochs or cfg["epochs"],
        imgsz=args.imgsz or cfg["imgsz"],
        batch=args.batch or cfg["batch"],
        output_dir=args.output_dir or cfg.get("output_dir", "runs/detect"),
    )
    print(f"Best weights: {best}")


if __name__ == "__main__":
    main()

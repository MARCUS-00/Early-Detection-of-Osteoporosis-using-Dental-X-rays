"""
CLI wrapper for YOLO-based ROI extraction on a single image.
All business logic lives in src/osteo/roi/yolo_extract.py.
"""
import argparse
import yaml


def main():
    parser = argparse.ArgumentParser(description="Extract left/right cortex ROIs using YOLO")
    parser.add_argument("--config", default="configs/yolo.yaml", help="Path to yolo.yaml")
    parser.add_argument("img_path", help="Input dental X-ray image path")
    parser.add_argument("--model-path", help="Override YOLO model path")
    parser.add_argument("--conf", type=float, help="Override confidence threshold")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_path = args.model_path or cfg.get("model_path")
    if not model_path:
        parser.error("model_path must be set in config or via --model-path")

    from osteo.roi.yolo_extract import roi_extraction

    left, right = roi_extraction(
        img_path=args.img_path,
        model_path=model_path,
        conf=args.conf or cfg.get("conf_threshold", 0.1),
    )
    print(f"Left ROI:  {left}")
    print(f"Right ROI: {right}")


if __name__ == "__main__":
    main()

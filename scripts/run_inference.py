"""
CLI wrapper for patch-based majority-vote inference.
All business logic lives in src/osteo/inference/predict_patches.py.

NOTE: This uses the ACTUAL verified inference path (patch majority vote,
no YOLO, no U-Net). The end-to-end pipeline is NOT implemented.
"""
import argparse
import yaml


def main():
    parser = argparse.ArgumentParser(
        description="Predict osteoporosis class from a dental X-ray (patch majority vote)"
    )
    parser.add_argument("--config", default="configs/classifier.yaml",
                        help="Path to classifier.yaml")
    parser.add_argument("img_path", help="Input dental X-ray image path")
    parser.add_argument("--model-path", help="Override model .h5 path")
    parser.add_argument("--class-names", nargs="+",
                        default=["Normal", "Osteopenia", "Osteoporosis"],
                        help="Class names in training order")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_path = args.model_path or cfg.get("model_filename", "osteoporosis_mobilenetv2.h5")

    from osteo.inference.predict_patches import predict_image

    prediction = predict_image(
        img_path=args.img_path,
        model_path=model_path,
        class_names=args.class_names,
    )
    print(f"Prediction: {prediction}")


if __name__ == "__main__":
    main()

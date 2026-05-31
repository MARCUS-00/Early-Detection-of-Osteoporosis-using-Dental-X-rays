"""
CLI wrapper for MobileNetV2 classifier evaluation.
All business logic lives in src/osteo/classification/evaluate.py.
"""
import argparse
import yaml


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MobileNetV2 classifier (per-patch accuracy)"
    )
    parser.add_argument("--config", default="configs/classifier.yaml",
                        help="Path to classifier.yaml")
    parser.add_argument("--test-dir", help="Override test data directory")
    parser.add_argument("--model-path", help="Override model .h5 path")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    test_dir = args.test_dir or cfg.get("test_dir")
    if not test_dir:
        parser.error("test_dir must be set in config or via --test-dir")

    from osteo.classification.evaluate import evaluate

    results = evaluate(
        test_dir=test_dir,
        model_path=args.model_path or cfg["model_filename"],
        batch_size=cfg["eval_batch_size"],
        img_size=tuple(cfg["img_size"]),
    )
    print(f"\nPer-patch accuracy: {results['accuracy']:.4f}")


if __name__ == "__main__":
    main()

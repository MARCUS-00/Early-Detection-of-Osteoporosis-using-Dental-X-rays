"""
CLI wrapper for MobileNetV2 classifier training.
All business logic lives in src/osteo/classification/train.py.
"""
import argparse
import yaml


def main():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 osteoporosis classifier")
    parser.add_argument("--config", default="configs/classifier.yaml",
                        help="Path to classifier.yaml")
    parser.add_argument("--train-dir", help="Override training data directory")
    parser.add_argument("--val-dir", help="Override validation data directory")
    parser.add_argument("--output-path", help="Override output .h5 path")
    parser.add_argument("--epochs", type=int, help="Override epochs")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_dir = args.train_dir or cfg.get("train_dir")
    val_dir = args.val_dir or cfg.get("val_dir")
    if not train_dir or not val_dir:
        parser.error("train_dir and val_dir must be set in config or via CLI")

    from osteo.classification.train import train

    saved = train(
        train_dir=train_dir,
        val_dir=val_dir,
        output_path=args.output_path or cfg["model_filename"],
        epochs=args.epochs or cfg["epochs"],
        batch_size=cfg["train_batch_size"],
        lr=cfg["learning_rate"],
        img_size=tuple(cfg["img_size"]),
    )
    print(f"Saved: {saved}")


if __name__ == "__main__":
    main()

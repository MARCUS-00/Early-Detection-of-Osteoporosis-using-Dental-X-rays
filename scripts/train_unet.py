"""
CLI wrapper for U-Net training.
All business logic lives in src/osteo/segmentation/train_unet.py.
"""
import argparse
import yaml


def main():
    parser = argparse.ArgumentParser(description="Train U-Net segmentation model")
    parser.add_argument("--config", default="configs/unet.yaml", help="Path to unet.yaml")
    parser.add_argument("--images-dir", help="Override images directory")
    parser.add_argument("--masks-dir", help="Override masks directory")
    parser.add_argument("--side", default="general",
                        choices=["left", "right", "general"],
                        help="Which side model to train")
    parser.add_argument("--output-path", help="Override output .pth path")
    parser.add_argument("--epochs", type=int, help="Override epochs")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    images_dir = args.images_dir or cfg.get("images_dir")
    masks_dir = args.masks_dir or cfg.get("masks_dir")
    if not images_dir or not masks_dir:
        parser.error("images_dir and masks_dir must be set in config or via CLI")

    from osteo.segmentation.train_unet import train_unet

    output_path = args.output_path or cfg["output_filenames"][args.side]
    saved = train_unet(
        images_dir=images_dir,
        masks_dir=masks_dir,
        side=args.side,
        output_path=output_path,
        epochs=args.epochs or cfg["epochs"],
        lr=cfg["learning_rate"],
        val_split=cfg["val_split"],
    )
    print(f"Saved: {saved}")


if __name__ == "__main__":
    main()

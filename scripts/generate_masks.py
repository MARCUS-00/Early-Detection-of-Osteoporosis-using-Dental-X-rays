"""
CLI wrapper for classical-CV pseudo-mask generation.
All business logic lives in src/osteo/segmentation/masks_classical.py.
"""
import argparse
from pathlib import Path
import cv2


def main():
    parser = argparse.ArgumentParser(
        description="Generate pseudo-ground-truth masks for a directory of ROI images"
    )
    parser.add_argument("images_dir", help="Directory of input images")
    parser.add_argument("output_dir", help="Directory to write mask images")
    args = parser.parse_args()

    from osteo.segmentation.masks_classical import create_final_mask
    from osteo.utils.io import ensure_dir

    out = ensure_dir(args.output_dir)
    images_dir = Path(args.images_dir)
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    paths = [p for p in images_dir.iterdir() if p.suffix.lower() in exts]
    if not paths:
        print(f"No images found in {images_dir}")
        return

    for img_path in sorted(paths):
        mask = create_final_mask(str(img_path))
        save_path = out / img_path.name
        cv2.imwrite(str(save_path), mask)
        print(f"Wrote mask: {save_path}")


if __name__ == "__main__":
    main()

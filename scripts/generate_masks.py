'''CLI wrapper for classical-CV pseudo-mask generation.
All business logic lives in src/osteo/segmentation/masks_classical.py.'''
import argparse, cv2, sys, os
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))


def main():
    parser = argparse.ArgumentParser(
        description='Generate pseudo-ground-truth masks for a directory of ROI images')
    parser.add_argument('images_dir'); parser.add_argument('output_dir')
    args = parser.parse_args()
    from osteo.segmentation.masks_classical import create_final_mask
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    exts = {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}
    paths = [p for p in Path(args.images_dir).iterdir() if p.suffix.lower() in exts]
    if not paths:
        print(f'No images found in {args.images_dir}')
        return
    for img_path in sorted(paths):
        mask = create_final_mask(str(img_path))
        save_path = out / img_path.name
        cv2.imwrite(str(save_path), mask)
        print(f'Wrote mask: {save_path}')

if __name__ == '__main__':
    main()

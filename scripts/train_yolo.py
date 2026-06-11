'''CLI wrapper for YOLOv8 cortex-roi training.
All business logic lives in src/osteo/roi/yolo_train.py.'''
import argparse, yaml, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 cortex-roi detector')
    parser.add_argument('--config', default='configs/yolo.yaml')
    parser.add_argument('--data-yaml'); parser.add_argument('--epochs', type=int)
    parser.add_argument('--imgsz',  type=int); parser.add_argument('--batch', type=int)
    parser.add_argument('--output-dir')
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    data_yaml = args.data_yaml or cfg['data_yaml']
    if not data_yaml:
        parser.error('data_yaml must be set in config or via --data-yaml')
    from osteo.roi.yolo_train import train_yolo
    best = train_yolo(
        data_yaml=data_yaml,
        model_weights=cfg.get('base_weights', 'yolov8n.pt'),
        epochs=args.epochs or cfg['epochs'],
        imgsz=args.imgsz or cfg['imgsz'],
        batch=args.batch or cfg['batch'],
        output_dir=args.output_dir or cfg.get('output_dir', 'runs/detect'),
    )
    print(f'Best weights: {best}')

if __name__ == '__main__':
    main()

'''
YOLOv8 training for cortex-roi detection — ported from ROI_Extraction.ipynb.

Two configurations are present in the source notebook:
  - Config A (default): epochs=50, imgsz=1024, batch=2
  - Config B:           epochs=30, imgsz=640,  batch=8

Both are exposed here; callers select via the config argument.
Dataset: Roboflow YOLOv8 export; 1 class: cortex_roi.
'''
from pathlib import Path


def train_yolo(
    data_yaml: str,
    model_weights: str = 'yolov8n.pt',
    epochs: int = 50,
    imgsz: int = 1024,
    batch: int = 2,
    output_dir: str = 'runs/detect',
) -> str:
    '''
    Train YOLOv8 on the cortex-roi dataset.
    Default hyperparameters match Config A from ROI_Extraction.ipynb.
    Config B values: epochs=30, imgsz=640, batch=8.
    Returns the path to the best weights file.
    '''
    from ultralytics import YOLO
    model = YOLO(model_weights)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=output_dir,
    )
    best = Path(results.save_dir) / 'weights' / 'best.pt'
    return str(best)

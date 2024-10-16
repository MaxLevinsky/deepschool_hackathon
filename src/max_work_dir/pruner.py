import os
from magnitude_pruner import prune
from ultralytics import YOLO
import torch


def run_pruner(dataset_yaml_path: str, model_path: str, output_path: str):
    cfg = {'data': os.path.join(dataset_yaml_path, 'dataset.yaml'), 
           'cfg': os.path.join(os.path.dirname(__file__), 'config/prune.yaml')}
    prune(
        model=YOLO(os.path.join(model_path, 'best.pt')),
        # dataset_dir='/home/nikitamarkov/deepschool/hackathon/data',
        output_dir=output_path,
        input_example = torch.rand(1,3,640,640),
        test_dataset='test',
        pruning_ratio=0.3,
        device='cuda',
        save_onnx=True,
        **cfg
    )


if __name__ == '__main__':
    run_pruner(dataset_yaml_path='/home/nikitamarkov/deepschool/hackathon/data',
          model_path='/home/nikitamarkov/deepschool/hackathon/',
          output_path='/home/nikitamarkov/deepschool_hackathon/models')
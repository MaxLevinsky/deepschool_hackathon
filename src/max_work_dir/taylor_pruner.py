import os
from copy import deepcopy
import warnings

import torch
import torch_pruning as tp
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.modules import Classify, Detect
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.torch_utils import initialize_weights

from utils import (replace_c2f_with_c2f_v2, 
                   train_v2, get_dataloader, 
                   get_classification_criterion, 
                   get_detection_criterion,
                   preprocess_batch,
                   get_test_performane_metric
                   )


def prune(
        model: YOLO, 
        dataset_dir: str, 
        output_dir: str,
        input_example: torch.Tensor, 
        task: str = 'detect', 
        test_dataset: str = 'val',
        pruning_ratio: float = 0.5, 
        batch_size: int = 16,
        save_onnx: bool = False,
        device: str = 'cuda',
        **yolo_cgf
        ) -> None:
    """
    The script for Taylor prunig of YOLOv8 detection and classification models.

    Parameters
    ----------
    model : YOLO
        A YOLO model.
    dataset_dir : str
        A path to a directory with train, val, test subsets.
    output_dir : str
        A directory for artifacts.
    input_example : torch.Tensor
        Example: input_example = torch.rand(1,3,224,224).
    task : str, optional
        One of [detect, classify], by default 'detect'.
    test_dataset: str,
        Test dataset split, one of ['test', 'val'], by default: 'val'.
    pruning_ratio : float, optional
        Global channel sparisty. Also known as pruning ratio, by default 0.5.
    batch_size : int, optional
        Batch size, by default 16.
    save_onnx: bool, optional
        Save a model to *.onnx, by default False.
    yolo_cfg: dict
        YOLO trainingg config, example: cfg = {'epochs': 30, 'warmup_epochs': 5, 'imgsz': 224, 'batch': 128}.

    Example:

        from captcha_engine.tools.optimization.yolo_pruning.taylor_pruner import prune
        from ultralytics import YOLO
        import torch

        yolo_cgf= {'epochs': 30, 'warmup_epochs': 5, 'imgsz': 224, 'batch': 128}

        prune(
            model=YOLO(<model_path>),
            dataset_dir=<dataset_dir>/train,
            output_dir=<output_dir>,
            input_example = torch.rand(1,3,224,224),
            pruning_ratio=0.75,
            task='classify',
            test_dataset='val',
            save_onnx=True,
            **yolo_cgf
        )
    """
    # Verify that a path to the dataset.yaml exists
    if not os.path.exists(model.model.args['data']):
        warnings.warn('A path to the dataset.yaml does not exist')
        if 'data' not in yolo_cgf:
            raise Exception(f'Pass the path to the yolo_cfg')
        else:
            model.model.args['data'] = yolo_cgf['data']

    model.model.to(device)
    input_example = input_example.to(device)
    
    # Evaluate the pruned model
    eval_model = deepcopy(model)
    baseline_test_metric = eval_model.val(split='test') if test_dataset == 'test' else eval_model.val()
    baseline_metric = get_test_performane_metric(baseline_test_metric, task)
    baseline_macs, baseline_nparams = tp.utils.count_ops_and_params(model.model, input_example.to(model.device))

    print('========================================================================')
    print(f'Baseline model metric - {baseline_metric}, MACs - {round(baseline_macs / 1e9, 3)} G,'
          f'#Params - {round(baseline_nparams / 1e6, 3)} M')
    print('========================================================================')

    cfg = get_cfg(DEFAULT_CFG)
    model.__setattr__("train_v2", train_v2.__get__(model))

    model.model.train()

    replace_c2f_with_c2f_v2(model.model)
    initialize_weights(model.model)

    head = Classify if task == 'classify' else Detect

    for _, param in model.model.named_parameters():
        param.requires_grad = True

    # Ignore the last layer
    ignored_layers = []
    for m in model.model.modules():
        if isinstance(m, (head,)):
            ignored_layers.append(m)

    pruner = tp.pruner.MetaPruner(
        model=model.model.to(device),
        example_inputs=input_example,
        importance=tp.importance.GroupTaylorImportance(),
        pruning_ratio=pruning_ratio,
        ignored_layers=ignored_layers,
    )

    model.model.train()

    for _, param in model.model.named_parameters():
        param.requires_grad = True

    dataloader = get_dataloader(model=model, dataset_path=os.path.join(dataset_dir, 'train'), cfg=cfg, batch_size=batch_size, rank=-1, task=task)
    model.model.args = cfg
    for batch in tqdm(dataloader):
        batch = preprocess_batch(batch, device)
        label = batch['cls'].to(device)
        out = model.model(batch['img'])
        if task == 'classify':
            loss = get_classification_criterion(out, label)
        else:
            loss = get_detection_criterion(model.model, out, batch)
        loss.backward()

    pruner.step()

    # Fine tuning
    model.train_v2(pruning=True, **yolo_cgf)

    # Evaluate the pruned model
    eval_model.model = deepcopy(model.model)
    pruned_test_metric = eval_model.val(split='test') if test_dataset == 'test' else eval_model.val()
    pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(eval_model.model, input_example.to(eval_model.device))

    pruned_metric = pruned_test_metric.top1 if task == 'classify' else pruned_test_metric.box.map50

    print('========================================================================')
    print(f'Pruned model metric - {round(pruned_metric, 3)}, MACs - {round(pruned_macs / 1e9, 3)} G,' 
          f'#Params - {round(pruned_nparams / 1e6, 3)} M')
    print('========================================================================')

    # Save pt 
    model.save(os.path.join(output_dir, 'taylor_pruned_model.pt'))

    # Save onnx
    if save_onnx:
        model.export(format="onnx", dynamic=True, simplify=False)

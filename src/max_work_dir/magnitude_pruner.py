import os
from copy import deepcopy
import warnings

import torch
import torch_pruning as tp
from ultralytics import YOLO
from ultralytics.utils.torch_utils import initialize_weights
from ultralytics.nn.modules import Classify, Detect

from utils import replace_c2f_with_c2f_v2, train_v2, get_test_performane_metric


def prune(
    model: YOLO,
    input_example: torch.Tensor,
    output_dir: str,
    device: str,
    task: str = 'detect',
    test_dataset: str = 'test',
    pruning_ratio: float = 0.5,
    global_pruning: bool = False,
    save_onnx: bool = False,
    **yolo_cgf,
    ) -> None:
    """
    The script for magnitude prunig of YOLOv8 detection and classification models.

    Parameters
    ----------
    model : YOLO
        A YOLO model.
    input_example : torch.Tensor.
        Example: input_example = torch.rand(1,3,224,224).
    output_dir : str
        A directory for artifacts.
    device: str
        device.
    task: str
        One of [detect, classify], by default 'detect'.
    test_dataset: str,
        Test dataset split, one of ['test', 'val'], by default: 'test'   
    pruning_ratio : float, optional
        Global channel sparisty. Also known as pruning ratio, by default 0.5.
    global_pruning: bool, optional
        Enable global pruning, by default: False.
    save_onnx: bool, optional
        Save a model to *.onnx, by default False.
    yolo_cfg: dict
        YOLO trainingg config, example: cfg = {'epochs': 30, 'warmup_epochs': 5, 'imgsz': 224, 'batch': 128}.

    Example:

        from captcha_engine.tools.optimization.yolo_pruning.magnitude_pruner import prune
        from ultralytics import YOLO
        import torch

        model = YOLO(<model_path>)
        input_example = torch.rand(1,3,224,224)
        task = 'classify'
        cfg = {'epochs': 2, 'warmup_epochs': 5, 'imgsz': 224, 'batch': 128}
        prune(model=model, input_example=input_example, task=task, test_dataset='val', output_dir=<output_dir>, save_onnx=True, **cfg)
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

    baseline_macs, baseline_nparams = tp.utils.count_ops_and_params(model.model, input_example)

    print('========================================================================')
    print(f'Baseline model metric - {baseline_metric}, MACs - {round(baseline_macs / 1e9, 3)} G,'
          f'#Params - {round(baseline_nparams / 1e6, 3)} M')
    print('========================================================================')

    model.__setattr__("train_v2", train_v2.__get__(model))
    model.model.train()
    replace_c2f_with_c2f_v2(model.model)
    initialize_weights(model.model)

    head = Classify if task == 'classify' else Detect

    for _, param in model.model.named_parameters():
        param.requires_grad = True

    # ignore the last layer
    ignored_layers = []
    for m in model.model.modules():
        if isinstance(m, (head,)):
            ignored_layers.append(m)

    pruner = tp.pruner.MetaPruner(
        model=model.model.to(device),
        example_inputs=input_example,
        global_pruning=global_pruning,
        importance=tp.importance.MagnitudeImportance(),
        pruning_ratio=pruning_ratio,
        ignored_layers=ignored_layers,
    )

    pruner.step()

    # Fine tuning
    model.train_v2(pruning=True, **yolo_cgf)

    # Evaluate the pruned model
    eval_model.model = deepcopy(model.model)
    pruned_test_metric = eval_model.val(split='test') if test_dataset == 'test' else eval_model.val()
    pruned_metric = get_test_performane_metric(pruned_test_metric, task)
    pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(eval_model.model, input_example.to(eval_model.device))

    print('========================================================================')
    print(f'Pruned model metric - {pruned_metric}, MACs - {round(pruned_macs / 1e9, 3)} G, ' 
          f'#Params - {round(pruned_nparams / 1e6, 3)} M')
    print('========================================================================')

    # Save
    model.save(os.path.join(output_dir, 'magnitude_spruned_model.pt'))

    # Save onnx
    if save_onnx:
        saved_model = YOLO(os.path.join(output_dir, 'magnitude_spruned_model.pt'))
        saved_model.export(format="onnx", dynamic=True, simplify=False)

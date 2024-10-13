import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Union

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO, __version__
from ultralytics.data.dataset import ClassificationDataset
from ultralytics.data.build import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.nn.modules import Bottleneck, C2f, Conv
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.utils import (DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER,
                               RANK, yaml_load)
from ultralytics.utils.checks import check_yaml
from ultralytics.utils.metrics import DetMetrics, ClassifyMetrics
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.torch_utils import is_parallel, torch_distributed_zero_first, de_parallel


class C2f_v2(nn.Module):
    # CSP Bottleneck with 2 convolutions
    # ch_in, ch_out, number, shortcut, groups, expansion
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=(
            (3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        # y = list(self.cv1(x).chunk(2, 1))
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


def infer_shortcut(bottleneck):
    c1 = bottleneck.cv1.conv.in_channels
    c2 = bottleneck.cv2.conv.out_channels
    return c1 == c2 and hasattr(bottleneck, 'add') and bottleneck.add


def transfer_weights(c2f, c2f_v2):
    c2f_v2.cv2 = c2f.cv2
    c2f_v2.m = c2f.m

    state_dict = c2f.state_dict()
    state_dict_v2 = c2f_v2.state_dict()

    # Transfer cv1 weights from C2f to cv0 and cv1 in C2f_v2
    old_weight = state_dict['cv1.conv.weight']
    half_channels = old_weight.shape[0] // 2
    state_dict_v2['cv0.conv.weight'] = old_weight[:half_channels]
    state_dict_v2['cv1.conv.weight'] = old_weight[half_channels:]

    # Transfer cv1 batchnorm weights and buffers from C2f to cv0 and cv1 in C2f_v2
    for bn_key in ['weight', 'bias', 'running_mean', 'running_var']:
        old_bn = state_dict[f'cv1.bn.{bn_key}']
        state_dict_v2[f'cv0.bn.{bn_key}'] = old_bn[:half_channels]
        state_dict_v2[f'cv1.bn.{bn_key}'] = old_bn[half_channels:]

    # Transfer remaining weights and buffers
    for key in state_dict:
        if not key.startswith('cv1.'):
            state_dict_v2[key] = state_dict[key]

    # Transfer all non-method attributes
    for attr_name in dir(c2f):
        attr_value = getattr(c2f, attr_name)
        if not callable(attr_value) and '_' not in attr_name:
            setattr(c2f_v2, attr_name, attr_value)

    c2f_v2.load_state_dict(state_dict_v2)


def replace_c2f_with_c2f_v2(module):
    for name, child_module in module.named_children():
        if isinstance(child_module, C2f):
            # Replace C2f with C2f_v2 w hile preserving its parameters
            shortcut = infer_shortcut(child_module.m[0])
            c2f_v2 = C2f_v2(child_module.cv1.conv.in_channels, child_module.cv2.conv.out_channels,
                            n=len(child_module.m), shortcut=shortcut,
                            g=child_module.m[0].cv2.conv.groups,
                            e=child_module.c / child_module.cv2.conv.out_channels)
            transfer_weights(child_module, c2f_v2)
            setattr(module, name, c2f_v2)
        else:
            replace_c2f_with_c2f_v2(child_module)


def save_model_v2(self: BaseTrainer):
    """
    Disabled half precision saving. originated from ultralytics/yolo/engine/trainer.py
    """
    ckpt = {
        'epoch': self.epoch,
        'best_fitness': self.best_fitness,
        'model': deepcopy(de_parallel(self.model)),
        'ema': deepcopy(self.ema.ema),
        'updates': self.ema.updates,
        'optimizer': self.optimizer.state_dict(),
        'train_args': vars(self.args),  # save as dict
        'date': datetime.now().isoformat(),
        'version': __version__}

    # Save last, best and delete
    torch.save(ckpt, self.last)
    if self.best_fitness == self.fitness:
        torch.save(ckpt, self.best)
    if (self.epoch > 0) and (self.save_period > 0) and (self.epoch % self.save_period == 0):
        torch.save(ckpt, self.wdir / f'epoch{self.epoch}.pt')
    del ckpt


def final_eval_v2(self: BaseTrainer):
    """
    originated from ultralytics/yolo/engine/trainer.py
    """
    for f in self.last, self.best:
        if f.exists():
            strip_optimizer_v2(f)  # strip optimizers
            if f is self.best:
                LOGGER.info(f'\nValidating {f}...')
                self.metrics = self.validator(model=f)
                self.metrics.pop('fitness', None)
                self.run_callbacks('on_fit_epoch_end')


def strip_optimizer_v2(f: Union[str, Path] = 'best.pt', s: str = '') -> None:
    """
    Disabled half precision saving. originated from ultralytics/yolo/utils/torch_utils.py
    """
    x = torch.load(f, map_location=torch.device('cpu'))
    # combine model args with default args, preferring model args
    args = {**DEFAULT_CFG_DICT, **x['train_args']}
    if x.get('ema'):
        x['model'] = x['ema']  # replace model with ema
    for k in 'optimizer', 'ema', 'updates':  # keys
        x[k] = None
    for p in x['model'].parameters():
        p.requires_grad = False
    x['train_args'] = {k: v for k, v in args.items(
    ) if k in DEFAULT_CFG_KEYS}  # strip non-default keys
    # x['model'].args = x['train_args']
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1E6  # filesize
    LOGGER.info(
        f"Optimizer stripped from {f},{f' saved as {s},' if s else ''} {mb:.1f}MB")


def train_v2(self: YOLO, pruning=False, **kwargs):
    """
    Disabled loading new model when pruning flag is set. originated from ultralytics/yolo/engine/model.py
    """

    self._check_is_pytorch_model()
    if self.session:  # Ultralytics HUB session
        if any(kwargs):
            LOGGER.warning(
                'WARNING ⚠️ using HUB training arguments, ignoring local training arguments.')
        kwargs = self.session.train_args
    overrides = self.overrides.copy()
    overrides.update(kwargs)
    if kwargs.get('cfg'):
        LOGGER.info(
            f"cfg file passed. Overriding default params with {kwargs['cfg']}.")
        overrides = yaml_load(check_yaml(kwargs['cfg']))
    overrides['mode'] = 'train'
    if not overrides.get('data'):
        raise AttributeError(
            "Dataset required but missing, i.e. pass 'data=coco128.yaml'")
    if overrides.get('resume'):
        overrides['resume'] = self.ckpt_path

    self.task = overrides.get('task') or self.task
    # TASK_MAP[self.task][1](overrides=overrides, _callbacks=self.callbacks)
    self.trainer = self.task_map[self.task]['trainer'](
        overrides=overrides, _callbacks=self.callbacks)

    if not pruning:
        if not overrides.get('resume'):  # manually set model only if not resuming
            self.trainer.model = self.trainer.get_model(
                weights=self.model if self.ckpt else None, cfg=self.model.yaml)
            self.model = self.trainer.model

    else:
        # pruning mode
        self.trainer.pruning = True
        self.trainer.model = self.model

        # replace some functions to disable half precision saving
        self.trainer.save_model = save_model_v2.__get__(self.trainer)
        self.trainer.final_eval = final_eval_v2.__get__(self.trainer)

    self.trainer.hub_session = self.session  # attach optional HUB session
    self.trainer.train()
    # Update model and cfg after training
    if RANK in (-1, 0):
        self.model, _ = attempt_load_one_weight(str(self.trainer.best))
        self.overrides = self.model.args
        self.metrics = getattr(self.trainer.validator, 'metrics', None)


def read_yaml(yaml_path: str):
    with open(yaml_path) as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(f'Can not open yaml file {yaml_path}, error {exc}')
            raise

    return data


def build_dataset(img_path, cfg, mode="train"):
        """Creates a ClassificationDataset instance given an image path, and mode (train/test etc.)."""
        return ClassificationDataset(root=img_path, args=cfg, augment=mode == "train", prefix=mode)


def get_dataloader(model, dataset_path, cfg, batch_size=16, rank=0, mode="train", task='classify'):
    """Returns PyTorch DataLoader with transforms to preprocess images for inference."""
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        if task == 'classify':
            dataset = build_dataset(dataset_path, cfg, mode)
        elif task == 'detect':
            model_args = read_yaml(model.model.args['data'])
            dataset = build_yolo_dataset(cfg, dataset_path, 32, model_args)

    loader = build_dataloader(dataset, batch_size, cfg.workers, rank=rank)
    # Attach inference transforms
    if mode != "train":
        if is_parallel(model):
            model.module.transforms = loader.dataset.torch_transforms
        else:
            model.transforms = loader.dataset.torch_transforms
    return loader


def preprocess_batch(batch, device):
    """Preprocesses a batch of images by scaling and converting to float."""
    batch['img'] = batch['img'].to(device, non_blocking=True).float() / 255
    return batch


def get_classification_criterion(preds: torch.Tensor, gt: torch.Tensor):
    loss = F.cross_entropy(preds, gt, reduction="mean")
    return loss


def get_detection_criterion(model: YOLO, preds: torch.Tensor, gt: torch.Tensor):
    loss_fn = v8DetectionLoss(de_parallel(model))
    loss, _ = loss_fn(preds, gt)
    return loss


def get_test_performane_metric(test_result: Union[DetMetrics, ClassifyMetrics], 
                               task: str = 'classify') -> Union[float, str]:
    """The script returns classification or detection metrics."""
    if task == 'classify':
        metric = round(test_result.top1, 3)
    elif task == 'detect':
        metric = f'MAP50 - {round(test_result.box.map50, 3)}, MAP50-95 - {round(test_result.box.map, 3)}'
    else:
        print(f'{task} task is not supported.')

    return metric

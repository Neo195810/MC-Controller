import cv2
import math
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.utils.impala_lib import misc
from src.utils.impala_lib import torch_util as tu
from src.utils.impala_lib.util import FanInInitReLULayer


def resize_image(img, target_resolution=(128, 128)):
    if type(img) == np.ndarray:
        img = cv2.resize(img, target_resolution, interpolation=cv2.INTER_LINEAR)
    elif type(img) == torch.Tensor:
        img = F.interpolate(img, size=target_resolution, mode='bilinear')
    else:
        raise ValueError
    return img


def resize_segmentation(seg, target_resolution=(128, 128)):
    if type(seg) == np.ndarray:
        seg = cv2.resize(seg, target_resolution, interpolation=cv2.INTER_NEAREST)
    elif type(seg) == torch.Tensor:
        if seg.dim() != 4:
            raise ValueError("segmentation tensor must be BCHW")
        seg = F.interpolate(seg.float(), size=target_resolution, mode='nearest')
    else:
        raise ValueError
    return seg


class SegCnnBasicBlock(nn.Module):
    """
    Residual block with goal gate and segmentation-driven spatial attention.
    """

    def __init__(
        self,
        inchan: int,
        goal_dim: int,
        seg_channels: int,
        seg_attn_channels: int = 16,
        init_scale: float = 1,
        log_scope: str = "",
        init_norm_kwargs: Dict = {},
        **kwargs,
    ):
        super().__init__()
        s = math.sqrt(init_scale)
        self.inchan = inchan
        self.goal_dim = goal_dim
        self.seg_channels = seg_channels
        attn_channels = max(1, min(seg_attn_channels, seg_channels))

        self.conv0 = FanInInitReLULayer(
            self.inchan,
            self.inchan,
            kernel_size=3,
            padding=1,
            init_scale=s,
            log_scope=f"{log_scope}/conv0",
            **init_norm_kwargs,
        )
        self.conv1 = FanInInitReLULayer(
            self.inchan,
            self.inchan,
            kernel_size=3,
            padding=1,
            init_scale=s,
            log_scope=f"{log_scope}/conv1",
            **init_norm_kwargs,
        )
        self.goal_gate = nn.Sequential(
            nn.Linear(self.goal_dim, self.inchan * 2),
            nn.ReLU(),
            nn.Linear(self.inchan * 2, self.inchan),
        )
        self.seg_attn = nn.Sequential(
            nn.Conv2d(self.seg_channels, attn_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(attn_channels, 1, kernel_size=3, padding=1),
        )

    def forward(self, x, goal_embeddings, seg=None):
        px = self.conv1(self.conv0(x))
        goal_gate = self.goal_gate(goal_embeddings).sigmoid().unsqueeze(2).unsqueeze(3)

        if seg is None:
            seg_gate = 1.0
        else:
            seg_gate = self.seg_attn(seg).sigmoid()

        return x + px * goal_gate * seg_gate


class SegCnnDownStack(nn.Module):
    """
    Downsampling stack with segmentation-aware residual blocks.
    """

    name = "Seg_Impala_CnnDownStack"

    def __init__(
        self,
        inchan: int,
        nblock: int,
        outchan: int,
        goal_dim: int,
        seg_channels: int,
        seg_attn_channels: int = 16,
        init_scale: float = 1,
        pool: bool = True,
        post_pool_groups: Optional[int] = None,
        log_scope: str = "",
        init_norm_kwargs: Dict = {},
        first_conv_norm: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.inchan = inchan
        self.outchan = outchan
        self.pool = pool
        first_conv_init_kwargs = deepcopy(init_norm_kwargs)
        if not first_conv_norm:
            first_conv_init_kwargs["group_norm_groups"] = None
            first_conv_init_kwargs["batch_norm"] = False
        self.firstconv = FanInInitReLULayer(
            inchan,
            outchan,
            kernel_size=3,
            padding=1,
            log_scope=f"{log_scope}/firstconv",
            **first_conv_init_kwargs,
        )
        self.post_pool_groups = post_pool_groups
        if post_pool_groups is not None:
            self.n = nn.GroupNorm(post_pool_groups, outchan)
        self.blocks = nn.ModuleList(
            [
                SegCnnBasicBlock(
                    outchan,
                    goal_dim=goal_dim,
                    seg_channels=seg_channels,
                    seg_attn_channels=seg_attn_channels,
                    init_scale=init_scale / math.sqrt(nblock),
                    log_scope=f"{log_scope}/block{i}",
                    init_norm_kwargs=init_norm_kwargs,
                    **kwargs,
                )
                for i in range(nblock)
            ]
        )

    def forward(self, x, goal_embeddings, seg=None):
        x = self.firstconv(x)
        if self.pool:
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
            if self.post_pool_groups is not None:
                x = self.n(x)

        seg_in = None
        if seg is not None:
            seg_in = resize_segmentation(seg, x.shape[-2:])

        x = tu.sequential(self.blocks, x, goal_embeddings, seg_in, diag_name=self.name)
        return x

    def output_shape(self, inshape):
        c, h, w = inshape
        assert c == self.inchan
        if self.pool:
            return (self.outchan, (h + 1) // 2, (w + 1) // 2)
        return (self.outchan, h, w)


class SegGoalImpalaCNN(nn.Module):
    """
    Goal-sensitive Impala CNN with segmentation-driven spatial attention.
    """

    name = "SegGoalImpalaCNN"

    def __init__(
        self,
        inshape: List[int],
        chans: List[int],
        outsize: int,
        nblock: int,
        goal_dim: int,
        seg_channels: int,
        seg_attn_channels: int = 16,
        init_norm_kwargs: Dict = {},
        dense_init_norm_kwargs: Dict = {},
        first_conv_norm: bool = False,
        **kwargs,
    ):
        super().__init__()
        h, w, c = inshape
        curshape = (c, h, w)
        self.stacks = nn.ModuleList()
        for i, outchan in enumerate(chans):
            stack = SegCnnDownStack(
                curshape[0],
                nblock=nblock,
                outchan=outchan,
                goal_dim=goal_dim,
                seg_channels=seg_channels,
                seg_attn_channels=seg_attn_channels,
                init_scale=math.sqrt(len(chans)),
                log_scope=f"downstack{i}",
                init_norm_kwargs=init_norm_kwargs,
                first_conv_norm=first_conv_norm if i == 0 else True,
                **kwargs,
            )
            self.stacks.append(stack)
            curshape = stack.output_shape(curshape)

        self.dense = FanInInitReLULayer(
            misc.intprod(curshape),
            outsize,
            layer_type="linear",
            log_scope="impala_final_dense",
            init_scale=1.4,
            **dense_init_norm_kwargs,
        )
        self.outsize = outsize

    def forward(self, x, goal_embeddings, seg=None):
        x = tu.sequential(self.stacks, x, goal_embeddings, seg, diag_name=self.name)
        x = tu.flatten_image(x)
        x = self.dense(x)
        return x


class SegGoalImpalaCNNWrapper(nn.Module):
    """
    Wrapper that matches the vision backbone interface and accepts segmentation maps.

    img: B x 3 x H x W
    seg: B x C x H x W (one-hot or probabilities); if B x H x W, will be expanded.
    goal_embeddings: B x goal_dim
    """

    def __init__(
        self,
        scale: str = '1x',
        seg_channels: int = 1,
        seg_num_classes: Optional[int] = None,
        seg_attn_channels: int = 16,
        goal_dim: int = 512,
        **kwargs,
    ):
        super().__init__()
        if seg_num_classes is not None:
            seg_channels = seg_num_classes

        if scale == '1x':
            net_config = {
                'hidsize': 1024,
                'img_shape': [128, 128, 3],
                'impala_chans': [16, 32, 32],
                'impala_kwargs': {'post_pool_groups': 1},
                'impala_width': 4,
                'init_norm_kwargs': {'batch_norm': False, 'group_norm_groups': 1},
            }
        elif scale == '3x':
            net_config = {
                'hidsize': 3072,
                'img_shape': [128, 128, 3],
                'impala_chans': [16, 32, 32],
                'impala_kwargs': {'post_pool_groups': 1},
                'impala_width': 12,
                'init_norm_kwargs': {'batch_norm': False, 'group_norm_groups': 1},
            }
        else:
            raise ValueError("scale must be '1x' or '3x'")

        hidsize = net_config['hidsize']
        img_shape = net_config['img_shape']
        impala_width = net_config['impala_width']
        impala_chans = net_config['impala_chans']
        impala_kwargs = net_config['impala_kwargs']
        init_norm_kwargs = net_config['init_norm_kwargs']

        chans = tuple(int(impala_width * c) for c in impala_chans)
        self.dense_init_norm_kwargs = deepcopy(init_norm_kwargs)
        if self.dense_init_norm_kwargs.get("group_norm_groups", None) is not None:
            self.dense_init_norm_kwargs.pop("group_norm_groups", None)
            self.dense_init_norm_kwargs["layer_norm"] = True
        if self.dense_init_norm_kwargs.get("batch_norm", False):
            self.dense_init_norm_kwargs.pop("batch_norm", False)
            self.dense_init_norm_kwargs["layer_norm"] = True

        self.seg_num_classes = seg_num_classes
        self.seg_channels = seg_channels

        self.cnn = SegGoalImpalaCNN(
            outsize=256,
            inshape=img_shape,
            chans=chans,
            nblock=2,
            goal_dim=goal_dim,
            seg_channels=seg_channels,
            seg_attn_channels=seg_attn_channels,
            init_norm_kwargs=init_norm_kwargs,
            dense_init_norm_kwargs=self.dense_init_norm_kwargs,
            first_conv_norm=False,
            **impala_kwargs,
            **kwargs,
        )

        self.linear = FanInInitReLULayer(
            256,
            hidsize,
            layer_type="linear",
            **self.dense_init_norm_kwargs,
        )

    def _prepare_seg(self, seg):
        if seg is None:
            return None
        if isinstance(seg, np.ndarray):
            seg = torch.from_numpy(seg)
        if seg.dim() == 3:
            if self.seg_num_classes is None:
                seg = seg.unsqueeze(1)
            else:
                seg = F.one_hot(seg.long(), self.seg_num_classes).permute(0, 3, 1, 2)
        elif seg.dim() == 4 and self.seg_num_classes is not None and seg.shape[1] == 1:
            seg = F.one_hot(seg[:, 0].long(), self.seg_num_classes).permute(0, 3, 1, 2)
        if seg.shape[1] != self.seg_channels:
            raise ValueError("seg channels do not match seg_channels")
        return seg.float()

    def forward(self, img, goal_embeddings, seg):
        assert len(img.shape) == 4
        img = resize_image(img, (128, 128))
        img = img.to(dtype=torch.float32) / 255.
        seg = self._prepare_seg(seg)
        if seg is not None:
            seg = resize_segmentation(seg, (128, 128))
        return self.linear(self.cnn(img, goal_embeddings, seg))


def create_seg_backbone(name, seg_channels=1, seg_num_classes=None, **kwargs):
    assert name in ['goal_impala_seg_1x', 'goal_impala_seg_3x'], (
        f"[x] backbone {name} is not supported!"
    )
    if name == 'goal_impala_seg_1x':
        return SegGoalImpalaCNNWrapper(
            scale='1x',
            seg_channels=seg_channels,
            seg_num_classes=seg_num_classes,
            **kwargs,
        )
    return SegGoalImpalaCNNWrapper(
        scale='3x',
        seg_channels=seg_channels,
        seg_num_classes=seg_num_classes,
        **kwargs,
    )

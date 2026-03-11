# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
# ---------------------------------------------------------------
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import OrderedDict
import attr


class ConvModule(nn.Module):
    """mmcv ConvModule drop-in: named submodules match mmcv (.conv / .bn / .activate)
    so that state_dict keys load correctly."""
    def __init__(self, in_channels, out_channels, kernel_size, norm_cfg=None, act_cfg=None, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels) if norm_cfg is not None else None
        use_relu = act_cfg is None or act_cfg.get('type', 'ReLU') == 'ReLU'
        self.activate = nn.ReLU(inplace=True) if use_relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activate is not None:
            x = self.activate(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class SegFormerHead(nn.Module):
    def __init__(self, feature_strides, in_channels, num_classes, in_index, dropout_ratio,
                 input_transform='multiple_select', **kwargs):
        super(SegFormerHead, self).__init__()
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        self.in_channels = in_channels
        self.in_index = in_index
        self.num_classes = num_classes

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim * 4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

    def forward(self, inputs):
        x = [inputs[i] for i in self.in_index]
        c1, c2, c3, c4 = x
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c) if self.dropout is not None else _c
        feature = x
        x = self.linear_pred(x)
        return x, feature


def resize(input, size=None, scale_factor=None, mode='nearest', align_corners=None):
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)

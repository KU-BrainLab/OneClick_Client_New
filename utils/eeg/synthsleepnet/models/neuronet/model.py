# -*- coding:utf-8 -*-
import torch
import numpy as np
import torch.nn as nn
from typing import List
from .encoder import FrameBackBone
from timm.models.vision_transformer import Block
from ..utils import get_2d_sincos_pos_embed_flexible
from functools import partial


class NeuroNetEncoder(nn.Module):
    def __init__(self, fs: int, second: int, time_window: int, time_step: float,
                 encoder_embed_dim: int, encoder_heads: int, encoder_depths: int):
        super().__init__()
        self.mlp_ratio = 4.0
        self.fs, self.second = fs, second
        self.time_window = time_window
        self.time_step = time_step

        _, self.num_patches, self.frame_size = self.make_frame(torch.randn(1, self.fs * self.second)).shape
        self.frame_backbone = FrameBackBone(fs=fs, window=time_window)
        self.patch_embed = nn.Linear(self.frame_backbone.feature_num, encoder_embed_dim)
        self.encoder_block = nn.ModuleList([
            Block(encoder_embed_dim, encoder_heads, self.mlp_ratio, qkv_bias=True,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6))
            for _ in range(encoder_depths)
        ])
        self.encoder_norm = nn.LayerNorm(encoder_embed_dim, eps=1e-6)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, encoder_embed_dim))

        self.cls_token.requires_grad = False
        self.pos_embed.requires_grad = False
        self.final_length = encoder_embed_dim

    def forward(self, x):
        x = self.make_frame(x)
        x = self.frame_backbone(x)
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for block in self.encoder_block:
            x = block(x)
        x = self.encoder_norm(x)
        return x

    def make_frame(self, x):
        size = self.fs * self.second
        step = int(self.time_step * self.fs)
        window = int(self.time_window * self.fs)
        frame = []
        for i in range(0, size, step):
            start_idx, end_idx = i, i + window
            sample = x[..., start_idx: end_idx]
            if sample.shape[-1] == window:
                frame.append(sample)
        frame = torch.stack(frame, dim=1)
        return frame

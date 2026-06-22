# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from typing import Dict
from timm.models.vision_transformer import Block
from ..utils import get_2d_sincos_pos_embed_flexible
from functools import partial
from einops.layers.torch import Rearrange


class SynthSleepNetEncoder(nn.Module):
    def __init__(self,
                 backbone_networks: Dict[str, nn.Module],
                 backbone_embed_dim: int, num_backbone_frames: int,
                 encoder_embed_dim: int, encoder_heads: int, encoder_depths: int):
        super().__init__()
        self.modal_names = list(backbone_networks.keys())
        self.backbone_networks = nn.ModuleDict(backbone_networks)
        self.modal_count, self.num_backbone_frames = len(self.backbone_networks), num_backbone_frames
        self.backbone_embed_dim, self.encoder_embed_dim = backbone_embed_dim, encoder_embed_dim

        self.input_size = (self.num_backbone_frames, self.encoder_embed_dim)
        self.patch_size = (1, self.encoder_embed_dim)
        self.grid_h = int(self.input_size[0] // self.patch_size[0])
        self.grid_w = int(self.input_size[1] // self.patch_size[1])
        self.num_patches = self.grid_h * self.grid_w
        self.mlp_ratio = 4.

        self.backbone_embedded = nn.ModuleDict({
            modal_name: nn.Sequential(nn.Linear(backbone_embed_dim, encoder_embed_dim),
                                      Rearrange('b t e -> b e t'),
                                      nn.BatchNorm1d(encoder_embed_dim),
                                      nn.ELU(),
                                      Rearrange('b e t -> b t e'),
                                      nn.Linear(encoder_embed_dim, encoder_embed_dim))
            for modal_name in self.modal_names
        })
        self.modal_token_dict = nn.ParameterDict({
            modal_name: nn.Parameter(torch.zeros(1, num_backbone_frames, encoder_embed_dim))
            for modal_name in self.modal_names
        })

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, encoder_embed_dim),
                                      requires_grad=False)
        self.multimodal_encoder_block = nn.ModuleList([
            Block(encoder_embed_dim, encoder_heads, self.mlp_ratio, qkv_bias=True,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6))
            for _ in range(encoder_depths)
        ])
        self.multimodal_encoder_norm = nn.LayerNorm(encoder_embed_dim, eps=1e-6)
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed_flexible(self.pos_embed.shape[-1],
                                                     (self.grid_h, self.grid_w), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward(self, data, fusion: bool = True):
        total_x = []
        for unimodal_name, unimodal_x in data.items():
            encoder_out = self.backbone_networks[unimodal_name](unimodal_x)
            encoder_emb = self.backbone_embedded[unimodal_name](encoder_out)
            x = encoder_emb[:, 1:, :] + self.modal_token_dict[unimodal_name] + self.pos_embed
            total_x.append(x)

        x = torch.cat(total_x, dim=1)
        for block in self.multimodal_encoder_block:
            x = block(x)
        x = self.multimodal_encoder_norm(x)

        if fusion:
            x = torch.mean(x, dim=1)
        return x

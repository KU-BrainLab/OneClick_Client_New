# -*- coding:utf-8 -*-
"""
SynthSleepNet 체크포인트 로더.
서버의 downstream/utils.py 로직을 로컬 패키지 임포트에 맞게 재구현.
"""
import torch
import torch.nn as nn
from collections import OrderedDict
from peft import get_peft_model, LoraConfig

from .models.neuronet.model import NeuroNetEncoder
from .models.unified.model import SynthSleepNetEncoder


def load_backbone(ckpt_path: str):
    """
    SynthSleepNet 사전학습 체크포인트에서 인코더 백본을 로드.

    Returns
    -------
    backbone : SynthSleepNetEncoder (frozen)
    ch_names : List[str]
    encoder_embed_dim : int
    """
    def _select(model_state, keyword):
        return OrderedDict(
            (k, v) for k, v in model_state.items() if keyword in k
        )

    def _remap(pretrained_subset, target_model):
        """pretrained_subset 값을 target_model 키 순서에 맞게 매핑."""
        new_state = OrderedDict()
        for (tgt_k, _), (_, src_v) in zip(
            target_model.state_dict().items(),
            pretrained_subset.items()
        ):
            new_state[tgt_k] = src_v
        return new_state

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    ch_names = ckpt['ch_names']
    lora_parameter = ckpt['lora_parameter']
    unimodal_parameter = ckpt['unimodal_parameter']
    multimodal_parameter = ckpt['multimodal_parameter']
    multimodal_model_state = ckpt['multimodal_model_state']

    # 1. 각 채널별 NeuroNetEncoder + LoRA 생성
    neuronet_dict = {}
    for ch_name in ch_names:
        neuronet = NeuroNetEncoder(**unimodal_parameter)
        peft_config = LoraConfig(
            r=lora_parameter['lora_r'],
            lora_alpha=lora_parameter['lora_alpha'],
            lora_dropout=lora_parameter['lora_dropout'],
            bias='none',
            use_rslora=True,
            init_lora_weights='gaussian',
            target_modules=['attn.proj'],
        )
        neuronet = get_peft_model(model=neuronet, peft_config=peft_config)
        neuronet_dict[ch_name] = neuronet

    # 2. SynthSleepNetEncoder 조립
    backbone = SynthSleepNetEncoder(
        backbone_networks=neuronet_dict,
        backbone_embed_dim=multimodal_parameter['backbone_embed_dim'],
        num_backbone_frames=multimodal_parameter['num_backbone_frames'],
        encoder_embed_dim=multimodal_parameter['encoder_embed_dim'],
        encoder_heads=multimodal_parameter['encoder_heads'],
        encoder_depths=multimodal_parameter['encoder_depths'],
    )

    # 3. 서브모듈별 가중치 로드
    backbone.backbone_networks.load_state_dict(
        _remap(_select(multimodal_model_state, 'backbone_networks'), backbone.backbone_networks)
    )
    backbone.backbone_embedded.load_state_dict(
        _remap(_select(multimodal_model_state, 'backbone_embedded'), backbone.backbone_embedded)
    )
    backbone.modal_token_dict.load_state_dict(
        _remap(_select(multimodal_model_state, 'modal_token_dict'), backbone.modal_token_dict)
    )
    backbone.multimodal_encoder_block.load_state_dict(
        _remap(_select(multimodal_model_state, 'multimodal_encoder_block'), backbone.multimodal_encoder_block)
    )
    backbone.multimodal_encoder_norm.load_state_dict(
        _remap(_select(multimodal_model_state, 'multimodal_encoder_norm'), backbone.multimodal_encoder_norm)
    )

    # 4. 백본 전체 freeze
    for param in backbone.parameters():
        param.requires_grad = False

    return backbone, ch_names, multimodal_parameter['encoder_embed_dim']


class SynthSleepClassifier(nn.Module):
    """백본 위에 MLP 분류 헤드를 붙인 최종 모델."""

    def __init__(self, backbone: SynthSleepNetEncoder, backbone_embed_dim: int, class_num: int):
        super().__init__()
        self.backbone = backbone
        hidden = backbone_embed_dim // 2
        self.fc = nn.Sequential(
            nn.Linear(backbone_embed_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ELU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ELU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden, class_num),
        )

    def forward(self, x):
        feat = self.backbone(x)   # {ch_name: Tensor[B, 3000]} → [B, embed_dim]
        return self.fc(feat)


def load_classifier(backbone_ckpt_path: str, linear_prob_ckpt_path: str, class_num: int = 5):
    """
    백본 + linear_prob 체크포인트를 로드하여 분류 준비된 모델 반환.
    """
    backbone, ch_names, embed_dim = load_backbone(backbone_ckpt_path)
    model = SynthSleepClassifier(backbone=backbone,
                                  backbone_embed_dim=embed_dim,
                                  class_num=class_num)

    lp_ckpt = torch.load(linear_prob_ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(lp_ckpt['model_state'])

    return model, ch_names

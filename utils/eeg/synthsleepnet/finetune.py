# -*- coding:utf-8 -*-
"""파인튜닝된 SynthSleepNet(수면단계) 로더 — 저자 downstream/fine_tuning/model.py 와
같은 구조를 CPU 에서 조립한다.

선형 프로브판과 다른 점:
  - 백본에 LoRA(frame backbone feature_layer 0/2, 멀티모달 블록 3 의 MLP)가
    라벨로 추가 학습돼 있다.
  - 연속 20 epoch 을 Mamba2 시간문맥 모듈이 함께 읽고, 20 개 전부의 단계를
    낸다(many-to-many).
논문(2502.17481) 표 III 기준 SHHS ACC 87.3 / MF1 81.6 (선형 프로브 80.0 / 70.0).
"""
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig

from .loader import load_backbone
from .mamba2_cpu import Mamba2CPU

TEMPORAL_CONTEXT_LENGTH = 20

# 저자 코드의 target_modules 그대로. 채널별 NeuroNet 안(frame backbone)과
# 멀티모달 인코더 마지막 블록의 MLP 에만 LoRA 가 얹혀 있다.
_LORA_TARGETS = [
    'base_model.model.frame_backbone.feature_layer.0',
    'base_model.model.frame_backbone.feature_layer.2',
    'multimodal_encoder_block.3.mlp.fc1',
    'multimodal_encoder_block.3.mlp.fc2',
]


class FineTunedSleepStager(nn.Module):
    def __init__(self, backbone, backbone_embed_dim, class_num=5,
                 temporal_context_length=TEMPORAL_CONTEXT_LENGTH):
        super().__init__()
        self.backbone = get_peft_model(
            model=backbone,
            peft_config=LoraConfig(
                r=4, lora_alpha=8, lora_dropout=0.05, bias='none',
                use_rslora=True, init_lora_weights='gaussian',
                target_modules=_LORA_TARGETS,
            ),
        )
        self.temporal_context_length = temporal_context_length
        hidden = backbone_embed_dim // 2
        self.mamba = Mamba2CPU(d_model=backbone_embed_dim, d_state=16, d_conv=4, expand=2)
        self.norm = nn.BatchNorm1d(backbone_embed_dim)
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
        """x: {ch_name: Tensor[B, T, 3000]} → logits [B, T, class_num]

        저자 forward 와 같은 계산이되, tokens.squeeze() 가 batch 1 에서 1차원이
        되어 BatchNorm1d 에 걸리는 문제를 피하려고 [B*T, D] 로 펴서 fc 에 넣는다.
        """
        T = self.temporal_context_length
        latents = []
        for i in range(T):
            sample = {ch: x_[:, i, :] for ch, x_ in x.items()}
            latents.append(self.norm(self.backbone(sample)))       # [B, D]
        seq = torch.stack(latents, dim=1)                             # [B, T, D]
        mixed = seq + self.mamba(seq)
        B_, _, D = mixed.shape
        out = self.fc(mixed.reshape(B_ * T, D))
        return out.reshape(B_, T, -1)


def load_finetuned(backbone_ckpt_path: str, finetune_ckpt_path: str, class_num: int = 5):
    """백본 사전학습 ckpt + 파인튜닝 ckpt → 추론 준비된 모델, 채널 이름."""
    backbone, ch_names, embed_dim = load_backbone(backbone_ckpt_path)
    ft = torch.load(finetune_ckpt_path, map_location='cpu', weights_only=False)
    tcl = ft['model_parameter'].get('temporal_context_length', TEMPORAL_CONTEXT_LENGTH)
    model = FineTunedSleepStager(backbone, embed_dim, class_num=class_num,
                                 temporal_context_length=tcl)
    # strict: 키 하나라도 어긋나면 구조 재현이 틀린 것이다 — 조용히 넘기지 않는다.
    model.load_state_dict(ft['model_state'], strict=True)
    model.eval()
    return model, ch_names

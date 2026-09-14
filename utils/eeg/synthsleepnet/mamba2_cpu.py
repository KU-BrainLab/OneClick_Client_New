# -*- coding:utf-8 -*-
"""Mamba2 의 순수 PyTorch 추론 구현.

원본 mamba_ssm.modules.mamba2.Mamba2 는 CUDA 커널(causal_conv1d, triton SSD)이
있어야만 돌아가 CPU 파이프라인에서는 import 조차 되지 않는다. 파인튜닝된
SynthSleepNet 의 시간문맥 모듈이 이 클래스라, 같은 파라미터 이름·형태로
CPU 에서 동일한 계산을 하도록 다시 썼다.

참조 (mamba_ssm 2.2.2):
  - modules/mamba2.py  forward 의 비융합 경로: in_proj → [z, xBC, dt] 분할,
    causal depthwise conv1d + SiLU, x/B/C 분할, 선택적 스캔, 게이트 RMSNorm,
    out_proj.
  - ops/selective_scan_interface.py selective_scan_ref: dt = softplus(dt + dt_bias),
    상태 s_t = exp(dt·A)·s_{t-1} + dt·B_t·x_t, y_t = C_t·s_t + D·x_t.
  - ops/triton/layernorm_gated.py rms_norm_ref (norm_before_gate=False):
    y = rmsnorm(y · silu(z)) · weight, eps 1e-5.

SSD(청크 스캔)는 위 재귀의 효율화일 뿐 수학적으로 같다. 우리는 시퀀스가
20 epoch 이라 재귀를 그대로 돈다 — 청크 알고리즘의 근사·정렬 조건이 없다.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNormGated(nn.Module):
    """mamba_ssm 의 RMSNormGated 와 같은 파라미터(weight)·계산. bias 없음."""

    def __init__(self, dim, eps=1e-5, group_size=None):
        super().__init__()
        self.eps = eps
        self.group_size = group_size or dim
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x, z):
        dtype = x.dtype
        x = x.float() * F.silu(z.float())                 # norm_before_gate=False
        g = self.group_size
        xg = x.reshape(*x.shape[:-1], x.shape[-1] // g, g)
        rstd = torch.rsqrt(xg.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        out = (xg * rstd).reshape(x.shape) * self.weight.float()
        return out.to(dtype)


class Mamba2CPU(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, headdim=64, ngroups=1):
        super().__init__()
        self.d_model, self.d_state, self.d_conv = d_model, d_state, d_conv
        self.d_inner = expand * d_model
        self.headdim, self.ngroups = headdim, ngroups
        assert self.d_inner % headdim == 0
        self.nheads = self.d_inner // headdim

        # 순서: [z, x, B, C, dt]  (원본과 동일)
        d_in_proj = 2 * self.d_inner + 2 * ngroups * d_state + self.nheads
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)
        conv_dim = self.d_inner + 2 * ngroups * d_state
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, kernel_size=d_conv,
                                groups=conv_dim, padding=d_conv - 1, bias=True)
        self.dt_bias = nn.Parameter(torch.zeros(self.nheads))
        self.A_log = nn.Parameter(torch.zeros(self.nheads))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.norm = RMSNormGated(self.d_inner, eps=1e-5, group_size=self.d_inner // ngroups)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, u):
        """u: [B, L, d_model] → [B, L, d_model]"""
        B_, L, _ = u.shape
        h, p, n, g = self.nheads, self.headdim, self.d_state, self.ngroups

        zxbcdt = self.in_proj(u)
        z, xBC, dt = torch.split(
            zxbcdt, [self.d_inner, self.d_inner + 2 * g * n, h], dim=-1)

        # causal depthwise conv: 왼쪽만 (d_conv-1) 패딩한 것과 같다.
        # nn.Conv1d 는 양쪽 패딩이라 앞 L 개만 취한다 (= causal_conv1d_fn).
        xBC = self.conv1d(xBC.transpose(1, 2))[..., :L].transpose(1, 2)
        xBC = F.silu(xBC)
        x, Bm, Cm = torch.split(xBC, [self.d_inner, g * n, g * n], dim=-1)

        x = x.reshape(B_, L, h, p)
        Bm = Bm.reshape(B_, L, g, n)
        Cm = Cm.reshape(B_, L, g, n)
        # ngroups=1: 모든 head 가 같은 B/C 를 쓴다
        Bm = Bm.repeat_interleave(h // g, dim=2)          # [B, L, h, n]
        Cm = Cm.repeat_interleave(h // g, dim=2)

        dt = F.softplus(dt.float() + self.dt_bias.float())  # [B, L, h]
        A = -torch.exp(self.A_log.float())                  # [h]

        x32 = x.float()
        state = torch.zeros(B_, h, p, n, dtype=torch.float32, device=u.device)
        ys = []
        for t in range(L):
            dA = torch.exp(dt[:, t] * A)                                   # [B, h]
            dBx = dt[:, t, :, None, None] * x32[:, t, :, :, None] \
                * Bm[:, t, :, None, :].float()                             # [B, h, p, n]
            state = state * dA[:, :, None, None] + dBx
            y_t = (state * Cm[:, t, :, None, :].float()).sum(-1)           # [B, h, p]
            y_t = y_t + self.D.float()[None, :, None] * x32[:, t]
            ys.append(y_t)
        y = torch.stack(ys, dim=1).reshape(B_, L, h * p).to(u.dtype)

        y = self.norm(y, z)
        return self.out_proj(y)

# -*- coding: utf-8 -*-
"""U-Sleep (Perslev et al., npj Digit. Med. 2021) 순수 PyTorch 추론 구현.

가중치는 SLEEPYLAND(Rossi et al., npj Digit. Med. 2026, MIT) 가 NSRR 17개
코호트 27,494건으로 학습해 공개한 TensorFlow/Keras h5 를 그대로 옮긴 것이다
(https://github.com/biomedical-signal-processing/sleepyland, usleepyland/model/).
층 구성·패딩·정규화 순서를 uSLEEPYLAND/utime/models/usleep.py 와 동일하게
맞췄고, TF 원본과 fp32 오차 1e-6 수준으로 검증했다 (scratch: compare_usleep.py).

원본과의 대응
  Conv2D(k=(9,1), 'same', activation) → Conv1d(k=9, pad 4) 뒤 ELU (활성화가 BN 앞)
  BatchNormalization(eps 1e-3)         → BatchNorm1d(eps=1e-3)
  PadStartToEvenLength → MaxPool(2)    → 홀수 길이면 앞에 1 패딩 후 max_pool1d(2); skip 연결은 패딩 후 텐서
  UpSampling2D(2, nearest)             → repeat_interleave(2)
  Conv2D(k=(2,1), 'same')              → 뒤에 1 패딩 후 Conv1d(k=2) (TF 는 짝수 커널 패딩을 뒤에 몰아 준다)
  CropToMatch + Concatenate([res, up]) → 앞쪽을 diff//2 + diff%2 만큼 잘라 맞추고 [res, up] 순 concat
  dense 1x1 tanh → AveragePooling(3840) → 1x1 ELU → 1x1 softmax
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SAMPLE_RATE = 128           # 학습 표본화율
PERIOD_SEC = 30
DATA_PER_PREDICTION = SAMPLE_RATE * PERIOD_SEC   # 3840 = 30 s 당 1 예측
STAGES = ('W', 'N1', 'N2', 'N3', 'REM')          # 출력 순서 (원클릭과 동일)


class USleep(nn.Module):
    def __init__(self, n_channels=1, n_classes=5, depth=12, init_filters=5,
                 complexity_factor=1.67, kernel_size=9, transition_window=1,
                 data_per_prediction=DATA_PER_PREDICTION, bn_eps=1e-3):
        super().__init__()
        assert kernel_size % 2 == 1 and transition_window % 2 == 1
        cf = math.sqrt(complexity_factor)
        self.depth = depth
        self.dpp = data_per_prediction
        self.n_channels = n_channels

        self.enc_conv, self.enc_bn = nn.ModuleList(), nn.ModuleList()
        f, cin, enc_filters = init_filters, n_channels, []
        for _ in range(depth):
            cout = int(f * cf)
            self.enc_conv.append(nn.Conv1d(cin, cout, kernel_size, padding=kernel_size // 2))
            self.enc_bn.append(nn.BatchNorm1d(cout, eps=bn_eps))
            enc_filters.append(cout)
            cin, f = cout, int(f * math.sqrt(2))
        cout = int(f * cf)
        self.bottom_conv = nn.Conv1d(cin, cout, kernel_size, padding=kernel_size // 2)
        self.bottom_bn = nn.BatchNorm1d(cout, eps=bn_eps)
        cin = cout

        self.up_conv1, self.up_bn1 = nn.ModuleList(), nn.ModuleList()
        self.up_conv2, self.up_bn2 = nn.ModuleList(), nn.ModuleList()
        for i in range(depth):
            f = int(math.ceil(f / math.sqrt(2)))
            cout = int(f * cf)
            self.up_conv1.append(nn.Conv1d(cin, cout, 2))
            self.up_bn1.append(nn.BatchNorm1d(cout, eps=bn_eps))
            self.up_conv2.append(nn.Conv1d(cout + enc_filters[depth - 1 - i], cout,
                                           kernel_size, padding=kernel_size // 2))
            self.up_bn2.append(nn.BatchNorm1d(cout, eps=bn_eps))
            cin = cout

        n_dense = int(n_classes * cf)
        self.dense = nn.Conv1d(cin, n_dense, 1)
        self.seq1 = nn.Conv1d(n_dense, n_classes, transition_window, padding=transition_window // 2)
        self.seq2 = nn.Conv1d(n_classes, n_classes, transition_window, padding=transition_window // 2)

    def forward(self, x):
        """x: [B, C, L] (128 Hz, L 은 dpp 의 배수) → 확률 [B, L // dpp, n_classes]."""
        if x.shape[-1] % self.dpp:
            raise ValueError(f'입력 길이 {x.shape[-1]} 가 {self.dpp} 의 배수가 아닙니다')
        res = []
        for conv, bn in zip(self.enc_conv, self.enc_bn):
            x = bn(F.elu(conv(x)))
            if x.shape[-1] % 2:
                x = F.pad(x, (1, 0))
            res.append(x)            # 원본은 짝수 패딩 *후* 텐서를 skip 연결로 쓴다
            x = F.max_pool1d(x, 2)
        x = self.bottom_bn(F.elu(self.bottom_conv(x)))
        for i in range(self.depth):
            x = x.repeat_interleave(2, dim=-1)
            x = self.up_bn1[i](F.elu(self.up_conv1[i](F.pad(x, (0, 1)))))
            r = res[self.depth - 1 - i]
            diff = x.shape[-1] - r.shape[-1]
            if diff > 0:
                start = diff // 2 + diff % 2
                x = x[..., start:start + r.shape[-1]]
            x = torch.cat([r, x], dim=1)
            x = self.up_bn2[i](F.elu(self.up_conv2[i](x)))
        x = torch.tanh(self.dense(x))
        x = F.avg_pool1d(x, self.dpp)
        x = F.elu(self.seq1(x))
        x = torch.softmax(self.seq2(x), dim=1)
        return x.transpose(1, 2)


def state_dict_from_keras_h5(path, depth=12):
    """uSLEEPYLAND 가 저장한 Keras h5(가중치 전용) → 이 모듈의 state_dict."""
    import h5py
    sd = {}

    def tensor(g, name):
        return torch.from_numpy(np.array(g[name], dtype=np.float32))

    with h5py.File(path, 'r') as f:
        def conv(kname, tname):
            g = f[kname][kname]
            k = np.array(g['kernel:0'], dtype=np.float32)        # (kh, 1, cin, cout)
            sd[tname + '.weight'] = torch.from_numpy(np.ascontiguousarray(k[:, 0].transpose(2, 1, 0)))
            sd[tname + '.bias'] = tensor(g, 'bias:0')

        def bn(kname, tname):
            g = f[kname][kname]
            sd[tname + '.weight'] = tensor(g, 'gamma:0')
            sd[tname + '.bias'] = tensor(g, 'beta:0')
            sd[tname + '.running_mean'] = tensor(g, 'moving_mean:0')
            sd[tname + '.running_var'] = tensor(g, 'moving_variance:0')

        for i in range(depth):
            conv(f'encoder_L{i}_conv1', f'enc_conv.{i}')
            bn(f'encoder_L{i}_BN1', f'enc_bn.{i}')
        conv('bottom_conv1', 'bottom_conv')
        bn('bottom_BN1', 'bottom_bn')
        for i in range(depth):
            conv(f'upsample_L{i}_conv1', f'up_conv1.{i}')
            bn(f'upsample_L{i}_BN1', f'up_bn1.{i}')
            conv(f'upsample_L{i}_conv2', f'up_conv2.{i}')
            bn(f'upsample_L{i}_BN2', f'up_bn2.{i}')
        conv('dense_classifier_out', 'dense')
        conv('sequence_conv_out_1', 'seq1')
        conv('sequence_conv_out_2', 'seq2')
    return sd


def load_model(state_dict_or_path, n_channels=1, **kwargs):
    """변환된 .pt(state_dict) 또는 원본 h5 로부터 eval 모드 모델을 만든다."""
    model = USleep(n_channels=n_channels, **kwargs)
    if isinstance(state_dict_or_path, str) and state_dict_or_path.endswith('.h5'):
        sd = state_dict_from_keras_h5(state_dict_or_path, depth=model.depth)
    elif isinstance(state_dict_or_path, str):
        sd = torch.load(state_dict_or_path, map_location='cpu')
        sd = sd.get('state_dict', sd)
    else:
        sd = state_dict_or_path
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if not m.endswith('num_batches_tracked')]
    if missing or unexpected:
        raise RuntimeError(f'가중치 키 불일치: missing={missing} unexpected={unexpected}')
    return model.eval()


def preprocess(x, sfreq, iqr_clip=20):
    """psg_utils 와 같은 순서의 전처리: 클리핑 → 128 Hz 리샘플 → RobustScaler.

    x: [n_channels, n_samples] 연속 신호(단위 무관, 채널별로 스케일링됨).
    반환: float32 [n_channels, n_samples * 128 / sfreq]
    """
    from scipy.signal import resample_poly
    x = np.asarray(x, dtype=np.float64)
    out = []
    for ch in x:
        q75, q25 = np.percentile(ch, [75, 25])
        thr = (q75 - q25) * iqr_clip
        if thr > 0:
            ch = np.clip(ch, -thr, thr)
        if int(sfreq) != SAMPLE_RATE:
            ch = resample_poly(ch, SAMPLE_RATE, int(sfreq))
        med = np.median(ch)
        q75, q25 = np.percentile(ch, [75, 25])
        scale = (q75 - q25) or 1.0
        out.append((ch - med) / scale)
    return np.stack(out).astype(np.float32)


@torch.no_grad()
def predict_probs(model, x, sfreq, channels_as_votes=True):
    """연속 신호 x[n_channels, n_samples] → epoch 별 확률 [n_epochs, 5].

    channels_as_votes=True 면 각 채널을 따로 넣어 확률을 평균한다 (저자·SLEEPYLAND
    의 --majority 와 같은 soft vote). False 면 채널을 모델 입력 채널로 함께 넣는다
    (EEG+EOG 다채널 모델용).
    """
    z = preprocess(x, sfreq)
    n_ep = z.shape[-1] // model.dpp
    z = torch.from_numpy(z[:, :n_ep * model.dpp])
    if channels_as_votes:
        probs = torch.stack([model(z[c:c + 1][None])[0] for c in range(z.shape[0])]).mean(0)
    else:
        probs = model(z[None])[0]
    return probs

import sys
import copy
from pathlib import Path

import mne
import torch
import numpy as np

# synthsleepnet 패키지를 로컬 패키지로 임포트
_EEG_DIR = Path(__file__).parent.parent
if str(_EEG_DIR) not in sys.path:
    sys.path.insert(0, str(_EEG_DIR))

from synthsleepnet.loader import load_classifier

# 서버 체크포인트 경로 (X: 드라이브 마운트 필요)
_SERVER_ROOT = Path(r'X:\Workspace\Chlee\MultiModal_for_Sleep\ckpt\multimodal\EEG2')
_BACKBONE_CKPT  = _SERVER_ROOT / 'model' / 'best_model.pth'
_LINEAR_CKPT    = _SERVER_ROOT / 'linear_prob' / 'sleep_stage' / 'best_model.pth'

# ch_names 매핑: 서버 학습 채널명 → 원클릭 채널 인덱스용 이름
_SERVER_TO_LOCAL = {
    'EEG_C4': 'C4',
    'EEG_C3': 'C3',
}

_model_cache = None   # 최초 1회만 로드


def _get_model():
    global _model_cache
    if _model_cache is None:
        print('[SynthSleepNet] 모델 로드 중...')
        model, ch_names = load_classifier(
            backbone_ckpt_path=str(_BACKBONE_CKPT),
            linear_prob_ckpt_path=str(_LINEAR_CKPT),
            class_num=5,
        )
        model.eval()
        _model_cache = (model, ch_names)
        print(f'[SynthSleepNet] 로드 완료. 채널: {ch_names}')
    return _model_cache


def compute_sleep_metrics(stage_list, epoch_sec: int = 30):
    sleep_labels = {1, 2, 3, 4}
    wake_label = 0

    n_epochs = len(stage_list)
    epoch_min = epoch_sec / 60.0
    tib = n_epochs * epoch_min

    try:
        sleep_onset_idx = next(i for i, s in enumerate(stage_list) if s in sleep_labels)
        sleep_latency = sleep_onset_idx * epoch_min
    except StopIteration:
        sleep_onset_idx = None
        sleep_latency = None

    rem_latency = 0
    if sleep_onset_idx is not None:
        try:
            rem_idx = next(i for i, s in enumerate(stage_list[sleep_onset_idx:], start=sleep_onset_idx) if s == 4)
            rem_latency = (rem_idx - sleep_onset_idx) * epoch_min
        except StopIteration:
            pass

    tst = sum(1 for s in stage_list if s in sleep_labels) * epoch_min

    waso = None
    if sleep_onset_idx is not None:
        waso = sum(1 for s in stage_list[sleep_onset_idx:] if s == wake_label) * epoch_min

    twt = None
    if sleep_latency is not None and waso is not None:
        twt = sleep_latency + waso

    sleep_eff = (tst / tib * 100.0) if tib > 0 else None

    return {
        'tib': tib,
        'tst': tst,
        'twt': twt,
        'waso': waso,
        'sleep_latency': sleep_latency,
        'rem_latency': rem_latency,
        'sleep_eff': sleep_eff,
    }


def get_sleep_staging(epoch_data, ch_list):
    epoch_data = copy.deepcopy(epoch_data)
    info = epoch_data.info

    # 스케일링 (median)
    scaler = mne.decoding.Scaler(info=info, scalings='median')
    data = scaler.fit_transform(epoch_data.get_data())  # [n_epochs, n_ch, n_times]

    # 모델 로드
    model, ch_names = _get_model()

    # 실제 epoch에 남아있는 채널 목록 (O1/O2 드롭 후 기준)
    actual_ch_names = epoch_data.info['ch_names']

    # 입력 딕셔너리 구성: {서버채널명: Tensor[n_epochs, 3000]}
    x = {}
    for server_ch in ch_names:
        local_ch = _SERVER_TO_LOCAL[server_ch]
        ch_idx = actual_ch_names.index(local_ch)
        arr = data[:, ch_idx, :]   # [n_epochs, n_times]

        # 모델은 fs=100, second=30 → 3000 샘플 기대
        expected = 3000
        if arr.shape[1] != expected:
            arr = _resample_to(arr, expected)

        x[server_ch] = torch.tensor(arr, dtype=torch.float32)

    # 추론
    with torch.no_grad():
        logits = model(x)                           # [n_epochs, 5]
        probs  = torch.softmax(logits, dim=-1)      # [n_epochs, 5]

    sleep_stage      = torch.argmax(probs, dim=-1).cpu().numpy().tolist()
    sleep_stage_prob = probs.cpu().numpy().tolist()

    # 통계
    total_epoch = len(sleep_stage)
    w_tst    = sleep_stage.count(0) / total_epoch * 100
    n1_tst   = sleep_stage.count(1) / total_epoch * 100
    n2_tst   = sleep_stage.count(2) / total_epoch * 100
    n3_tst   = sleep_stage.count(3) / total_epoch * 100
    nrem_tst = n1_tst + n2_tst + n3_tst
    rem_tst  = sleep_stage.count(4) / total_epoch * 100

    w_min    = sleep_stage.count(0) * 30 / 60
    n1_min   = sleep_stage.count(1) * 30 / 60
    n2_min   = sleep_stage.count(2) * 30 / 60
    n3_min   = sleep_stage.count(3) * 30 / 60
    nrem_min = n1_min + n2_min + n3_min
    rem_min  = sleep_stage.count(4) * 30 / 60

    sleep_summary = compute_sleep_metrics(sleep_stage, 30)
    sleep_summary['sleep_tst'] = [n1_tst, n2_tst, n3_tst, nrem_tst, rem_tst]
    sleep_summary['sleep_min'] = [n1_min, n2_min, n3_min, nrem_min, rem_min]

    return {
        'sleep_stage':      sleep_stage,
        'sleep_stage_prob': sleep_stage_prob,
        'sleep_summary':    sleep_summary,
    }


def _resample_to(arr: np.ndarray, target_len: int) -> np.ndarray:
    """에포크 배열을 target_len 샘플로 리샘플링 (scipy)."""
    from scipy.signal import resample
    return resample(arr, target_len, axis=1)

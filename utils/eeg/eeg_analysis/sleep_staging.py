import sys
import copy
from pathlib import Path

import mne
import torch
import numpy as np

# synthsleepnet 패키지를 로컬 패키지로 임포트할 수 있게 경로만 잡아둔다.
# 실제 임포트는 _get_model() 안에서 한다 — synthsleepnet/loader.py 가 peft 를
# 최상단에서 불러오고, peft 는 transformers, huggingface_hub 로 이어진다.
# 여기서 임포트하면 기본 모델(NeuroNet)만 쓰는 사람도 그 의존성이 깨졌을 때
# 파이프라인 전체가 실행조차 안 된다(실제로 발생한 장애다).
_EEG_DIR = Path(__file__).parent.parent
if str(_EEG_DIR) not in sys.path:
    sys.path.insert(0, str(_EEG_DIR))

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:      # neuronet 패키지가 프로젝트 루트에 있다
    sys.path.insert(0, str(_PROJECT_ROOT))

_CKPT_ROOT = _PROJECT_ROOT / 'synthsleepnet' / 'ckpt' / 'multimodal' / 'EEG2'
_BACKBONE_CKPT = _CKPT_ROOT / 'model' / 'best_model.pth'
_LINEAR_CKPT   = _CKPT_ROOT / 'linear_prob' / 'sleep_stage' / 'best_model.pth'
_FT_CKPT       = _CKPT_ROOT / 'fine_tuning' / 'sleep_stage' / 'best_model.pth'

_NEURONET_CKPT_ROOT = _PROJECT_ROOT / 'neuronet' / 'ckpt'
_NEURONET_N_FOLDS   = 5

# U-Sleep: SLEEPYLAND(NSRR 17개 코호트 27,494건) 학습 가중치를 PyTorch 로 옮긴 것.
# 저장소에 포함돼 있어(각 12.5MB, MIT) 서버에 따로 복사할 필요가 없다.
_USLEEP_CKPT_DIR = _EEG_DIR / 'usleep' / 'ckpt'
_USLEEP_CKPT     = _USLEEP_CKPT_DIR / 'usleep-nsrr-2024_eeg.pt'
# EEG 단일채널 모델을 채널마다 따로 돌려 확률을 평균한다(저자의 majority vote).
# 학습 채널군에 C3/C4·F3/F4·P3/P4(M1/M2/AVG 유도)가 모두 들어 있다.
_USLEEP_CHANNELS = ('C3', 'C4', 'F3', 'F4')

# 선택 가능한 수면단계 모델
MODEL_SYNTHSLEEPNET    = 'synthsleepnet'        # 선형 프로브 (epoch 단독)
MODEL_SYNTHSLEEPNET_FT = 'synthsleepnet_ft'     # 파인튜닝 + 20 epoch 시간문맥
MODEL_NEURONET         = 'neuronet'
MODEL_USLEEP           = 'usleep'               # U-Sleep (SLEEPYLAND 가중치, EEG 단일채널 soft vote)
AVAILABLE_MODELS    = (MODEL_SYNTHSLEEPNET, MODEL_SYNTHSLEEPNET_FT, MODEL_NEURONET, MODEL_USLEEP)
# 기본값은 U-Sleep (2026-09-29). NSRR 17개 코호트 27,494건으로 학습된 가중치라
# 학습 데이터 폭이 가장 넓고, 로컬 측정 16건에서 단계 전이가 가장 적고(18/100 epoch)
# 단계별 delta 파워 순서가 가장 잘 지켜졌으며(ρ 0.56), 100 epoch 채점에 0.5 s 다.
# SHHS1 eval 31명 정확도는 86.2 로 SynthSleepNet 파인튜닝판(87.0)과 동급.
# 가중치(12.5MB)는 저장소에 포함돼 있어 없을 일이 없지만, 없거나 추론이 실패하면
# 파인튜닝판 → 선형 프로브 순으로 대체한다(get_sleep_staging).
# SynthSleepNet 계열은 SHHS C3/C4 학습, NeuroNet 은 Sleep-EDFX(Fpz-Cz) 학습 — 예비용.
DEFAULT_MODEL       = MODEL_USLEEP

# 두 모델 모두 C4, C3 두 채널만 쓴다.
# ch_names 매핑: 서버 학습 채널명 → 원클릭 채널 인덱스용 이름
_SERVER_TO_LOCAL = {
    'EEG_C4': 'C4',
    'EEG_C3': 'C3',
}
_NEURONET_CHANNELS = ('C4', 'C3')

_model_cache = None            # SynthSleepNet, 최초 1회만 로드
_ft_cache = None               # SynthSleepNet 파인튜닝판, 최초 1회만 로드
_neuronet_cache = None         # NeuroNet 5-fold, 최초 1회만 로드
_usleep_cache = None           # U-Sleep, 최초 1회만 로드


def _get_model():
    """SynthSleepNet 을 로드한다. 이 모델을 고를 때만 peft/transformers 가 필요하다."""
    global _model_cache
    if _model_cache is None:
        try:
            from synthsleepnet.loader import load_classifier
        except ImportError as e:
            raise ImportError(
                f"SynthSleepNet 을 불러오지 못했습니다: {e}\n"
                f"이 모델은 peft/transformers/huggingface_hub 가 필요합니다.\n"
                f"  pip install -r requirements.txt\n"
                f"로 버전을 맞추거나, 기본 모델을 쓰려면 "
                f"--SLEEP_MODEL {MODEL_NEURONET} 로 실행하세요 "
                f"(NeuroNet 은 이 의존성이 필요 없습니다)."
            ) from e
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


def _get_finetuned():
    """파인튜닝 SynthSleepNet(20 epoch 문맥)을 로드한다. peft/transformers 필요."""
    global _ft_cache
    if _ft_cache is None:
        try:
            from synthsleepnet.finetune import load_finetuned
        except ImportError as e:
            raise ImportError(
                f"SynthSleepNet 파인튜닝판을 불러오지 못했습니다: {e}\n"
                f"pip install -r requirements.txt 로 버전을 맞추거나 "
                f"--SLEEP_MODEL {MODEL_NEURONET} 로 실행하세요."
            ) from e
        if not _FT_CKPT.exists():
            raise FileNotFoundError(
                f"파인튜닝 체크포인트가 없습니다: {_FT_CKPT}\n"
                f"연구실 저장소의 ckpt/multimodal/EEG2/fine_tuning/sleep_stage/ 를 복사하세요.")
        print('[SynthSleepNet-FT] 모델 로드 중...')
        model, ch_names = load_finetuned(backbone_ckpt_path=str(_BACKBONE_CKPT),
                                         finetune_ckpt_path=str(_FT_CKPT), class_num=5)
        _ft_cache = (model, ch_names)
        print(f'[SynthSleepNet-FT] 로드 완료. 채널: {ch_names}, '
              f'문맥 {model.temporal_context_length} epoch')
    return _ft_cache


def _probs_synthsleepnet_ft(data, actual_ch_names):
    """20 epoch 문맥 모델의 epoch 별 확률.

    저자 평가 방식(비중첩 20-epoch 청크 + 끝에 맞춘 꼬리 청크)을 따르되,
    꼬리 청크와 겹치는 epoch 은 두 예측의 확률을 평균한다. 20 epoch 미만인
    기록은 마지막 epoch 을 반복해 채우고 그 출력은 버린다 — 모델이 요구하는
    길이일 뿐, 실제 기록은 보통 1시간 이상이라 걸릴 일이 없다.
    """
    model, ch_names = _get_finetuned()
    T = model.temporal_context_length
    xs = {srv: _pick_channel(data, actual_ch_names, _SERVER_TO_LOCAL[srv], 3000)
          for srv in ch_names}
    n = next(iter(xs.values())).shape[0]
    m = max(n, T)
    if n < T:
        xs = {k: torch.cat([v, v[-1:].repeat(T - n, 1)], dim=0) for k, v in xs.items()}
    starts = list(range(0, m - T + 1, T))
    if starts[-1] + T < m:
        starts.append(m - T)                      # 꼬리 청크
    prob_sum = torch.zeros(m, 5)
    count = torch.zeros(m, 1)
    with torch.no_grad():
        for b in range(0, len(starts), 16):
            batch = starts[b:b + 16]
            xb = {k: torch.stack([v[s:s + T] for s in batch]) for k, v in xs.items()}
            out = torch.softmax(model(xb), dim=-1)          # [b, T, 5]
            for i, s in enumerate(batch):
                prob_sum[s:s + T] += out[i]
                count[s:s + T] += 1
    return (prob_sum / count)[:n]


def _get_usleep():
    """U-Sleep(PyTorch 이식판)을 로드한다. torch/scipy 외 의존성 없음."""
    global _usleep_cache
    if _usleep_cache is None:
        from usleep.usleep_torch import load_model
        if not _USLEEP_CKPT.exists():
            raise FileNotFoundError(f'U-Sleep 가중치가 없습니다: {_USLEEP_CKPT} (git pull 로 받아집니다)')
        print('[U-Sleep] 모델 로드 중...')
        _usleep_cache = load_model(str(_USLEEP_CKPT), n_channels=1)
        print(f'[U-Sleep] 로드 완료. 채널: {_USLEEP_CHANNELS} (각각 채점 후 확률 평균)')
    return _usleep_cache


def _probs_usleep(raw_epochs, actual_ch_names, sfreq):
    """U-Sleep 확률 [n_epochs, 5].

    raw_epochs: 스케일링 전 epoch 데이터 [n_epochs, n_ch, n_times]. U-Sleep 은 자체 전처리
    (20 IQR 클리핑 → 128 Hz 리샘플 → RobustScaler)를 연속 신호에 적용하므로 epoch 을
    시간축으로 이어 붙여 넣는다. 밤 전체를 한 번에 처리하는 완전 합성곱 모델이라 길이 제한이 없다.
    """
    from usleep.usleep_torch import predict_probs
    model = _get_usleep()
    chans = [c for c in _USLEEP_CHANNELS if c in actual_ch_names]
    if not chans:
        raise ValueError(f'U-Sleep 입력 채널 {_USLEEP_CHANNELS} 이 없습니다: {actual_ch_names}')
    idx = [actual_ch_names.index(c) for c in chans]
    x = np.asarray(raw_epochs)[:, idx, :].transpose(1, 0, 2).reshape(len(idx), -1)  # [n_ch, n_ep*n_times]
    return predict_probs(model, x, sfreq, channels_as_votes=True)


def _get_neuronet():
    """NeuroNet 5-fold 앙상블을 로드한다 (구버전 모델).

    fold 마다 사전학습 백본(NeuroNet) + linear probe 를 조립한다.
    """
    global _neuronet_cache
    if _neuronet_cache is not None:
        return _neuronet_cache

    from neuronet.model import NeuroNet, NeuroNetEncoderWrapper, Classifier

    print(f'[NeuroNet] 모델 로드 중... ({_NEURONET_N_FOLDS}-fold 앙상블)')
    models = []
    for i in range(_NEURONET_N_FOLDS):
        ckpt = torch.load(_NEURONET_CKPT_ROOT / str(i) / 'model' / 'best_model.pth',
                          map_location='cpu', weights_only=False)
        mp = ckpt['model_parameter']
        pretrained = NeuroNet(**mp)
        pretrained.load_state_dict(ckpt['model_state'])

        backbone = NeuroNetEncoderWrapper(
            fs=mp['fs'], second=mp['second'],
            time_window=mp['time_window'], time_step=mp['time_step'],
            frame_backbone=pretrained.frame_backbone,
            patch_embed=pretrained.autoencoder.patch_embed,
            encoder_block=pretrained.autoencoder.encoder_block,
            encoder_norm=pretrained.autoencoder.encoder_norm,
            cls_token=pretrained.autoencoder.cls_token,
            pos_embed=pretrained.autoencoder.pos_embed,
            final_length=pretrained.autoencoder.embed_dim,
        )
        model = Classifier(backbone=backbone,
                           backbone_final_length=pretrained.autoencoder.embed_dim)
        lp = torch.load(_NEURONET_CKPT_ROOT / str(i) / 'linear_prob' / 'best_model.pth',
                        map_location='cpu', weights_only=False)
        model.load_state_dict(lp['model_state'])
        model.eval()
        models.append(model)

    _neuronet_cache = (models, mp['fs'] * mp['second'])
    print(f'[NeuroNet] 로드 완료. 채널: {list(_NEURONET_CHANNELS)}, '
          f'입력 길이: {_neuronet_cache[1]}')
    return _neuronet_cache


def _pick_channel(data, actual_ch_names, ch, expected_len):
    """실제 채널 이름으로 인덱싱해 [n_epochs, expected_len] 텐서를 만든다.

    analysis.py 는 O1/O2 를 drop 한 13채널 epoch 을 넘기면서 ch_list 는 15채널짜리를
    그대로 넘긴다. 구버전 NeuroNet 코드가 ch_list.index('C4')=11 로 인덱싱하는 바람에
    13채널 배열의 11번(=T4)을 C4 로 착각해 먹이고 있었다. 반드시 epoch 에 실제로
    남아있는 채널 이름으로 찾아야 한다.
    """
    arr = data[:, actual_ch_names.index(ch), :]
    if arr.shape[1] != expected_len:
        arr = _resample_to(arr, expected_len)
    return torch.tensor(arr, dtype=torch.float32)


def _probs_synthsleepnet(data, actual_ch_names):
    model, ch_names = _get_model()
    x = {srv: _pick_channel(data, actual_ch_names, _SERVER_TO_LOCAL[srv], 3000)
         for srv in ch_names}
    with torch.no_grad():
        return torch.softmax(model(x), dim=-1)      # [n_epochs, 5]


def _probs_neuronet(data, actual_ch_names):
    """5-fold x 2채널 평균 확률.

    구버전은 fold 마다 softmax(C4)+softmax(C3) 를 그대로 더해서 행 합이 2가 됐다.
    여기서는 채널 수로 나눠 합이 1이 되게 한다. argmax 는 단조변환이라 바뀌지 않고,
    SynthSleepNet 출력과 스케일이 같아져 sleep_stage_prob 를 두 모델 간에 비교할 수 있다.
    """
    models, in_len = _get_neuronet()
    xs = [_pick_channel(data, actual_ch_names, ch, in_len) for ch in _NEURONET_CHANNELS]
    with torch.no_grad():
        per_fold = [
            torch.stack([torch.softmax(m(x), dim=-1) for x in xs]).mean(dim=0)
            for m in models
        ]
        return torch.stack(per_fold).mean(dim=0)    # [n_epochs, 5]


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


def get_sleep_staging(epoch_data, ch_list, model=DEFAULT_MODEL):
    """수면단계 추론.

    model: 'synthsleepnet_ft' (SHHS1 학습, 파인튜닝 + 20 epoch 시간문맥, 기본값)
           'synthsleepnet'    (SHHS1 학습, 선형 프로브, epoch 단독)
           'neuronet'         (Sleep-EDFX 학습, 5-fold 앙상블, 구버전)
           'usleep'           (U-Sleep, NSRR 17개 코호트 학습(SLEEPYLAND), EEG 단일채널 soft vote)
    ch_list 는 하위호환을 위해 남겨두지만 쓰지 않는다 — 실제 채널 이름으로 인덱싱한다.
    """
    if model == MODEL_USLEEP and not _USLEEP_CKPT.exists():
        print(f'[SleepStaging] U-Sleep 가중치가 없어 {MODEL_SYNTHSLEEPNET_FT} 로 대체합니다: {_USLEEP_CKPT}')
        model = MODEL_SYNTHSLEEPNET_FT
    if model == MODEL_SYNTHSLEEPNET_FT and not _FT_CKPT.exists():
        # 서버에 파인튜닝 체크포인트가 아직 없어도 분석이 죽지 않게 선형 프로브로 내린다.
        print(f'[SleepStaging] 파인튜닝 체크포인트가 없어 {MODEL_SYNTHSLEEPNET} 로 대체합니다: {_FT_CKPT}')
        model = MODEL_SYNTHSLEEPNET
    print(f'[SleepStaging] 모델: {model}')
    epoch_data = copy.deepcopy(epoch_data)
    info = epoch_data.info

    # SynthSleepNet 계열은 SHHS 를 0.5-40 Hz 로 걸러 학습했다. 파이프라인은
    # 0.5-60 Hz 라 수면단계 입력에만 40 Hz 를 맞춘다 (저자 전처리와 같이 epoch
    # 단위로 건다). NeuroNet 경로는 종전 그대로 둔다.
    if model in (MODEL_SYNTHSLEEPNET, MODEL_SYNTHSLEEPNET_FT):
        epoch_data.load_data()
        epoch_data.filter(l_freq=None, h_freq=40., verbose=False)

    # 스케일링 (median)
    raw_epochs = epoch_data.get_data()                  # U-Sleep 은 자체 전처리를 쓴다
    scaler = mne.decoding.Scaler(info=info, scalings='median')
    data = scaler.fit_transform(raw_epochs)             # [n_epochs, n_ch, n_times]

    # 실제 epoch 에 남아있는 채널 목록 (O1/O2 드롭 후 기준).
    # ch_list 인자는 15채널짜리라 인덱스가 어긋난다 — 쓰지 않는다(_pick_channel 주석 참고).
    actual_ch_names = epoch_data.info['ch_names']

    if model == MODEL_SYNTHSLEEPNET:
        probs = _probs_synthsleepnet(data, actual_ch_names)
    elif model == MODEL_SYNTHSLEEPNET_FT:
        try:
            probs = _probs_synthsleepnet_ft(data, actual_ch_names)
        except Exception:
            # 서버 환경(peft/transformers 버전 등)에서 파인튜닝판 로드·추론이 깨져도
            # 분석 전체가 죽지 않게 선형 프로브로 내린다. 원인은 로그에 남긴다.
            import traceback
            traceback.print_exc()
            print(f'[SleepStaging] 파인튜닝판 실패 → {MODEL_SYNTHSLEEPNET} 로 대체합니다 (위 traceback 확인)')
            model = MODEL_SYNTHSLEEPNET
            probs = _probs_synthsleepnet(data, actual_ch_names)
    elif model == MODEL_NEURONET:
        probs = _probs_neuronet(data, actual_ch_names)
    elif model == MODEL_USLEEP:
        try:
            probs = _probs_usleep(raw_epochs, actual_ch_names, info['sfreq'])
        except Exception:
            # 분석 전체가 죽지 않게 선형 프로브로 내린다 (40 Hz 저역통과 없이 들어가지만
            # 종전 동작과 같다). 원인은 로그에 남긴다.
            import traceback
            traceback.print_exc()
            print(f'[SleepStaging] U-Sleep 실패 → {MODEL_SYNTHSLEEPNET} 로 대체합니다 (위 traceback 확인)')
            model = MODEL_SYNTHSLEEPNET
            probs = _probs_synthsleepnet(data, actual_ch_names)
    else:
        raise ValueError(
            f"알 수 없는 수면단계 모델: {model!r} (가능: {', '.join(AVAILABLE_MODELS)})")

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

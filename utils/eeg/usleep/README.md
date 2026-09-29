# U-Sleep (PyTorch 이식판)

원클릭 수면단계 옵션 `usleep`. 의존성은 torch, numpy, scipy 뿐이다.

## 출처
- 구조: Perslev et al., *U-Sleep: resilient high-frequency sleep staging*, npj Digit. Med. 4, 72 (2021).
  원저자 가중치는 비공개(웹서비스 전용).
- 가중치: Rossi et al., *SLEEPYLAND*, npj Digit. Med. 9, 55 (2026) 가 같은 구조를
  NSRR 17개 코호트 27,494건으로 학습해 MIT 로 공개한 것.
  https://github.com/biomedical-signal-processing/sleepyland (`usleepyland/model/`)
  - `ckpt/usleep-nsrr-2024_eeg.pt` ← `u-sleep-nsrr-2024_eeg/model/@epoch_877_val_dice_0.74960.h5` (기본값, 논문 판)
  - `ckpt/usleep-nsrr-2022_eeg.pt` ← `u-sleep-nsrr-2022_eeg/model/@epoch_1653_val_dice_0.79550.h5`
  - 라이선스 원문: `LICENSE-SLEEPYLAND.txt` (MIT, Copyright (c) 2024 Luigi Fiorillo)
- 변환: `usleep_torch.state_dict_from_keras_h5()` (h5py 로 Keras h5 를 읽어 층별로 옮김).
  각 .pt 의 `meta` 에 원본 URL·sha256·하이퍼파라미터가 들어 있다.

## 검증 (2026-09-29)
- TF 2.15 + uSLEEPYLAND 원본 모델 정의로 같은 h5 를 로드한 참조 출력과 대조:
  난수 입력 35/40/121 epoch(홀수 길이 패딩·크롭 경로 포함), 1채널·2채널 모델 모두
  확률 max|diff| ≤ 1.5e-6, argmax 100% 일치. 층별 출력도 encoder 부터 1e-6 수준.
- SHHS1 eval 31명(C4+C3 soft vote): 2024_eeg ACC 86.2 / MF1 79.9 / κ 0.807,
  2022_eeg 88.3 / 82.3 / 0.838 (같은 구간 SynthSleepNet FT 87.0 / 81.0 / 0.822).
  단, SHHS1 은 SLEEPYLAND 학습 데이터에 포함돼 있어 U-Sleep 에 유리한 비교다.

## 입력 규약 (원본 psg_utils 와 동일)
1. 채널별 |x| > 20·IQR 클리핑 → 2. `scipy.signal.resample_poly` 로 128 Hz → 3. RobustScaler(중앙값 0, IQR 1)
4. 길이는 3840 샘플(30 s)의 배수, 밤 전체를 한 번에 통과(완전 합성곱)
5. EEG 단일채널 모델을 채널마다 따로 돌려 확률 평균 = 원저자의 majority vote

원클릭에서는 `sleep_staging._USLEEP_CHANNELS`(기본 C3, C4, F3, F4) 를 각각 넣는다.
학습 채널군에 C3/C4·F3/F4·P3/P4·Fp1/Fp2 (M1/M2/AVG/Cz 유도) 가 모두 들어 있다.

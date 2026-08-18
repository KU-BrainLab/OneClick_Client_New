import os
import pyhrv
import biosppy

import warnings
import matplotlib.cbook
import matplotlib as mpl

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pyhrv.time_domain as td
import pyhrv.frequency_domain as fd
from scipy.ndimage import median_filter
from matplotlib.projections import register_projection

from matplotlib.ticker import FixedLocator, LogFormatter, ScalarFormatter
from matplotlib.scale import FuncScale


######################## Custom Import ########################
from .charts.radar_chart import radar_chart
from .charts.heart_rate_heatplot import heart_rate_heatplot
###############################################################

sns.set()
sns.set_palette("muted")
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)


def remove_noise_beats(filtered, rpeaks, sfreq, corr_min=0.5,
                       amp_lo=0.25, amp_hi=4.0, max_bad_frac=0.3):
    """Template 상관 + R 진폭으로 노이즈 위의 가짜 R-peak 를 제거한다.

    센서 접촉 불량 구간에서는 QRS 가 아예 없는데도 검출기가 노이즈 위에
    그럴듯한 간격의 가짜 피크를 찍는다. interval 값만 보는 보정으로는 이런
    구간을 거를 수 없으므로, 각 비트 파형을 중앙값 template 과 비교해
    상관이 낮거나(모양이 QRS 가 아님) R 진폭이 비정상인 피크를 먼저 제거한다.
    제거 후 남는 긴 구멍은 correct_rpeak_artifacts 의 valid 마스크가 제외한다.

    비정상 비트가 max_bad_frac 를 넘으면 template 자체를 신뢰할 수 없으므로
    아무것도 제거하지 않는다 (기록 전체가 이상한 경우 안전장치).

    Returns (rpeaks_kept, n_removed)
    """
    rpeaks = np.asarray(rpeaks, dtype=int)
    before = int(0.2 * sfreq)
    after = int(0.4 * sfreq)
    ok_win = (rpeaks - before >= 0) & (rpeaks + after < len(filtered))
    if ok_win.sum() < 10:
        return rpeaks, 0
    idx = np.where(ok_win)[0]

    beats = np.stack([filtered[r - before: r + after] for r in rpeaks[idx]])
    beats = beats - beats.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(beats, axis=1)
    norm[norm == 0] = 1e-12
    beats = beats / norm[:, None]
    template = np.median(beats, axis=0)
    template = template - template.mean()
    template = template / max(np.linalg.norm(template), 1e-12)
    corr = beats @ template

    amp = np.abs(filtered[rpeaks[idx]])
    med_amp = np.median(amp)
    bad = (corr < corr_min) | (amp < amp_lo * med_amp) | (amp > amp_hi * med_amp)

    if bad.mean() > max_bad_frac:
        print(f'[SQI] 비정상 비트 {100 * bad.mean():.0f}% -> template 신뢰 불가, SQI 생략')
        return rpeaks, 0

    keep = np.ones(len(rpeaks), dtype=bool)
    keep[idx[bad]] = False
    return rpeaks[keep], int(bad.sum())


def correct_rpeak_artifacts(rpeaks, sfreq, dev_ms=250.0, local_window=11,
                            max_insert=2, nni_min=300.0, nni_max=2000.0):
    """R-peak 시각(ms) 수준의 아티팩트 보정 (Kubios threshold 방식 유사).

    국소 중앙값(local_window-beat rolling median) 대비 |RR - median| > dev_ms 인
    interval 을 아티팩트로 보고, 유형별로 보정한다:
      - extra  : 짧은 interval 쌍의 합 ~ median      -> 가운데 가짜 피크 삭제
      - missed : RR ~ k*median (k-1 <= max_insert)   -> 놓친 피크 삽입
      - ectopic: 반대 부호 short-long 쌍의 합이 정상 2박자 -> 피크 재배치

    보정할 수 없는 구간(장시간 노이즈, 센서 dropout 등)은 valid 마스크에서
    제외하기만 하고 가짜 비트를 만들어내지 않는다. max_insert 를 제한하는
    이유도 같다: 검출기가 길게 놓친 구간을 등간격 가짜 비트로 채우면 RMSSD 가
    실제보다 작아진다.

    Returns
    -------
    t_ms : (n_peaks,) 보정된 R-peak 시각 (ms, float)
    valid : (n_peaks-1,) bool, 사용 가능한 NN interval 마스크
    n_fixed : 보정 횟수
    """
    t = np.asarray(rpeaks, dtype=float) / float(sfreq) * 1000.0
    if len(t) < local_window:
        rr = np.diff(t)
        return t, (rr >= nni_min) & (rr <= nni_max), 0

    banned = []          # 보정 실패로 제외 확정된 interval 의 시작 시각(ms)
    n_fixed = 0
    for _ in range(2 * len(t)):
        rr = np.diff(t)
        med = median_filter(rr, size=local_window, mode='nearest')
        dev = rr - med
        bad = np.abs(dev) > dev_ms
        for b in banned:
            j = int(np.searchsorted(t, b))
            if j < len(bad) and abs(t[j] - b) < 1e-6:
                bad[j] = False
        if not bad.any():
            break
        i = int(np.argmax(np.abs(dev) * bad))   # 가장 심한 아티팩트부터 처리

        fixed = False
        if dev[i] < 0:
            # extra beat: 가짜 피크가 정상 interval 을 둘로 쪼갠 경우 -> 병합
            if i + 1 < len(rr) and abs(rr[i] + rr[i + 1] - med[i]) <= dev_ms:
                t = np.delete(t, i + 1)
                fixed = True
            elif i >= 1 and abs(rr[i - 1] + rr[i] - med[i]) <= dev_ms:
                t = np.delete(t, i)
                fixed = True
        else:
            # missed beat: 피크를 놓쳐 interval 이 k배로 늘어난 경우 -> 삽입
            k = int(round(rr[i] / max(med[i], 1.0)))
            if 2 <= k <= max_insert + 1 and abs(rr[i] / k - med[i]) <= dev_ms:
                t = np.insert(t, i + 1, t[i] + np.arange(1, k) * rr[i] / k)
                fixed = True
        if not fixed:
            # ectopic/misaligned: 반대 부호 쌍만 재배치 (정상 RSA 는 같은 부호로
            # 천천히 변하므로 건드리지 않는다)
            for a, b2 in ((i, i + 1), (i - 1, i)):
                if 0 <= a and b2 < len(rr) and dev[a] * dev[b2] < 0:
                    pair_sum = rr[a] + rr[b2]
                    if abs(pair_sum - (med[a] + med[b2])) <= dev_ms:
                        t[a + 1] = t[a] + pair_sum / 2.0
                        fixed = True
                        break

        if fixed:
            n_fixed += 1
        else:
            banned.append(t[i])

    # 2차 보정: 보상성 휴지를 동반한 조기박동(PAC/PVC) 쌍.
    # 고정 임계값(dev_ms)보다 작은 편차라도, 국소 변동성(MAD) 대비 뚜렷한
    # 반대부호 short-long 쌍이면서 합이 정상 2박자에 가까우면 이소성 박동으로
    # 보고 재배치한다. 정상 RSA 는 여러 박자에 걸쳐 천천히 변하므로 한 박자
    # 만에 반대부호로 튀는 쌍과 구분된다.
    for _ in range(len(t)):
        rr = np.diff(t)
        med = median_filter(rr, size=local_window, mode='nearest')
        dev = rr - med
        scale = 1.4826 * median_filter(np.abs(dev), size=91, mode='nearest')
        thr = np.clip(3.5 * scale, 100.0, dev_ms)
        cand = np.where((dev[:-1] * dev[1:] < 0) &
                        (np.abs(dev[:-1]) > thr[:-1]) &
                        (np.abs(dev[1:]) > thr[1:]))[0]
        cand = [i for i in cand
                if abs(rr[i] + rr[i + 1] - (med[i] + med[i + 1])) <= dev_ms]
        if not cand:
            break
        i = max(cand, key=lambda j: abs(dev[j]) + abs(dev[j + 1]))
        t[i + 1] = t[i] + (rr[i] + rr[i + 1]) / 2.0
        n_fixed += 1

    rr = np.diff(t)
    med = median_filter(rr, size=local_window, mode='nearest')
    valid = (np.abs(rr - med) <= dev_ms) & (rr >= nni_min) & (rr <= nni_max)
    return t, valid, n_fixed


class ECGFeatureExtractor:
    def __init__(
            self,
            data_path,
            save_path,
            sfreq=125,
            age=18,
            sex='male'
    ):
        self.data_path = data_path
        self.save_path = save_path
        self.sfreq = sfreq
        self.age = age
        self.sex= sex

        data = pd.read_csv(data_path)

        trigger = data.iloc[:, 1]
        filtered_trigger = np.where(trigger > 0)[0]
        self.rows = data.shape[0]
        ecg = data.iloc[:, 0]

        self.filtered_trigger = filtered_trigger
        self.ecg = ecg

    def get_image_encoder(self, tmp_name):
        import base64
        with open(tmp_name, 'rb') as f:
            im_bytes = f.read()
        im_b64 = base64.b64encode(im_bytes).decode("utf8")
        return im_b64

    def extract(self):
        n_phases = len(self.filtered_trigger)
        nni, rmssd = self.whole()

        baseline_hrv, baseline_psd = self.baseline()
        baseline_hrv.update({
            'psd': baseline_psd,
            'heart_rate': self.get_image_encoder(os.path.join(self.save_path, 'fig1_Baseline.png')),
            'comparison': self.get_image_encoder(os.path.join(self.save_path, 'fig2_Baseline.png')),
        })

        if n_phases >= 3:
            stimulation1_hrv, stimulation1_psd = self.stimulation1()
            stimulation1_hrv.update({
                'psd': stimulation1_psd,
                'heart_rate': self.get_image_encoder(os.path.join(self.save_path, 'fig1_Stimulation1.png')),
                'comparison': self.get_image_encoder(os.path.join(self.save_path, 'fig2_Stimulation1.png')),
            })
            recovery1_hrv, recovery1_psd = self.recovery1()
            recovery1_hrv.update({
                'psd': recovery1_psd,
                'heart_rate': self.get_image_encoder(os.path.join(self.save_path, 'fig1_Recovery1.png')),
                'comparison': self.get_image_encoder(os.path.join(self.save_path, 'fig2_Recovery1.png')),
            })
        else:
            stimulation1_hrv = {}
            recovery1_hrv = {}

        if n_phases >= 5:
            stimulation2_hrv, stimulation2_psd = self.stimulation2()
            stimulation2_hrv.update({
                'psd': stimulation2_psd,
                'heart_rate': self.get_image_encoder(os.path.join(self.save_path, 'fig1_Stimulation2.png')),
                'comparison': self.get_image_encoder(os.path.join(self.save_path, 'fig2_Stimulation2.png')),
            })
            recovery2_hrv, recovery2_psd = self.recovery2()
            recovery2_hrv.update({
                'psd': recovery2_psd,
                'heart_rate': self.get_image_encoder(os.path.join(self.save_path, 'fig1_Recovery2.png')),
                'comparison': self.get_image_encoder(os.path.join(self.save_path, 'fig2_Recovery2.png')),
            })
        else:
            stimulation2_hrv = {}
            recovery2_hrv = {}

        sample = {
            'nni': nni, 'rmssd': rmssd,
            'baseline': baseline_hrv,
            'stimulation1': stimulation1_hrv,
            'recovery1': recovery1_hrv,
            'stimulation2': stimulation2_hrv,
            'recovery2': recovery2_hrv
        }

        self.filtered_trigger //= 7500
        return sample, self.filtered_trigger.tolist()

    # baseline-stimulation1  부분만 feature extract 해서 저장
    def baseline(self):
        print('baseline')
        end_idx = self.filtered_trigger[1] if len(self.filtered_trigger) > 1 else len(self.ecg)
        baseline_ecg = self.ecg[:end_idx]
        return self.feature_extract(baseline_ecg, phase='Baseline')

    # stimulation1-recovery1 부분만 feature extract 해서 저장
    def stimulation1(self):
        print('stimulation1')
        stimulation1_ecg = self.ecg[self.filtered_trigger[1]:self.filtered_trigger[2]]
        return self.feature_extract(stimulation1_ecg, phase='Stimulation1')

    # recovery1-stimulation2 부분만 feature extract 해서 저장
    def recovery1(self):
        print('recovery1')
        end_idx = self.filtered_trigger[3] if len(self.filtered_trigger) > 3 else len(self.ecg)
        recovery1_ecg = self.ecg[self.filtered_trigger[2]:end_idx]
        return self.feature_extract(recovery1_ecg, phase='Recovery1')

    # stimulation2-recovery2  부분만 feature extract 해서 저장
    def stimulation2(self):
        print('stimulation2')
        stimulation2_ecg = self.ecg[self.filtered_trigger[3]:self.filtered_trigger[4]]
        return self.feature_extract(stimulation2_ecg, phase='Stimulation2')

    # recovery2-end  부분만 feature extract 해서 저장
    def recovery2(self):
        recovery2_ecg = self.ecg[self.filtered_trigger[4]:]
        return self.feature_extract(recovery2_ecg, phase='Recovery2')

    def whole(self):
        _, filtered_ecg, rpeaks = biosppy.signals.ecg.ecg(self.ecg, show=False, sampling_rate=self.sfreq)[:3]

        # 노이즈 위 가짜 피크 제거 -> R-peak 시각 보정 + 아티팩트 interval 제외
        rpeaks, n_noise = remove_noise_beats(filtered_ecg, rpeaks, self.sfreq)
        t_ms, valid, n_fixed = correct_rpeak_artifacts(rpeaks, self.sfreq)
        nni_all = np.diff(t_ms)
        self.whole_nni = nni_all[valid].tolist()
        print(f'[whole] beats={len(rpeaks)}, noise_removed={n_noise}, fixed={n_fixed}, '
              f'excluded={100 * (1 - valid.mean()):.2f}%')

        # 보정된 R-peak 위치 (샘플 단위, 윈도우 분할용)
        rpeaks_corr = t_ms * self.sfreq / 1000.0

        # sliding RMSSD
        window_size = int(self.sfreq * 300)  # 5 min
        step_size = int(self.sfreq * 10)  # 10 sec

        start_idx, end_idx = 0, window_size
        trigger_idx = 0
        rmssd = []
        trigger_list = []

        signal_len = len(self.ecg)  # or len(filtered_ecg) if same length

        while end_idx <= signal_len:
            # trigger가 처음 들어간 시점 탐지
            if (trigger_idx < len(self.filtered_trigger)) and (self.filtered_trigger[trigger_idx] <= start_idx):
                trigger_idx += 1
                trigger_list.append(trigger_idx)
            else:
                trigger_list.append(0)

            # pick corrected rpeaks inside this window
            left = np.searchsorted(rpeaks_corr, start_idx, side='left')
            right = np.searchsorted(rpeaks_corr, end_idx, side='left')

            # RMSSD from window intervals (피크 left..right-1 사이의 interval 은
            # nni_all[left:right-1] 이고, valid 한 것만 사용)
            if right - left >= 3:
                nni_win = nni_all[left:right - 1][valid[left:right - 1]]

                if len(nni_win) >= 2:
                    rmssd_val = np.sqrt(np.mean(np.diff(nni_win) ** 2))
                else:
                    rmssd_val = np.nan
            else:
                rmssd_val = np.nan

            rmssd.append(rmssd_val)

            start_idx += step_size
            end_idx += step_size

        # original code tried to overwrite last trigger entry
        if len(trigger_list) > 0:
            trigger_list[-1] = trigger_idx

        return self.whole_nni, rmssd

    def feature_extract(self, ecg, whole=False, phase=''):
        # R-peak 검출: biosppy 기본 파이프라인 (FIR 0.67-45Hz bandpass +
        # Hamilton segmenter + local-max 보정). 이전에 쓰던
        # mne.preprocessing.find_ecg_events 는 MEG/EEG 심장 아티팩트 탐지용이라
        # 노이즈 구간에서 비트를 통째로 놓쳐 RMSSD 가 크게 부풀었다.
        _, filtered_ecg, rpeaks = biosppy.signals.ecg.ecg(ecg, show=False, sampling_rate=self.sfreq)[:3]

        # 노이즈 위 가짜 피크 제거 -> R-peak 시각 보정 + 아티팩트 interval 제외
        # (고정 400-1500ms 컷 대체: 국소 중앙값 기반이라 심박수와 무관하게
        # 놓친/가짜/이소성 비트를 잡는다)
        rpeaks, n_noise = remove_noise_beats(filtered_ecg, rpeaks, self.sfreq)
        t_ms, valid, n_fixed = correct_rpeak_artifacts(rpeaks, self.sfreq)
        nni = np.diff(t_ms)[valid]
        print(f'[{phase}] beats={len(rpeaks)}, noise_removed={n_noise}, fixed={n_fixed}, '
              f'excluded={100 * (1 - valid.mean()):.2f}%')

        if whole is False:
            params = ['sdnn', 'rmssd', 'sdsd', 'fft_ratio', 'pnn50']
            fig = heart_rate_heatplot(nni=nni, age=int(self.age), gender=str(self.sex), show=False)
            fig[0].savefig(os.path.join(self.save_path, f'fig1_{phase}.png'))
            plt.close('all')
            _, frequency, power = fd.welch_psd(nni=nni, show=False, mode='dev')

            idx = np.where(frequency < 0.4)[0]
            self.frequency = frequency[idx]
            self.power = power[idx]

            if phase == 'Baseline':
                self.baseline_nni = nni
                radar_chart(
                    nni=nni, comparison_nni=self.whole_nni[len(nni):], parameters=params, legend=True,
                    reference_label='Baseline', comparison_label='Rest',
                    save_path=os.path.join(self.save_path, f'fig2_{phase}.png')
                )
                plt.close('all')
            else:
                radar_chart(
                    nni=nni, comparison_nni=self.baseline_nni, parameters=params, legend=True,
                    reference_label=phase, comparison_label='Baseline',
                    save_path=os.path.join(self.save_path, f'fig2_{phase}.png')
                )
                plt.close('all')

        rmssd = td.rmssd(nni=nni)['rmssd']
        sdnn = td.sdnn(nni=nni)['sdnn']
        sdsd = td.sdsd(nni=nni)['sdsd']
        nn50 = td.nn50(nni=nni)['nn50']
        pnn50 = td.nn50(nni=nni)['pnn50']
        tri_index = td.triangular_index(nni=nni, show=False)['tri_index']

        fd_hrv, _, _ = fd.welch_psd(nni=nni, show=False, mode='dev')
        vlf_rel_power, lf_rel_power, hf_rel_power = fd_hrv['fft_rel']
        lh_ratio = fd_hrv["fft_ratio"]
        norm_lf = fd_hrv["fft_norm"][0]
        norm_hf = fd_hrv["fft_norm"][1]
        plt.close('all')

        data = {
            'sdnn': sdnn,
            'rmssd': rmssd,
            'sdsd': sdsd,
            'nn50': nn50,
            'pnn50': pnn50,
            'tri_index': tri_index,
            'vlf_rel_power': vlf_rel_power,
            'lf_rel_power': lf_rel_power,
            'hf_rel_power': hf_rel_power,
            'lh_ratio': lh_ratio,
            'norm_lf': norm_lf,
            'norm_hf': norm_hf,
        }

        if whole is False:
            psd_data = {
                'frequency': list(self.frequency),
                'power': list(self.power)
            }
            return data, psd_data

        else:
            return data


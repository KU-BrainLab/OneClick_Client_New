# -*- coding:utf-8 -*-
import os
import copy
import mne
import warnings
import matplotlib.cbook

import numpy as np

from scipy import signal
from scipy.signal import welch
from scipy.integrate import simps
from mne.preprocessing import ICA
from brainflow.data_filter import DataFilter
from mne_connectivity import spectral_connectivity_epochs, degree
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)


class FAA(object):
    def __init__(
            self,
            data_path,
            sfreq=125,
            filter_order=3,
            low_cut=0.5,
            high_cut=40,
            channel_names=None,
            channel_types=None,
            psd_relative=True,
    ):
        self.data_path = data_path
        self.sfreq = sfreq
        self.filter_order = filter_order
        self.low_cut = low_cut
        self.high_cut = high_cut
        self.psd_relative = psd_relative

        if channel_names is None:
            self.ch_names = ['Fp1', 'F7', 'F3', 'T3', 'C3', 'Cz', 'P3', 'O1', 'Fp2', 'F4', 'F8', 'C4', 'T4', 'P4', 'O2']
            self.ch_types = ['eeg'] * 15

        else:
            self.ch_names = channel_names
            self.ch_types = channel_types

        # 데이터 구조 = EEG (0-15열), ECG (16열), Trigger (-1열)
        data = DataFilter.read_file(data_path)

        # Trigger = {0: 기본, 1이상: 사용자 지정 trigger 신호}
        trigger = data[-1, :]
        self.filtered_trigger = np.where(trigger > 0)[0]  # 사용자 지정 trigger 신호 추출

        # EEG 신호 처리
        self.eeg_data = data[1:16, self.filtered_trigger[0]:] / 1e6  # 시작-끝 데이터 추출
        b, a = signal.butter(self.filter_order, [self.low_cut, self.high_cut], btype="bandpass", fs=self.sfreq)
        filtered_eeg_data = signal.filtfilt(b, a, self.eeg_data, axis=1)
        filtered_eeg_data = self.make_mne_class(filtered_eeg_data)
        filtered_eeg_data.plot()
        self.filtered_eeg = filtered_eeg_data.get_data()

        # 총 실험 시간 출력 (기기 결함 있을 시, 실험 시간이 비정상적으로 출력됨)
        experiment_time_s = self.filtered_eeg.shape[1] / sfreq  # seconds
        experiment_time_m = int(experiment_time_s / 60)  # minutes
        print(f'Total experiments time = {experiment_time_m} minutes')

        check_trigger = self.filtered_trigger.tolist() + [self.filtered_eeg.shape[1]]
        protocol, exp_time = [], []
        for idx in range(len(check_trigger) - 1):
            duration = (check_trigger[idx + 1] - check_trigger[idx]) // 7500
            exp_time.append(duration)
            if idx == 0:
                print(f'Baseline  {duration} min')
                # protocol.append('Baseline')
            elif idx % 2 == 1:
                print(f'Stimulation{(idx // 2) + 1}  {duration} min')
                # protocol.append(f'Eye-Open')
            else:
                print(f'Recovery{(idx // 2)}  {duration} min')
                # protocol.append(f'Eye-Closed')

        self.protocol = protocol
        self.exp_time = exp_time
        self.check_trigger = check_trigger
        self.check_trigger[0] = 0

    def make_mne_class(self, signal):
        # create EEG information
        eeg_info = mne.create_info(ch_names=self.ch_names, sfreq=self.sfreq, ch_types="eeg")
        signal = mne.io.RawArray(signal, info=eeg_info, verbose=False)

        # set montage
        montage = mne.channels.make_standard_montage('standard_1020')
        signal.set_montage(montage)

        return signal

    def psd_analysis(self, signal, low, high):
        # Welch's method를 사용하여 PSD 계산
        frequencies, psd = welch(signal, self.sfreq, axis=1)

        idx_band = np.logical_and(frequencies >= low, frequencies < high)
        bp = simps(psd[:, idx_band], dx=frequencies[1] - frequencies[0])

        if self.psd_relative:
            sum = 0
            for level, (l, h) in enumerate([(0.5, 4), (4, 8), (8, 13), (13, 30)]):
                idx = np.logical_and(frequencies >= l, frequencies < h)
                if level == 0:
                    sum = simps(psd[:, idx], dx=frequencies[1] - frequencies[0])
                else:
                    sum += simps(psd[:, idx], dx=frequencies[1] - frequencies[0])
            bp /= sum

        return bp

    def get_faa_by_section(self):
        """
        각 구간(Baseline, Stimulation1, Recovery1, ...)별로 FAA (alpha band)를 계산하여 출력
        FAA = log(Right alpha power) - log(Left alpha power)
        """
        alpha_band = (8, 13)
        left_channels = ['Fp1', 'F7', 'F3']
        right_channels = ['Fp2', 'F4', 'F8']

        left_indices = [self.ch_names.index(ch) for ch in left_channels]
        right_indices = [self.ch_names.index(ch) for ch in right_channels]

        section_faa = {}

        for i in range(len(self.check_trigger) - 1):
            start = self.check_trigger[i]
            end = self.check_trigger[i + 1]
            label = self.protocol[i]

            # 해당 구간의 EEG 데이터 추출
            print(self.filtered_eeg.shape, start, end)
            segment_left = self.filtered_eeg[left_indices, start:end]
            segment_right = self.filtered_eeg[right_indices, start:end]

            # alpha band power 계산
            left_psd = self.psd_analysis(segment_left, *alpha_band)
            right_psd = self.psd_analysis(segment_right, *alpha_band)

            # FAA 계산
            left_alpha = np.mean(np.log(left_psd + 1e-12))
            right_alpha = np.mean(np.log(right_psd + 1e-12))
            faa = right_alpha - left_alpha

            section_faa[label] = faa
            print(f'{label}: FAA = {faa:.4f} (Right {right_alpha:.4f} - Left {left_alpha:.4f})')
        return section_faa


def get_psd_analysis(data_, sfreq, low, high):
    frequencies, psd = welch(data_, sfreq, axis=1)

    idx_band = np.logical_and(frequencies >= low, frequencies < high)
    bp = simps(psd[:, idx_band], dx=frequencies[1] - frequencies[0])

    if True:
        sum_ = 0
        for level, (l, h) in enumerate([(0.5, 4), (4, 8), (8, 13), (13, 30)]):
            idx = np.logical_and(frequencies >= l, frequencies < h)
            if level == 0:
                sum_ = simps(psd[:, idx], dx=frequencies[1] - frequencies[0])
            else:
                sum_ += simps(psd[:, idx], dx=frequencies[1] - frequencies[0])
        bp /= sum_
    return bp


def get_faa(data_, ch_names, sfreq):
    alpha_band = (8, 13)
    l_channels, r_channels = ['Fp1', 'F7', 'F3'], ['Fp2', 'F4', 'F8']
    l_indices, r_indices = [ch_names.index(ch) for ch in l_channels], \
                           [ch_names.index(ch) for ch in r_channels]

    data_ = data_.get_data()
    data_ = copy.deepcopy(data_)
    segment_l, segment_r = data_[:, l_indices, :], data_[:, r_indices, :]
    total_l_alpha, total_r_alpha = [], []
    for sl, sr in zip(segment_l, segment_r):
        l_alpha = get_psd_analysis(sl, sfreq, low=8, high=13)
        r_alpha = get_psd_analysis(sr, sfreq, low=8, high=13)
        total_l_alpha.append(l_alpha)
        total_r_alpha.append(r_alpha)

    total_l_alpha, total_r_alpha = np.array(total_l_alpha), np.array(total_r_alpha)
    total_l_alpha, total_r_alpha = np.mean(np.log(total_l_alpha + 1e-12)), np.mean(np.log(total_r_alpha + 1e-12))
    faa = total_r_alpha - total_l_alpha
    return faa


def plot_faa_on_brain(faa_value, title):
    # 이미지 불러오기
    brain_img_path = './test.png'
    img = mpimg.imread(brain_img_path)
    height, width, _ = img.shape

    # 좌/우 전두엽 위치 (이미지 좌표 기준으로 수동 지정)
    left_coord = (int(width * 0.35), int(height * 0.25))  # Fp1~F3 근처
    right_coord = (int(width * 0.62), int(height * 0.25))  # Fp2~F4 근처

    # 컬러 매핑 설정
    cmap = plt.get_cmap('bwr')
    norm = plt.Normalize(-0.5, 0.5)

    left_color = cmap(norm(-faa_value))  # FAA < 0: 좌측 우세
    right_color = cmap(norm(faa_value))  # FAA > 0: 우측 우세

    # 시각화
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    ax.set_title(f"{title}: {faa_value:.2f}", fontsize=15)

    # 좌/우 영역 색상 표시
    ax.scatter(*left_coord, color=left_color, s=4500, label='Left frontal', alpha=0.45)
    ax.scatter(*right_coord, color=right_color, s=4500,  label='Right frontal', alpha=0.45)

    # 색상바
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('FAA (Right - Left)', fontsize=12)

    ax.axis('off')
    plt.tight_layout()
    plt.show()


def epoching(data, epoch_duration=30., artifact_rejection=False):
    epoched = mne.make_fixed_length_epochs(raw=data, duration=epoch_duration,
                                           reject_by_annotation=artifact_rejection)
    return epoched


if __name__ == "__main__":
    import uuid
    data_path = r'D:\Project\OneClick_Client\database\ch\2024-05-28-1430.csv'
    data = DataFilter.read_file(data_path)
    eeg_data = data[1:16, :] / 1e6

    ch_list = ['Fp1', 'F7', 'F3', 'T3', 'C3', 'Cz', 'P3', 'O1', 'Fp2', 'F4', 'F8', 'C4', 'T4', 'P4', 'O2']
    exp_names = ['baseline', 'stimulation1', 'recovery1', 'stimulation2', 'recovery2']
    sfreq, rfreq = 125, 100
    eeg_info = mne.create_info(ch_names=ch_list, sfreq=sfreq, ch_types="eeg")
    data_ = mne.io.RawArray(eeg_data, info=eeg_info)
    # data_.drop_channels(['O1', 'O2'])
    filter_data = data_.copy().filter(l_freq=1, h_freq=60.)

    montage = mne.channels.make_standard_montage('standard_1020')
    filter_data.set_montage(montage)

    # independent component analysis (artifact remove)
    ica = ICA(n_components=15 - 2, max_iter='auto', random_state=7)
    ica.fit(filter_data)

    muscle_idx_auto, scores = ica.find_bads_muscle(filter_data)
    ica.apply(filter_data, exclude=muscle_idx_auto)

    # resampling (125 Hz -> 100 Hz)
    filter_data = filter_data.copy().resample(rfreq)
    epoch_data = epoching(data=filter_data)
    myuuid = uuid.uuid4()

    epoch_data = epoch_data.get_data()

    sample_size = epoch_data.shape[0]
    step = sample_size // 5
    for i in range(5):
        start, end = i * step, (i+1) * step
        sample = epoch_data[start: end, ...]
        raw = mne.EpochsArray(sample, info=eeg_info)
        faa = get_faa(raw, ch_names=ch_list, sfreq=sfreq)
        plot_faa_on_brain(faa, exp_names[i])


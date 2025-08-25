import os
import biosppy
import warnings
import matplotlib.cbook

import numpy as np
import pandas as pd
import seaborn as sns
import pyhrv.tools as tools
import matplotlib.pyplot as plt

from brainflow.data_filter import DataFilter

warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)


class CleanUpECG:
    def __init__(self, data_path, sfreq=125):
        self.data_path = data_path
        self.sfreq = sfreq

        # 데이터 구조 = EEG (0-15열), ECG (16열), Trigger (-1열)
        print(data_path)
        data = DataFilter.read_file(data_path)

        # Trigger = {0: 기본, 1이상: 사용자 지정 trigger 신호}
        trigger = data[-1, :]
        filtered_trigger = np.where(trigger > 0)[0]  # 사용자 지정 trigger 신호 추출

        # ECG 신호 처리
        ecg = data[16, filtered_trigger[0]:] / 1e6  # 시작-끝 데이터 추출
        t, filtered_ecg, rpeaks = biosppy.signals.ecg.ecg(ecg, show=False, sampling_rate=sfreq)[:3]

        self.t = t  # (signal_length, ) 각 점들을 "초" 단위로 변환
        self.trigger = trigger[filtered_trigger[0]:]  # 트리거의 시작 부분부터 가져오기
        self.filtered_ecg = filtered_ecg  # (signal_length, )
        self.filtered_trigger = filtered_trigger  # (signal_length, )
        self.rpeaks = rpeaks  # (reaks, ) R-peak location indices. ex) [77, 183. 291 ...]
        self.nni = tools.nn_intervals(t[rpeaks])  # (reapks-1, ) R-peak 사이의 시간

        # nni 값이 이상할 때 search 하기 위한 부분
        self.low_search = True
        self.high_search = True

        # 총 실험 시간 출력 (기기 결함 있을 시, 실험 시간이 비정상적으로 출력됨)
        experiment_time_s = len(filtered_ecg) / sfreq  # seconds
        experiment_time_m = int(experiment_time_s / 60)  # minutes
        print(f'Total experiments time = {experiment_time_m} minutes')

    def outlier_detection(self):
        self.plot_nni()

        while (self.low_search is True) or (self.high_search is True):
            if self.low_search is True:
                lowest = np.min(self.nni)
                low_range = input(f" Low outlier를 제거 하시겠습니까? [ 최소 nni 값: {lowest} ] ( x / low 범위 입력 ):  ")
                if low_range in ['n', 'x', 'ㅌ', 'ㅜ']:
                    self.low_search = False
                    if low_range == "n":
                        self.low_search = True

                # nni가 low_range 보다 작은 부분 detection
                else:
                    low_nni_idx = np.where(self.nni < int(low_range))[0]
                    self.check_outlier_nni(outlier_nni_idx=low_nni_idx.tolist())
                    self.plot_nni()

            if self.high_search is True:
                highest = np.max(self.nni)
                high_range = input(f" High outlier를 제거 하시겠습니까? [ 최대 nni 값: {highest} ] ( x / high 범위 입력 ):  ")
                if high_range in ['n', 'x', 'X', 'ㅌ', 'ㅜ']:
                    self.high_search = False
                    if high_range == "n":
                        self.high_search = True

                # nni가 high_range 보다 작은 부분 detection
                else:
                    high_nni_idx = np.where(self.nni > int(high_range))[0]
                    self.check_outlier_nni(outlier_nni_idx=high_nni_idx.tolist())
                    self.plot_nni()

    def check_outlier_nni(self, outlier_nni_idx):
        # 제거할 idx 리스트
        remove_idx = []

        if len(outlier_nni_idx) == 0:
            print(' Outlier 값이 없습니다. ')

        else:
            while len(outlier_nni_idx) != 0:
                # outlier nni 들을 graph 상에 빨간색으로 표시하기 위한 리스트
                rpeak_red_idx = [outlier_nni_idx[0], outlier_nni_idx[0] + 1]

                # out of index 예외 처리
                start_idx = 0 if outlier_nni_idx[0] - 4 < 0 else outlier_nni_idx[0] - 4
                end_idx = len(self.nni) - 1 if outlier_nni_idx[0] + 20 > len(self.nni) - 1 else outlier_nni_idx[0] + 20

                # 처리한 outlier nni를 리스트에서 삭제
                del outlier_nni_idx[0]

                # 25개의 nni 내 에서의 outlier 추출
                del_list = []
                if len(outlier_nni_idx) > 0:
                    for i in range(len(outlier_nni_idx)):
                        if end_idx > outlier_nni_idx[i]:
                            rpeak_red_idx += [outlier_nni_idx[i], outlier_nni_idx[i] + 1]
                            del_list.append(i)
                        else:
                            break

                for index in sorted(del_list, reverse=True):
                    del outlier_nni_idx[index]

                rpeak_red_idx = set(rpeak_red_idx)

                # outlier nni 들을 graph로 출력
                plt.figure(figsize=(15, 5))
                txt_position = np.max(self.filtered_ecg[self.rpeaks[start_idx]: self.rpeaks[end_idx + 1]])

                for n, i in enumerate(range(start_idx, end_idx + 2)):
                    plt.axvline(self.rpeaks[i] / self.sfreq, color='green')
                    plt.text(self.rpeaks[i] / self.sfreq, txt_position * 1.02, str(n))
                for i in rpeak_red_idx:
                    plt.axvline(self.rpeaks[i] / self.sfreq, color='red')

                sns.lineplot(
                    x=[i / self.sfreq for i in range(self.rpeaks[start_idx], self.rpeaks[end_idx + 1])],
                    y=self.filtered_ecg[self.rpeaks[start_idx]: self.rpeaks[end_idx + 1]]
                )
                plt.show()

                # outlier nni를 제거할 idx들을 추가하는 부분
                remove_input = input('  제거할 범위 입력하세요 (없을 경우 x 입력) [ex 1,10,12,15]: ')
                if remove_input in ['x', 'ㅌ', 'X', ""]:
                    pass
                else:
                    remove_input = remove_input.split(',')
                    remove_input = [int(i) for i in remove_input]

                    for i in remove_input:
                        remove_idx.append(self.rpeaks[start_idx + i])

            # outlier nni 제거 부분
            trigger_adjustment_value = 0  # 신호 제거 시 trigger가 제거 되지 않게 해주는 변수
            signal_indexes, trigger_indexes = [], []
            for i in range(int(len(remove_idx) / 2)):
                for j in range(remove_idx[i * 2], remove_idx[i * 2 + 1]):
                    signal_indexes.append(j)
                    if j in self.filtered_trigger:
                        trigger_adjustment_value += 1
                    trigger_indexes.append(j + trigger_adjustment_value)
            self.filtered_ecg = np.delete(self.filtered_ecg, signal_indexes)
            self.trigger = np.delete(self.trigger, trigger_indexes)
            self.filtered_trigger = np.where(self.trigger > 0)[0]

            self.t, self.filtered_signal, self.rpeaks = biosppy.signals.ecg.ecg(
                self.filtered_ecg,
                show=False,
                sampling_rate=self.sfreq,
            )[:3]
            self.nni = tools.nn_intervals(self.t[self.rpeaks])

    def plot_nni(self):
        plt.figure(figsize=(12, 5))
        plt.plot(self.nni)
        plt.show()

    def save_filtered_data(self, save_path):
        file_name = os.path.split(self.data_path)[-1]
        df_data = {
            'ecg': self.filtered_ecg,
            'trigger': self.trigger
        }
        df = pd.DataFrame(df_data)
        df.to_csv(os.path.join(save_path, file_name), index=False)
        return os.path.join(save_path, file_name)


if __name__ == "__main__":
    data = r'E:\VNS_code\database\eeg_ecg'
    data_path = os.path.join(data, 'LYS', '2023-06-05-0826_noise.csv')
    a = CleanUpECG(data_path)
    a.outlier_detection()
    a.save_filtered_data(r'E:\vns\cleaned_ecg_data')

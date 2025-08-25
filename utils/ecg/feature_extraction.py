import os
import biosppy
import warnings
import matplotlib.cbook

import numpy as np
import pandas as pd
import seaborn as sns
import pyhrv.tools as tools
import matplotlib.pyplot as plt
import pyhrv.time_domain as td
import pyhrv.frequency_domain as fd

sns.set()
sns.set_palette("muted")
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)


class ECGFeatureExtractor:
    def __init__(
            self,
            data_path,
            save_path,
            sfreq=125,
    ):
        self.data_path = data_path
        self.save_path = save_path
        self.sfreq = sfreq

        data = pd.read_csv(data_path)
        trigger = data.iloc[:, 1]
        filtered_trigger = np.where(trigger > 0)[0]
        ecg = data.iloc[:, 0]

        self.filtered_trigger = filtered_trigger
        self.ecg = ecg

        self.aa = int((self.filtered_trigger[2] - self.filtered_trigger[1]) / 4)

    # baseline-stimulation1  부분만 feature extract 해서 저장
    def baseline(self):
        baseline_ecg = self.ecg[:self.filtered_trigger[1]]
        self.feature_extract(baseline_ecg, 'baseline.csv')

    # stimulation1-recovery1 부분만 feature extract 해서 저장
    def stimulation1(self):
        # stimulation1_ecg = self.ecg[self.filtered_trigger[1]:self.filtered_trigger[2]]
        stimulation1_ecg = self.ecg[self.filtered_trigger[1]:self.filtered_trigger[1] + self.aa]
        self.feature_extract(stimulation1_ecg, 'stimulation1.csv')

    # recovery1-stimulation2 부분만 feature extract 해서 저장
    def recovery1(self):
        # recovery1_ecg = self.ecg[self.filtered_trigger[2]:self.filtered_trigger[3]]
        recovery1_ecg = self.ecg[self.filtered_trigger[1] + self.aa:self.filtered_trigger[1] + self.aa * 2]
        self.feature_extract(recovery1_ecg, 'recovery1.csv')

    # stimulation2-recovery2  부분만 feature extract 해서 저장
    def stimulation2(self):
        # stimulation2_ecg = self.ecg[self.filtered_trigger[3]:self.filtered_trigger[4]]
        stimulation2_ecg = self.ecg[self.filtered_trigger[1] + self.aa * 2:self.filtered_trigger[1] + self.aa * 3]
        self.feature_extract(stimulation2_ecg, 'stimulation2.csv')

    # recovery2-end  부분만 feature extract 해서 저장
    def recovery2(self):
        # recovery2_ecg = self.ecg[self.filtered_trigger[4]:]
        recovery2_ecg = self.ecg[self.filtered_trigger[1] + self.aa * 3:]
        self.feature_extract(recovery2_ecg, 'recovery2.csv')

    # baseline 라인 앞의 5분만 feature extract 해서 저장
    def baseline_five_minutes(self):
        five_minutes_ecg = self.ecg[:300 * self.sfreq]
        self.feature_extract(five_minutes_ecg, 'baseline_5min.csv')

    # baseline 이후 부터 마지막 recovery 시작 할 때까지 feature extract 해서 저장
    def stimulation_all(self):
        stimulation_ecg = self.ecg[self.filtered_trigger[1]:self.filtered_trigger[-1]]
        self.feature_extract(stimulation_ecg, 'stimulation_all.csv')

    # 마지막 stimulation 이후 부터 recovery 끝날 때까지 feature extract 해서 저장
    def recovery_all(self):
        recovery_ecg = self.ecg[self.filtered_trigger[-1]:]
        self.feature_extract(recovery_ecg, 'recovery_all.csv')

    # window 5분으로 10초씩 이동하면서 feature extraction 하고 그래프 저장
    def moving_window_five_minutes(self, protocol=None):
        feature = ['RMSSD', 'LH Ratio', 'LF', 'HF']
        if protocol is None:
            protocol = ['baseline', 'stimulation', 'recovery']

        start_idx, end_idx = 0, self.sfreq * 300
        trigger_idx = 0
        rmssd, lh_ratio, lf, hf, trigger_list = [], [], [], [], []

        while True:
            ecg = self.ecg[start_idx:end_idx]
            if len(ecg) < self.sfreq * 300:
                trigger_list[-1] = trigger_idx
                break

            # trigger가 처음 들어간 시점 탐지
            if (trigger_idx < len(self.filtered_trigger)) and (self.filtered_trigger[trigger_idx] <= start_idx):
                trigger_idx += 1
                trigger_list.append(trigger_idx)
            else:
                trigger_list.append(0)

            df = self.feature_extract(ecg, file_name=None)
            rmssd.append(df['rmssd'].item())
            lh_ratio.append(df['lh_ratio'].item())
            lf.append(df['norm_lf'].item())
            hf.append(df['norm_hf'].item())

            start_idx += self.sfreq * 10
            end_idx += self.sfreq * 10

        trigger = np.where(np.array(trigger_list) > 0)[0]

        for n, param in enumerate([rmssd, lh_ratio, lf, hf]):
            plt.figure(figsize=(12, 5))

            for phase in range(protocol.__len__()):
                start_time = (trigger[phase] * 10) / 60
                end_time = (trigger[phase + 1] * 10) / 60
                if phase == protocol.__len__() - 1:
                    end_time += 5

                if phase == 0:
                    data = param[trigger[phase]:trigger[phase + 1] - 14]
                elif phase == protocol.__len__() - 1:
                    data = param[trigger[phase] - 15:trigger[phase + 1]]
                else:
                    data = param[trigger[phase] - 15:trigger[phase + 1] - 14]

                x = np.linspace(start_time, end_time, len(data))
                sns.lineplot(x=x, y=data, label=protocol[phase], linewidth=3)

            plt.title(feature[n], fontdict={'size': 17})
            plt.legend(fontsize=13)
            plt.tick_params(axis='y', direction="inout", labelsize=13, length=10, pad=7)
            plt.tick_params(axis='x', direction="inout", labelsize=13, length=10, pad=7)
            plt.xlabel('Time (m)', fontdict={'size': 15})
            plt.ylabel(feature[n], fontdict={'size': 15})
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_path, f'{feature[n]}'), dpi=300)

    def feature_extract(self, ecg, file_name):
        t, _, rpeaks = biosppy.signals.ecg.ecg(ecg, show=False, sampling_rate=self.sfreq)[:3]
        nni = tools.nn_intervals(t[rpeaks])

        sdnn = td.sdnn(nni=nni)['sdnn']
        rmssd = td.rmssd(nni=nni)['rmssd']
        sdsd = td.sdsd(nni=nni)['sdsd']
        nn50 = td.nn50(nni=nni)['nn50']
        pnn50 = td.nn50(nni=nni)['pnn50']
        tri_index = td.triangular_index(nni=nni, show=False)['tri_index']

        fd_hrv, _, _ = fd.welch_psd(nni=nni, show=False, mode='dev')
        vlf_rel_power, lf_rel_power, hf_rel_power = fd_hrv['fft_rel']
        lh_ratio = fd_hrv["fft_ratio"]
        norm_lf = fd_hrv["fft_norm"][0]
        norm_hf = fd_hrv["fft_norm"][1]

        name, date = os.path.split(self.data_path)
        name = os.path.split(name)[-1]
        date = date.split('-')[:3]

        df_data = {
            'subject': name,
            'date': f'{date[0]}-{date[1]}-{date[2]}',
            'sdnn': [sdnn],
            'rmssd': [rmssd],
            'sdsd': [sdsd],
            'nn50': [nn50],
            'pnn50': [pnn50],
            'tri_index': [tri_index],
            'vlf_rel_power': [vlf_rel_power],
            'lf_rel_power': [lf_rel_power],
            'hf_rel_power': [hf_rel_power],
            'lh_ratio': [lh_ratio],
            'norm_lf': [norm_lf],
            'norm_hf': [norm_hf],
        }
        df = pd.DataFrame(df_data)

        if file_name:
            if os.path.exists(os.path.join(self.save_path, file_name)):
                df_total = pd.read_csv(os.path.join(self.save_path, file_name))
                df = pd.concat([df_total, df], ignore_index=True)

            df.to_csv(os.path.join(self.save_path, file_name), index=False)

        else:
            return df

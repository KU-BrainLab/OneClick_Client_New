import os
import pyhrv
import biosppy

import warnings
import matplotlib.cbook
import matplotlib as mpl

import numpy as np
import pandas as pd
import seaborn as sns
import pyhrv.tools as tools
import matplotlib.pyplot as plt
import pyhrv.time_domain as td
import pyhrv.frequency_domain as fd
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
        nni, rmssd = self.whole()
        baseline_hrv, baseline_psd = self.baseline()
        baseline_hrv.update({
            'psd': baseline_psd,
            'heart_rate': self.get_image_encoder(os.path.join(self.save_path, 'fig1_Baseline.png')),
            'comparison': self.get_image_encoder(os.path.join(self.save_path, 'fig2_Baseline.png')),
        })
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

        sample = {
            'nni': nni, 'rmssd': rmssd,
            'baseline': baseline_hrv,
            'stimulation1': stimulation1_hrv,
            'recovery1': recovery1_hrv,
            'stimulation2': stimulation2_hrv,
            'recovery2': recovery2_hrv
        }

        return sample

    # baseline-stimulation1  부분만 feature extract 해서 저장
    def baseline(self):
        print('baseline')
        baseline_ecg = self.ecg[:self.filtered_trigger[1]]
        return self.feature_extract(baseline_ecg, phase='Baseline')

    # stimulation1-recovery1 부분만 feature extract 해서 저장
    def stimulation1(self):
        print('stimulation1')
        stimulation1_ecg = self.ecg[self.filtered_trigger[1]:self.filtered_trigger[2]]
        return self.feature_extract(stimulation1_ecg, phase='Stimulation1')

    # recovery1-stimulation2 부분만 feature extract 해서 저장
    def recovery1(self):
        print('recovery1')
        recovery1_ecg = self.ecg[self.filtered_trigger[2]:self.filtered_trigger[3]]
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
        t, _, rpeaks = biosppy.signals.ecg.ecg(self.ecg, show=False, sampling_rate=self.sfreq)[:3]
        nni = tools.nn_intervals(t[rpeaks])
        filtered_arr = nni[(nni >= 400) & (nni <= 1500)]
        self.whole_nni = filtered_arr.tolist()



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

            df = self.feature_extract(ecg, whole=True)
            rmssd.append(df['rmssd'])

            start_idx += self.sfreq * 10
            end_idx += self.sfreq * 10

        return self.whole_nni, rmssd

    def feature_extract(self, ecg, whole=False, phase=''):
        t, _, rpeaks = biosppy.signals.ecg.ecg(ecg, show=False, sampling_rate=self.sfreq)[:3]
        nni = tools.nn_intervals(t[rpeaks])

        # nni = np.clip(nni, 400, 1200) # + np.random.randint(1,15, size=nni.shape)
        nni = nni[(nni >= 400) & (nni <= 1500)]

        if whole is False:
            params = ['sdnn', 'rmssd', 'sdsd', 'fft_ratio', 'pnn50']            
            fig = heart_rate_heatplot(nni=nni, age=int(self.age), gender=str(self.sex), show=False)
            fig[0].savefig(os.path.join(self.save_path, f'fig1_{phase}.png'))
            plt.close('all')
            _, frequency, power = fd.welch_psd(rpeaks=t[rpeaks], show=False, mode='dev')

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

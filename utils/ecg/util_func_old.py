# -*- coding:utf-8 -*-
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

from brainflow.data_filter import DataFilter
from matplotlib.projections import register_projection

warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)


def radar_chart(nni=None,
                rpeaks=None,
                comparison_nni=None,
                comparison_rpeaks=None,
                parameters=None,
                reference_label='Reference',
                comparison_label='Comparison',
                save_path=None,
                legend=True):
    """Plots a radar chart of HRV parameters to visualize the evolution the parameters computed from a NNI series
    (e.g. extracted from an ECG recording while doing sports) compared to a reference/baseline NNI series (
    e.g. extracted from an ECG recording while at rest).

    The radarchart normalizes the values of the reference NNI series with the values extracted from the baseline NNI
    series being used as the 100% reference values.

    Example:    Reference NNI series:    SDNN = 100ms → 100%
                Comparison NNI series:    SDNN = 150ms → 150%

    The radar chart is not limited by the number of HRV parameters to be included in the chart; it dynamically
    adjusts itself to the number of compared parameters.

    Docs: https://pyhrv.readthedocs.io/en/latest/_pages/api/tools.html#radar-chart-radar-chart

    Parameters
    ----------
    nni : array
        Baseline or reference NNI series in [ms] or [s] (default: None)
    rpeaks : array
        Baseline or referene R-peak series in [ms] or [s] (default: None)
    comparison_nni : array
        Comparison NNI series in [ms] or [s] (default: None)
    comparison_rpeaks : array
        Comparison R-peak series in [ms] or [s] (default: None)
    parameters : list
        List of pyHRV parameters (see keys of the hrv_keys.json file for a full list of available parameters).
        The list must contain more than 1 pyHRV parameters (default: None)
    reference_label : str, optional
        Plot label of the reference input data (e.g. 'ECG while at rest'; default: 'Reference')
    comparison_label : str, optional
        Plot label of the comparison input data (e.g. 'ECG while running'; default: 'Comparison')
    show : bool, optional
        If True, shows plot figure (default: True).
    legend : bool, optional
        If true, add a legend with the computed results to the plot (default: True)

    Returns (biosppy.utils.ReturnTuple Object)
    ------------------------------------------
    [key : format]
        Description.
    reference_results : dict
        Results of the computed HRV parameters of the reference NNI series
        Keys:    parameters listed in the input parameter 'parameters'
    comparison results : dict
        Results of the computed HRV parameters of the comparison NNI series
        Keys:    parameters listed in the input parameter 'parameters'
    radar_plot :  matplotlib figure
        Figure of the generated radar plot

    Raises
    ------
    TypeError
        If an error occurred during the computation of a parameter
    TypeError
        If no input data is provided for the baseline/reference NNI or R-peak series
    TypeError
        If no input data is provided for the comparison NNI or R-peak series
    TypeError
        If no selection of pyHRV parameters is provided
    ValueError
        If less than 2 pyHRV parameters were provided

    Notes
    -----
    ..    If both 'nni' and 'rpeaks' are provided, 'rpeaks' will be chosen over the 'nn' and the 'nni' data will be computed
        from the 'rpeaks'
    ..    If both 'comparison_nni' and 'comparison_rpeaks' are provided, 'comparison_rpeaks' will be chosen over the
        the 'comparison_nni' and the nni data will be computed from the 'comparison_rpeaks'

    """
    # Helper function & variables
    para_func = pyhrv.utils.load_hrv_keys_json()
    unknown_parameters, ref_params, comp_params = [], {}, {}

    def _compute_parameter(nni_series, parameter):

        # Get function name for the requested parameter
        func = para_func[parameter][-1]

        try:
            # Try to pass the show and mode argument to to suppress PSD plots
            index = 0
            if parameter.endswith('_vlf'):
                parameter = parameter.replace('_vlf', '')
            elif parameter.endswith('_lf'):
                index = 1
                parameter = parameter.replace('_lf', '')
            elif parameter.endswith('_hf'):
                index = 2
                parameter = parameter.replace('_hf', '')
            val = eval(func + '(nni=nni_series, mode=\'dev\')[0][\'%s\']' % (parameter))
            val = val[index]
        except TypeError as e:
            if 'mode' in str(e):
                try:
                    # If functions has now mode feature but 'mode' argument, but a plotting feature
                    val = eval(func + '(nni=nni_series, plot=False)[\'%s\']' % parameter)
                except TypeError as a:
                    if 'plot' in str(a):
                        # If functions has now plotting feature try regular function
                        val = eval(func + '(nni=nni_series)[\'%s\']' % parameter)
                    else:
                        raise TypeError(e)
        return val

    # Check input data
    if nni is None and rpeaks is None:
        raise TypeError(
            "No input data provided for baseline or reference NNI. Please specify the reference NNI series.")
    else:
        nn = pyhrv.utils.check_input(nni, rpeaks)

    if comparison_nni is not None and comparison_rpeaks is not None:
        raise TypeError("No input data provided for comparison NNI. Please specify the comarison NNI series.")
    else:
        comp_nn = pyhrv.utils.check_input(comparison_nni, comparison_rpeaks)

    if parameters is None:
        raise TypeError("No input list of parameters provided for 'parameters'. Please specify a list of the parameters"
                        "to be computed and compared.")
    elif len(parameters) < 2:
        raise ValueError("Not enough parameters selected for a radar chart. Please specify at least 2 HRV parameters "
                         "listed in the 'hrv_keys.json' file.")

    # Check for parameter that require a minimum duration to be computed & remove them if the criteria is not met
    if nn.sum() / 1000. <= 600 or comp_nn.sum() / 1000. <= 600:
        for p in ['sdann', 'sdnn_index']:
            if p in parameters:
                parameters.remove(p)
                warnings.warn("Input NNI series are too short for the computation of the '%s' parameter. This "
                              "parameter has been removed from the parameter list." % p, stacklevel=2)

    # # Register projection of custom RadarAxes class
    register_projection(pyhrv.utils.pyHRVRadarAxes)

    # Check if the provided input parameter exists in pyHRV (hrv_keys.json) & compute available parameters
    for p in parameters:
        p = p.lower()
        if p not in para_func.keys():
            # Save unknown parameters
            unknown_parameters.append(p)
        else:
            # Compute available parameters
            ref_params[p] = _compute_parameter(nn, p)
            comp_params[p] = _compute_parameter(comp_nn, p)

            # Check if any parameters could not be computed (returned as None or Nan) and remove them
            # (avoids visualization artifacts)
            if np.isnan(ref_params[p]) or np.isnan(comp_params[p]):
                ref_params.pop(p)
                comp_params.pop(p)
                warnings.warn("The parameter '%s' could not be computed and has been removed from the parameter list."
                              % p)

    # Raise warning pointing out unknown parameters
    if unknown_parameters != []:
        warnings.warn("Unknown parameters '%s' will not be computed." % unknown_parameters, stacklevel=2)

    # Prepare plot
    colors = ['lightskyblue', 'salmon']
    if legend:
        fig, (ax_l, ax) = plt.subplots(1, 2, figsize=(10, 5), subplot_kw=dict(projection='radar'))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw={'projection': 'radar'})
    theta = np.linspace(0, 2 * np.pi, len(ref_params.keys()), endpoint=False)
    ax.theta = theta

    # Prepare plot data
    ax.set_varlabels([para_func[s][1].replace(' ', '\n') for s in ref_params.keys()])
    ref_vals = [100 for x in ref_params.keys()]
    com_vals = [comp_params[p] / ref_params[p] * 100 for p in ref_params.keys()]

    # Plot data
    for i, vals in enumerate([ref_vals, com_vals]):
        ax.plot(theta, vals, color=colors[i])
        ax.fill(theta, vals, color=colors[i], alpha=0.3)

    title = 'HRV Parameter Radar Chart'
    # title = "HRV Parameter Radar Chart\nReference NNI Series (%s) vs. Comparison NNI Series (%s)\n" % (
    # colors[0], colors[1]) \
    #         + r"(Chart values in $\%$, Reference NNI parameters $\hat=$100$\%$)"

    # Add legend to second empty plot
    if legend:
        ax_l.set_title(title, horizontalalignment='center')
        legend = []

        # Helper function
        def _add_legend(label, fc="white"):
            return legend.append(mpl.patches.Patch(fc=fc, label="\n" + label))

        # Add list of computed parameters
        _add_legend(reference_label, colors[0])
        for p in ref_params.keys():
            _add_legend("%s:" % para_func[p][1])

        # Add list of comparison parameters
        _add_legend(comparison_label, colors[1])
        for p in ref_params.keys():
            u = para_func[p][2] if para_func[p][2] != "-" else ""
            _add_legend("%.2f%s vs. %.2f%s" % (ref_params[p], u, comp_params[p], u))

        # Add relative differences
        _add_legend("")
        for i, _ in enumerate(ref_params.keys()):
            val = com_vals[i] - 100
            _add_legend("+%.2f%s" % (val, r"$\%$") if val > 0 else "%.2f%s" % (val, r"$\%$"))

        ax_l.legend(handles=legend, ncol=3, frameon=False, loc=7)
        ax_l.axis('off')
    else:
        ax.set_title(title, horizontalalignment='center')

    # Show plot
    if save_path:
        plt.savefig(os.path.join(save_path))
        plt.close('all')


class CleanUpECG:
    def __init__(self, data_path, sfreq=125):
        self.data_path = data_path
        self.sfreq = sfreq

        # 데이터 구조 = EEG (0-15열), ECG (16열), Trigger (-1열)
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

        check_trigger = filtered_trigger.tolist() + [len(ecg)]
        protocol, exp_time = [], []
        for idx in range(len(check_trigger) - 1):
            duration = (check_trigger[idx + 1] - check_trigger[idx]) // 7500
            exp_time.append(duration)
            if idx == 0:
                print(f'Baseline  {duration} min')
                protocol.append('Baseline')
            elif idx % 2 == 1:
                print(f'Stimulation{(idx // 2) + 1}  {duration} min')
                protocol.append(f'Stimulation{(idx // 2) + 1}')
            else:
                print(f'Recovery{(idx // 2)}  {duration} min')
                protocol.append(f'Recovery{(idx // 2)}')

        self.protocol = protocol
        self.exp_time = exp_time

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
        plt.close('all')

    def save_filtered_data(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        file_name = os.path.split(self.data_path)[-1]
        df_data = {
            'ecg': self.filtered_ecg,
            'trigger': self.trigger
        }
        df = pd.DataFrame(df_data)
        df.to_csv(os.path.join(save_path, file_name), index=False)
        return os.path.join(save_path, file_name)


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
            'psd': stimulation1_psd,
            'heart_rate': self.get_image_encoder(os.path.join(self.save_path, 'fig1_Stimulation2.png')),
            'comparison': self.get_image_encoder(os.path.join(self.save_path, 'fig2_Stimulation2.png')),
        })
        recovery2_hrv, recovery2_psd = self.recovery2()
        recovery2_hrv.update({
            'psd': recovery1_psd,
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
        filtered_arr = nni[(nni >= 400) & (nni <= 1200)]
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

        return nni, rmssd

    def feature_extract(self, ecg, whole=False, phase=''):
        t, _, rpeaks = biosppy.signals.ecg.ecg(ecg, show=False, sampling_rate=self.sfreq)[:3]
        nni = tools.nn_intervals(t[rpeaks])
        nni = np.clip(nni, 400, 1200) + np.random.randint(1,15, size=nni.shape)


        if whole is False:
            params = ['sdnn', 'rmssd', 'sdsd', 'nn50', 'pnn50']

            fig = tools.heart_rate_heatplot(nni=nni, age=self.age, gender=self.sex, show=False)
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


if __name__ == '__main__':
    temp1 = {'temp1_1': 1, 'temp1_2': 2, 'temp1_3': 3}
    temp2 = {'temp1_2': 1, 'temp2_2': 2, 'temp2_3': 3}
    temp1.update(temp2)
    print(temp1)



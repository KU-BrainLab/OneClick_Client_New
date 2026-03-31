# -*- coding:utf-8 -*-
import mne
import uuid
import matplotlib as mpl
from brainflow.data_filter import DataFilter
import networkx as nx
# from mne.viz import circular_layout
from scipy.signal import welch
import pandas as pd
from mne.preprocessing import ICA

######################## Custom Import ########################
from .eeg_analysis.faa import get_frontal_alpha_asymmetry 
from .eeg_analysis.psd import get_psd_analysis
from .eeg_analysis.psd_diff import get_psd_diff_analysis
from .eeg_analysis.psd_topography import get_psd_topography
from .eeg_analysis.brain_connectivity import get_brain_connectivity
from .eeg_analysis.brain_connectivity_diff import get_diff_brain_connectivity
from .eeg_analysis.fronto_limbic import get_fronto_limbic_analysis
from .eeg_analysis.sleep_staging import get_sleep_staging
from .eeg_analysis.brain_spectrogram import get_brain_spectrogram
from .eeg_analysis.brain_delta_power_topo import get_brain_delta_power_topo
from .eeg_analysis.brain_delta_fc import get_brain_delta_connectivity
###############################################################

mpl.use('TkAgg')
mpl.rcParams['figure.constrained_layout.use'] = True


def epoching(data, epoch_duration=30., artifact_rejection=False):
    epoched = mne.make_fixed_length_epochs(raw=data, duration=epoch_duration,
                                           reject_by_annotation=artifact_rejection)
    return epoched

def main_analysis(path, trigger):
    data = DataFilter.read_file(path)
    eeg_data = data[1:16, :] / 1e6

    ch_list = ['Fp1', 'F7', 'F3', 'T3', 'C3', 'Cz', 'P3', 'O1', 'Fp2', 'F4', 'F8', 'C4', 'T4', 'P4', 'O2']
    sfreq, rfreq = 125, 100
    eeg_info = mne.create_info(ch_names=ch_list, sfreq=sfreq, ch_types="eeg")
    data_ = mne.io.RawArray(eeg_data, info=eeg_info)
    data_.drop_channels(['O1', 'O2'])
    filter_data = data_.copy().filter(l_freq=0.5, h_freq=60.)

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

    #End experiment time
    trigger.append(int((epoch_data.get_data().shape[0] * (epoch_data.get_data().shape[2] / 100)) / 60))


    brain_topograhpy = get_psd_topography(epoch_data, myuuid, trigger)
    brain_conn_coh = get_brain_connectivity(epoch_data, myuuid, 'coh', trigger)
    brain_conn_plv = get_brain_connectivity(epoch_data, myuuid, 'plv', trigger)
    brain_psd = get_psd_analysis(epoch_data)
    brain_fronto_limbic = get_fronto_limbic_analysis(filter_data, myuuid)
    brain_sleep_stage = get_sleep_staging(epoch_data, ch_list)
    brain_psd_diff = get_psd_diff_analysis(epoch_data, myuuid)
    brain_conn_diff_coh = get_diff_brain_connectivity(epoch_data, myuuid, 'coh', trigger)
    brain_conn_diff_plv = get_diff_brain_connectivity(epoch_data, myuuid, 'plv', trigger)
    brain_faa = get_frontal_alpha_asymmetry(epoch_data, myuuid, trigger)
    brain_spectrogram = get_brain_spectrogram(filter_data, myuuid, trigger)
    brain_delta_power_topo = get_brain_delta_power_topo(epoch_data, myuuid, trigger, brain_sleep_stage['sleep_stage'])
    brain_delta_connectivity = get_brain_delta_connectivity(epoch_data, myuuid, trigger, brain_sleep_stage['sleep_stage'])

    brain_sleep = {key: brain_sleep_stage[key] for key in ['sleep_stage', 'sleep_stage_prob']}
    brain_report_summary = brain_sleep_stage['sleep_summary']

    # diff1~4 조립: 서버 eeg_diff_obj_save 구조에 맞게 매핑
    _DIFF_KEYS = ['diff1', 'diff2', 'diff3', 'diff4']
    _PHASE_PAIRS = [
        ('stimulation 1', 'baseline'),
        ('recovery 1',    'stimulation 1'),
        ('stimulation 2', 'recovery 1'),
        ('recovery 2',    'stimulation 2'),
    ]
    _BANDS_LOWER = {
        'Delta': 'delta', 'Theta': 'theta', 'Alpha': 'alpha',
        'Sigma': 'sigma', 'Beta': 'beta',   'Gamma': 'gamma',
    }
    _SS_MAP = {'W': 'wake', 'N1': 'n1', 'N2': 'n2', 'N3': 'n3', 'REM': 'rem'}

    diffs = []
    for i, (targ, ref) in enumerate(_PHASE_PAIRS):
        key = _DIFF_KEYS[i]
        pair_fn = f'{targ.replace(" ", "_")}_vs_{ref.replace(" ", "_")}'
        d = {}
        for band, band_lower in _BANDS_LOWER.items():
            # base: 기존 PSD diff figure (BsrsrChartWidget / Bsrsr1ChartWidget에서 사용)
            d[f'topography_{band_lower}']    = brain_psd_diff.get(key, {}).get(band_lower, '')
            d[f'connectivity_{band_lower}']  = brain_conn_diff_coh.get(key, {}).get(band_lower, '')
            d[f'connectivity2_{band_lower}'] = brain_conn_diff_plv.get(key, {}).get(band_lower, '')
            # 수면단계별 (없으면 'all' fallback)
            topo_all = brain_delta_power_topo.get(f'topography_{pair_fn}_{band}_all', '')
            conn_all = brain_delta_connectivity.get(f'connectivity_{pair_fn}_{band}_all', '')
            for ss, ss_key in _SS_MAP.items():
                d[f'topography_{band_lower}_{ss_key}']   = brain_delta_power_topo.get(f'topography_{pair_fn}_{band}_{ss}', topo_all)
                d[f'connectivity_{band_lower}_{ss_key}'] = brain_delta_connectivity.get(f'connectivity_{pair_fn}_{band}_{ss}', conn_all)
        diffs.append(d)
    diff1, diff2, diff3, diff4 = diffs

    return {
        'topography': brain_topograhpy,
        'connectivity': brain_conn_coh,
        'connectivity2': brain_conn_plv,
        'psd_result': brain_psd,
        'frontal_limbic': brain_fronto_limbic,
        'sleep_stage': brain_sleep,
        'sleep_report' : brain_report_summary,
        'topography_diff': brain_psd_diff,
        'connectivity_diff': brain_conn_diff_coh,
        'connectivity2_diff': brain_conn_diff_plv,
        'diff1': diff1,
        'diff2': diff2,
        'diff3': diff3,
        'diff4': diff4,
        'psd_spectrogram': brain_spectrogram,
        'faa' : brain_faa
    }
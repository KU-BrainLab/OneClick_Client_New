# -*- coding:utf-8 -*-
import mne
import uuid
import matplotlib as mpl
from brainflow.data_filter import DataFilter
import networkx as nx
# from mne.viz import circular_layout
from scipy.signal import welch
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
###############################################################

mpl.use('TkAgg')
mpl.rcParams['figure.constrained_layout.use'] = True


def epoching(data, epoch_duration=30., artifact_rejection=False):
    epoched = mne.make_fixed_length_epochs(raw=data, duration=epoch_duration,
                                           reject_by_annotation=artifact_rejection)
    return epoched

def main_analysis(path):
    data = DataFilter.read_file(path)
    eeg_data = data[1:16, :] / 1e6

    ch_list = ['Fp1', 'F7', 'F3', 'T3', 'C3', 'Cz', 'P3', 'O1', 'Fp2', 'F4', 'F8', 'C4', 'T4', 'P4', 'O2']
    sfreq, rfreq = 125, 100
    eeg_info = mne.create_info(ch_names=ch_list, sfreq=sfreq, ch_types="eeg")
    data_ = mne.io.RawArray(eeg_data, info=eeg_info)
    data_.drop_channels(['O1', 'O2'])
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


    brain_topograhpy = get_psd_topography(epoch_data, myuuid)
    brain_conn_coh = get_brain_connectivity(epoch_data, myuuid, 'coh')
    brain_conn_plv = get_brain_connectivity(epoch_data, myuuid, 'plv')
    brain_psd = get_psd_analysis(epoch_data)
    brain_fronto_limbic = get_fronto_limbic_analysis(filter_data, myuuid)
    brain_sleep_stage = get_sleep_staging(epoch_data, ch_list)
    brain_psd_diff = get_psd_diff_analysis(epoch_data, myuuid)
    brain_conn_diff_coh = get_diff_brain_connectivity(epoch_data, myuuid, 'coh')
    brain_conn_diff_plv = get_diff_brain_connectivity(epoch_data, myuuid, 'plv')
    brain_faa = get_frontal_alpha_asymmetry(epoch_data, myuuid)
    
    brain_sleep = {key: brain_sleep_stage[key] for key in ['sleep_stage', 'sleep_stage_prob']}
    #brain_report_summary = brain_sleep_stage['sleep_summary']

    return {
        'topography': brain_topograhpy,
        'connectivity': brain_conn_coh,
        'connectivity2': brain_conn_plv,
        'psd_result': brain_psd,
        'frontal_limbic': brain_fronto_limbic,
        'sleep_stage': brain_sleep,
        'topography_diff': brain_psd_diff,
        'connectivity_diff': brain_conn_diff_coh,
        'connectivity2_diff': brain_conn_diff_plv,
        'faa' : brain_faa
    }
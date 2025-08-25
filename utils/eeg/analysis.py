# -*- coding:utf-8 -*-
import mne
import os
import uuid
import copy
import torch
import base64
import numpy as np
import matplotlib as mpl
from brainflow.data_filter import DataFilter
import networkx as nx
from scipy.signal import hilbert
# from mne.viz import circular_layout
from mne.preprocessing import ICA
from mne_connectivity import spectral_connectivity_epochs
from mne_connectivity.viz import plot_connectivity_circle
import matplotlib.pyplot as plt
from neuronet.model import NeuroNet, NeuroNetEncoderWrapper, Classifier
from scipy.integrate import simpson
import cv2

mpl.use('TkAgg')
mpl.rcParams['figure.constrained_layout.use'] = True


def center_crop(img, dim):
    width, height = img.shape[1], img.shape[0]

    # process crop width and height for max available dimension
    crop_width = dim[0] if dim[0] < img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0]
    mid_x, mid_y = int(width / 2), int(height / 2)
    cw2, ch2 = int(crop_width / 2), int(crop_height / 2)
    crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    return crop_img


def epoching(data, epoch_duration=30., artifact_rejection=False):
    epoched = mne.make_fixed_length_epochs(raw=data, duration=epoch_duration,
                                           reject_by_annotation=artifact_rejection)
    return epoched


def get_psd_topography(epoch_data, uuid):
    from sklearn.preprocessing import StandardScaler
    epoch_data = copy.deepcopy(epoch_data)
    eeg_info = epoch_data.info
    epoch_data = epoch_data.get_data()
    sample_size = epoch_data.shape[0]
    step = sample_size // 5
    exp_names = ['baseline', 'stimulation1', 'recovery1', 'stimulation2', 'recovery2']
    bands = {'delta': [0, 4], 'theta': [4, 8], 'alpha': [8, 13], 'beta': [13, 30], 'gamma': [30, 40]}
    files = {exp_name: {band_name: None for band_name in bands.keys()} for exp_name in exp_names}

    # 1. [Power Spectrum Density]
    for i in range(5):
        start, end = i * step, (i+1) * step
        sample = epoch_data[start: end, ...]
        raw = mne.EpochsArray(sample, info=eeg_info)

        for band_name, band_range in bands.items():
            spectrum = raw.compute_psd(
                method='welch',
                fmin=band_range[0],
                fmax=band_range[1]
            )
            psds, freqs = spectrum.get_data(return_freqs=True)
            psds = 10 * np.log10(psds)
            psds_mean = psds.mean(axis=0)

            freq_res = freqs[1] - freqs[0]
            idx_band = np.logical_and(freqs >= band_range[0], freqs <= band_range[1])
            total_power = []
            for psds_mean_sample in psds_mean:
                bp = simpson(psds_mean_sample[idx_band], dx=freq_res)
                total_power.append(bp)
            total_power = np.array(total_power)
            scaler = StandardScaler()
            total_power = scaler.fit_transform(total_power.reshape(-1, 1))
            total_power = total_power.reshape(-1)

            fig, ax = plt.subplots(1, figsize=(3.5, 3.5))
            mne.viz.plot_topomap(total_power, eeg_info, ch_type='eeg',
                                 names=eeg_info['ch_names'], show=False,
                                 axes=ax)

            tmp_name = os.path.join('image', 'psd', '{}_{}_{}.jpg'.format(uuid, exp_names[i], band_name))
            tmp_name = os.path.abspath(tmp_name)
            fig.savefig(tmp_name)
            plt.close('all')
            plt.clf()

            with open(tmp_name, 'rb') as f:
                im_bytes = f.read()
            im_b64 = base64.b64encode(im_bytes).decode("utf8")
            files[exp_names[i]][band_name] = im_b64

    return files


def get_brain_connectivity(epoch_data, uuid):
    epoch_data = copy.deepcopy(epoch_data)
    eeg_info = epoch_data.info
    sfreq = eeg_info['sfreq']
    epoch_data = epoch_data.get_data()
    sample_size = epoch_data.shape[0]
    step = sample_size // 5
    exp_names = ['baseline', 'stimulation1', 'recovery1', 'stimulation2', 'recovery2']
    bands = {'delta': [0, 4], 'theta': [4, 8], 'alpha': [8, 13], 'beta': [13, 30], 'gamma': [30, 40]}
    files = {exp_name: {band_name: None for band_name in bands.keys()} for exp_name in exp_names}

    for i in range(5):
        start, end = i * step, (i+1) * step
        sample = epoch_data[start: end, ...]
        raw = mne.EpochsArray(sample, info=eeg_info)

        for band_name, band_range in bands.items():
            fc_method = 'coh'

            start, end = i * step, (i+1) * step
            sample = epoch_data[start: end, ...]
            raw = mne.EpochsArray(sample, info=eeg_info)
            con = spectral_connectivity_epochs(data=raw, names=raw.info['ch_names'], sfreq=sfreq,
                                               mode="multitaper",
                                               method=fc_method, fmin=band_range[0], fmax=band_range[1],
                                               faverage=True,
                                               mt_adaptive=True, n_jobs=1)
            conmat = con.get_data(output="dense")[:, :, 0]
            tmp_name = os.path.join('image', 'connectivity', '{}_{}_{}.jpg'.format(uuid, exp_names[i], band_name))
            tmp_name = os.path.abspath(tmp_name)

            fig, ax = plt.subplots(1, figsize=(5, 5), subplot_kw={'projection': 'polar'})
            plot_connectivity_circle(conmat, raw.info['ch_names'], n_lines=50,
                                     facecolor='white', textcolor='black',
                                     colormap='gray_r',
                                     ax=ax,
                                     colorbar=False,
                                     show=False,
                                     node_linewidth=1)
            fig.savefig(tmp_name)
            image = cv2.imread(tmp_name)
            crop_img = center_crop(image, (370, 370))
            cv2.imwrite(tmp_name, crop_img)

            plt.close('all')
            plt.clf()

            with open(tmp_name, 'rb') as f:
                im_bytes = f.read()
            im_b64 = base64.b64encode(im_bytes).decode("utf8")
            files[exp_names[i]][band_name] = im_b64
    plt.close('all')
    plt.clf()
    return files


def get_diff_brain_connectivity(epoch_data, uuid):
    def connecitivy(sample, band_range, eeg_info):
        fc_method = 'coh'
        raw = mne.EpochsArray(sample, info=eeg_info)
        con = spectral_connectivity_epochs(data=raw, names=raw.info['ch_names'], sfreq=sfreq,
                                           mode="multitaper",
                                           method=fc_method, fmin=band_range[0], fmax=band_range[1],
                                           faverage=True,
                                           mt_adaptive=True, n_jobs=1)
        conmat = con.get_data(output="dense")[:, :, 0]
        return conmat

    epoch_data = copy.deepcopy(epoch_data)
    eeg_info = epoch_data.info
    sfreq = eeg_info['sfreq']
    epoch_data = epoch_data.get_data()
    sample_size = epoch_data.shape[0]
    step = sample_size // 5
    exp_names = ['diff1', 'diff2', 'diff3', 'diff4']
    bands = {'delta': [0, 4], 'theta': [4, 8], 'alpha': [8, 13], 'beta': [13, 30], 'gamma': [30, 40]}
    files = {exp_name: {band_name: None for band_name in bands.keys()} for exp_name in exp_names}

    epochs = []
    for i in range(5):
        start, end = i * step, (i+1) * step
        sample = epoch_data[start: end, ...]
        epoch = mne.EpochsArray(sample, info=eeg_info)
        epochs.append(epoch)

    for diff_i, (i_epoch, j_epoch) in enumerate(zip(epochs[:-1], epochs[1:])):
        for band_name, band_range in bands.items():
            conmat_1 = connecitivy(i_epoch, band_range, eeg_info)
            conmat_2 = connecitivy(j_epoch, band_range, eeg_info)
            conmat_d = conmat_2 - conmat_1
            tmp_name = os.path.join('image', 'connectivity_diff', '{}_{}_{}.jpg'.format(uuid,
                                                                                        exp_names[diff_i], band_name))
            tmp_name = os.path.abspath(tmp_name)

            fig, ax = plt.subplots(1, figsize=(5, 5), subplot_kw={'projection': 'polar'})
            plot_connectivity_circle(conmat_d, i_epoch.info['ch_names'], n_lines=50,
                                     facecolor='white', textcolor='black',
                                     colormap='gray_r',
                                     ax=ax,
                                     colorbar=False,
                                     show=False,
                                     node_linewidth=1)
            fig.savefig(tmp_name)
            image = cv2.imread(tmp_name)
            crop_img = center_crop(image, (370, 370))
            cv2.imwrite(tmp_name, crop_img)

            plt.close('all')
            plt.clf()

            with open(tmp_name, 'rb') as f:
                im_bytes = f.read()
            im_b64 = base64.b64encode(im_bytes).decode("utf8")
            files[exp_names[diff_i]][band_name] = im_b64
    return files


def get_brain_connectivity2(epoch_data, uuid):
    epoch_data = copy.deepcopy(epoch_data)
    eeg_info = epoch_data.info
    sfreq = eeg_info['sfreq']
    epoch_data = epoch_data.get_data()
    sample_size = epoch_data.shape[0]
    step = sample_size // 5
    exp_names = ['baseline', 'stimulation1', 'recovery1', 'stimulation2', 'recovery2']
    bands = {'delta': [0, 4], 'theta': [4, 8], 'alpha': [8, 13], 'beta': [13, 30], 'gamma': [30, 40]}
    files = {exp_name: {band_name: None for band_name in bands.keys()} for exp_name in exp_names}

    for i in range(5):
        start, end = i * step, (i+1) * step
        sample = epoch_data[start: end, ...]
        raw = mne.EpochsArray(sample, info=eeg_info)

        for band_name, band_range in bands.items():
            fc_method = 'plv'

            start, end = i * step, (i+1) * step
            sample = epoch_data[start: end, ...]
            raw = mne.EpochsArray(sample, info=eeg_info)
            con = spectral_connectivity_epochs(data=raw, names=raw.info['ch_names'], sfreq=sfreq,
                                               mode="multitaper",
                                               method=fc_method, fmin=band_range[0], fmax=band_range[1],
                                               faverage=True,
                                               mt_adaptive=True, n_jobs=1)
            conmat = con.get_data(output="dense")[:, :, 0]
            tmp_name = os.path.join('image', 'connectivity2', '{}_{}_{}.jpg'.format(uuid, exp_names[i], band_name))
            tmp_name = os.path.abspath(tmp_name)

            fig, ax = plt.subplots(1, figsize=(5, 5), subplot_kw={'projection': 'polar'})
            plot_connectivity_circle(conmat, raw.info['ch_names'], n_lines=50,
                                     facecolor='white', textcolor='black',
                                     colormap='gray_r',
                                     ax=ax,
                                     colorbar=False,
                                     show=False,
                                     node_linewidth=1)
            fig.savefig(tmp_name)
            image = cv2.imread(tmp_name)
            crop_img = center_crop(image, (370, 370))
            cv2.imwrite(tmp_name, crop_img)

            plt.close('all')
            plt.clf()

            with open(tmp_name, 'rb') as f:
                im_bytes = f.read()
            im_b64 = base64.b64encode(im_bytes).decode("utf8")
            files[exp_names[i]][band_name] = im_b64
    plt.close('all')
    plt.clf()
    return files


def get_diff_brain_connectivity2(epoch_data, uuid):
    def connecitivy(sample, band_range, eeg_info):
        fc_method = 'plv'
        raw = mne.EpochsArray(sample, info=eeg_info)
        con = spectral_connectivity_epochs(data=raw, names=raw.info['ch_names'], sfreq=sfreq,
                                           mode="multitaper",
                                           method=fc_method, fmin=band_range[0], fmax=band_range[1],
                                           faverage=True,
                                           mt_adaptive=True, n_jobs=1)
        conmat = con.get_data(output="dense")[:, :, 0]
        return conmat

    epoch_data = copy.deepcopy(epoch_data)
    eeg_info = epoch_data.info
    sfreq = eeg_info['sfreq']
    epoch_data = epoch_data.get_data()
    sample_size = epoch_data.shape[0]
    step = sample_size // 5
    exp_names = ['diff1', 'diff2', 'diff3', 'diff4']
    bands = {'delta': [0, 4], 'theta': [4, 8], 'alpha': [8, 13], 'beta': [13, 30], 'gamma': [30, 40]}
    files = {exp_name: {band_name: None for band_name in bands.keys()} for exp_name in exp_names}

    epochs = []
    for i in range(5):
        start, end = i * step, (i+1) * step
        sample = epoch_data[start: end, ...]
        epoch = mne.EpochsArray(sample, info=eeg_info)
        epochs.append(epoch)

    for diff_i, (i_epoch, j_epoch) in enumerate(zip(epochs[:-1], epochs[1:])):
        for band_name, band_range in bands.items():
            conmat_1 = connecitivy(i_epoch, band_range, eeg_info)
            conmat_2 = connecitivy(j_epoch, band_range, eeg_info)
            conmat_d = conmat_2 - conmat_1
            tmp_name = os.path.join('image', 'connectivity_diff2', '{}_{}_{}.jpg'.format(uuid,
                                                                                         exp_names[diff_i], band_name))
            tmp_name = os.path.abspath(tmp_name)

            fig, ax = plt.subplots(1, figsize=(5, 5), subplot_kw={'projection': 'polar'})
            plot_connectivity_circle(conmat_d, i_epoch.info['ch_names'], n_lines=50,
                                     facecolor='white', textcolor='black',
                                     colormap='gray_r',
                                     ax=ax,
                                     colorbar=False,
                                     show=False,
                                     node_linewidth=1)
            fig.savefig(tmp_name)
            image = cv2.imread(tmp_name)
            crop_img = center_crop(image, (370, 370))
            cv2.imwrite(tmp_name, crop_img)

            plt.close('all')
            plt.clf()

            with open(tmp_name, 'rb') as f:
                im_bytes = f.read()
            im_b64 = base64.b64encode(im_bytes).decode("utf8")
            files[exp_names[diff_i]][band_name] = im_b64
    return files


def get_psd_analysis(epoch_data):
    def get_related_power(psds_, freqs_, freq_range):
        freq_res = freqs_[1] - freqs_[0]
        idx_band = np.logical_and(freqs_ >= freq_range[0], freqs_ <= freq_range[1])
        bp = simpson(psds_[idx_band], dx=freq_res)
        # bp /= simpson(psds_, dx=freq_res)
        full_band = np.logical_and(freqs_ >= 0, freqs_ <= 40)
        bp /= simpson(psds_[full_band], dx=freq_res)
        return bp

    epoch_data = copy.deepcopy(epoch_data)
    raw_spectrum = epoch_data.compute_psd(method='welch', fmin=0, fmax=40)
    mean_spectrum = raw_spectrum.average()
    psds, freqs = mean_spectrum.get_data(return_freqs=True)
    psds = 10 * np.log10(psds)
    psds_mean, psds_std = psds.mean(axis=0), psds.std(axis=0)

    p1 = get_related_power(psds_mean, freqs, freq_range=[0, 4])
    p2 = get_related_power(psds_mean, freqs, freq_range=[4, 8])
    p3 = get_related_power(psds_mean, freqs, freq_range=[8, 12])
    p4 = get_related_power(psds_mean, freqs, freq_range=[12, 30])
    p5 = get_related_power(psds_mean, freqs, freq_range=[30, 40])

    result = {
        'raw_psd': {'mean': psds_mean, 'std': psds_std, 'freqs': freqs},
        'related_psd': [p1, p2, p3, p4, p5],
        'region_psd': get_region_psd(epoch_data),
    }
    return result


def get_region_psd(epoch_data):
    def get_related_power(psds_, freqs_, freq_range):
        freq_res = freqs_[1] - freqs_[0]
        idx_band = np.logical_and(freqs_ >= freq_range[0], freqs_ <= freq_range[1])
        bp = simpson(psds_[idx_band], dx=freq_res)
        # bp /= simpson(psds_, dx=freq_res)
        full_band = np.logical_and(freqs_ >= 0, freqs_ <= 40)
        bp /= simpson(psds_[full_band], dx=freq_res)
        return bp

    # 4. [Left/Right Power Spectum Density]
    epoch_data = copy.deepcopy(epoch_data)
    ch_list = epoch_data.info['ch_names']
    sfreq = epoch_data.info['sfreq']
    bands = {'delta': [0, 4], 'theta': [4, 8], 'alpha': [8, 13], 'beta': [13, 30], 'gamma': [30, 40]}
    r_brain_region = ['Fp2', 'F8', 'F4', 'C4', 'T4', 'P4']
    l_brain_region = ['Fp1', 'F7', 'F3', 'C3', 'T3', 'P3']
    r_brain_region_idx = np.array([ch_list.index(ch_name) for ch_name in r_brain_region])
    l_brain_region_idx = np.array([ch_list.index(ch_name) for ch_name in l_brain_region])
    r_brain_eeg = mne.EpochsArray(epoch_data.get_data()[:, r_brain_region_idx, :],
                                  mne.create_info(ch_names=r_brain_region, sfreq=sfreq, ch_types="eeg"))
    l_brain_eeg = mne.EpochsArray(epoch_data.get_data()[:, l_brain_region_idx, :],
                                  mne.create_info(ch_names=l_brain_region, sfreq=sfreq, ch_types="eeg"))

    r_raw_spectrum = r_brain_eeg.compute_psd(method='welch', fmin=0, fmax=40)
    r_mean_spectrum = r_raw_spectrum.average()
    r_psds, freqs = r_mean_spectrum.get_data(return_freqs=True)
    r_psds = 10 * np.log10(r_psds)
    r_psds_mean = r_psds.mean(axis=0)

    r_p1 = get_related_power(r_psds_mean, freqs, freq_range=[0, 4])
    r_p2 = get_related_power(r_psds_mean, freqs, freq_range=[4, 8])
    r_p3 = get_related_power(r_psds_mean, freqs, freq_range=[8, 12])
    r_p4 = get_related_power(r_psds_mean, freqs, freq_range=[12, 30])
    r_p5 = get_related_power(r_psds_mean, freqs, freq_range=[30, 40])

    # r_all = r_p1 + r_p2 + r_p3 + r_p4 + r_p5
    # r_p1, r_p2, r_p3, r_p4, r_p5 = r_p1/r_all, r_p2/r_all, r_p3/r_all, r_p4/r_all, r_p5/r_all

    l_raw_spectrum = l_brain_eeg.compute_psd(method='welch', fmin=0, fmax=40)
    l_mean_spectrum = l_raw_spectrum.average()
    l_psds, freqs = l_mean_spectrum.get_data(return_freqs=True)
    l_psds = 10 * np.log10(l_psds)
    l_psds_mean = l_psds.mean(axis=0)

    l_p1 = get_related_power(l_psds_mean, freqs, freq_range=[0, 4])
    l_p2 = get_related_power(l_psds_mean, freqs, freq_range=[4, 8])
    l_p3 = get_related_power(l_psds_mean, freqs, freq_range=[8, 12])
    l_p4 = get_related_power(l_psds_mean, freqs, freq_range=[12, 30])
    l_p5 = get_related_power(l_psds_mean, freqs, freq_range=[30, 40])

    # l_all = l_p1 + l_p2 + l_p3 + l_p4 + l_p5
    # l_p1, l_p2, l_p3, l_p4, l_p5 = l_p1 / l_all, l_p2 / l_all, l_p3 / l_all, l_p4 / l_all, l_p5 / l_all

    result = {'right': [r_p1, r_p2, r_p3, r_p4, r_p5], 'left': [l_p1, l_p2, l_p3, l_p4, l_p5]}
    return result


def compute_plv(data):
    n_channels, n_samples = data.shape
    plv_matrix = np.zeros((n_channels, n_channels))
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            phase_diff = np.angle(hilbert(data[i])) - np.angle(hilbert(data[j]))
            plv = np.abs(np.sum(np.exp(1j * phase_diff)) / n_samples)
            plv_matrix[i, j] = plv
            plv_matrix[j, i] = plv
    return plv_matrix


def get_fronto_limbic_analysis(epoch_data, uuid):
    wanted_ch_list = ['F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'T3', 'T4']
    wanted_ch_indices = [2, 10, 1, 11, 0, 8, 3, 12]
    freq_bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (12, 30),
        'gamma': (30, 49)
    }

    epoch_data = copy.deepcopy(epoch_data)
    result = {}
    for band_name, band_freqs in freq_bands.items():
        filtered_data = copy.deepcopy(epoch_data)
        filtered_data = filtered_data.load_data().filter(l_freq=band_freqs[0], h_freq=band_freqs[1]).get_data()
        montage = mne.channels.make_standard_montage('standard_1020')
        info = mne.create_info(wanted_ch_list, sfreq=125, ch_types='eeg')
        info.set_montage(montage)

        matrix = compute_plv(filtered_data)

        fig, ax = plt.subplots(figsize=(10, 10))
        # Plot empty head layout instead of using plot_sensors and adjust the position of the channels
        pos = {ch: (loc[0], loc[1] - 0.025) for ch, loc in
               zip(info.ch_names, info.get_montage().get_positions()['ch_pos'].values())}
        pos_array = np.array(list(pos.values()))

        # Plot the head layout without markers
        mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='w')
        mne.viz.plot_topomap(np.zeros(len(wanted_ch_list)), pos_array,
                             show=False, axes=ax, mask=np.ones(len(wanted_ch_list)), mask_params=mask_params)
        G = nx.Graph()
        for idx, ch in enumerate(wanted_ch_list):
            G.add_node(ch, pos=pos[ch])

        for i in range(len(wanted_ch_list)):
            for j in range(i + 1, len(wanted_ch_list)):
                weight = matrix[i, j]
                G.add_edge(wanted_ch_list[i], wanted_ch_list[j], weight=weight)

        edges = G.edges(data=True)
        colors = [edge[2]['weight'] for edge in edges]
        nx.draw(G, pos, with_labels=False, node_size=100, node_color='gray', ax=ax, font_size=12, edge_color=colors,
                edge_cmap=plt.cm.viridis, edge_vmin=0, edge_vmax=1)

        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, orientation='horizontal', label='PLV')

        tmp_name = os.path.join('image', 'fronto_limbic', '{}_{}.jpg'.format(uuid, band_name))
        tmp_name = os.path.abspath(tmp_name)
        fig.savefig(tmp_name)

        with open(tmp_name, 'rb') as f:
            im_bytes = f.read()
        im_b64 = base64.b64encode(im_bytes).decode("utf8")
        result[band_name] = im_b64
        plt.close('all')
        plt.clf()
    return result


def get_sleep_staging(epoch_data, ch_list):
    epoch_data = copy.deepcopy(epoch_data)
    info = epoch_data.info
    epoch_data = epoch_data.get_data()

    scaler = mne.decoding.Scaler(info=info, scalings='median')
    epoch_data = scaler.fit_transform(epoch_data)

    epoch_data1 = epoch_data[:, ch_list.index('C4'), :].squeeze()
    epoch_data1 = torch.tensor(epoch_data1, dtype=torch.float32)

    epoch_data2 = epoch_data[:, ch_list.index('C3'), :].squeeze()
    epoch_data2 = torch.tensor(epoch_data2, dtype=torch.float32)

    outs = []
    for i in range(5):
        # 1. Prepared Pretrained Model
        ckpt_path = r'C:\Users\brainlab\Desktop\OneClick_Client\neuronet\ckpt\\' + str(i) + r'\model\best_model.pth'
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model_parameter = ckpt['model_parameter']
        pretrained_model = NeuroNet(**model_parameter)
        pretrained_model.load_state_dict(ckpt['model_state'])

        # 2. Encoder Wrapper
        backbone = NeuroNetEncoderWrapper(
            fs=model_parameter['fs'], second=model_parameter['second'],
            time_window=model_parameter['time_window'], time_step=model_parameter['time_step'],
            frame_backbone=pretrained_model.frame_backbone,
            patch_embed=pretrained_model.autoencoder.patch_embed,
            encoder_block=pretrained_model.autoencoder.encoder_block,
            encoder_norm=pretrained_model.autoencoder.encoder_norm,
            cls_token=pretrained_model.autoencoder.cls_token,
            pos_embed=pretrained_model.autoencoder.pos_embed,
            final_length=pretrained_model.autoencoder.embed_dim
        )

        # 3. Generator Classifier
        model = Classifier(backbone=backbone,
                           backbone_final_length=pretrained_model.autoencoder.embed_dim)
        ckpt_path = r'C:\Users\brainlab\Desktop\OneClick_Client\neuronet\ckpt\\' + str(i) + r'\linear_prob\best_model.pth'
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['model_state'])

        with torch.no_grad():
            model.eval()
            out1 = model(epoch_data1)
            out1 = torch.softmax(out1, dim=-1)

            out2 = model(epoch_data2)
            out2 = torch.softmax(out2, dim=-1)
            outs.append(out1 + out2)
    outs = torch.stack(outs)
    outs = torch.mean(outs, dim=0)
    sleep_stage = torch.argmax(outs, dim=-1)
    sleep_stage_prob = outs
    sleep_stage = sleep_stage.cpu().detach().numpy()
    sleep_stage_prob = sleep_stage_prob.cpu().detach().numpy()
    return {
        'sleep_stage': list(sleep_stage),
        'sleep_stage_prob': list([list(prob) for prob in sleep_stage_prob])
    }


def get_psd_diff_analysis(epoch_data, uuid):
    from sklearn.preprocessing import StandardScaler

    def psd(raw, band_range):
        spectrum = raw.compute_psd(method='welch', fmin=0.5, fmax=50)
        psds, freqs = spectrum.get_data(return_freqs=True)
        psds = 10 * np.log10(psds)
        psds_mean = psds.mean(axis=0)

        freq_res = freqs[1] - freqs[0]
        idx_band = np.logical_and(freqs >= band_range[0], freqs <= band_range[1])

        result = []
        for psds_mean_sample in psds_mean:
            total_power = simpson(psds_mean_sample, dx=freq_res)
            rel_power = simpson(psds_mean_sample[idx_band], dx=freq_res)
            result.append((rel_power / total_power))
        result = np.array(result)
        scaler = StandardScaler()
        result = scaler.fit_transform(result.reshape(-1, 1))
        result = result.reshape(-1)
        return result

    epoch_data = copy.deepcopy(epoch_data)
    info = epoch_data.info
    epoch_data = epoch_data.get_data()
    sample_size = epoch_data.shape[0]
    step = sample_size // 5
    exp_names = ['diff1', 'diff2', 'diff3', 'diff4']
    freq_bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 12), 'beta': (12, 30), 'gamma': (30, 49)}
    files = {exp_name: {band_name: None for band_name in freq_bands.keys()} for exp_name in exp_names}

    epochs = []
    for i in range(5):
        start, end = i * step, (i+1) * step
        sample = epoch_data[start: end, ...]
        epoch = mne.EpochsArray(sample, info=info)
        epochs.append(epoch)

    for diff_i, (i_epoch, j_epoch) in enumerate(zip(epochs[:-1], epochs[1:])):
        for band_name, bands in freq_bands.items():
            i_psd = psd(i_epoch, bands)
            j_psd = psd(j_epoch, bands)
            total_power = j_psd - i_psd

            fig, ax = plt.subplots(1, figsize=(3.5, 3.5))
            mne.viz.plot_topomap(total_power, info, ch_type='eeg',
                                 names=info['ch_names'], show=False,
                                 axes=ax)
            tmp_name = os.path.join('image', 'psd_diff', '{}_{}_{}.jpg'.format(uuid, exp_names[diff_i], band_name))
            fig.savefig(tmp_name)
            with open(tmp_name, 'rb') as f:
                im_bytes = f.read()
            im_b64 = base64.b64encode(im_bytes).decode("utf8")
            files[exp_names[diff_i]][band_name] = im_b64
            plt.close('all')
            plt.clf()
    return files


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

    file1 = get_psd_topography(epoch_data, myuuid)
    file2 = get_brain_connectivity(epoch_data, myuuid)
    file3 = get_brain_connectivity2(epoch_data, myuuid)
    psd_result = get_psd_analysis(epoch_data)
    fronto_limbic_result = get_fronto_limbic_analysis(filter_data, myuuid)
    sleep_stage_result = get_sleep_staging(epoch_data, ch_list)
    file4 = get_psd_diff_analysis(epoch_data, myuuid)
    file5 = get_diff_brain_connectivity(epoch_data, myuuid)
    file6 = get_diff_brain_connectivity2(epoch_data, myuuid)

    return {
        'topography': file1,
        'connectivity': file2,
        'connectivity2': file3,
        'psd_result': psd_result,
        'frontal_limbic': fronto_limbic_result,
        'sleep_stage': sleep_stage_result,
        'topography_diff': file4,
        'connectivity_diff': file5,
        'connectivity2_diff': file6
    }


if __name__ == '__main__':
    file_ = '2023-07-18-1137.csv'
    data_path_ = r'D:\Project\BioSignal_Analysis_cli\database\ch'
    path_ = os.path.join(data_path_, file_)
    main_analysis(path_)

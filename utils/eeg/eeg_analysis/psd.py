import numpy as np
from scipy.integrate import simpson
import copy
import mne

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
    psds += np.abs(np.min(psds))
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
    r_psds += np.abs(np.min(r_psds))
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
    l_psds += np.abs(np.min(l_psds))
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

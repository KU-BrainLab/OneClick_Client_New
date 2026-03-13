import copy
import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
import base64
import os

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
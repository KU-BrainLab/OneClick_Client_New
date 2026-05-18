import numpy as np
from scipy.integrate import simpson
import copy
import mne
import matplotlib.pyplot as plt
import os
import base64

def get_psd_diff_analysis(epoch_data, uuid, trigger=None):
    from sklearn.preprocessing import StandardScaler

    def psd(raw, band_range):
        spectrum = raw.compute_psd(method='welch', fmin=0.5, fmax=40)
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
    exp_names = ['diff1', 'diff2', 'diff3', 'diff4']
    freq_bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 12), 'beta': (12, 30), 'gamma': (30, 40), 'sigma': (12, 15)}
    files = {exp_name: {band_name: None for band_name in freq_bands.keys()} for exp_name in exp_names}

    n_phases = len(trigger) - 1 if (trigger is not None and len(trigger) > 1) else 5
    epochs = []
    for i in range(n_phases):
        start = trigger[i] * 2
        end = trigger[i+1] * 2
        sample = epoch_data[start:end, ...]
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

    # 존재하지 않는 phase의 diff는 빈 문자열로 채움
    for diff_i in range(max(len(epochs) - 1, 0), len(exp_names)):
        for band_name in freq_bands:
            files[exp_names[diff_i]][band_name] = ''

    return files
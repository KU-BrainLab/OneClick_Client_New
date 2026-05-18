import mne
import numpy as np
import warnings
import matplotlib
import matplotlib.pyplot as plt
from mne.time_frequency import psd_array_multitaper
from pathlib import Path
import os
import base64

SPEC_WINDOW_SEC = 10.0
SPEC_STEP_SEC   = 2.0

def get_brain_spectrogram(raw: mne.io.RawArray, uuid, trigger):
    print('Plotting spectrograms...')
    sfreq  = raw.info['sfreq']
    CHS    = raw.ch_names
    win_n  = int(SPEC_WINDOW_SEC * sfreq)
    step_n = int(SPEC_STEP_SEC   * sfreq)

    phase_names = ['baseline', 'stimulation1', 'recovery1', 'stimulation2', 'recovery2']
    n_phases = len(trigger) - 1
    boundaries = {}
    for i in range(n_phases):
        boundaries[phase_names[i]] = (trigger[i], trigger[i+1])

    result = {}

    os.makedirs(os.path.join('image', 'spectrogram'), exist_ok=True)
    raw_data = raw.get_data()
    n_samples = raw_data.shape[1]

    # sliding window 인덱스
    starts    = np.arange(0, n_samples - win_n + 1, step_n)
    t_centers = (starts + win_n / 2) / sfreq  # 각 window 중심 시간 (초)

    # (n_windows, n_ch, win_n)
    win_array = np.stack([raw_data[:, s:s + win_n] for s in starts])

    # multitaper PSD — bandwidth=2 Hz (수면 저주파 분리 최적)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        psds, freqs = psd_array_multitaper(
            win_array, sfreq=sfreq, fmin=0.5, fmax=45.0,
            bandwidth=2.0, verbose=False
        )
    # psds: (n_windows, n_ch, n_freqs) → dB
    psds_db = 10 * np.log10(psds + 1e-30)

    stage_lines = {s_min: name for name, (s_min, _) in boundaries.items()}
    t_min = t_centers / 60

    vmin = float(np.percentile(psds_db, 2))
    vmax = float(np.percentile(psds_db, 98))


    for ci, ch in enumerate(CHS):
        fig, ax = plt.subplots(figsize=(14, 4), constrained_layout=True)
        ch_data = psds_db[:, ci, :].T   # (n_freqs, n_windows)
        im = ax.imshow(ch_data, aspect='auto', origin='lower',
                       extent=[t_min[0], t_min[-1], freqs[0], freqs[-1]],
                       vmin=vmin, vmax=vmax, cmap='RdYlBu_r')
        for t_stage, name in stage_lines.items():
            if t_min[0] <= t_stage <= t_min[-1]:
                ax.axvline(t_stage, color='white', linewidth=2.5, linestyle='--')
                ax.text(t_stage + 0.1, freqs[-1] * 0.92, name,
                        color='white', fontsize=11, va='top', fontweight='bold')
        fig.colorbar(im, ax=ax, label='Power (dB)', orientation='horizontal', pad=0.05)
        ax.set_title(f'Spectrogram — {ch}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Frequency (Hz)')


        tmp_name = os.path.join('image', 'spectrogram', '{}_{}.jpg'.format(uuid, ch))
        tmp_name = os.path.abspath(tmp_name)
        fig.savefig(tmp_name)
        
        with open(tmp_name, 'rb') as f:
            im_bytes = f.read()
        im_b64 = base64.b64encode(im_bytes).decode("utf8")
        result[ch] = im_b64
        plt.close('all')
        plt.clf()

        print(ch)
        
    return result
import copy
import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import os
import networkx as nx
import base64
import pandas as pd
import matplotlib.colors as mcolors

SLEEP_STAGES = ['W', 'N1', 'N2', 'N3', 'REM']
SLEEP_STAGE_NAMES = {0: 'W', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}

FREQS_BANDS = {
    'Delta': (0.5,  4),
    'Theta': (4,    8),
    'Alpha': (8,   12),
    'Sigma': (12,  15),
    'Beta' : (12,  30),
    'Gamma': (30,  45),
}

STAGES = ['baseline', 'stimulation 1', 'recovery 1', 'stimulation 2', 'recovery 2']

BANDS = list(FREQS_BANDS.keys())

PHASE_PAIRS = [
    ('stimulation 1', 'baseline'),
    ('recovery 1',    'stimulation 1'),
    ('stimulation 2', 'recovery 1'),
    ('recovery 2',    'stimulation 2'),
]

MIN_POWER_EPOCHS = 1

def compute_band_power(epochs: mne.Epochs, df) -> tuple:
    """
    Returns
    -------
    df_power : pd.DataFrame
        columns = ['stage', 'sleep_stage', 'band', 'ch', 'power']
    info : mne.Info
        EEG 채널만 남긴 info
    CHS : list
        EEG 채널명 리스트
    """
    # pick()은 채널 구성을 바꾸므로 preload 필요
    if not epochs.preload:
        epochs.load_data()

    # 원본 epochs는 건드리지 않고 복사본에서 EEG만 선택
    eeg_epochs = epochs.copy().pick('eeg')

    info = eeg_epochs.info.copy()
    CHS = list(info['ch_names'])
    rows = []

    md = df.reset_index(drop=True).copy()

    # epoch 개수와 metadata 길이 일치 여부 확인
    if len(md) != len(eeg_epochs):
        raise ValueError(
            f"df 길이({len(md)})와 epochs 개수({len(eeg_epochs)})가 다릅니다. "
            "mask 인덱싱이 맞지 않을 수 있습니다."
        )

    # 필요한 컬럼 확인
    required_cols = {'stage', 'sleep_stage'}
    missing_cols = required_cols - set(md.columns)
    if missing_cols:
        raise ValueError(f"df에 필요한 컬럼이 없습니다: {missing_cols}")

    for stage in STAGES:
        for ss in SLEEP_STAGES + ['all']:
            if ss == 'all':
                mask = (md['stage'] == stage).to_numpy()
            else:
                mask = ((md['stage'] == stage) & (md['sleep_stage'] == ss)).to_numpy()

            if mask.sum() < MIN_POWER_EPOCHS:
                continue

            ep = eeg_epochs[mask]

            if len(ep) < MIN_POWER_EPOCHS:
                continue

            spectrum = ep.compute_psd(
                method='welch',
                fmin=0.5,
                fmax=45.0,
                verbose=False
            )
            psds, freqs = spectrum.get_data(return_freqs=True)
            # psds shape: (n_epochs, n_channels, n_freqs)

            if psds.size == 0:
                continue

            psds_mean = psds.mean(axis=0)  # (n_channels, n_freqs)

            bp_all = {}
            for band, (flo, fhi) in FREQS_BANDS.items():
                idx = np.logical_and(freqs >= flo, freqs <= fhi)

                if not np.any(idx):
                    bp_all[band] = np.zeros(len(CHS), dtype=float)
                else:
                    bp_all[band] = psds_mean[:, idx].mean(axis=1)

            total = np.zeros(len(CHS), dtype=float)
            for band in BANDS:
                if band not in bp_all:
                    raise KeyError(f"BANDS에 있는 '{band}'가 FREQS_BANDS에 없습니다.")
                total += bp_all[band]

            total = total + 1e-12

            for band in BANDS:
                norm_bp = bp_all[band] / total
                for ch_i, ch in enumerate(CHS):
                    rows.append({
                        'stage': stage,
                        'sleep_stage': ss,
                        'band': band,
                        'ch': ch,
                        'power': float(norm_bp[ch_i])
                    })

    df_power = pd.DataFrame(rows, columns=['stage', 'sleep_stage', 'band', 'ch', 'power'])
    return df_power, info, CHS

def assign_stage_metadata(epochs, boundaries):
    n_epochs = len(epochs)
    stages   = ['unknown'] * n_epochs
    for stage_name, (start_min, end_min) in boundaries.items():
        s = int(start_min * 60 / 30)
        e = int(end_min   * 60 / 30)
        for i in range(min(s, n_epochs), min(e, n_epochs)):
            stages[i] = stage_name
    df = pd.DataFrame({'stage': stages, 'sleep_stage': 'unknown'})
    return df


def get_brain_delta_power_topo(epochs, uuid, trigger, sleep_labels_int):

    os.makedirs(os.path.join('image', 'delta_power_topo'), exist_ok=True)

    boundaries = {}
    boundaries['baseline']       = (trigger[0], trigger[1])
    boundaries['stimulation 1']  = (trigger[1], trigger[2])
    boundaries['recovery 1']     = (trigger[2], trigger[3])
    boundaries['stimulation 2']  = (trigger[3], trigger[4])
    boundaries['recovery 2']     = (trigger[4], trigger[5])
    df = assign_stage_metadata(epochs, boundaries)
    df['sleep_stage'] = [SLEEP_STAGE_NAMES[l] for l in sleep_labels_int]
    
    #before compute band power
    df_power, info, CHS = compute_band_power(epochs, df)

    print("#########################################################")
    print(df)
    result = {}

    all_deltas = []
    for band in BANDS:
        for targ, ref in PHASE_PAIRS:
            for ss in SLEEP_STAGES + ['all']:
                ref_df  = df_power[(df_power['stage'] == ref)  & (df_power['sleep_stage'] == ss)]
                targ_df = df_power[(df_power['stage'] == targ) & (df_power['sleep_stage'] == ss)]
                if ref_df.empty or targ_df.empty:
                    continue
                pv_ref  = ref_df.pivot_table(values='power', index='ch', columns='band').reindex(CHS)
                pv_targ = targ_df.pivot_table(values='power', index='ch', columns='band').reindex(CHS)
                if band not in pv_ref.columns:
                    continue
                diff = (pv_targ[band] - pv_ref[band]).values
                all_deltas.extend(diff[~np.isnan(diff)].tolist())
    global_vlim = float(np.percentile(np.abs(all_deltas), 97)) if all_deltas else 0.01
    norm     = mcolors.Normalize(vmin=-global_vlim, vmax=global_vlim)
    colormap = plt.cm.get_cmap('RdBu_r')
    print(f'    Power global vlim: ±{global_vlim:.4f}')

    for targ, ref in PHASE_PAIRS:
        pair_fn = f'{targ.replace(" ", "_")}_vs_{ref.replace(" ", "_")}'

        for ss in SLEEP_STAGES + ['all']:
            ref_df  = df_power[(df_power['stage'] == ref)  & (df_power['sleep_stage'] == ss)]
            targ_df = df_power[(df_power['stage'] == targ) & (df_power['sleep_stage'] == ss)]
            no_data = ref_df.empty or targ_df.empty

            if no_data:
                pv_diff = None
            else:
                pv_ref  = ref_df.pivot_table(values='power', index='ch', columns='band').reindex(CHS)
                pv_targ = targ_df.pivot_table(values='power', index='ch', columns='band').reindex(CHS)
                pv_diff = pv_targ - pv_ref

            for band in BANDS:
                fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
                title = f'Δ Power [{band}] [{ss}]\n{targ} − {ref}'

                has_data = (not no_data and pv_diff is not None
                            and band in pv_diff.columns
                            and pv_diff[band].notna().any())

                if has_data:
                    vals = pv_diff[band].values
                    mne.viz.plot_topomap(vals, info, axes=ax,
                                        vlim=(-global_vlim, global_vlim),
                                        cmap='RdBu_r', show=False)
                else:
                    mne.viz.plot_topomap(np.zeros(len(CHS)), info, axes=ax,
                                        vlim=(-global_vlim, global_vlim),
                                        cmap='RdBu_r', show=False, contours=0)
                    ax.text(0, 0, 'No data', ha='center', va='center',
                            fontsize=13, color='gray', fontweight='bold',
                            transform=ax.transData)

                # colorbar — ScalarMappable로 범위 명시 (항상 동일)
                sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
                sm.set_array([])
                fig.colorbar(sm, ax=ax, shrink=0.8, label='Δ norm. power',
                             orientation='horizontal', pad=0.05)

                ax.set_title(title, fontsize=11, fontweight='bold')
                
                tmp_name = os.path.join('image', 'delta_power_topo', '{}_{}_{}_{}.jpg'.format(uuid, pair_fn, ss, band))
                tmp_name = os.path.abspath(tmp_name)
                fig.savefig(tmp_name)

                with open(tmp_name, 'rb') as f:
                    im_bytes = f.read()
                im_b64 = base64.b64encode(im_bytes).decode("utf8")
                title = f'topography_{pair_fn}_{band}_{ss}'
                print(title)
                result[title] = im_b64
                plt.close('all')
                plt.clf()

    return result
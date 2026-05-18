import mne
import mne_connectivity
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.signal import hilbert
import matplotlib.colors as mcolors
import warnings
import os
import base64

COMPUTE_AEC   = True
SPECTRAL_FCS  = []
METHODS    = SPECTRAL_FCS + (['AEC'] if COMPUTE_AEC else [])
STAGES = ['baseline', 'stimulation 1', 'recovery 1', 'stimulation 2', 'recovery 2']
SLEEP_STAGES = ['W', 'N1', 'N2', 'N3', 'REM']
SLEEP_STAGE_NAMES = {0: 'W', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
FC_EPOCH_LEN  = 10.0
FC_OVERLAP    = FC_EPOCH_LEN / 2
RFREQ  = 100
AEC_EPOCH_LEN = 20.0
MIN_FC_SUB       = 3
FREQS_BANDS = {
    'Delta': (0.5,  4),
    'Theta': (4,    8),
    'Alpha': (8,   12),
    'Sigma': (12,  15),
    'Beta' : (12,  30),
    'Gamma': (30,  45),
}
PHASE_PAIRS = [
    ('stimulation 1', 'baseline'),
    ('recovery 1',    'stimulation 1'),
    ('stimulation 2', 'recovery 1'),
    ('recovery 2',    'stimulation 2'),
]
BANDS = list(FREQS_BANDS.keys())
ALL_FREQS  = [f for band in FREQS_BANDS.values() for f in band]
FMIN, FMAX = min(ALL_FREQS), max(ALL_FREQS)

def _aec(epochs: mne.Epochs, ch_names: list) -> pd.DataFrame:
    rows = []
    for band, (flo, fhi) in FREQS_BANDS.items():
        ep_f    = epochs.copy().filter(flo, fhi, verbose=False)
        n_times = int(AEC_EPOCH_LEN * RFREQ)
        data_3d = ep_f.get_data(copy=False)
        sub     = []
        for ep_data in data_3d:
            for t0 in range(0, ep_data.shape[1] - n_times + 1, n_times):
                sub.append(ep_data[:, t0:t0 + n_times])
        if not sub:
            continue
        sub_ep   = mne.EpochsArray(np.stack(sub), epochs.info, verbose=False)
        envs     = np.abs(hilbert(sub_ep.get_data(), axis=-1))
        env_mean = envs.mean(axis=2)
        for i in range(len(ch_names)):
            for j in range(i + 1, len(ch_names)):
                corr = np.corrcoef(env_mean[:, i], env_mean[:, j])[0, 1]
                rows.append({'ch1': ch_names[i], 'ch2': ch_names[j],
                             'band': band, 'value': float(corr)})
    return pd.DataFrame(rows)


def _spectral_fc(epochs: mne.Epochs, method: str, ch_names: list) -> pd.DataFrame:
    con   = mne_connectivity.spectral_connectivity_epochs(
        epochs, method=method, mode='multitaper',
        fmin=FMIN, fmax=FMAX, faverage=False, mt_adaptive=True, verbose=False)
    data  = con.get_data()
    freqs = np.array(con.freqs)
    rows  = []
    k = 0
    for i in range(len(ch_names)):
        for j in range(i + 1, len(ch_names)):
            for band, (flo, fhi) in FREQS_BANDS.items():
                idx = np.logical_and(freqs >= flo, freqs <= fhi)
                val = float(data[k, idx].mean()) if idx.any() else np.nan
                rows.append({'ch1': ch_names[i], 'ch2': ch_names[j],
                             'band': band, 'value': val})
            k += 1
    return pd.DataFrame(rows).groupby(['ch1', 'ch2', 'band'])['value'].mean().reset_index()



def compute_fc(epochs: mne.Epochs, df) -> dict:
    """
    Returns fc_db: {method: {(stage, sleep_stage): df}}
    """
    if not epochs.preload:
        epochs.load_data()

    eeg_epochs = epochs.copy().pick('eeg')
    ch_names = list(eeg_epochs.ch_names)
    fc_db = defaultdict(dict)
    md = df.reset_index(drop=True).copy()
    
    for method in [m for m in METHODS if m != 'AEC']:
        print(f'    {method}...')
        for stage in STAGES:
            for ss in SLEEP_STAGES:
                mask = (md['stage'] == stage) & (md['sleep_stage'] == ss)
                ep = epochs[mask]
                if len(ep) == 0:
                    continue

                data_3d = ep.get_data()
                n_times = int(FC_EPOCH_LEN * RFREQ)
                sub = []
                for ep_data in data_3d:
                    for t0 in range(0, ep_data.shape[1] - n_times + 1,
                                   int((FC_EPOCH_LEN - FC_OVERLAP) * RFREQ)):
                        sub.append(ep_data[:, t0:t0 + n_times])
                if len(sub) < MIN_FC_SUB:
                    continue
                sub_ep = mne.EpochsArray(np.stack(sub), ep.info, verbose=False)
                fc_db[method][(stage, ss)] = _spectral_fc(sub_ep, method, ch_names)

    if COMPUTE_AEC:
        print('    AEC...')
        for stage in STAGES:
            for ss in SLEEP_STAGES + ['all']:
                if ss == 'all':
                    mask = (md['stage'] == stage)
                else:
                    mask = (md['stage'] == stage) & (md['sleep_stage'] == ss)
                ep = epochs[mask]
                if len(ep) == 0:
                    continue
                fc_db['AEC'][(stage, ss)] = _aec(ep, ch_names)

    return fc_db

def _draw_head_outline(ax):
    """머리 윤곽선(원 + 코)을 그린다."""
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', lw=1.5, zorder=1)
    # 코
    ax.plot([-.08, 0, .08], [0.97, 1.12, 0.97], 'k-', lw=1.5, zorder=1)
    # 귀
    for sign in (-1, 1):
        ear_x = sign * np.array([1.00, 1.06, 1.08, 1.06, 1.00])
        ear_y = np.array([0.15, 0.10, 0.00, -0.10, -0.15])
        ax.plot(ear_x, ear_y, 'k-', lw=1.5, zorder=1)

def _get_sensor_pos_2d(CHS: list) -> np.ndarray:
    """
    standard_1020 몬타주 3D 좌표 → 2D 방위각 등거리 투영.
    MNE plot_topomap과 동일한 방법:
      - Cz = 중앙, 코(anterior +y) = 위쪽, 오른쪽(+x) = 우측
      - 최외곽 채널이 반경 0.9 (머리 원 반경 1.0 기준)
    """
    montage = mne.channels.make_standard_montage('standard_1020')
    ch_pos  = montage.get_positions()['ch_pos']   # {ch_name: np.array([x, y, z])}

    pos3d = np.array([ch_pos.get(ch, np.zeros(3)) for ch in CHS])

    # 단위 구면으로 정규화
    norms = np.linalg.norm(pos3d, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    p = pos3d / norms                             # (n_ch, 3)

    # 방위각 등거리 투영
    # elevation: z 축 위쪽 방향으로의 각도 (-π/2 ~ π/2)
    # azimuth  : xy 평면에서의 방향 (anterior=+y → azimuth=π/2 → 위쪽)
    elevation = np.arcsin(np.clip(p[:, 2], -1.0, 1.0))
    azimuth   = np.arctan2(p[:, 1], p[:, 0])     # +y(nose)=π/2 → 2D 상단
    r2d       = np.pi / 2 - elevation             # Cz=0, equator=π/2

    pos2d = np.column_stack([r2d * np.cos(azimuth),
                             r2d * np.sin(azimuth)])

    # 최외곽 채널 → 반경 0.9
    max_r = np.max(np.linalg.norm(pos2d, axis=1)) or 1.0
    return pos2d / max_r * 0.9

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

def get_brain_delta_connectivity(epoch_data, uuid, trigger, sleep_labels_int):

    os.makedirs(os.path.join('image', 'delta_connectivity'), exist_ok=True)

    stage_names = ['baseline', 'stimulation 1', 'recovery 1', 'stimulation 2', 'recovery 2']
    n_phases = len(trigger) - 1
    boundaries = {}
    for i in range(n_phases):
        boundaries[stage_names[i]] = (trigger[i], trigger[i+1])
    df = assign_stage_metadata(epoch_data, boundaries)
    df['sleep_stage'] = [SLEEP_STAGE_NAMES[l] for l in sleep_labels_int]

    fc_db = compute_fc(epoch_data, df)

    CHS = ['Fp1', 'F7', 'F3', 'T3', 'C3', 'Cz', 'P3', 'Fp2', 'F4', 'F8', 'C4', 'T4', 'P4']

    pos = _get_sensor_pos_2d(CHS)   # (n_ch, 2)
    result = {}
    # ── method 별 단일 global vlim 사전 계산 (모든 band/phase/sleep_stage 포함)
    global_fc_vlim: dict = {}
    for method in METHODS:
        method_db = fc_db.get(method, {})
        all_deltas = []
        for band in BANDS:
            for targ, ref in PHASE_PAIRS:
                for ss in SLEEP_STAGES + ['all']:
                    ref_key  = (ref,  ss)
                    targ_key = (targ, ss)
                    if not method_db or ref_key not in method_db or targ_key not in method_db:
                        continue
                    df_m = method_db[targ_key].merge(
                        method_db[ref_key], on=['ch1', 'ch2', 'band'],
                        suffixes=('_targ', '_ref'))
                    df_m['delta'] = df_m['value_targ'] - df_m['value_ref']
                    vals = df_m.loc[df_m['band'] == band, 'delta'].dropna().values
                    all_deltas.extend(vals.tolist())
        global_fc_vlim[method] = float(np.percentile(np.abs(all_deltas), 97)) if all_deltas else 0.01
        global_fc_vlim[method] = global_fc_vlim[method] or 0.01

    for method in METHODS:
        method_db = fc_db.get(method, {})

        for targ, ref in PHASE_PAIRS:
            pair_fn = f'{targ.replace(" ", "_")}_vs_{ref.replace(" ", "_")}'
            for ss in SLEEP_STAGES + ['all']:
                ref_key  = (ref,  ss)
                targ_key = (targ, ss)

                # delta 행렬 계산 (데이터 없으면 None)
                M = None
                if method_db and ref_key in method_db and targ_key in method_db:
                    df_ref  = method_db[ref_key]
                    df_targ = method_db[targ_key]
                    df_m    = df_targ.merge(df_ref, on=['ch1', 'ch2', 'band'],
                                            suffixes=('_targ', '_ref'))
                    df_m['delta'] = df_m['value_targ'] - df_m['value_ref']
                else:
                    df_m = None

                for band in BANDS:
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.set_aspect('equal')
                    ax.axis('off')
                    _draw_head_outline(ax)

                    # nodes (항상 표시)
                    ax.scatter(pos[:, 0], pos[:, 1],
                               c='white', s=120, zorder=3,
                               edgecolors='black', linewidths=0.8)
                    for i, ch in enumerate(CHS):
                        ax.text(pos[i, 0], pos[i, 1], ch,
                                fontsize=5.5, ha='center', va='center', zorder=4)

                    vlim     = global_fc_vlim[method]
                    norm     = mcolors.Normalize(vmin=-vlim, vmax=vlim)
                    colormap = plt.cm.get_cmap('RdBu_r')

                    # colorbar (항상 표시 — 범위 통일)
                    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
                    sm.set_array([])
                    fig.colorbar(sm, ax=ax, shrink=0.6, label=f'Δ {method}',
                                orientation='horizontal', pad=0.05)

                    no_data = True
                    if df_m is not None:
                        data_band = df_m[df_m['band'] == band]
                        if not data_band.empty:
                            band_M = data_band.pivot_table(
                                index='ch1', columns='ch2', values='delta') \
                                .reindex(index=CHS, columns=CHS).to_numpy()
                            with warnings.catch_warnings():
                                warnings.simplefilter('ignore', RuntimeWarning)
                                band_M = np.nanmean(np.stack([band_M, band_M.T]), axis=0)
                            vals = band_M[~np.isnan(band_M)]
                            if vals.size > 0:
                                no_data = False
                                # edges
                                for i in range(len(CHS)):
                                    for j in range(i + 1, len(CHS)):
                                        val = band_M[i, j]
                                        if np.isnan(val):
                                            continue
                                        color = colormap(norm(val))
                                        lw    = 0.4 + 2.5 * abs(val) / vlim
                                        ax.plot([pos[i, 0], pos[j, 0]],
                                                [pos[i, 1], pos[j, 1]],
                                                color=color, linewidth=lw,
                                                alpha=0.75, zorder=2)

                    if no_data:
                        ax.text(0, 0, 'No data', ha='center', va='center',
                                fontsize=13, color='gray', fontweight='bold',
                                zorder=5)

                    ax.set_title(f'Δ FC [{method}] [{band}] [{ss}]\n{targ} − {ref}',
                                 fontsize=9, fontweight='bold')
                    ax.set_xlim(-1.25, 1.25)
                    ax.set_ylim(-1.25, 1.30)

                    tmp_name = os.path.join('image', 'delta_connectivity', '{}_{}_{}_{}.jpg'.format(uuid, pair_fn, ss, band))
                    tmp_name = os.path.abspath(tmp_name)
                    fig.savefig(tmp_name)

                    with open(tmp_name, 'rb') as f:
                        im_bytes = f.read()
                    im_b64 = base64.b64encode(im_bytes).decode("utf8")
                    title = f'connectivity_{pair_fn}_{band}_{ss}'
                    print(title)
                    result[title] = im_b64
                    plt.close('all')
                    plt.clf()

    return result
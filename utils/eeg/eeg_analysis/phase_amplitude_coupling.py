# -*- coding: utf-8 -*-
import copy
import os
import base64

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
from joblib import Parallel, delayed


# ── 상수 ──────────────────────────────────────────
PHASE_FREQS = np.arange(0.5, 20.5, 1.0)   # 위상: 0.5 ~ 20 Hz
AMP_FREQS   = np.arange(4.0,  46.0, 1.0)  # 진폭: 4  ~ 45 Hz
BW          = 1.0                           # 각 bin 대역폭
N_BINS      = 18


# ── 신호처리 ──────────────────────────────────────

def _bandpass(data, sfreq, low, high, order=4):
    nyq = sfreq / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, data, axis=-1)


def _modulation_index(phase_sig, amp_sig, n_bins=N_BINS):
    """벡터화된 Modulation Index (Tort et al. 2010).

    이전: 18-bin Python for loop
    개선: numpy bincount → 18x 속도 향상
    """
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_idx   = np.clip(np.searchsorted(bin_edges[1:], phase_sig), 0, n_bins - 1)
    weights   = np.bincount(bin_idx, weights=amp_sig, minlength=n_bins).astype(float)
    counts    = np.bincount(bin_idx,                  minlength=n_bins).astype(float)
    amp_mean  = np.where(counts > 0, weights / counts, 0.0)
    amp_norm  = amp_mean / (amp_mean.sum() + 1e-30)
    return float(np.sum(amp_norm * np.log(amp_norm * n_bins + 1e-30)) / np.log(n_bins))


def _compute_phase_row(pf, flat, sfreq, amp_signals):
    """단일 phase freq에 대해 모든 amp freq의 MI를 계산.

    joblib worker 함수 — amp_signals는 사전에 계산된 리스트.
    phase freq >= amp freq 인 셀은 NaN (물리적으로 무의미한 구간).
    """
    plo        = max(pf - BW, 0.3)
    phi        = pf + BW
    inst_phase = np.angle(hilbert(_bandpass(flat, sfreq, plo, phi), axis=-1)).reshape(-1)
    row = np.full(len(AMP_FREQS), np.nan)
    for ai, af in enumerate(AMP_FREQS):
        if af <= pf:
            continue   # phase freq >= amp freq → 건너뜀
        row[ai] = _modulation_index(inst_phase, amp_signals[ai])
    return row


def _compute_comodulogram(epoch_arr, sfreq):
    """(n_epochs, n_ch, n_times) → (n_phase_freqs, n_amp_freqs) MI 행렬.

    최적화:
      ① epoch × ch 이중 루프 제거 → reshape(-1) 로 flatten 후 일괄 처리
      ② amp bandpass를 사전에 1회만 계산 → 20 phase freq × 재사용
          (이전: 20 × 42 = 840회 amp 필터링  →  개선: 42회)
      ③ MI binning을 numpy bincount로 벡터화
          (이전: 18-step Python loop  →  개선: 순수 numpy)
      ④ phase freq 루프를 joblib threads로 병렬화
          (prefer='threads': Windows 호환, numpy/scipy는 GIL 해제)
    """
    # ① flatten: (n_epochs, n_ch, n_times) → (n_epochs*n_ch, n_times)
    flat = epoch_arr.reshape(-1, epoch_arr.shape[-1])

    # ② amp 신호 사전 계산 (42회, 이후 20번 재사용)
    print('    [PAC] Precomputing amplitude signals...')
    amp_signals = []
    for af in AMP_FREQS:
        alo = af - BW
        ahi = min(af + BW, sfreq / 2.0 - 1)
        env = np.abs(hilbert(_bandpass(flat, sfreq, alo, ahi), axis=-1)).reshape(-1)
        amp_signals.append(env)

    # ③④ phase freq별 MI 행 병렬 계산
    print('    [PAC] Computing MI matrix...')
    rows = Parallel(n_jobs=-1, prefer='threads')(
        delayed(_compute_phase_row)(pf, flat, sfreq, amp_signals)
        for pf in PHASE_FREQS
    )

    return np.vstack(rows)   # (n_pf, n_af)


def _compute_phase_amplitude_hist(epoch_arr, sfreq, phase_band, amp_band, n_bins=N_BINS):
    """위상 구간별 평균 진폭 분포. 완전 벡터화 (루프 없음).

    이전: epoch × ch × bin Python 3중 루프
    개선: flatten + numpy bincount → 루프 완전 제거
    """
    flat      = epoch_arr.reshape(-1, epoch_arr.shape[-1])
    bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1)

    inst_phase = np.angle(
        hilbert(_bandpass(flat, sfreq, *phase_band), axis=-1)
    ).reshape(-1) % (2 * np.pi)

    inst_amp = np.abs(
        hilbert(_bandpass(flat, sfreq, *amp_band), axis=-1)
    ).reshape(-1)

    bin_idx  = np.clip(np.searchsorted(bin_edges[1:], inst_phase), 0, n_bins - 1)
    weights  = np.bincount(bin_idx, weights=inst_amp, minlength=n_bins).astype(float)
    counts   = np.bincount(bin_idx,                   minlength=n_bins).astype(float)
    amp_mean = np.where(counts > 0, weights / counts, 0.0)
    return amp_mean / (amp_mean.sum() + 1e-30)


# ── 콘솔 출력 ─────────────────────────────────────

def _print_phase_amplitude_table(amp_norm, label, n_bins=N_BINS):
    uniform     = 1.0 / n_bins
    bin_edges   = np.linspace(0, 2 * np.pi, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    print(f"\n[Phase-Amplitude — {label}]")
    print(f"  {'Phase (deg)':>12}  {'Amp (norm)':>12}  {'vs Uniform':>12}")
    print(f"  {'-'*40}")
    for k in range(n_bins):
        deg    = np.degrees(bin_centers[k])
        diff   = amp_norm[k] - uniform
        marker = '▲' if diff > 0 else '▽'
        print(f"  {deg:>12.1f}  {amp_norm[k]:>12.6f}  {diff:>+12.6f} {marker}")
    print(f"  {'Uniform':>12}   {uniform:.6f}")


def _roi_mi(mi_matrix, pf_low, pf_high, af_low, af_high):
    """comodulogram 행렬에서 ROI 영역의 평균 MI를 반환."""
    p_mask = (PHASE_FREQS >= pf_low) & (PHASE_FREQS <= pf_high)
    a_mask = (AMP_FREQS   >= af_low) & (AMP_FREQS   <= af_high)
    return float(mi_matrix[np.ix_(p_mask, a_mask)].mean())


# ── 시각화 ────────────────────────────────────────

def _to_b64(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf8')


def _add_roi_boxes(ax):
    """δ–θ, β–γ 관심 영역 강조 박스. X=phase, Y=amp."""
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle(xy=(0.5, 4),  width=3.5, height=4,
                            linewidth=2, edgecolor='white', facecolor='none', zorder=5))
    ax.text(0.7, 4.3, 'δ–θ', color='white', fontsize=8, fontweight='bold', zorder=6)

    ax.add_patch(Rectangle(xy=(13, 30), width=17,  height=15,
                            linewidth=2, edgecolor='white', facecolor='none', zorder=5))
    ax.text(13.5, 30.5, 'β–γ', color='white', fontsize=8, fontweight='bold', zorder=6)


def _plot_comodulogram(mi_matrix, title, save_path, cmap_name='jet', vmin=0, vmax=None):
    """단일 comodulogram 저장."""
    cmap = plt.cm.get_cmap(cmap_name).copy()
    cmap.set_bad(color='#1a1a1a')   # NaN(phase>=amp) 구간 — 어두운 회색
    if vmax is None:
        vmax = float(np.nanmax(mi_matrix))
        vmax = vmax if vmax > 0 else 0.01

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im = ax.imshow(mi_matrix.T, aspect='auto', origin='lower',
                   cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=[PHASE_FREQS[0], PHASE_FREQS[-1],
                           AMP_FREQS[0],   AMP_FREQS[-1]])
    ax.grid(False)
    _add_roi_boxes(ax)
    ax.set_xlabel('Phase frequency (Hz)', fontsize=10)
    ax.set_ylabel('Amplitude frequency (Hz)', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    fig.colorbar(im, ax=ax, label='Modulation Index (MI)')
    fig.savefig(save_path, dpi=100)
    plt.close('all')
    plt.clf()


def _plot_diff_comodulogram(diff_mat, title, save_path):
    """차이 comodulogram 저장 (diverging colormap)."""
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad(color='#1a1a1a')   # NaN(phase>=amp) 구간 — 어두운 회색
    vabs = float(np.nanmax(np.abs(diff_mat)))
    vabs = vabs if vabs > 0 else 0.01

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im = ax.imshow(diff_mat.T, aspect='auto', origin='lower',
                   cmap=cmap, vmin=-vabs, vmax=vabs,
                   extent=[PHASE_FREQS[0], PHASE_FREQS[-1],
                           AMP_FREQS[0],   AMP_FREQS[-1]])
    ax.grid(False)
    _add_roi_boxes(ax)
    ax.set_xlabel('Phase frequency (Hz)', fontsize=10)
    ax.set_ylabel('Amplitude frequency (Hz)', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    fig.colorbar(im, ax=ax, label='ΔMI (B − A)')
    fig.savefig(save_path, dpi=100)
    plt.close('all')
    plt.clf()


def _plot_phase_amplitude_hist(amp_norm, phase_label, amp_label, title, save_path, n_bins=N_BINS):
    uniform     = 1.0 / n_bins
    std_val     = amp_norm.std()
    bin_centers = np.linspace(0, 2 * np.pi, n_bins, endpoint=False) + np.pi / n_bins

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.bar(bin_centers, amp_norm, width=2 * np.pi / n_bins,
           color='steelblue', edgecolor='white', linewidth=0.5, align='center')
    ax.axhline(uniform,           color='red', linewidth=1.5, linestyle='-',  label='Uniform (no PAC)')
    ax.axhline(uniform + std_val, color='red', linewidth=1.0, linestyle='--')
    ax.axhline(uniform - std_val, color='red', linewidth=1.0, linestyle='--')
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'], fontsize=10)
    ax.set_xlabel('Phase', fontsize=11)
    ax.set_ylabel('Average amplitude for a given phase', fontsize=10)
    ax.set_title(f'{title}\n({phase_label} phase × {amp_label} amplitude)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    fig.savefig(save_path, dpi=100)
    plt.close('all')
    plt.clf()


# ── 메인 함수 ─────────────────────────────────────

def get_phase_amplitude_coupling(epoch_data, uuid, trigger, sleep_labels_int=None):
    """
    PAC 분석 — baseline / stimulation / recovery 각각 comodulogram +
    stimulation-baseline / recovery-stimulation diff comodulogram.

    반환 dict:
        {
          'baseline':                      { 'comodulogram': b64, 'delta_theta': b64, 'beta_gamma': b64 },
          'stimulation':                   { 'comodulogram': b64, 'delta_theta': b64, 'beta_gamma': b64 },
          'recovery':                      { 'comodulogram': b64, 'delta_theta': b64, 'beta_gamma': b64 },
          'diff_stimulation_vs_baseline':  b64,
          'diff_recovery_vs_stimulation':  b64,
        }
    """
    print('Computing PAC...')

    epoch_data = copy.deepcopy(epoch_data)
    sfreq      = epoch_data.info['sfreq']
    raw_arr    = epoch_data.get_data()          # (n_epochs, n_ch, n_times)

    n_phases = len(trigger) - 1 if (trigger is not None and len(trigger) > 1) else 1

    phase_map = {
        'baseline':    (0, 1),
        'stimulation': (1, 2),
        'recovery':    (2, 3),
    }

    mi_cache = {}
    result   = {}

    for label, (ti, tj) in phase_map.items():
        empty = {'comodulogram': '', 'delta_theta': '', 'beta_gamma': ''}

        if ti >= n_phases or tj > n_phases:
            result[label] = empty
            continue

        start  = trigger[ti] * 2
        end    = trigger[tj] * 2
        sample = raw_arr[start:end, ...]

        if sample.shape[0] == 0:
            result[label] = empty
            continue

        print(f'  [{label}] n_epochs={sample.shape[0]}, n_ch={sample.shape[1]}')

        # ── Comodulogram ──
        mi = _compute_comodulogram(sample, sfreq)
        mi_cache[label] = mi

        como_path = os.path.abspath(
            os.path.join('image', 'pac', f'{uuid}_{label}_comodulogram.jpg')
        )
        _plot_comodulogram(mi, title=f'PAC Comodulogram — {label}', save_path=como_path)

        # ── Delta-Theta histogram ──
        dt_hist = _compute_phase_amplitude_hist(sample, sfreq,
                                                phase_band=(0.5, 4.0), amp_band=(4.0, 8.0))
        dt_path = os.path.abspath(
            os.path.join('image', 'pac', f'{uuid}_{label}_delta_theta.jpg')
        )
        _plot_phase_amplitude_hist(dt_hist, 'Delta', 'Theta',
                                   title=f'Delta–Theta PAC — {label}', save_path=dt_path)
        _print_phase_amplitude_table(dt_hist, f'{label} / Delta-Theta')

        # ── Beta-Gamma histogram ──
        bg_hist = _compute_phase_amplitude_hist(sample, sfreq,
                                                phase_band=(13.0, 30.0), amp_band=(30.0, 45.0))
        bg_path = os.path.abspath(
            os.path.join('image', 'pac', f'{uuid}_{label}_beta_gamma.jpg')
        )
        _plot_phase_amplitude_hist(bg_hist, 'Beta', 'Gamma',
                                   title=f'Beta–Gamma PAC — {label}', save_path=bg_path)
        _print_phase_amplitude_table(bg_hist, f'{label} / Beta-Gamma')

        result[label] = {
            'comodulogram': _to_b64(como_path),
            'delta_theta':  _to_b64(dt_path),
            'beta_gamma':   _to_b64(bg_path),
        }
        print(f'  PAC done: {label}')

    # ── Diff comodulograms ──
    for diff_key, (phase_b, phase_a) in [
        ('diff_stimulation_vs_baseline', ('stimulation', 'baseline')),
        ('diff_recovery_vs_stimulation', ('recovery',    'stimulation')),
    ]:
        if phase_b not in mi_cache or phase_a not in mi_cache:
            result[diff_key] = ''
            continue

        diff_mat  = mi_cache[phase_b] - mi_cache[phase_a]
        diff_path = os.path.abspath(
            os.path.join('image', 'pac', f'{uuid}_{diff_key}.jpg')
        )
        title = diff_key.replace('diff_', '').replace('_vs_', ' − ').replace('_', ' ')
        _plot_diff_comodulogram(diff_mat, title=f'PAC ΔMI — {title}', save_path=diff_path)
        result[diff_key] = _to_b64(diff_path)
        print(f'  PAC diff done: {diff_key}')

    # ── ROI MI 요약 콘솔 출력 ──
    print('\n' + '='*56)
    print('  PAC MI Summary  (ROI 영역 평균)')
    print('='*56)
    print(f"  {'Phase':<14}  {'Delta-Theta (δ–θ)':>18}  {'Beta-Gamma (β–γ)':>18}")
    print(f"  {'':─<14}  {'phase 0.5–4 / amp 4–8':>18}  {'phase 13–20 / amp 30–45':>18}")
    print(f"  {'-'*56}")
    for lbl in ['baseline', 'stimulation', 'recovery']:
        if lbl in mi_cache:
            dt_mi = _roi_mi(mi_cache[lbl], 0.5,  4.0,  4.0,  8.0)
            bg_mi = _roi_mi(mi_cache[lbl], 13.0, 20.0, 30.0, 45.0)
            print(f"  {lbl:<14}  {dt_mi:>18.6f}  {bg_mi:>18.6f}")
        else:
            print(f"  {lbl:<14}  {'(no data)':>18}  {'(no data)':>18}")

    diff_pairs = [
        ('stim − base', 'stimulation', 'baseline'),
        ('rec  − stim', 'recovery',    'stimulation'),
    ]
    if any(a in mi_cache and b in mi_cache for _, a, b in diff_pairs):
        print(f"  {'-'*56}")
        for row_label, phase_b, phase_a in diff_pairs:
            if phase_b in mi_cache and phase_a in mi_cache:
                d    = mi_cache[phase_b] - mi_cache[phase_a]
                dt_d = _roi_mi(d, 0.5,  4.0,  4.0,  8.0)
                bg_d = _roi_mi(d, 13.0, 20.0, 30.0, 45.0)
                dt_m = '▲' if dt_d > 0 else '▽'
                bg_m = '▲' if bg_d > 0 else '▽'
                print(f"  {row_label:<14}  {dt_d:>+17.6f}{dt_m}  {bg_d:>+17.6f}{bg_m}")
    print('='*56 + '\n')

    return result

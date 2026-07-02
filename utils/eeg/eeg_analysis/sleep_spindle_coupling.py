# -*- coding:utf-8 -*-
"""
SO-Spindle Coupling — OneClick phase-aware wrapper
Based on Schreiner et al. (2021) & Staresina et al. (2015)

채널: P3, P4 (tVNS_SOSpindle 프로젝트와 동일)
수면단계: N1, N2, N3, REM 각각 독립 계산
"""

import numpy as np
import mne
from scipy.signal import hilbert
from scipy.stats import circmean

# ── 상수 ──────────────────────────────────────────────────────────────
COUPLING_CHANNELS    = ['P3', 'P4']
SO_BAND              = (0.16, 1.25)
SPINDLE_BAND         = (12.0, 16.0)
UP_STATE_WINDOW_DEG  = 180.0   # ±90° from SO peak
EPOCH_DUR_SEC        = 30.0

STAGE_N1  = {1}
STAGE_N2  = {2}
STAGE_N3  = {3}
STAGE_REM = {4}

_STAGES = [
    ('n1',  STAGE_N1,  'N1'),
    ('n2',  STAGE_N2,  'N2'),
    ('n3',  STAGE_N3,  'N3'),
    ('rem', STAGE_REM, 'REM'),
]

PHASE_NAMES = ['baseline', 'stimulation1', 'recovery1', 'stimulation2', 'recovery2']

_EMPTY_RESULT = {
    'coupled_ratio':  None,
    'MRL':            None,
    'mean_phase_deg': None,
    'n_SO':           0,
    'n_spindle':      0,
    'n_coupled':      0,
    'stage_used':     None,
}


# ══════════════════════════════════════════════════════════════════════
#  내부 신호 처리 함수
# ══════════════════════════════════════════════════════════════════════

def _bandpass(raw: mne.io.BaseRaw, ch: str, l: float, h: float) -> np.ndarray:
    return (raw.copy().pick(ch)
               .filter(l_freq=l, h_freq=h, method='iir',
                       iir_params={'order': 3, 'ftype': 'butter'}, verbose=False)
               .get_data()[0])


def _bad_mask_from_annotations(raw: mne.io.BaseRaw) -> np.ndarray:
    """BAD_ annotation → boolean mask (BAD_boundary 포함)."""
    mask = np.zeros(raw.n_times, dtype=bool)
    sfreq = raw.info['sfreq']
    for ann in raw.annotations:
        if ann['description'].startswith('BAD'):
            s = max(0, int(ann['onset'] * sfreq))
            e = min(raw.n_times, int((ann['onset'] + ann['duration']) * sfreq))
            mask[s:e] = True
    return mask


def _detect_SOs(raw, bad_mask, ch, amp_pct=75):
    sfreq = raw.info['sfreq']
    sig   = _bandpass(raw, ch, *SO_BAND)

    sign = np.sign(sig); sign[sign == 0] = 1
    zc   = np.where((sign[:-1] > 0) & (sign[1:] < 0))[0]

    min_s, max_s = int(0.8 * sfreq), int(3.0 * sfreq)
    cands = [(zc[i], zc[i+1]) for i in range(len(zc)-1)
             if min_s <= zc[i+1] - zc[i] <= max_s]

    valid = [(s, e) for s, e in cands
             if not bad_mask[s:min(e, len(bad_mask))].any()]
    if not valid:
        return []

    details = []
    for s, e in valid:
        seg = sig[s:e]
        ti, pi = int(np.argmin(seg)), int(np.argmax(seg))
        details.append({'start_sample': s, 'end_sample': e,
                        'trough_sample': s+ti, 'peak_sample': s+pi,
                        'amplitude': seg[pi] - seg[ti]})

    thr  = np.percentile([d['amplitude'] for d in details], amp_pct)
    SOs  = []
    for d in details:
        if d['amplitude'] >= thr:
            d['trough_time'] = d['trough_sample'] / sfreq
            d['peak_time']   = d['peak_sample']   / sfreq
            SOs.append(d)
    return SOs


def _detect_spindles(raw, bad_mask, ch, amp_pct=75):
    sfreq = raw.info['sfreq']
    sig   = _bandpass(raw, ch, *SPINDLE_BAND)
    n     = len(sig)

    hw = int(0.2 * sfreq) // 2
    cs = np.concatenate([[0], np.cumsum(sig**2)])
    sw = np.maximum(np.arange(n) - hw, 0)
    ew = np.minimum(np.arange(n) + hw, n)
    rms = np.sqrt((cs[ew] - cs[sw]) / (ew - sw))

    rms_c = rms.copy(); rms_c[bad_mask] = np.nan
    thr   = np.nanpercentile(rms_c, amp_pct)

    above = np.where(~np.isnan(rms_c), rms_c > thr, False).astype(int)
    diff  = np.diff(above, prepend=0, append=0)
    segs  = list(zip(np.where(diff == 1)[0], np.where(diff == -1)[0]))

    min_s, max_s = int(0.5 * sfreq), int(3.0 * sfreq)
    spindles = []
    for s, e in segs:
        if not (min_s <= e - s <= max_s): continue
        if bad_mask[s:e].any(): continue
        pi = int(np.argmax(rms[s:e]))
        spindles.append({'start_sample': s, 'end_sample': e,
                         'peak_sample': s+pi, 'peak_time': (s+pi)/sfreq,
                         'peak_rms': float(rms[s+pi])})
    return spindles


def _detect_coupling(SOs, spindles, raw, ch):
    if not SOs or not spindles:
        return []
    sfreq    = raw.info['sfreq']
    so_filt  = _bandpass(raw, ch, *SO_BAND)
    so_phase = np.angle(hilbert(so_filt))
    half_win = np.deg2rad(UP_STATE_WINDOW_DEG / 2.0)

    sp_peaks = np.array([sp['peak_sample'] for sp in spindles])
    events   = []
    for so in SOs:
        in_so = np.where((sp_peaks >= so['start_sample']) &
                         (sp_peaks <= so['end_sample']))[0]
        if len(in_so) == 0: continue

        kept = [j for j in in_so
                if abs(so_phase[spindles[j]['peak_sample']]) <= half_win]
        if not kept: continue

        best = kept[np.argmax([spindles[j]['peak_rms'] for j in kept])]
        sp   = spindles[best]
        events.append({
            'so': so, 'spindle': sp,
            'trough_sample':          so['trough_sample'],
            'trough_time':            so['trough_time'],
            'spindle_peak_time':      sp['peak_time'],
            'spindle_peak_phase_deg': float(np.degrees(so_phase[sp['peak_sample']])),
            'time_diff':              sp['peak_time'] - so['trough_time'],
        })
    return events


def _summarize(SOs, spindles, coupling_events, raw, ch):
    phases = np.array([
        np.angle(hilbert(_bandpass(raw, ch, *SO_BAND)))[ev['spindle']['peak_sample']]
        for ev in coupling_events
    ]) if coupling_events else np.array([])

    n_so, n_sp, n_c = len(SOs), len(spindles), len(coupling_events)
    if len(phases) > 0:
        mrl            = float(np.abs(np.mean(np.exp(1j * phases))))
        mean_phase_deg = float(np.degrees(circmean(phases, low=-np.pi, high=np.pi)))
    else:
        mrl = mean_phase_deg = None

    return {
        'n_SO':           n_so,
        'n_spindle':      n_sp,
        'n_coupled':      n_c,
        'coupled_ratio':  n_c / max(n_so, 1) if n_so > 0 else None,
        'MRL':            mrl,
        'mean_phase_deg': mean_phase_deg,
    }


def _run_one_channel(raw, bad_mask, ch):
    """단일 채널 SO-Spindle coupling 계산."""
    if ch not in raw.ch_names:
        return None
    SOs      = _detect_SOs(raw, bad_mask, ch)
    spindles = _detect_spindles(raw, bad_mask, ch)
    events   = _detect_coupling(SOs, spindles, raw, ch)
    return _summarize(SOs, spindles, events, raw, ch)


# ══════════════════════════════════════════════════════════════════════
#  Phase 분리 + 수면단계 필터링
# ══════════════════════════════════════════════════════════════════════

def _extract_phase_raw(raw, t_start_min, t_end_min, sleep_stages,
                       stage_filter):
    """
    phase 구간에서 stage_filter에 해당하는 30초 epoch만 연결한 Raw 반환.
    연결 경계는 BAD_boundary annotation으로 표시.
    """
    sfreq     = raw.info['sfreq']
    ep_start  = int(t_start_min * 2)          # 2 epochs/min
    ep_end    = int(t_end_min   * 2)
    n_stages  = len(sleep_stages)

    valid_eps = [i for i in range(ep_start, min(ep_end, n_stages))
                 if sleep_stages[i] in stage_filter]
    if not valid_eps:
        return None, None

    raws = []
    for ep_idx in valid_eps:
        t0 = ep_idx * EPOCH_DUR_SEC
        t1 = t0 + EPOCH_DUR_SEC
        t1 = min(t1, raw.times[-1])
        if t1 - t0 < 1.0:
            continue
        raws.append(raw.copy().crop(tmin=t0, tmax=t1, include_tmax=False))

    if not raws:
        return None, None

    raw_cat  = mne.concatenate_raws(raws, verbose=False)
    bad_mask = _bad_mask_from_annotations(raw_cat)
    return raw_cat, bad_mask


def _compute_one_condition(raw, t_start_min, t_end_min, sleep_stages,
                           stage_set, stage_label):
    """특정 수면단계 조건 하나에 대한 coupling 계산. 데이터 없으면 None 반환."""
    raw_ph, bad_mask = _extract_phase_raw(
        raw, t_start_min, t_end_min, sleep_stages, stage_set)
    if raw_ph is None:
        return None

    ch_results = {}
    for ch in COUPLING_CHANNELS:
        res = _run_one_channel(raw_ph, bad_mask, ch)
        if res is not None:
            ch_results[ch] = res

    if not ch_results:
        return None

    def _avg(key):
        vals = [r[key] for r in ch_results.values() if r[key] is not None]
        return float(np.mean(vals)) if vals else None

    return {
        'coupled_ratio':  _avg('coupled_ratio'),
        'MRL':            _avg('MRL'),
        'mean_phase_deg': _avg('mean_phase_deg'),
        'n_SO':           int(np.mean([r['n_SO']      for r in ch_results.values()])),
        'n_spindle':      int(np.mean([r['n_spindle'] for r in ch_results.values()])),
        'n_coupled':      int(np.mean([r['n_coupled'] for r in ch_results.values()])),
        'stage_used':     stage_label,
        **{ch: r for ch, r in ch_results.items()},
    }


def _coupling_for_phase(raw, t_start_min, t_end_min, sleep_stages):
    """
    한 phase에 대해 N1/N2/N3/REM 각각 독립 계산.
    Returns {'n1': {...}|None, 'n2': {...}|None, 'n3': {...}|None, 'rem': {...}|None}
    """
    return {
        key: _compute_one_condition(raw, t_start_min, t_end_min,
                                    sleep_stages, stage_set, label)
        for key, stage_set, label in _STAGES
    }


# ══════════════════════════════════════════════════════════════════════
#  공개 API
# ══════════════════════════════════════════════════════════════════════

def get_spindle_coupling_per_phase(filter_data, trigger, sleep_stages):
    """
    Phase별 SO-Spindle coupling 정량 지표 계산.

    Parameters
    ----------
    filter_data  : mne.io.RawArray  (100Hz, ICA cleaned)
    trigger      : list[int]  분 단위 경계 (end 포함, analysis.py와 동일)
    sleep_stages : list[int]  epoch별 수면단계 (0=W,1=N1,2=N2,3=N3,4=REM)

    Returns
    -------
    dict  {phase_name: {'n2': {metrics}|None, 'all': {metrics}|None}}
    """
    n_phases = len(trigger) - 1
    results  = {}

    for i in range(n_phases):
        phase_name = PHASE_NAMES[i] if i < len(PHASE_NAMES) else f'phase{i}'
        t_start    = trigger[i]
        t_end      = trigger[i + 1]
        print(f'[Spindle Coupling] {phase_name}  ({t_start}~{t_end} min) ...')
        try:
            results[phase_name] = _coupling_for_phase(
                filter_data, t_start, t_end, sleep_stages)
            for key, _, label in _STAGES:
                r = results[phase_name][key]
                print(f'  {label:3s}: coupled_ratio={r["coupled_ratio"] if r else "-"}  '
                      f'MRL={r["MRL"] if r else "-"}')
        except Exception as e:
            print(f'  [경고] {phase_name} 커플링 계산 실패: {e}')
            results[phase_name] = {'n2': None, 'all': None}

    return results

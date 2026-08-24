# -*- coding:utf-8 -*-
"""실험 phase 구성의 단일 정의.

지금까지 phase 이름 목록이 파일마다 복제돼 있었고 표기도 다섯 가지로 갈렸다
('stimulation1' / 'stimulation 1' / 'stimulation' / 'faa_stimulation1' /
'Stimulation1'). 6-phase 모드를 추가하면서 그 목록들을 여기 한 곳으로 모은다.

지원 모드 (phase 개수 -> 구성):
  1: baseline
  3: baseline, stimulation1, recovery1
  5: baseline, stimulation1, recovery1, stimulation2, recovery2
  6: baseline, stimulation1, stimulation2, stimulation3, stimulation4, recovery

6-phase 가 recovery1/recovery2 대신 'recovery' 라는 새 키를 쓰는 이유:
키 하나가 어디서든 정확히 하나의 뜻을 갖게 하기 위해서다. recovery1 을
재사용하면 5-phase 에선 '1차 회복'(3번째), 6-phase 에선 '회복'(6번째)이 되어
서버·프론트가 모드를 알아야만 라벨과 순서를 정할 수 있게 된다.
"""

# phase 개수 -> 정식 키 목록 (업로드 payload / DB 컬럼 / 프론트 모델과 일치)
PHASE_SETS = {
    1: ['baseline'],
    3: ['baseline', 'stimulation1', 'recovery1'],
    5: ['baseline', 'stimulation1', 'recovery1', 'stimulation2', 'recovery2'],
    6: ['baseline', 'stimulation1', 'stimulation2',
        'stimulation3', 'stimulation4', 'recovery'],
}

# 서버·프론트가 "존재하는 것만 걸러서" 순회할 때 쓰는 합집합 순서.
# 5-phase 세션은 stim3/stim4/recovery 가 비어 있고, 6-phase 세션은
# recovery1/recovery2 가 비어 있으므로, 걸러낸 결과의 상대 순서가
# 두 모드 모두 올바른 시간 순서가 된다.
PHASE_SUPERSET = ['baseline', 'stimulation1', 'recovery1', 'stimulation2',
                  'recovery2', 'stimulation3', 'stimulation4', 'recovery']

# 표시 라벨. 키 하나 = 라벨 하나 (모드와 무관).
PHASE_LABELS_KO = {
    'baseline':     '기준',
    'stimulation1': '1차 자극',
    'recovery1':    '1차 회복',
    'stimulation2': '2차 자극',
    'recovery2':    '2차 회복',
    'stimulation3': '3차 자극',
    'stimulation4': '4차 자극',
    'recovery':     '회복',
}

# ECG 차트 파일명 등에 쓰는 대문자 표기 (fig1_Baseline.png ...)
PHASE_TITLES = {
    'baseline':     'Baseline',
    'stimulation1': 'Stimulation1',
    'recovery1':    'Recovery1',
    'stimulation2': 'Stimulation2',
    'recovery2':    'Recovery2',
    'stimulation3': 'Stimulation3',
    'stimulation4': 'Stimulation4',
    'recovery':     'Recovery',
}


def phase_names(n_phases):
    """phase 개수에 맞는 정식 키 목록. 미지원 개수는 앞에서부터 잘라 쓰되
    모자라면 phase{i} 로 채운다 (기존 sleep_spindle_coupling 의 fallback 동작)."""
    if n_phases in PHASE_SETS:
        return list(PHASE_SETS[n_phases])
    base = PHASE_SETS[5]
    return [base[i] if i < len(base) else 'phase%d' % i for i in range(n_phases)]


def phase_names_spaced(n_phases):
    """brain_delta_* 모듈이 내부 키·파일명(pair_fn)에 쓰는 공백 표기.
    'stimulation1' -> 'stimulation 1'. 숫자 없는 키(baseline/recovery)는 그대로."""
    out = []
    for name in phase_names(n_phases):
        if name[-1].isdigit() and not name.startswith('phase'):
            out.append(name[:-1] + ' ' + name[-1])
        else:
            out.append(name)
    return out


def diff_pairs(n_phases, spaced=False):
    """인접 phase 쌍 (targ, ref) 목록 — diff1 부터 순서대로.

    5-phase: (stim1,base) (rec1,stim1) (stim2,rec1) (rec2,stim2)      -> diff1~4
    6-phase: (stim1,base) (stim2,stim1) (stim3,stim2) (stim4,stim3)
             (recovery,stim4)                                          -> diff1~5
    """
    names = phase_names_spaced(n_phases) if spaced else phase_names(n_phases)
    return [(names[i + 1], names[i]) for i in range(len(names) - 1)]


# diff 슬롯은 DB 컬럼(EEG_DIFF_1~5_ID)과 payload 키로 고정돼 있다.
# 6-phase 의 인접 쌍이 5개라 diff5 까지 필요하다.
MAX_DIFFS = 5
DIFF_KEYS = ['diff%d' % (i + 1) for i in range(MAX_DIFFS)]


def faa_names(n_phases):
    """FAA 결과 dict 키 ('faa_' 접두어 표기)."""
    return ['faa_' + name for name in phase_names(n_phases)]

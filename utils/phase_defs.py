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
  9: baseline, stimulation1, recovery1, stimulation2, recovery2,
     stimulation3, recovery3, stimulation4, recovery4

6-phase 가 recovery1/recovery2 대신 'recovery' 라는 새 키를 쓰는 이유:
키 하나가 어디서든 정확히 하나의 뜻을 갖게 하기 위해서다. recovery1 을
재사용하면 5-phase 에선 '1차 회복'(3번째), 6-phase 에선 '회복'(6번째)이 되어
서버·프론트가 모드를 알아야만 라벨과 순서를 정할 수 있게 된다.

9-phase 가 새로 쓰는 키는 recovery3, recovery4 둘뿐이다. 앞 다섯 개는 5-phase 와,
stimulation3/stimulation4 는 6-phase 와 뜻이 같아 그대로 쓴다. 6-phase 의
'recovery' 와 9-phase 의 'recovery4' 는 둘 다 4차 자극 뒤의 회복이지만 별개
키다(같은 세션에 함께 나오지 않는다).

모드를 고르는 인자는 없다. phase 개수는 마커 개수로만 정하고(phase_names),
클라이언트 ECG·EEG 와 프론트가 같은 규칙을 쓴다.
"""

# phase 개수 -> 정식 키 목록 (업로드 payload / DB 컬럼 / 프론트 모델과 일치)
PHASE_SETS = {
    1: ['baseline'],
    3: ['baseline', 'stimulation1', 'recovery1'],
    5: ['baseline', 'stimulation1', 'recovery1', 'stimulation2', 'recovery2'],
    6: ['baseline', 'stimulation1', 'stimulation2',
        'stimulation3', 'stimulation4', 'recovery'],
    9: ['baseline', 'stimulation1', 'recovery1', 'stimulation2', 'recovery2',
        'stimulation3', 'recovery3', 'stimulation4', 'recovery4'],
}

# 한 세션이 가질 수 있는 phase 의 최대 개수. 마커가 이보다 많으면 앞
# MAX_PHASES 개만 쓰고 마지막 phase 가 녹화 끝까지 간다 (phase_names 참고).
MAX_PHASES = 9

# 서버·프론트가 "존재하는 것만 걸러서" 순회할 때 쓰는 합집합 순서 (10키).
# 5·9-phase 세션은 'recovery' 가 비어 있고, 6-phase 세션은 recovery1~4 가
# 비어 있으므로, 세션에 없는 키를 걸러낸 결과의 상대 순서는 어느 모드든
# (중간에 끝난 2/4/7/8 개짜리 세션도) 올바른 시간 순서가 된다.
PHASE_SUPERSET = ['baseline', 'stimulation1', 'recovery1', 'stimulation2',
                  'recovery2', 'stimulation3', 'recovery3', 'stimulation4',
                  'recovery4', 'recovery']

# 표시 라벨. 키 하나 = 라벨 하나 (모드와 무관).
PHASE_LABELS_KO = {
    'baseline':     '기준',
    'stimulation1': '1차 자극',
    'recovery1':    '1차 회복',
    'stimulation2': '2차 자극',
    'recovery2':    '2차 회복',
    'stimulation3': '3차 자극',
    'recovery3':    '3차 회복',
    'stimulation4': '4차 자극',
    'recovery4':    '4차 회복',
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
    'recovery3':    'Recovery3',
    'stimulation4': 'Stimulation4',
    'recovery4':    'Recovery4',
    'recovery':     'Recovery',
}

# 자리가 좁은 곳(phase 가 7개 이상일 때의 스펙트로그램 라벨·콘솔 표)에 쓰는
# 짧은 표기. 프론트 PhaseCol.short 와 같은 문자열이다.
PHASE_SHORT = {
    'baseline':     'Baseline',
    'stimulation1': 'Stim1',
    'recovery1':    'Rec1',
    'stimulation2': 'Stim2',
    'recovery2':    'Rec2',
    'stimulation3': 'Stim3',
    'recovery3':    'Rec3',
    'stimulation4': 'Stim4',
    'recovery4':    'Rec4',
    'recovery':     'Rec',
}


def phase_names(n_phases):
    """phase 개수(= 마커 개수)에 맞는 정식 키 목록.

    클라이언트 ECG·EEG 와 프론트가 같은 규칙으로 판정한다:
      - 6 이면 6-phase 구성.
      - 그 밖에는 9-phase 순서의 앞 min(n, 9) 개. 1/3/5/9 는 정식 모드와
        같고, 2/4/7/8 은 '중간에 끝난 세션' 이라 앞에서부터 그 개수만큼의
        phase 로 본다.
      - 9 를 넘어도 9개까지만 돌려준다. 호출하는 쪽이 앞 9개 마커만 쓰고
        마지막 phase(recovery4)를 녹화 끝까지 가게 해야 한다.
      - 0 이하이면 1-phase (baseline).

    예전에는 모자란 자리를 'phase5' 같은 임시 이름으로 채웠는데, 서버가
    모르는 키라 업로드에서 조용히 버려졌다. 더 이상 만들지 않는다.

    알려진 한계: 마커 6개에서 끝난 9-phase 세션은 6-phase 로 해석된다
    (개수만으로는 구분할 수 없다).
    """
    if n_phases in PHASE_SETS:
        return list(PHASE_SETS[n_phases])
    return list(PHASE_SETS[MAX_PHASES][:max(1, min(n_phases, MAX_PHASES))])


def _spaced(name):
    """'stimulation1' -> 'stimulation 1'. 숫자 없는 키(baseline/recovery)는 그대로."""
    return name[:-1] + ' ' + name[-1] if name[-1].isdigit() else name


def phase_names_spaced(n_phases):
    """brain_delta_* 모듈이 내부 키·파일명(pair_fn)에 쓰는 공백 표기.
    'stimulation1' -> 'stimulation 1'. 숫자 없는 키(baseline/recovery)는 그대로."""
    return [_spaced(name) for name in phase_names(n_phases)]


# 합집합의 공백 표기. brain_delta_* 의 STAGES 는 반드시 이것을 가져다 쓴다 —
# 모듈마다 손으로 복제해 두면 phase 를 추가할 때 빠지고, 빠져도 예외 없이
# 그 phase 가 낀 수면단계별 diff 그림만 전부 'No data' 가 된다.
PHASE_SUPERSET_SPACED = [_spaced(name) for name in PHASE_SUPERSET]


def diff_pairs(n_phases, spaced=False):
    """인접 phase 쌍 (targ, ref) 목록 — diff1 부터 순서대로.

    5-phase: (stim1,base) (rec1,stim1) (stim2,rec1) (rec2,stim2)      -> diff1~4
    6-phase: (stim1,base) (stim2,stim1) (stim3,stim2) (stim4,stim3)
             (recovery,stim4)                                          -> diff1~5
    9-phase: (stim1,base) (rec1,stim1) (stim2,rec1) (rec2,stim2)
             (stim3,rec2) (rec3,stim3) (stim4,rec3) (rec4,stim4)       -> diff1~8

    슬롯은 위치 기반이라 같은 diffN 의 뜻이 모드마다 다르다 (diff5 는 6-phase
    에서 recovery-stim4, 9-phase 에서 stim3-rec2). 라벨은 반드시 그 세션의
    phase 목록에서 유도한다.
    """
    names = phase_names_spaced(n_phases) if spaced else phase_names(n_phases)
    return [(names[i + 1], names[i]) for i in range(len(names) - 1)]


# diff 슬롯은 DB 컬럼(EEG_DIFF_1~8_ID)과 payload 키로 고정돼 있다 (EEG 만.
# HRV 에는 diff 가 없다). 인접 쌍이 6-phase 는 5개, 9-phase 는 8개라 diff8
# 까지 필요하다. 쌍이 없는 슬롯은 빈 dict 로 나간다.
MAX_DIFFS = 8
DIFF_KEYS = ['diff%d' % (i + 1) for i in range(MAX_DIFFS)]


def faa_names(n_phases):
    """FAA 결과 dict 키 ('faa_' 접두어 표기)."""
    return ['faa_' + name for name in phase_names(n_phases)]

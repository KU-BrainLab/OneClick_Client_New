# -*- coding: utf-8 -*-
"""utils/phase_defs.py 단위 테스트 — phase 레지스트리의 계약(1/3/5/6/9-phase).

서버 CONNECTIVITY_PHASES·프론트 phase_defs.dart 와 손으로 맞추는 값들이라, 한쪽만
바뀌면 예외 없이 phase 가 빠지거나 라벨이 어긋난다. 여기서 클라이언트 쪽 값을 못박는다.

    python -m unittest tests.test_phase_defs -v
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils import phase_defs as P    # noqa: E402

NINE = ['baseline', 'stimulation1', 'recovery1', 'stimulation2', 'recovery2',
        'stimulation3', 'recovery3', 'stimulation4', 'recovery4']
SIX = ['baseline', 'stimulation1', 'stimulation2', 'stimulation3', 'stimulation4', 'recovery']
SUPERSET = NINE + ['recovery']


def in_superset_order(names):
    """서버·프론트가 하는 일: 합집합을 순서대로 돌며 세션에 있는 키만 남긴다."""
    return [k for k in P.PHASE_SUPERSET if k in names]


class RegistryTests(unittest.TestCase):
    def test_supported_modes(self):
        self.assertEqual(sorted(P.PHASE_SETS), [1, 3, 5, 6, 9])
        self.assertEqual(P.PHASE_SETS[9], NINE)
        self.assertEqual(P.MAX_PHASES, 9)
        self.assertEqual(P.MAX_PHASES, max(P.PHASE_SETS))

    def test_existing_modes_unchanged(self):
        # 회귀 금지: 기존 1/3/5/6-phase 의 키와 순서.
        self.assertEqual(P.phase_names(1), ['baseline'])
        self.assertEqual(P.phase_names(3), ['baseline', 'stimulation1', 'recovery1'])
        self.assertEqual(P.phase_names(5), ['baseline', 'stimulation1', 'recovery1',
                                            'stimulation2', 'recovery2'])
        self.assertEqual(P.phase_names(6), SIX)
        for n in (1, 3, 5, 6, 9):
            self.assertEqual(P.phase_names(n), P.PHASE_SETS[n])

    def test_superset_is_ten_keys_in_contract_order(self):
        self.assertEqual(P.PHASE_SUPERSET, SUPERSET)
        self.assertEqual(len(set(P.PHASE_SUPERSET)), 10)
        # 모든 모드의 키를 합치면 정확히 합집합이다 (남는 키도 빠진 키도 없다).
        self.assertEqual(set().union(*P.PHASE_SETS.values()), set(P.PHASE_SUPERSET))

    def test_filtering_superset_gives_time_order_in_every_mode(self):
        # 세션에 없는 키를 걸러내면 어느 모드든 남은 것의 순서가 시간 순서여야 한다.
        for n, names in P.PHASE_SETS.items():
            self.assertEqual(in_superset_order(names), names, n)
        # 중간에 끝난 세션(2/4/7/8)과 마커가 넘친 세션(10 이상)도 마찬가지.
        for n in range(0, 13):
            names = P.phase_names(n)
            self.assertEqual(in_superset_order(names), names, n)

    def test_no_label_or_title_missing(self):
        keys = set(P.PHASE_SUPERSET)
        self.assertEqual(set(P.PHASE_LABELS_KO), keys)
        self.assertEqual(set(P.PHASE_TITLES), keys)
        self.assertEqual(set(P.PHASE_SHORT), keys)
        for table in (P.PHASE_LABELS_KO, P.PHASE_TITLES, P.PHASE_SHORT):
            self.assertEqual(len(set(table.values())), 10, '라벨은 키마다 달라야 한다')

    def test_new_labels(self):
        self.assertEqual(P.PHASE_LABELS_KO['recovery3'], '3차 회복')
        self.assertEqual(P.PHASE_LABELS_KO['recovery4'], '4차 회복')
        self.assertEqual(P.PHASE_TITLES['recovery3'], 'Recovery3')
        self.assertEqual(P.PHASE_TITLES['recovery4'], 'Recovery4')
        self.assertEqual(P.PHASE_SHORT['recovery3'], 'Rec3')
        self.assertEqual(P.PHASE_SHORT['recovery4'], 'Rec4')

    def test_existing_labels_unchanged(self):
        self.assertEqual({k: P.PHASE_LABELS_KO[k] for k in SUPERSET if k not in ('recovery3', 'recovery4')}, {
            'baseline': '기준', 'stimulation1': '1차 자극', 'recovery1': '1차 회복',
            'stimulation2': '2차 자극', 'recovery2': '2차 회복', 'stimulation3': '3차 자극',
            'stimulation4': '4차 자극', 'recovery': '회복'})
        self.assertEqual({k: P.PHASE_TITLES[k] for k in SUPERSET if k not in ('recovery3', 'recovery4')}, {
            'baseline': 'Baseline', 'stimulation1': 'Stimulation1', 'recovery1': 'Recovery1',
            'stimulation2': 'Stimulation2', 'recovery2': 'Recovery2', 'stimulation3': 'Stimulation3',
            'stimulation4': 'Stimulation4', 'recovery': 'Recovery'})


class PhaseCountRuleTests(unittest.TestCase):
    """계약 3절: 6 이면 6-phase, 그 밖에는 9-phase 순서의 앞 min(n, 9) 개."""

    def test_no_marker_is_one_phase(self):
        self.assertEqual(P.phase_names(0), ['baseline'])

    def test_interrupted_sessions_are_prefixes_of_nine(self):
        self.assertEqual(P.phase_names(2), ['baseline', 'stimulation1'])
        self.assertEqual(P.phase_names(4), ['baseline', 'stimulation1', 'recovery1', 'stimulation2'])
        self.assertEqual(P.phase_names(7), NINE[:7])
        self.assertEqual(P.phase_names(7)[-1], 'recovery3')
        self.assertEqual(P.phase_names(8), NINE[:8])
        self.assertEqual(P.phase_names(8)[-1], 'stimulation4')

    def test_too_many_markers_are_cut_to_nine(self):
        self.assertEqual(P.phase_names(10), NINE)
        self.assertEqual(P.phase_names(12), NINE)

    def test_six_is_the_six_phase_set_not_a_prefix_of_nine(self):
        # 알려진 한계: 마커 6개에서 끝난 9-phase 세션은 6-phase 로 해석된다.
        self.assertEqual(P.phase_names(6), SIX)
        self.assertNotEqual(P.phase_names(6), NINE[:6])

    def test_length_rule_and_no_placeholder_names(self):
        for n in range(0, 21):
            names = P.phase_names(n)
            self.assertEqual(len(names), max(1, min(n, 9)), n)
            # 'phase5' 같은 임시 이름은 서버가 모르는 키라 더 이상 만들지 않는다.
            self.assertTrue(set(names) <= set(P.PHASE_SUPERSET), (n, names))

    def test_returns_a_copy(self):
        for n in (5, 7, 9, 12):
            P.phase_names(n).append('x')
            self.assertNotIn('x', P.phase_names(n))
        self.assertEqual(P.PHASE_SETS[9], NINE)


class DerivedNameTests(unittest.TestCase):
    def test_spaced_names(self):
        self.assertEqual(P.phase_names_spaced(9), [
            'baseline', 'stimulation 1', 'recovery 1', 'stimulation 2', 'recovery 2',
            'stimulation 3', 'recovery 3', 'stimulation 4', 'recovery 4'])
        self.assertEqual(P.phase_names_spaced(6), [
            'baseline', 'stimulation 1', 'stimulation 2', 'stimulation 3', 'stimulation 4',
            'recovery'])
        self.assertEqual(P.phase_names_spaced(5), [
            'baseline', 'stimulation 1', 'recovery 1', 'stimulation 2', 'recovery 2'])

    def test_spaced_superset_covers_every_mode(self):
        # brain_delta_* 의 STAGES 가 이것을 쓴다. 여기 없는 stage 는 예외 없이
        # 계산에서 빠져 그 phase 가 낀 diff 그림이 'No data' 가 된다.
        self.assertEqual(P.PHASE_SUPERSET_SPACED, [
            'baseline', 'stimulation 1', 'recovery 1', 'stimulation 2', 'recovery 2',
            'stimulation 3', 'recovery 3', 'stimulation 4', 'recovery 4', 'recovery'])
        for n in range(0, 13):
            self.assertTrue(set(P.phase_names_spaced(n)) <= set(P.PHASE_SUPERSET_SPACED), n)

    def test_faa_names(self):
        self.assertEqual(P.faa_names(9), ['faa_' + k for k in NINE])
        self.assertEqual(P.faa_names(9)[-3:], ['faa_recovery3', 'faa_stimulation4', 'faa_recovery4'])
        self.assertEqual(P.faa_names(5), ['faa_baseline', 'faa_stimulation1', 'faa_recovery1',
                                          'faa_stimulation2', 'faa_recovery2'])
        self.assertEqual(P.faa_names(6), ['faa_' + k for k in SIX])


class DiffTests(unittest.TestCase):
    def test_diff_keys(self):
        self.assertEqual(P.MAX_DIFFS, 8)
        self.assertEqual(P.MAX_DIFFS, P.MAX_PHASES - 1)
        self.assertEqual(P.DIFF_KEYS, ['diff1', 'diff2', 'diff3', 'diff4',
                                       'diff5', 'diff6', 'diff7', 'diff8'])

    def test_nine_phase_pairs(self):
        # diffK = phase[K] - phase[K-1]. (targ, ref) 순서.
        pairs = P.diff_pairs(9)
        self.assertEqual(len(pairs), 8)
        self.assertEqual(pairs[0], ('stimulation1', 'baseline'))        # diff1
        self.assertEqual(pairs[1], ('recovery1', 'stimulation1'))       # diff2
        self.assertEqual(pairs[2], ('stimulation2', 'recovery1'))       # diff3
        self.assertEqual(pairs[3], ('recovery2', 'stimulation2'))       # diff4
        self.assertEqual(pairs[4], ('stimulation3', 'recovery2'))       # diff5
        self.assertEqual(pairs[5], ('recovery3', 'stimulation3'))       # diff6
        self.assertEqual(pairs[6], ('stimulation4', 'recovery3'))       # diff7
        self.assertEqual(pairs[7], ('recovery4', 'stimulation4'))       # diff8

    def test_nine_phase_pairs_spaced(self):
        self.assertEqual(P.diff_pairs(9, spaced=True), [
            ('stimulation 1', 'baseline'), ('recovery 1', 'stimulation 1'),
            ('stimulation 2', 'recovery 1'), ('recovery 2', 'stimulation 2'),
            ('stimulation 3', 'recovery 2'), ('recovery 3', 'stimulation 3'),
            ('stimulation 4', 'recovery 3'), ('recovery 4', 'stimulation 4')])

    def test_existing_pairs_unchanged(self):
        self.assertEqual(P.diff_pairs(1), [])
        self.assertEqual(P.diff_pairs(3), [('stimulation1', 'baseline'),
                                           ('recovery1', 'stimulation1')])
        self.assertEqual(P.diff_pairs(5), [('stimulation1', 'baseline'),
                                           ('recovery1', 'stimulation1'),
                                           ('stimulation2', 'recovery1'),
                                           ('recovery2', 'stimulation2')])
        self.assertEqual(P.diff_pairs(6), [('stimulation1', 'baseline'),
                                           ('stimulation2', 'stimulation1'),
                                           ('stimulation3', 'stimulation2'),
                                           ('stimulation4', 'stimulation3'),
                                           ('recovery', 'stimulation4')])
        # 9-phase 의 diff1~4 는 5-phase 와 같다.
        self.assertEqual(P.diff_pairs(9)[:4], P.diff_pairs(5))

    def test_pairs_never_exceed_slots(self):
        for n in range(0, 21):
            pairs = P.diff_pairs(n)
            self.assertEqual(len(pairs), len(P.phase_names(n)) - 1, n)
            self.assertLessEqual(len(pairs), P.MAX_DIFFS, n)


if __name__ == '__main__':
    unittest.main(verbosity=2)

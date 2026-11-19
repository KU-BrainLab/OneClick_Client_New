# -*- coding: utf-8 -*-
"""phase 개수(= 마커 개수)별 구간 나누기 단위 테스트 — 측정 파일 없이.

ECGFeatureExtractor.extract() 에서 HRV 계산(feature_extract/whole)과 그림 읽기만
가짜로 바꾸고, 마커 1~12개에서 실제로 잘리는 ECG 구간을 본다. 구간이 겹치거나
비면 예외도 경고도 없이 HRV 값만 틀리게 올라가므로 여기서 잡는다
(9-phase 를 넣을 때 recovery2 가 녹화 끝까지 늘어나는 식).

    python -m unittest tests.test_phase_modes -v
"""
import contextlib
import io
import os
import sys
import unittest
from unittest import mock

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils import phase_defs as P    # noqa: E402

MIN = 7500                 # 125 Hz x 60 초 = 1분
REC_ROWS = 60 * MIN        # 60분 녹화
NINE = ['baseline', 'stimulation1', 'recovery1', 'stimulation2', 'recovery2',
        'stimulation3', 'recovery3', 'stimulation4', 'recovery4']
CONTRACT5 = ['baseline', 'stimulation1', 'recovery1', 'stimulation2', 'recovery2']


def marker_rows(n):
    """정리된 CSV 기준 마커 행. 첫 마커는 0행, 나머지는 정분에서 조금(3.2초~) 뒤."""
    step = (60 // n) * MIN
    return [0] + [i * step + 400 + 13 * i for i in range(1, n)]


def run_extract(markers, n_rows=REC_ROWS):
    """extract() 를 돌려 (잘린 구간 목록, payload, 반환 trigger, 콘솔 출력)을 돌려준다.

    ecg 값을 행 번호로 두어, feature_extract 가 받은 조각의 처음·끝 값으로
    [시작, 끝) 행을 그대로 알 수 있게 한다.
    """
    from utils.ecg.feature_extraction import ECGFeatureExtractor
    ext = object.__new__(ECGFeatureExtractor)       # __init__ 은 CSV 를 읽는다 — 건너뛴다
    ext.sfreq = 125
    ext.save_path = '.'
    ext.ecg = pd.Series(np.arange(n_rows))
    ext.filtered_trigger = np.asarray(markers, dtype=np.int64)
    segments = []

    def fake_feature_extract(seg, whole=False, phase=''):
        seg = np.asarray(seg)
        segments.append((phase, int(seg[0]), int(seg[-1]) + 1))
        return {'rmssd': float(len(seg))}, {'frequency': [], 'power': []}

    ext.feature_extract = fake_feature_extract
    ext.whole = lambda: ([], [])
    ext.get_image_encoder = lambda path: os.path.basename(path)

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        sample, trigger = ext.extract()
    return segments, sample, trigger, out.getvalue()


class EcgSegmentTests(unittest.TestCase):
    def test_segments_are_contiguous_and_cover_the_recording(self):
        for n in range(1, 13):
            segments, _, _, _ = run_extract(marker_rows(n))
            self.assertEqual(segments[0][1], 0, n)
            self.assertEqual(segments[-1][2], REC_ROWS, n)
            for (_, _, end), (_, start, _) in zip(segments, segments[1:]):
                self.assertEqual(end, start, (n, segments))

    def test_phase_i_runs_from_marker_i_to_marker_i_plus_1(self):
        for n in range(1, 13):
            markers = marker_rows(n)
            names = P.phase_names(n)
            segments, _, _, _ = run_extract(markers)
            self.assertEqual([title for title, _, _ in segments],
                             [P.PHASE_TITLES[k] for k in names], n)
            # 앞 len(names) 개 마커만 경계로 쓴다. 마지막 phase 는 녹화 끝까지.
            self.assertEqual([start for _, start, _ in segments], markers[:len(names)], n)

    def test_payload_keys_equal_phase_names(self):
        for n in range(1, 13):
            names = P.phase_names(n)
            _, sample, _, _ = run_extract(marker_rows(n))
            filled = [k for k, v in sample.items() if isinstance(v, dict) and v]
            self.assertEqual(filled, names, n)                     # 키와 순서(시간 순서)
            self.assertEqual([k for k in P.PHASE_SUPERSET if sample.get(k)], names, n)
            self.assertIn('nni', sample)
            self.assertIn('rmssd', sample)
            # 서버 5-phase 계약 키는 없는 phase 라도 빈 dict 로 항상 있다.
            for key in CONTRACT5:
                self.assertIn(key, sample, n)
                if key not in names:
                    self.assertEqual(sample[key], {}, (n, key))
            self.assertTrue(set(sample) <= {'nni', 'rmssd'} | set(P.PHASE_SUPERSET), n)

    def test_returned_trigger_has_one_minute_per_used_marker(self):
        for n in range(1, 13):
            markers = marker_rows(n)
            names = P.phase_names(n)
            _, _, trigger, _ = run_extract(markers)
            self.assertEqual(len(trigger), len(names), n)          # EEG 가 이 길이로 phase 를 센다
            self.assertEqual(len(trigger), min(n, 9), n)
            self.assertEqual(trigger, [row // MIN for row in markers[:len(names)]], n)
            self.assertTrue(all(type(t) is int for t in trigger), n)

    def test_more_than_nine_markers_are_cut_to_nine(self):
        for n in (10, 11, 12):
            markers = marker_rows(n)
            segments, sample, trigger, log = run_extract(markers)
            self.assertEqual(len(segments), 9, n)
            self.assertEqual(len(trigger), 9, n)
            self.assertEqual([k for k in P.PHASE_SUPERSET if sample.get(k)], NINE, n)
            # 9번째 phase(recovery4)는 뒤에 마커가 더 있어도 녹화 끝까지 간다.
            self.assertEqual(segments[-1], ('Recovery4', markers[8], REC_ROWS), n)
            self.assertIn('경고', log, n)
            self.assertIn('마커가 %d개' % n, log, n)
            self.assertIn('%d개(' % (n - 9), log, n)                # 무엇을 무시했는지

    def test_no_warning_up_to_nine_markers(self):
        for n in range(1, 10):
            _, _, _, log = run_extract(marker_rows(n))
            self.assertNotIn('경고', log, n)

    def test_recovery2_runs_to_the_end_only_when_it_is_last(self):
        m5 = marker_rows(5)
        segments, _, _, _ = run_extract(m5)
        self.assertEqual(segments[4], ('Recovery2', m5[4], REC_ROWS))
        # 9-phase 에서 recovery2 는 5번째(중간) 구간이다. 끝까지 가면 안 된다.
        for n in (7, 8, 9, 12):
            m = marker_rows(n)
            segments, _, _, _ = run_extract(m)
            self.assertEqual(segments[4], ('Recovery2', m[4], m[5]), n)

    def test_existing_modes_unchanged(self):
        # 회귀 금지: 1/3/5/6-phase 의 구간과 payload 키 순서.
        expected_titles = {
            1: ['Baseline'],
            3: ['Baseline', 'Stimulation1', 'Recovery1'],
            5: ['Baseline', 'Stimulation1', 'Recovery1', 'Stimulation2', 'Recovery2'],
            6: ['Baseline', 'Stimulation1', 'Stimulation2', 'Stimulation3', 'Stimulation4',
                'Recovery'],
        }
        expected_keys = {
            1: ['nni', 'rmssd'] + CONTRACT5,
            3: ['nni', 'rmssd'] + CONTRACT5,
            5: ['nni', 'rmssd'] + CONTRACT5,
            6: ['nni', 'rmssd'] + CONTRACT5 + ['stimulation3', 'stimulation4', 'recovery'],
        }
        for n, titles in expected_titles.items():
            m = marker_rows(n)
            segments, sample, trigger, _ = run_extract(m)
            bounds = m + [REC_ROWS]
            self.assertEqual(segments, [(titles[i], bounds[i], bounds[i + 1]) for i in range(n)], n)
            self.assertEqual(list(sample), expected_keys[n], n)
            self.assertEqual(trigger, [row // MIN for row in m], n)

    def test_nine_phase_payload(self):
        m = marker_rows(9)
        segments, sample, trigger, _ = run_extract(m)
        bounds = m + [REC_ROWS]
        self.assertEqual(segments, [(P.PHASE_TITLES[NINE[i]], bounds[i], bounds[i + 1])
                                    for i in range(9)])
        self.assertEqual(list(sample), ['nni', 'rmssd'] + NINE)
        self.assertEqual(trigger, [0, 6, 12, 18, 24, 30, 36, 42, 48])
        for key, title in (('recovery3', 'Recovery3'), ('recovery4', 'Recovery4')):
            self.assertEqual(sample[key]['heart_rate'], 'fig1_%s.png' % title)
            self.assertEqual(sample[key]['comparison'], 'fig2_%s.png' % title)
            self.assertIn('psd', sample[key])

    def test_interrupted_sessions_use_the_nine_phase_prefix(self):
        # 2/4/7/8 은 중간에 끝난 세션: 앞에서부터 그 개수만큼, 마지막이 끝까지.
        for n, last in ((2, 'Stimulation1'), (4, 'Stimulation2'),
                        (7, 'Recovery3'), (8, 'Stimulation4')):
            m = marker_rows(n)
            segments, sample, trigger, _ = run_extract(m)
            self.assertEqual(len(segments), n)
            self.assertEqual(segments[-1], (last, m[-1], REC_ROWS), n)
            self.assertEqual([k for k in P.PHASE_SUPERSET if sample.get(k)], NINE[:n], n)
            self.assertEqual(len(trigger), n)


class CropTriggerRuleTests(unittest.TestCase):
    """main.analyze_eeg_with_crop: 크롭본에서 다시 센 마커 목록도 ECG 와 같은 규칙으로
    자른 뒤 개수를 비교한다. 안 그러면 마커 10개 이상인 세션이 매번 원본으로 폴백한다."""

    ROWS = 70 * MIN

    def run_crop(self, ecg_trigger, cropped_trigger):
        import main
        calls, logs = [], []

        def fake_eeg_analysis(path, trigger, sleep_model=None):
            calls.append((os.path.basename(path), list(trigger)))
            return {'ok': True}

        trigger = list(ecg_trigger)
        with mock.patch.object(main, 'auto_crop_csv',
                               return_value=([], list(cropped_trigger))), \
                mock.patch.object(main, 'count_rows', return_value=self.ROWS), \
                mock.patch.object(main, 'eeg_analysis', side_effect=fake_eeg_analysis):
            result = main.analyze_eeg_with_crop('data', 'x.csv', trigger, crop=True,
                                                log=logs.append)
        self.assertEqual(result, {'ok': True})
        return calls, logs, trigger, main.TEMP_FILE_NAME

    def test_twelve_markers_do_not_fall_back(self):
        ecg = [0, 5, 10, 15, 20, 25, 30, 35, 40]                    # extract() 가 9개로 자른 것
        cropped = [0, 5, 10, 15, 19, 24, 29, 34, 39, 44, 49, 54]    # 크롭본에는 12개 그대로
        calls, logs, trigger, temp = self.run_crop(ecg, cropped)
        self.assertEqual(calls, [(temp, cropped[:9])])              # 크롭본 + 앞 9개
        self.assertFalse([msg for msg in logs if '폴백' in msg], logs)
        self.assertEqual(trigger, ecg + [70])                       # 서버용: 경계값 10개

    def test_up_to_nine_markers_pass_through(self):
        for n in range(1, 10):
            ecg = [i * 5 for i in range(n)]
            cropped = [max(i * 5 - 1, 0) for i in range(n)]
            calls, logs, trigger, temp = self.run_crop(ecg, cropped)
            self.assertEqual(calls, [(temp, cropped)], n)
            self.assertFalse([msg for msg in logs if '폴백' in msg], n)
            self.assertEqual(trigger, ecg + [70], n)

    def test_real_mismatch_still_falls_back_to_the_source(self):
        ecg = [0, 10, 25, 30, 45]
        calls, logs, trigger, _ = self.run_crop(ecg, [0, 10, 24, 29])
        self.assertEqual(calls, [('x.csv', ecg)])
        self.assertTrue([msg for msg in logs if '폴백' in msg], logs)
        self.assertEqual(trigger, ecg + [70])

    def test_ecg_failure_stays_one_phase(self):
        calls, _, trigger, temp = self.run_crop([], list(range(0, 60, 5)))
        self.assertEqual(calls, [(temp, [])])
        self.assertEqual(trigger, [0, 70])


class DeltaStagesTests(unittest.TestCase):
    """brain_delta_* 의 STAGES 는 phase_defs 에서 나온다. 손으로 복제하면 새 phase 가
    빠지고, 빠지면 예외 없이 그 phase 가 낀 수면단계별 diff 그림이 'No data' 가 된다."""

    def test_stages_cover_every_phase_of_every_mode(self):
        from utils.eeg.eeg_analysis import brain_delta_fc, brain_delta_power_topo
        for module in (brain_delta_fc, brain_delta_power_topo):
            self.assertEqual(module.STAGES, P.PHASE_SUPERSET_SPACED, module.__name__)
            for n in range(0, 13):
                missing = set(P.phase_names_spaced(n)) - set(module.STAGES)
                self.assertFalse(missing, (module.__name__, n, missing))
            self.assertIn('recovery 3', module.STAGES)
            self.assertIn('recovery 4', module.STAGES)


if __name__ == '__main__':
    unittest.main()

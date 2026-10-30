# -*- coding: utf-8 -*-
"""원격 제출 경로 통합 테스트 — analysis_server(실제 HTTP) + remote_gui.run_submission(실제 압축·업로드·폴링).

파이프라인(main.py) 대신 진행 로그를 천천히 찍는 스텁을 CLIENT_DIR 에 두고 돈다.
확인하는 것:
  - gzip 으로 보낸 CSV 가 서버에서 원본과 같은 바이트로 풀려 저장된다
  - 파이프라인이 끝나기 전에 로그 줄이 제출 PC 에 도착한다(-u 비버퍼링 + 짧은 폴링)
  - 상태 문구에 경과 시간·대기 순번이 들어가고, 끝나면 소요 시간 요약이 찍힌다
  - 압축 없이 보낸 옛 클라이언트도 그대로 받는다

    python -m unittest tests.test_remote_pipeline -v
"""
import gzip
import io
import json
import os
import queue
import sys
import tempfile
import threading
import time
import unittest

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

STUB_MAIN = '''# -*- coding: utf-8 -*-
import sys, time
print("step 1 (args: %d)" % len(sys.argv))
time.sleep(4)            # 이 사이에 제출 PC 가 'step 1' 을 봐야 한다
print("step 2")
print("[RESULT] exp_pk=42")
'''


class RemotePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix='oneclick_test_')
        with io.open(os.path.join(cls.tmp, 'main.py'), 'w', encoding='utf-8') as f:
            f.write(STUB_MAIN)
        os.environ['ONECLICK_CLIENT_DIR'] = cls.tmp
        os.environ['ONECLICK_CLIENT_PYTHON'] = sys.executable
        import analysis_server as srv          # 환경변수를 읽은 뒤 import
        cls.srv = srv
        cls.httpd = srv.ThreadingHTTPServer(('127.0.0.1', 0), srv.Handler)
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        threading.Thread(target=srv._worker, daemon=True).start()
        # 측정 CSV 흉내: 숫자가 많아 gzip 이 잘 듣는다 (약 3MB)
        cls.csv = os.path.join(cls.tmp, 'sample.csv')
        with open(cls.csv, 'wb') as f:
            for i in range(60000):
                f.write(('%d,' % i + ','.join('%.3f' % ((i * k) % 97 / 7.0) for k in range(1, 17)) + '\n').encode())
        cls.params = {'name': 'TEST', 'age': '30', 'birth': '1996-01-01', 'sex': 'male',
                      'measurement_date': '2026-10-30 10:00'}

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()

    def base(self):
        return '127.0.0.1:%d' % self.port

    def test_gzip_submission_streams_log_and_reports_timing(self):
        import remote_gui as g
        g.POLL_INTERVAL_SEC = 1
        q = queue.Queue()
        t0 = time.time()
        g.run_submission(self.base(), self.csv, self.params, q)
        msgs = []
        while True:
            try:
                msgs.append(q.get_nowait())
            except queue.Empty:
                break
        kinds = [m[0] for m in msgs]
        self.assertEqual(kinds[-1], 'finish')
        self.assertTrue(msgs[-1][1], msgs[-1])
        self.assertIn('42', msgs[-1][2])                      # exp_pk 전달
        logs = ''.join(m[1] for m in msgs if m[0] == 'log')
        statuses = [m[1] for m in msgs if m[0] == 'status']
        # 압축 로그·소요 시간 요약
        self.assertIn('MB', logs)
        self.assertTrue(any(('압축' in s) for s in statuses), statuses[:5])
        self.assertIn('소요 시간', logs)
        # 끝나기 전에 'step 1' 을 봤는지: step 1 로그 메시지 뒤에 '분석 중' 상태가 또 와야 한다
        idx_step1 = next(i for i, m in enumerate(msgs) if m[0] == 'log' and 'step 1' in m[1])
        idx_step2 = next(i for i, m in enumerate(msgs) if m[0] == 'log' and 'step 2' in m[1])
        self.assertLess(idx_step1, idx_step2)
        between = [m for m in msgs[idx_step1:idx_step2] if m[0] == 'status']
        self.assertTrue(any('경과' in s[1] and '마지막 로그' in s[1] for s in between),
                        '실시간 로그·경과 표시가 없음: %r' % between[:3])
        # 서버 쪽: gzip 으로 받아 원본과 같은 바이트로 풀렸는지 (테스트 순서와 무관하게 최신 작업을 본다)
        jobs_dir = os.path.join(self.tmp, 'jobs')
        job = max(int(d) for d in os.listdir(jobs_dir)
                  if d.isdigit() and os.path.exists(os.path.join(jobs_dir, d, 'status.json')))
        st = json.load(open(os.path.join(jobs_dir, str(job), 'status.json'), encoding='utf-8'))
        self.assertEqual(st['compressed'], 'gzip')
        self.assertEqual(st['size'], os.path.getsize(self.csv))
        self.assertLess(st['size_wire'], st['size'] / 2)
        with open(os.path.join(jobs_dir, str(job), 'upload.csv'), 'rb') as a, open(self.csv, 'rb') as b:
            self.assertEqual(a.read(), b.read())
        self.assertLess(time.time() - t0, 60)

    def test_old_server_gets_raw_upload(self):
        # 서버가 gzip 을 모르는 버전으로 보이면(server_time 없음) 압축하지 않고 보낸다
        import remote_gui as g
        from unittest import mock
        g.POLL_INTERVAL_SEC = 1
        q = queue.Queue()
        with mock.patch.object(g, '_server_supports_gzip', return_value=False):
            g.run_submission(self.base(), self.csv, self.params, q)
        msgs = []
        while True:
            try:
                msgs.append(q.get_nowait())
            except queue.Empty:
                break
        self.assertTrue(msgs[-1][1], msgs[-1])
        logs = ''.join(m[1] for m in msgs if m[0] == 'log')
        self.assertNotIn('압축:', logs)
        jobs_dir = os.path.join(self.tmp, 'jobs')
        job = max(int(d) for d in os.listdir(jobs_dir)
                  if d.isdigit() and os.path.exists(os.path.join(jobs_dir, d, 'status.json')))
        st = json.load(open(os.path.join(jobs_dir, str(job), 'status.json'), encoding='utf-8'))
        self.assertIsNone(st['compressed'])
        self.assertEqual(st['size'], os.path.getsize(self.csv))

    def test_raw_submission_and_status_fields(self):
        # 옛 클라이언트(비압축) 호환 + 상태 응답의 시각·순번 필드
        with open(self.csv, 'rb') as f:
            r = requests.post('http://%s/jobs' % self.base(), params=self.params, data=f, timeout=60)
        self.assertEqual(r.status_code, 200, r.text)
        job = r.json()['job']
        self.assertIn('queue_ahead', r.json())
        st = requests.get('http://%s/jobs/%d' % (self.base(), job), timeout=30).json()
        for key in ('submitted', 'started', 'finished', 'queue_ahead', 'log_bytes', 'server_time'):
            self.assertIn(key, st)
        deadline = time.time() + 60
        while st['status'] not in ('done', 'failed') and time.time() < deadline:
            time.sleep(0.5)
            st = requests.get('http://%s/jobs/%d' % (self.base(), job), timeout=30).json()
        self.assertEqual(st['status'], 'done', st)
        self.assertEqual(st['exp_pk'], 42)
        self.assertGreaterEqual(st['finished'], st['started'])
        self.assertGreaterEqual(st['started'], st['submitted'])
        with open(os.path.join(self.tmp, 'jobs', str(job), 'upload.csv'), 'rb') as a, open(self.csv, 'rb') as b:
            self.assertEqual(a.read(), b.read())

    def test_capability_probe(self):
        st = requests.get('http://%s/jobs/0' % self.base(), timeout=30).json()
        self.assertIs(st['gzip'], True)
        self.assertEqual(st['job'], 0)
        for key in ('queue_length', 'server_time', 'server'):
            self.assertIn(key, st)
        import remote_gui as g
        self.assertTrue(g._server_supports_gzip('http://%s' % self.base(), True))
        self.assertFalse(g._server_supports_gzip('http://127.0.0.1:9', True))     # 연결 불가 → 비압축

    def test_bad_gzip_is_rejected(self):
        r = requests.post('http://%s/jobs' % self.base(), params=dict(self.params, compressed='gzip'),
                          data=b'not gzip at all' * 100, timeout=60)
        self.assertEqual(r.status_code, 400)
        self.assertIn('gzip', r.json()['error'])
        r = requests.post('http://%s/jobs' % self.base(), params=dict(self.params, compressed='zip'),
                          data=b'x', timeout=60)
        self.assertEqual(r.status_code, 400)

    def test_queue_ahead_counts_running_and_pending(self):
        body = gzip.compress(b'a,b\n1,2\n' * 1000)
        p = dict(self.params, compressed='gzip')
        first = requests.post('http://%s/jobs' % self.base(), params=p, data=body, timeout=60).json()
        second = requests.post('http://%s/jobs' % self.base(), params=p, data=body, timeout=60).json()
        self.assertGreaterEqual(second['queue_ahead'], 1)         # 앞 작업(돌고 있거나 대기)이 하나 이상
        st = requests.get('http://%s/jobs/%d' % (self.base(), second['job']), timeout=30).json()
        if st['status'] == 'queued':
            self.assertGreaterEqual(st['queue_ahead'], 1)
        deadline = time.time() + 60
        while time.time() < deadline:
            st = requests.get('http://%s/jobs/%d' % (self.base(), second['job']), timeout=30).json()
            if st['status'] in ('done', 'failed'):
                break
            time.sleep(0.5)
        self.assertEqual(st['status'], 'done')
        self.assertEqual(st['queue_ahead'], 0)
        self.assertIsNotNone(first['job'])


if __name__ == '__main__':
    unittest.main()

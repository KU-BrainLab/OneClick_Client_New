# -*- coding: utf-8 -*-
"""원격 제출 경로 통합 테스트 — analysis_server(실제 HTTP) + remote_gui.run_submission(실제 압축·업로드·폴링).

파이프라인(main.py) 대신 진행 로그를 천천히 찍는 스텁을 CLIENT_DIR 에 두고 돈다.
확인하는 것:
  - gzip 으로 보낸 CSV 가 서버에서 원본과 같은 바이트로 풀려 저장된다
  - 파이프라인이 끝나기 전에 로그 줄이 제출 PC 에 도착한다(-u 비버퍼링 + 짧은 폴링)
  - 로그가 서버 꼬리 창(8000바이트)보다 커져도 중복·누락 없이 따라간다
  - 상태 문구에 경과 시간·대기 순번이 들어가고, 끝나면 소요 시간 요약이 찍힌다
  - 압축 없이 보낸 옛 클라이언트도 그대로 받는다
  - 압축을 푼 크기가 상한을 넘는 본문(압축 폭탄)은 413 으로 거절하고 작업 폴더를 남기지 않는다
  - 상태 기록이 실패해도 워커 스레드가 살아 다음 작업을 돌린다

    python -m unittest tests.test_remote_pipeline -v
"""
import gzip
import io
import json
import os
import queue
import re
import shutil
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

STUB_MAIN = '''# -*- coding: utf-8 -*-
import os, sys, time
mode = os.environ.get('ONECLICK_STUB_MODE', '')
if mode == 'big':
    # 로그 창(8000바이트)보다 큰 출력: 100줄 -> 3초 -> 400줄 한꺼번에 -> 3초 -> 50줄 (한 줄 41~42바이트)
    for i in range(100):
        print('line %04d ' % i + 'x' * 30)
    time.sleep(3)
    for i in range(100, 500):
        print('line %04d ' % i + 'x' * 30)
    time.sleep(3)
    for i in range(500, 550):
        print('line %04d ' % i + 'x' * 30)
    print('[RESULT] exp_pk=43')
else:
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
        cls.jobs_dir = os.path.join(cls.tmp, 'jobs')
        os.makedirs(cls.jobs_dir)
        cls.httpd = srv.ThreadingHTTPServer(('127.0.0.1', 0), srv.Handler)
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        threading.Thread(target=srv._worker, daemon=True).start()
        # 측정 CSV 흉내: 숫자가 많아 gzip 이 잘 듣는다 (약 6MB)
        cls.csv = os.path.join(cls.tmp, 'sample.csv')
        with open(cls.csv, 'wb') as f:
            for i in range(60000):
                f.write(('%d,' % i + ','.join('%.3f' % ((i * k) % 97 / 7.0) for k in range(1, 17)) + '\n').encode())
        cls.params = {'name': 'TEST', 'age': '30', 'birth': '1996-01-01', 'sex': 'male',
                      'measurement_date': '2026-10-30 10:00'}

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        os.environ.pop('ONECLICK_STUB_MODE', None)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # ── 도우미 ──────────────────────────────────────────────────────────
    def base(self):
        return '127.0.0.1:%d' % self.port

    def url(self, path):
        return 'http://%s%s' % (self.base(), path)

    @staticmethod
    def drain(q):
        msgs = []
        while True:
            try:
                msgs.append(q.get_nowait())
            except queue.Empty:
                return msgs

    def latest_job(self):
        """status.json 이 있는 가장 최근 작업 번호 (테스트 순서와 무관하게)."""
        return max(int(d) for d in os.listdir(self.jobs_dir)
                   if d.isdigit() and os.path.exists(os.path.join(self.jobs_dir, d, 'status.json')))

    def read_status(self, job):
        with open(os.path.join(self.jobs_dir, str(job), 'status.json'), encoding='utf-8') as f:
            return json.load(f)

    def post_small(self):
        body = gzip.compress(b'a,b\n1,2\n' * 1000)
        r = requests.post(self.url('/jobs'), params=dict(self.params, compressed='gzip'),
                          data=body, timeout=60)
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()['job']

    def wait_done(self, job, limit=60):
        deadline = time.time() + limit
        st = requests.get(self.url('/jobs/%d' % job), timeout=30).json()
        while st['status'] not in ('done', 'failed') and time.time() < deadline:
            time.sleep(0.5)
            st = requests.get(self.url('/jobs/%d' % job), timeout=30).json()
        return st

    def run_gui(self, **env):
        import remote_gui as g
        g.POLL_INTERVAL_SEC = 1
        g.set_language('ko')
        q = queue.Queue()
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            g.run_submission(self.base(), self.csv, self.params, q)
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        return self.drain(q)

    # ── 테스트 ──────────────────────────────────────────────────────────
    def test_gzip_submission_streams_log_and_reports_timing(self):
        t0 = time.time()
        msgs = self.run_gui()
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
        # 서버 쪽: gzip 으로 받아 원본과 같은 바이트로 풀렸는지
        job = self.latest_job()
        st = self.read_status(job)
        self.assertEqual(st['compressed'], 'gzip')
        self.assertEqual(st['size'], os.path.getsize(self.csv))
        self.assertLess(st['size_wire'], st['size'] / 2)
        with open(os.path.join(self.jobs_dir, str(job), 'upload.csv'), 'rb') as a, open(self.csv, 'rb') as b:
            self.assertEqual(a.read(), b.read())
        self.assertLess(time.time() - t0, 60)

    def test_log_beyond_tail_window_is_followed_without_gaps(self):
        # run.log 가 8000바이트 창을 넘어도 (a) 보여준 바이트 + 건너뛴 바이트 = 파일 전체,
        # (b) 같은 줄이 두 번 나오지 않고, (c) 첫 묶음과 마지막 창은 빠짐없이 보인다.
        msgs = self.run_gui(ONECLICK_STUB_MODE='big')
        self.assertTrue(msgs[-1][1], msgs[-1])
        self.assertIn('43', msgs[-1][2])
        pipe = ''.join(m[1] for m in msgs if m[0] == 'log' and m[1].isascii())      # 스텁 출력은 ASCII 뿐
        skipped = sum(int(re.search(r'(\d+)', m[1]).group(1))
                      for m in msgs if m[0] == 'log' and '건너뜀' in m[1])
        log_size = os.path.getsize(os.path.join(self.jobs_dir, str(self.latest_job()), 'run.log'))
        self.assertEqual(len(pipe.encode('utf-8')) + skipped, log_size)
        nums = [int(n) for n in re.findall(r'line (\d{4}) x{30}\r?\n', pipe)]     # 윈도에서는 줄끝이 CRLF
        self.assertEqual(len(nums), len(set(nums)), '같은 줄이 두 번 나왔다')
        self.assertTrue(set(range(100)) <= set(nums), '첫 묶음 누락')
        self.assertTrue(set(range(330, 550)) <= set(nums), '마지막 창 누락')
        self.assertIn('[RESULT] exp_pk=43', pipe)

    def test_old_server_gets_raw_upload(self):
        # 서버가 gzip 을 모르는 버전으로 보이면 압축하지 않고 보낸다
        import remote_gui as g
        with mock.patch.object(g, '_server_supports_gzip', return_value=False):
            msgs = self.run_gui()
        self.assertTrue(msgs[-1][1], msgs[-1])
        logs = ''.join(m[1] for m in msgs if m[0] == 'log')
        self.assertNotIn('압축:', logs)
        st = self.read_status(self.latest_job())
        self.assertIsNone(st['compressed'])
        self.assertEqual(st['size'], os.path.getsize(self.csv))

    def test_raw_submission_and_status_fields(self):
        # 옛 클라이언트(비압축) 호환 + 상태 응답의 시각·순번·로그 창 필드
        with open(self.csv, 'rb') as f:
            r = requests.post(self.url('/jobs'), params=self.params, data=f, timeout=60)
        self.assertEqual(r.status_code, 200, r.text)
        job = r.json()['job']
        self.assertIn('queue_ahead', r.json())
        st = requests.get(self.url('/jobs/%d' % job), timeout=30).json()
        for key in ('submitted', 'started', 'finished', 'queue_ahead', 'log_bytes', 'log_from', 'server_time'):
            self.assertIn(key, st)
        st = self.wait_done(job)
        self.assertEqual(st['status'], 'done', st)
        self.assertEqual(st['exp_pk'], 42)
        self.assertGreaterEqual(st['finished'], st['started'])
        self.assertGreaterEqual(st['started'], st['submitted'])
        self.assertEqual(st['log_from'], max(0, st['log_bytes'] - self.srv.LOG_TAIL_BYTES))
        self.assertEqual(len(st['log_tail'].encode('utf-8')), st['log_bytes'] - st['log_from'])
        with open(os.path.join(self.jobs_dir, str(job), 'upload.csv'), 'rb') as a, open(self.csv, 'rb') as b:
            self.assertEqual(a.read(), b.read())

    def test_capability_probe(self):
        st = requests.get(self.url('/jobs/0'), timeout=30).json()
        self.assertIs(st['gzip'], True)
        self.assertEqual(st['job'], 0)
        for key in ('queue_length', 'server_time', 'server'):
            self.assertIn(key, st)
        import remote_gui as g
        self.assertTrue(g._server_supports_gzip('http://%s' % self.base(), True))
        self.assertFalse(g._server_supports_gzip('http://127.0.0.1:9', True))     # 연결 불가 → 비압축
        fake = mock.Mock(status_code=200)
        fake.json.return_value = ['not', 'a', 'dict']                               # 엉뚱한 JSON → 비압축
        with mock.patch.object(g.requests, 'get', return_value=fake):
            self.assertFalse(g._server_supports_gzip('http://x', True))

    def test_bad_gzip_is_rejected(self):
        before = set(os.listdir(self.jobs_dir))
        r = requests.post(self.url('/jobs'), params=dict(self.params, compressed='gzip'),
                          data=b'not gzip at all' * 100, timeout=60)
        self.assertEqual(r.status_code, 400)
        self.assertIn('gzip', r.json()['error'])
        r = requests.post(self.url('/jobs'), params=dict(self.params, compressed='zip'),
                          data=b'x', timeout=60)
        self.assertEqual(r.status_code, 400)
        # 잘린 gzip (끝나지 않은 스트림)
        body = gzip.compress(b'a,b\n1,2\n' * 1000)
        r = requests.post(self.url('/jobs'), params=dict(self.params, compressed='gzip'),
                          data=body[:-10], timeout=60)
        self.assertEqual(r.status_code, 400, r.text)
        self.assertEqual(set(os.listdir(self.jobs_dir)), before, '거절된 작업 폴더가 남았다')

    def test_decompression_bomb_is_rejected(self):
        srv = self.srv
        before = set(os.listdir(self.jobs_dir))
        limit, srv.MAX_UPLOAD_BYTES = srv.MAX_UPLOAD_BYTES, 4 << 20
        try:
            body = gzip.compress(b'0' * (6 << 20))          # 전송 수 KB, 풀면 6MB > 상한 4MB
            self.assertLess(len(body), 64 << 10)
            r = requests.post(self.url('/jobs'), params=dict(self.params, compressed='gzip'),
                              data=body, timeout=60)
        finally:
            srv.MAX_UPLOAD_BYTES = limit
        self.assertEqual(r.status_code, 413, r.text)
        self.assertIn('너무 큽니다', r.json()['error'])
        self.assertEqual(set(os.listdir(self.jobs_dir)), before, '거절된 작업 폴더가 남았다')
        self.assertEqual(requests.get(self.url('/jobs/0'), timeout=30).status_code, 200)   # 서버는 계속 산다

    def test_queue_ahead_counts_running_and_pending(self):
        first = self.post_small()
        r = requests.post(self.url('/jobs'), params=dict(self.params, compressed='gzip'),
                          data=gzip.compress(b'a,b\n1,2\n' * 1000), timeout=60).json()
        self.assertGreaterEqual(r['queue_ahead'], 1)         # 앞 작업(돌고 있거나 대기)이 하나 이상
        st = requests.get(self.url('/jobs/%d' % r['job']), timeout=30).json()
        if st['status'] == 'queued':
            self.assertGreaterEqual(st['queue_ahead'], 1)
        st = self.wait_done(r['job'])
        self.assertEqual(st['status'], 'done')
        self.assertEqual(st['queue_ahead'], 0)
        self.assertLess(first, r['job'])

    def test_worker_survives_status_write_failure(self):
        srv = self.srv
        orig = srv._write_status
        state = {'fail': True}

        def flaky(job_id, **kw):
            if state['fail'] and kw.get('status') == 'running':
                state['fail'] = False
                raise OSError('disk full')
            return orig(job_id, **kw)

        with mock.patch.object(srv, '_write_status', flaky):
            job = self.post_small()
            st = self.wait_done(job)
        self.assertEqual(st['status'], 'failed', st)
        self.assertIn('disk full', st['error'])
        self.assertFalse(state['fail'])
        st2 = self.wait_done(self.post_small())                # 워커가 살아 다음 작업을 돌린다
        self.assertEqual(st2['status'], 'done', st2)

    def test_gzip_temp_is_removed_on_failure(self):
        import remote_gui as g
        tmpdir = tempfile.gettempdir()

        def boom(done, total):
            raise OSError('disk full')
        before = {f for f in os.listdir(tmpdir) if f.startswith('oneclick_')}
        with self.assertRaises(OSError):
            g._gzip_to_temp(self.csv, boom)
        after = {f for f in os.listdir(tmpdir) if f.startswith('oneclick_')}
        self.assertEqual(before, after, '쓰다 만 임시 gz 가 남았다')

    def test_cli_submit_and_poll_follow_log(self):
        # 명령행(remote_submit)도 같은 서버에 압축 제출하고, 창을 넘는 로그를 빠짐없이 찍는다
        import contextlib
        import remote_submit as cli
        cli.POLL_INTERVAL_SEC = 1
        out = io.StringIO()
        os.environ['ONECLICK_STUB_MODE'] = 'big'
        try:
            with contextlib.redirect_stdout(out):
                job = cli.submit(self.base(), self.csv, self.params)
                rc = cli.poll(self.base(), job)
        finally:
            os.environ.pop('ONECLICK_STUB_MODE', None)
        text = out.getvalue()
        self.assertEqual(rc, 0, text[-500:])
        self.assertIn('압축:', text)
        self.assertIn('실험 번호 43', text)
        nums = [int(n) for n in re.findall(r'line (\d{4}) x{30}\r?\n', text)]
        self.assertEqual(len(nums), len(set(nums)), '같은 줄이 두 번 나왔다')
        self.assertTrue(set(range(100)) <= set(nums), '첫 묶음 누락')
        self.assertTrue(set(range(330, 550)) <= set(nums), '마지막 창 누락')
        st = self.read_status(job)
        self.assertEqual(st['compressed'], 'gzip')
        self.assertEqual(st['size'], os.path.getsize(self.csv))


if __name__ == '__main__':
    unittest.main()

# -*- coding:utf-8 -*-
"""GPU 분석 서버의 작업 접수 서비스.

측정 PC 는 원본 CSV 와 피험자 정보만 제출하고, 분석(main.py 파이프라인)은
이 서버가 수행한다. 완료되면 파이프라인이 기존 경로 그대로 Django 서버에
결과를 등록하므로 웹 서버 쪽은 아무 변경이 없다.

표준 라이브러리만 쓴다 — GPU 서버에 파이프라인 환경 외에 아무것도 더 설치할
필요가 없어야 한다. 그래서 multipart 대신 CSV 를 요청 본문(raw bytes)으로
받고, 피험자 정보는 쿼리스트링으로 받는다.

API
  POST /jobs?name=..&age=..&birth=..&sex=..&measurement_date=..[&옵션]
       본문 = CSV 바이트
       -> {"job": <id>, "status": "queued"}
  GET  /jobs/<id>
       -> {"job", "status": queued|running|done|failed,
           "exp_pk", "error", "log_tail"}

실행
  python analysis_server.py [--port 8500]
  환경변수:
    ONECLICK_CLIENT_DIR    파이프라인 저장소 경로 (기본: 이 파일이 있는 곳)
    ONECLICK_CLIENT_PYTHON 파이프라인을 돌릴 python (기본: 이 서버의 python)

분석은 한 번에 하나만 돈다. 파이프라인이 data/temp.csv 같은 공유 파일을
쓰기 때문에 동시 실행하면 서로를 덮어쓴다.

주의: 인증이 없다. 연구실 내부망 전용이며 외부에 노출하면 안 된다.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Queue
from urllib.parse import urlparse, parse_qs

from utils.ai_report import QUESTIONNAIRE_SCALES

CLIENT_DIR = os.environ.get(
    'ONECLICK_CLIENT_DIR', os.path.dirname(os.path.abspath(__file__)))
CLIENT_PYTHON = os.environ.get('ONECLICK_CLIENT_PYTHON', sys.executable)
JOBS_DIR = os.path.join(CLIENT_DIR, 'jobs')

# 제출 시 받는 파라미터. 나머지는 무시한다.
SUBJECT_FIELDS = ('name', 'age', 'birth', 'sex', 'measurement_date')
OPTION_FIELDS = ('stimulus', 'sleep_model', 'crop')
QUESTIONNAIRE_FIELDS = tuple(k for k, _l, _d in QUESTIONNAIRE_SCALES)

# 업로드 상한. 측정 CSV 실측이 130~165MB 라 여유를 둔다.
MAX_UPLOAD_BYTES = 600 * 1024 * 1024

_queue = Queue()
_lock = threading.Lock()


def _job_dir(job_id):
    return os.path.join(JOBS_DIR, str(job_id))


def _status_path(job_id):
    return os.path.join(_job_dir(job_id), 'status.json')


def _read_status(job_id):
    try:
        with open(_status_path(job_id), encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write_status(job_id, **updates):
    """상태를 디스크에 남긴다 — 서버가 재시작해도 결과 조회가 가능해야 한다."""
    with _lock:
        st = _read_status(job_id) or {}
        st.update(updates)
        tmp = _status_path(job_id) + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(st, f, ensure_ascii=False)
        os.replace(tmp, _status_path(job_id))
    return st


def _next_job_id():
    os.makedirs(JOBS_DIR, exist_ok=True)
    existing = [int(d) for d in os.listdir(JOBS_DIR) if d.isdigit()]
    return (max(existing) + 1) if existing else 1


def _build_argv(job_id, params):
    """작업 파라미터 -> main.py 인자."""
    argv = [
        CLIENT_PYTHON, os.path.join(CLIENT_DIR, 'main.py'),
        '--NAME', params['name'],
        '--AGE', params['age'],
        '--BIRTH', params['birth'],
        '--SEX', params['sex'],
        '--MEASUREMENT_DATE', params['measurement_date'],
        '--FILE_NAME', 'job_%d.csv' % job_id,
    ]
    if params.get('stimulus'):
        argv += ['--STIMULUS', params['stimulus']]
    if params.get('sleep_model'):
        argv += ['--SLEEP_MODEL', params['sleep_model']]
    # argparse 의 type=bool 은 비어 있지 않은 모든 문자열을 True 로 읽는다.
    # 끄는 방법은 빈 문자열뿐이다 — main.py 의 기존 계약이라 여기서 맞춘다.
    if str(params.get('crop', '')).lower() in ('0', 'false', 'no'):
        argv += ['--CROP_MODE', '']
    for key in QUESTIONNAIRE_FIELDS:
        if params.get(key):
            argv += ['--' + key, params[key]]
    return argv


def _worker():
    """작업을 순서대로 하나씩 처리한다."""
    while True:
        job_id = _queue.get()
        params = (_read_status(job_id) or {}).get('params', {})
        log_path = os.path.join(_job_dir(job_id), 'run.log')
        _write_status(job_id, status='running', started=time.time())
        print('[작업 %d] 분석 시작 — %s' % (job_id, params.get('name', '?')))

        # 파이프라인은 자기 data/ 폴더의 파일만 읽는다. 작업 파일을 복사해 두고
        # 끝나면 지운다 (원본은 jobs/<id>/ 에 남는다).
        data_csv = os.path.join(CLIENT_DIR, 'data', 'job_%d.csv' % job_id)
        try:
            os.makedirs(os.path.dirname(data_csv), exist_ok=True)
            src = os.path.join(_job_dir(job_id), 'upload.csv')
            with open(src, 'rb') as fin, open(data_csv, 'wb') as fout:
                while True:
                    chunk = fin.read(1 << 20)
                    if not chunk:
                        break
                    fout.write(chunk)

            with open(log_path, 'ab') as logf:
                proc = subprocess.Popen(
                    _build_argv(job_id, params),
                    cwd=CLIENT_DIR, stdout=logf, stderr=subprocess.STDOUT)
                rc = proc.wait()

            if rc == 0:
                exp_pk = None
                with open(log_path, 'rb') as f:
                    m = re.findall(rb'\[RESULT\] exp_pk=(\d+)', f.read())
                    if m:
                        exp_pk = int(m[-1])
                _write_status(job_id, status='done', exp_pk=exp_pk,
                              finished=time.time())
                print('[작업 %d] 완료 exp_pk=%s' % (job_id, exp_pk))
            else:
                _write_status(job_id, status='failed',
                              error='파이프라인 종료 코드 %d — run.log 확인' % rc,
                              finished=time.time())
                print('[작업 %d] 실패 rc=%d' % (job_id, rc))
        except Exception as e:
            _write_status(job_id, status='failed', error=str(e),
                          finished=time.time())
            print('[작업 %d] 예외: %s' % (job_id, e))
        finally:
            try:
                if os.path.exists(data_csv):
                    os.remove(data_csv)
            except OSError:
                pass
            _queue.task_done()


def _recover_stale_jobs():
    """재시작 시, 돌다 만 작업을 실패로 정리한다 (프로세스는 이미 없다)."""
    if not os.path.isdir(JOBS_DIR):
        return
    for d in os.listdir(JOBS_DIR):
        if not d.isdigit():
            continue
        st = _read_status(int(d)) or {}
        if st.get('status') in ('queued', 'running'):
            _write_status(int(d), status='failed',
                          error='분석 서버 재시작으로 중단됨 — 다시 제출해 주세요.')
            print('[복구] 작업 %s 를 실패로 정리' % d)


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):        # 기본 stderr 로그를 짧게
        print('[http] %s %s' % (self.command, self.path.split('?')[0]))

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path.rstrip('/') != '/jobs':
            return self._json(404, {'error': 'POST /jobs 만 지원합니다.'})

        qs = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        params = {}
        for key in SUBJECT_FIELDS + OPTION_FIELDS + QUESTIONNAIRE_FIELDS:
            v = qs.get(key)
            if v is not None and v.strip() != '':
                params[key] = v
        missing = [k for k in SUBJECT_FIELDS if k not in params]
        if missing:
            return self._json(400, {'error': '필수 항목 누락: %s' % ', '.join(missing)})

        length = int(self.headers.get('Content-Length') or 0)
        if length <= 0:
            return self._json(400, {'error': '본문에 CSV 를 실어 보내 주세요.'})
        if length > MAX_UPLOAD_BYTES:
            return self._json(413, {'error': '파일이 너무 큽니다 (%dMB).'
                                             % (length >> 20)})

        job_id = _next_job_id()
        os.makedirs(_job_dir(job_id), exist_ok=True)
        dst = os.path.join(_job_dir(job_id), 'upload.csv')
        remain = length
        with open(dst, 'wb') as f:
            while remain > 0:
                chunk = self.rfile.read(min(1 << 20, remain))
                if not chunk:
                    break
                f.write(chunk)
                remain -= len(chunk)
        if remain > 0:
            return self._json(400, {'error': '본문이 %d바이트 부족하게 끊겼습니다.'
                                             % remain})

        _write_status(job_id, status='queued', params=params,
                      submitted=time.time(), size=length)
        _queue.put(job_id)
        print('[접수] 작업 %d — %s (%.1fMB)'
              % (job_id, params.get('name'), length / 1e6))
        return self._json(200, {'job': job_id, 'status': 'queued'})

    def do_GET(self):
        m = re.fullmatch(r'/jobs/(\d+)/?', urlparse(self.path).path)
        if not m:
            return self._json(404, {'error': 'GET /jobs/<id> 만 지원합니다.'})
        job_id = int(m.group(1))
        st = _read_status(job_id)
        if st is None:
            return self._json(404, {'error': '작업 %d 없음' % job_id})

        tail = ''
        log_path = os.path.join(_job_dir(job_id), 'run.log')
        if os.path.exists(log_path):
            try:
                with open(log_path, 'rb') as f:
                    f.seek(0, os.SEEK_END)
                    f.seek(max(0, f.tell() - 8000))
                    tail = f.read().decode('utf-8', 'replace')
            except OSError:
                pass
        return self._json(200, {
            'job': job_id,
            'status': st.get('status'),
            'exp_pk': st.get('exp_pk'),
            'error': st.get('error'),
            'log_tail': tail,
        })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8500)
    args = ap.parse_args()

    _recover_stale_jobs()
    threading.Thread(target=_worker, daemon=True).start()

    print('분석 서버 시작 — 포트 %d' % args.port)
    print('  파이프라인 : %s' % CLIENT_DIR)
    print('  python     : %s' % CLIENT_PYTHON)
    print('  작업 폴더  : %s' % JOBS_DIR)
    ThreadingHTTPServer(('0.0.0.0', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()

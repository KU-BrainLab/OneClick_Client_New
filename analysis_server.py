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
       -> {"job", "status": queued|running|done|failed, "exp_pk", "error",
           "log_tail"(run.log 꼬리), "log_bytes"(run.log 전체 크기), "log_from"(꼬리 시작 위치),
           "submitted"/"started"/"finished"(시각), "queue_ahead"(앞 작업 수), "server_time"}
  GET  /jobs/0
       -> 서버 기능 목록 {"gzip": true, "queue_length", "server_time"} (압축 업로드 지원 확인용)

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
import shutil
import subprocess
import zlib
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

# 업로드 상한. 측정 CSV 실측이 130~165MB 라 여유를 둔다. gzip 으로 오면 전송 크기와
# 풀린 크기 둘 다 이 상한을 넘지 못한다.
MAX_UPLOAD_BYTES = 600 * 1024 * 1024
# 상태 조회가 돌려주는 run.log 꼬리 길이. 제출 PC 는 log_bytes·log_from 으로 이 창에서
# 새로 늘어난 부분만 골라 보여준다 (remote_gui.LogFollower).
LOG_TAIL_BYTES = 8000


class _TooLarge(Exception):
    """압축을 푼 본문이 상한을 넘었다."""

_queue = Queue()
_lock = threading.Lock()
_current_job = None            # 지금 돌고 있는 작업 번호 (대기 순번 계산용)


def _queue_ahead(job_id):
    """job_id 보다 먼저 접수돼 아직 안 끝난 작업 수 (돌고 있는 것 포함)."""
    with _queue.mutex:
        pending = [j for j in _queue.queue if j < job_id]
    running = _current_job
    return len(pending) + (1 if running is not None and running < job_id else 0)


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


def _create_job():
    """새 작업 번호를 정하고 폴더까지 만든다. 번호 결정과 폴더 생성을 한 잠금 안에서 해야
    동시에 들어온 두 제출이 같은 번호를 받지 않는다."""
    with _lock:
        os.makedirs(JOBS_DIR, exist_ok=True)
        existing = [int(d) for d in os.listdir(JOBS_DIR) if d.isdigit()]
        job_id = (max(existing) + 1) if existing else 1
        os.makedirs(_job_dir(job_id))
    return job_id


def _build_argv(job_id, params):
    """작업 파라미터 -> main.py 인자."""
    # -u: 파이프라인 stdout 이 파일로 가면 파이썬이 8KB 단위로 버퍼링해 run.log 가 한참 뒤에
    # 몰아서 채워진다. 끄지 않으면 제출 PC 에서 로그가 '실시간' 으로 안 보인다.
    argv = [
        CLIENT_PYTHON, '-u', os.path.join(CLIENT_DIR, 'main.py'),
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
    """작업을 순서대로 하나씩 처리한다.

    어떤 예외에도 이 스레드는 죽지 않는다 — 죽으면 서버는 접수만 하고 아무것도 돌리지 않는
    채로 멀쩡해 보인다. 상태 기록(디스크 쓰기)이 실패하는 경우까지 여기서 받는다.
    """
    global _current_job
    while True:
        job_id = _queue.get()
        _current_job = job_id
        try:
            _run_job(job_id)
        except Exception as e:
            print('[작업 %d] 예외: %s' % (job_id, e))
            try:
                _write_status(job_id, status='failed', error=str(e), finished=time.time())
            except Exception as e2:
                print('[작업 %d] 상태 기록 실패: %s' % (job_id, e2))
        finally:
            _current_job = None
            _queue.task_done()


def _run_job(job_id):
    params = (_read_status(job_id) or {}).get('params', {})
    log_path = os.path.join(_job_dir(job_id), 'run.log')
    _write_status(job_id, status='running', started=time.time())
    print('[작업 %d] 분석 시작 - %s' % (job_id, params.get('name', '?')))

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

        env = dict(os.environ, PYTHONUNBUFFERED='1')     # -u 와 같은 뜻, 하위 프로세스까지
        with open(log_path, 'ab') as logf:
            proc = subprocess.Popen(
                _build_argv(job_id, params),
                cwd=CLIENT_DIR, stdout=logf, stderr=subprocess.STDOUT, env=env)
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
    finally:
        try:
            if os.path.exists(data_csv):
                os.remove(data_csv)
        except OSError:
            pass


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

    def _reject(self, code, obj, remain=0):
        """본문을 다 읽기 전에 거절한다. 남은 본문(remain 바이트)은 읽어 버린 뒤 응답한다 —
        안 읽고 답하면 중계(Django)가 연결 리셋을 받아 이 안내 대신 '연결할 수 없습니다' 로
        바꿔 버린다. 상한을 넘는 본문은 읽지 않고 응답 뒤 연결을 닫는다."""
        if remain > MAX_UPLOAD_BYTES:
            self.close_connection = True
        else:
            while remain > 0:
                chunk = self.rfile.read(min(1 << 20, remain))
                if not chunk:
                    self.close_connection = True
                    break
                remain -= len(chunk)
        return self._json(code, obj)

    def log_message(self, fmt, *args):        # 기본 stderr 로그를 짧게
        print('[http] %s %s' % (self.command, self.path.split('?')[0]))

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path.rstrip('/') != '/jobs':
            return self._json(404, {'error': 'POST /jobs 만 지원합니다.'})

        try:
            length = int(self.headers.get('Content-Length') or 0)
        except ValueError:
            length = -1
        if length <= 0:
            return self._reject(400, {'error': '본문에 CSV 를 실어 보내 주세요.'})
        if length > MAX_UPLOAD_BYTES:
            return self._reject(413, {'error': '파일이 너무 큽니다 (%dMB).'
                                               % (length >> 20)}, length)

        qs = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        params = {}
        for key in SUBJECT_FIELDS + OPTION_FIELDS + QUESTIONNAIRE_FIELDS:
            v = qs.get(key)
            if v is not None and v.strip() != '':
                params[key] = v
        missing = [k for k in SUBJECT_FIELDS if k not in params]
        if missing:
            return self._reject(400, {'error': '필수 항목 누락: %s' % ', '.join(missing)},
                                length)

        # compressed=gzip: 제출 PC 가 CSV 를 gzip 으로 보낸다(150MB 급 CSV 가 4~6배 줄어
        # 교외 회선에서 업로드가 그만큼 빨라진다). 여기서 풀어 upload.csv 로 저장한다.
        compressed = (qs.get('compressed') or '').strip().lower()
        if compressed not in ('', 'gzip'):
            return self._reject(400, {'error': '지원하지 않는 압축: %s' % compressed}, length)
        decomp = zlib.decompressobj(16 + zlib.MAX_WBITS) if compressed == 'gzip' else None

        job_id = _create_job()
        dst = os.path.join(_job_dir(job_id), 'upload.csv')
        remain = length
        raw_size = 0
        try:
            with open(dst, 'wb') as f:
                while remain > 0:
                    chunk = self.rfile.read(min(1 << 20, remain))
                    if not chunk:
                        break
                    remain -= len(chunk)
                    if decomp is None:
                        raw_size += len(chunk)
                        f.write(chunk)
                        continue
                    # 1MB 씩 끊어 푼다. 한꺼번에 풀면 압축률이 극단적인 본문(0 으로만 채운
                    # 파일 등)이 메모리를 다 차지한다. 풀린 크기도 업로드 상한으로 막는다.
                    while chunk:
                        out = decomp.decompress(chunk, 1 << 20)
                        raw_size += len(out)
                        if raw_size > MAX_UPLOAD_BYTES:
                            raise _TooLarge()
                        f.write(out)
                        chunk = decomp.unconsumed_tail
                if decomp is not None:
                    out = decomp.flush()
                    raw_size += len(out)
                    if raw_size > MAX_UPLOAD_BYTES:
                        raise _TooLarge()
                    f.write(out)
                    if remain == 0 and not decomp.eof:
                        raise zlib.error('gzip 스트림이 끝나지 않았습니다')
                    if decomp.unused_data:
                        raise zlib.error('gzip 스트림 뒤에 데이터가 %d바이트 더 있습니다'
                                         % len(decomp.unused_data))
        except _TooLarge:
            shutil.rmtree(_job_dir(job_id), ignore_errors=True)
            return self._reject(413, {'error': '압축을 푼 파일이 너무 큽니다 (%dMB 초과).'
                                               % (MAX_UPLOAD_BYTES >> 20)}, remain)
        except zlib.error as e:
            shutil.rmtree(_job_dir(job_id), ignore_errors=True)   # status.json 없는 빈 작업을 남기지 않는다
            return self._reject(400, {'error': 'gzip 해제 실패: %s' % e}, remain)
        if remain > 0:
            shutil.rmtree(_job_dir(job_id), ignore_errors=True)
            self.close_connection = True
            return self._json(400, {'error': '본문이 %d바이트 부족하게 끊겼습니다.'
                                             % remain})

        _write_status(job_id, status='queued', params=params,
                      submitted=time.time(), size=raw_size, size_wire=length,
                      compressed=compressed or None)
        _queue.put(job_id)
        print('[접수] 작업 %d - %s (%.1fMB%s)'
              % (job_id, params.get('name'), raw_size / 1e6,
                 ', 전송 %.1fMB gzip' % (length / 1e6) if decomp else ''))
        return self._json(200, {'job': job_id, 'status': 'queued',
                                'queue_ahead': _queue_ahead(job_id)})

    def do_GET(self):
        m = re.fullmatch(r'/jobs/(\d+)/?', urlparse(self.path).path)
        if not m:
            return self._json(404, {'error': 'GET /jobs/<id> 만 지원합니다.'})
        job_id = int(m.group(1))
        if job_id == 0:
            # 기능 조회. 제출 PC 가 압축 업로드를 써도 되는지 여기로 확인한다(옛 서버는 404).
            # 작업 번호는 1 부터라 0 은 비어 있고, 중계(Django)도 <int> 경로라 그대로 통과한다.
            return self._json(200, {'job': 0, 'server': 'oneclick-analysis', 'gzip': True,
                                    'queue_length': _queue_ahead(1 << 62),
                                    'server_time': time.time()})
        st = _read_status(job_id)
        if st is None:
            return self._json(404, {'error': '작업 %d 없음' % job_id})

        tail, log_bytes, log_from = '', 0, 0
        log_path = os.path.join(_job_dir(job_id), 'run.log')
        try:
            with open(log_path, 'rb') as f:
                f.seek(0, os.SEEK_END)
                log_bytes = f.tell()
                log_from = max(0, log_bytes - LOG_TAIL_BYTES)
                f.seek(log_from)
                # 크기와 꼬리를 같은 열기에서, 정확히 그 길이만큼 읽는다 — 제출 PC 가
                # log_bytes 로 '새로 늘어난 바이트' 를 세어 꼬리에서 잘라 내기 때문이다.
                tail = f.read(log_bytes - log_from).decode('utf-8', 'replace')
        except OSError:            # 아직 run.log 가 없다(대기열)
            pass
        # 시각·순번은 제출 PC 가 '어디서 시간이 가는지'(업로드/대기/분석)를 보여 주는 데 쓴다.
        return self._json(200, {
            'job': job_id,
            'status': st.get('status'),
            'exp_pk': st.get('exp_pk'),
            'error': st.get('error'),
            'log_tail': tail,
            'log_bytes': log_bytes,
            'log_from': log_from,
            'submitted': st.get('submitted'),
            'started': st.get('started'),
            'finished': st.get('finished'),
            'queue_ahead': _queue_ahead(job_id) if st.get('status') == 'queued' else 0,
            'server_time': time.time(),
        })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8500)
    args = ap.parse_args()

    _recover_stale_jobs()
    threading.Thread(target=_worker, daemon=True).start()

    print('분석 서버 시작 - 포트 %d' % args.port)
    print('  파이프라인 : %s' % CLIENT_DIR)
    print('  python     : %s' % CLIENT_PYTHON)
    print('  작업 폴더  : %s' % JOBS_DIR)
    ThreadingHTTPServer(('0.0.0.0', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()

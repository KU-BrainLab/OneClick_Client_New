# -*- coding:utf-8 -*-
"""측정 PC 용 제출 도구 — CSV 와 피험자 정보를 GPU 분석 서버로 보낸다.

분석은 GPU 서버(analysis_server.py)가 수행하고, 결과 등록은 그쪽 파이프라인이
기존 경로 그대로 처리한다. 이 스크립트는 requests 만 있으면 돌아간다 —
측정 PC 에 mne/torch 같은 무거운 환경이 더 이상 필요 없다.

사용 예
  python remote_submit.py --FILE data/2026-08-21-1706.csv \
      --NAME 홍길동 --AGE 30 --BIRTH 1996-01-01 --SEX male \
      --MEASUREMENT_DATE "2026-08-21 17:06" --ISI 11 --PSQI 10

--SERVER 는 두 형태를 받는다:
  내부망에서 직접        --SERVER <연구실서버IP>:8500
  외부에서 Django 중계   --SERVER https://<웹서버주소>:8443/api/v1/exp/analysis
외부망은 연구실 서버에 직접 닿지 못하므로 웹서버의 중계 경로를 쓴다.
개인정보가 인터넷 구간을 지나므로 외부 경로는 https 가 의무다. 서버 인증서는
함께 배포되는 oneclick-ca.pem(저장소 루트)으로 검증한다.
어느 쪽이든 이 스크립트 입장에선 '<SERVER>/jobs' 로 붙는 것이라 동작은 같다.
"""
import argparse
import gzip
import os
import shutil
import sys
import tempfile
import time

import requests

from utils.ai_report import QUESTIONNAIRE_SCALES, api_base, tls_verify

DEFAULT_SERVER = os.environ.get('ONECLICK_ANALYSIS_SERVER', '127.0.0.1:8500')

POLL_INTERVAL_SEC = 15
# 분석 실측이 수 분 + 대기열을 감안한다.
POLL_LIMIT_SEC = 3600


def get_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--SERVER', default=DEFAULT_SERVER,
                    help='분석 서버 주소. 내부망 직접이면 <IP>:8500, 외부에서는 '
                         'Django 중계 https://<웹서버>:8443/api/v1/exp/analysis '
                         '(기본: %s 또는 ONECLICK_ANALYSIS_SERVER)'
                         % DEFAULT_SERVER)
    ap.add_argument('--FILE', required=True, help='측정 원본 CSV 경로')
    ap.add_argument('--NAME', required=True)
    ap.add_argument('--AGE', required=True)
    ap.add_argument('--BIRTH', required=True, help='YYYY-MM-DD')
    ap.add_argument('--SEX', required=True, choices=['male', 'female'])
    ap.add_argument('--MEASUREMENT_DATE', required=True, help='"YYYY-MM-DD HH:MM"')
    ap.add_argument('--STIMULUS', default=None)
    ap.add_argument('--SLEEP_MODEL', default=None,
                    choices=[None, 'neuronet', 'synthsleepnet', 'synthsleepnet_ft', 'usleep'])
    ap.add_argument('--NO_CROP', action='store_true',
                    help='노이즈 크롭 없이 원본 그대로 분석')
    # 설문 — 모르는 항목은 넣지 말 것 (0 은 실제 0점으로 저장된다)
    for key, label, desc in QUESTIONNAIRE_SCALES:
        ap.add_argument('--' + key, default=None, help='%s %s' % (label, desc))
    return ap.parse_args()


def build_params(args):
    params = {
        'name': args.NAME,
        'age': str(args.AGE),
        'birth': args.BIRTH,
        'sex': args.SEX,
        'measurement_date': args.MEASUREMENT_DATE,
    }
    if args.STIMULUS:
        params['stimulus'] = args.STIMULUS
    if args.SLEEP_MODEL:
        params['sleep_model'] = args.SLEEP_MODEL
    if args.NO_CROP:
        params['crop'] = 'false'
    for key, _label, _desc in QUESTIONNAIRE_SCALES:
        v = getattr(args, key)
        if v is not None and str(v).strip() != '':
            params[key] = str(v)
    return params


def _gzip_to_temp(csv_path):
    """CSV 를 gzip 임시 파일로. 실패하면 쓰다 만 파일을 지우고 예외를 올린다. 호출자가 지운다."""
    tmp = tempfile.NamedTemporaryFile(prefix='oneclick_', suffix='.csv.gz', delete=False)
    try:
        with open(csv_path, 'rb') as fin, gzip.GzipFile(fileobj=tmp, mode='wb', compresslevel=3) as gz:
            shutil.copyfileobj(fin, gz, 1 << 20)
    except BaseException:
        tmp.close()
        try:
            os.remove(tmp.name)
        except OSError:
            pass
        raise
    tmp.close()
    return tmp.name


def submit(server, csv_path, params, use_gzip=True):
    base = api_base(server)
    size_mb = os.path.getsize(csv_path) / 1e6
    print('제출: %s (%.1fMB) -> %s' % (os.path.basename(csv_path), size_mb, server))
    send_path, send_params, tmp = csv_path, params, None
    if use_gzip:
        # 옛 분석 서버(compressed 를 모름)는 gzip 바이트를 그대로 저장해 분석이 깨진다. 새 서버만
        # GET /jobs/0 에 기능 목록({'gzip': true})을 주므로 그때만 압축한다.
        try:
            probe = requests.get('%s/jobs/0' % base, timeout=15, verify=tls_verify(base))
            body = probe.json()
            use_gzip = probe.status_code == 200 and isinstance(body, dict) and body.get('gzip') is True
        except (requests.RequestException, ValueError):
            use_gzip = False
        if not use_gzip:
            print('서버가 압축 업로드를 지원하지 않아 원본으로 보냅니다.')
    if use_gzip:
        # 측정 CSV(130~165MB)는 gzip 으로 4~6배 줄어 업로드가 그만큼 빨라진다. 서버가 풀어 저장한다.
        t0 = time.time()
        try:
            tmp = _gzip_to_temp(csv_path)
        except (OSError, ValueError) as e:
            print('압축 실패(%s) - 원본으로 보냅니다.' % e)
        else:
            send_path, send_params = tmp, dict(params, compressed='gzip')
            print('압축: %.1fMB -> %.1fMB (%.0f초)' % (size_mb, os.path.getsize(tmp) / 1e6, time.time() - t0))
    try:
        with open(send_path, 'rb') as f:
            # 파일을 본문으로 스트리밍한다 — 서버가 표준 라이브러리만으로 받는다.
            r = requests.post('%s/jobs' % base, params=send_params, data=f,
                              timeout=600, verify=tls_verify(base))
    finally:
        if tmp is not None:
            try:
                os.remove(tmp)
            except OSError:
                pass
    r.raise_for_status()
    body = r.json()
    if 'job' not in body:
        raise SystemExit('제출 실패: %s' % body)
    print('접수됨 — 작업 %s' % body['job'])
    return body['job']


class _LogFollower:
    """상태 응답에서 아직 안 찍은 로그만 골라낸다. remote_gui.LogFollower 와 같은 규칙이다
    (remote_gui 는 실행 파일로 묶이느라 저장소 모듈을 import 하지 않아 둘로 둔다).
    서버는 run.log 꼬리 8000바이트만 주므로 글자 수 비교로는 로그가 창보다 커지는 순간 멈춘다.
    파일 크기(log_bytes)로 새로 늘어난 바이트를 세어 꼬리 끝에서 잘라 낸다."""

    def __init__(self):
        self.seen = 0

    def take(self, st):
        tail = st.get('log_tail') or ''
        total = st.get('log_bytes')
        if total is None:                          # 옛 서버
            if len(tail) > self.seen:
                new, self.seen = tail[self.seen:], len(tail)
                return new, 0
            if len(tail) < self.seen:
                self.seen = len(tail)
            return '', 0
        if total <= self.seen:
            return '', 0
        new = total - self.seen
        self.seen = total
        window = total - (st.get('log_from') or 0)
        if new >= window:
            return tail, new - window
        return tail.encode('utf-8')[-new:].decode('utf-8', 'replace'), 0


def poll(server, job_id):
    base = api_base(server)
    deadline = time.time() + POLL_LIMIT_SEC
    follow = _LogFollower()
    last_status = None
    at_line_start = True
    while time.time() < deadline:
        try:
            r = requests.get('%s/jobs/%s' % (base, job_id), timeout=30,
                             verify=tls_verify(base))
            st = r.json()
        except (requests.RequestException, ValueError) as e:
            print('[폴링] 연결 오류(%s) — 계속 시도합니다' % type(e).__name__)
            time.sleep(POLL_INTERVAL_SEC)
            continue

        # Django 중계를 거칠 때는 분석 서버 장애가 예외가 아니라
        # {'error': ...} JSON(502 등)으로 온다. 일시 장애로 보고 재시도한다 —
        # 본 로그 위치(follow)를 지우면 복구 후 로그가 통째로 다시 찍힌다.
        if not isinstance(st, dict) or st.get('status') is None:
            print('[폴링] 중계 오류(%s) — 계속 시도합니다'
                  % ((st.get('error') if isinstance(st, dict) else None) or 'HTTP %s' % r.status_code))
            time.sleep(POLL_INTERVAL_SEC)
            continue

        new, skipped = follow.take(st)
        if skipped:
            if not at_line_start:
                sys.stdout.write('\n')
            print('(... 로그 %d바이트 건너뜀 - 전체는 서버의 run.log)' % skipped)
            at_line_start = True
        if new:
            sys.stdout.write(new)          # 바이트 단위 조각이라 줄 중간에서 끊길 수 있다 — 그대로 잇는다
            sys.stdout.flush()
            at_line_start = new.endswith('\n')

        if st.get('status') != last_status:
            last_status = st.get('status')
            if not at_line_start:
                sys.stdout.write('\n')
                at_line_start = True
            print('── 상태: %s ──' % last_status)
            deadline = time.time() + POLL_LIMIT_SEC      # 대기열 → 분석 으로 넘어가면 제한 시간을 다시 센다

        if last_status == 'done':
            print('분석 완료 — 실험 번호 %s. 웹에서 확인할 수 있습니다.'
                  % st.get('exp_pk'))
            return 0
        if last_status == 'failed':
            print('분석 실패: %s' % st.get('error'))
            return 1
        time.sleep(POLL_INTERVAL_SEC)

    print('제한 시간(%d분) 안에 끝나지 않았습니다. 작업 %s 는 서버에서 계속 '
          '진행될 수 있으니 나중에 GET /jobs/%s 로 확인하세요.'
          % (POLL_LIMIT_SEC // 60, job_id, job_id))
    return 2


def main():
    args = get_args()
    if not os.path.exists(args.FILE):
        raise SystemExit('파일이 없습니다: %s' % args.FILE)
    job_id = submit(args.SERVER, args.FILE, build_params(args))
    sys.exit(poll(args.SERVER, job_id))


if __name__ == '__main__':
    main()

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
  외부에서 Django 중계   --SERVER <웹서버주소>:8000/api/v1/exp/analysis
외부망은 연구실 서버에 직접 닿지 못하므로 웹서버의 중계 경로를 쓴다.
어느 쪽이든 이 스크립트 입장에선 '<SERVER>/jobs' 로 붙는 것이라 동작은 같다.
"""
import argparse
import os
import sys
import time

import requests

from utils.ai_report import QUESTIONNAIRE_SCALES

DEFAULT_SERVER = os.environ.get('ONECLICK_ANALYSIS_SERVER', '127.0.0.1:8500')

POLL_INTERVAL_SEC = 15
# 분석 실측이 수 분 + 대기열을 감안한다.
POLL_LIMIT_SEC = 3600


def get_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--SERVER', default=DEFAULT_SERVER,
                    help='분석 서버 주소. 내부망 직접이면 <IP>:8500, 외부에서는 '
                         'Django 중계 <웹서버>:8000/api/v1/exp/analysis '
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
                    choices=[None, 'neuronet', 'synthsleepnet'])
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


def submit(server, csv_path, params):
    size_mb = os.path.getsize(csv_path) / 1e6
    print('제출: %s (%.1fMB) -> %s' % (os.path.basename(csv_path), size_mb, server))
    with open(csv_path, 'rb') as f:
        # 파일을 본문으로 스트리밍한다 — 서버가 표준 라이브러리만으로 받는다.
        r = requests.post('http://%s/jobs' % server, params=params, data=f,
                          timeout=600)
    r.raise_for_status()
    body = r.json()
    if 'job' not in body:
        raise SystemExit('제출 실패: %s' % body)
    print('접수됨 — 작업 %s' % body['job'])
    return body['job']


def poll(server, job_id):
    deadline = time.time() + POLL_LIMIT_SEC
    seen = 0                      # 이미 출력한 로그 길이
    last_status = None
    while time.time() < deadline:
        try:
            r = requests.get('http://%s/jobs/%s' % (server, job_id), timeout=30)
            st = r.json()
        except requests.RequestException as e:
            print('[폴링] 연결 오류(%s) — 계속 시도합니다' % type(e).__name__)
            time.sleep(POLL_INTERVAL_SEC)
            continue

        tail = st.get('log_tail') or ''
        if len(tail) > seen:
            new = tail[seen:] if seen else tail
            sys.stdout.write(new if new.endswith('\n') else new + '\n')
            seen = len(tail)
        elif len(tail) < seen:    # 서버 쪽 tail 윈도가 밀렸다
            seen = len(tail)

        if st.get('status') != last_status:
            last_status = st.get('status')
            print('── 상태: %s ──' % last_status)

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

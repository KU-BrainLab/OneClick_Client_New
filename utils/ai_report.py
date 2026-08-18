# -*- coding:utf-8 -*-
"""측정 업로드 직후 AI 리포트를 미리 만들어 두는 단계.

리포트 생성에는 3~4분이 걸린다. 예전에는 웹에서 리포트를 열 때 생성을 시작해서,
임상의가 화면 앞에서 그 시간을 그대로 기다려야 했다. 게다가 그 3분 동안 연결에는
한 바이트도 흐르지 않아 중간 장비가 세션을 끊는 일이 잦았고, 다 만들어 놓고
전달만 실패하는 경우가 반복됐다.

측정을 올리는 시점에는 아무도 기다리지 않는다. 그래서 여기서 만들어 서버 DB 에
저장해 두고, 웹에서는 즉시 받아가게 한다.

이 단계가 실패해도 측정 데이터는 이미 올라가 있다. 따라서 예외를 밖으로 던지지
않고 경고만 남긴다 — 리포트는 웹에서 다시 만들 수 있다.
"""
import json
import time

import requests

# 서버가 리포트를 만드는 데 쓰는 시간(약 210초)보다 넉넉히 잡는다.
POST_TIMEOUT_SEC = 420

# POST 가 끊겼을 때 결과를 회수하는 폴링. 서버는 계속 만들고 있으므로
# 짧은 GET 으로 물어보면 된다.
POLL_INTERVAL_SEC = 10
POLL_LIMIT_SEC = 420


def _report_url(server, pk):
    """생성 전용 창구.

    /ai-report/ 는 로그인을 요구한다(리포트 본문에 환자 정보가 들어가므로).
    측정 장비는 토큰이 없으므로, 본문 없이 생성만 맡기는 이 경로를 쓴다.
    """
    return 'http://{}/api/v1/exp/{}/ai-report/generate/'.format(server, pk)


def extract_pk(create_response):
    """실험 생성 응답에서 pk 를 꺼낸다. 못 찾으면 None."""
    try:
        body = create_response.json()
    except (ValueError, AttributeError):
        return None
    pk = body.get('pk') if isinstance(body, dict) else None
    return pk


def _poll(server, pk, log):
    """POST 가 끊겼을 때 저장된 결과를 회수한다."""
    deadline = time.time() + POLL_LIMIT_SEC
    while time.time() < deadline:
        time.sleep(POLL_INTERVAL_SEC)
        try:
            r = requests.get(_report_url(server, pk), timeout=30)
        except requests.RequestException:
            continue                      # 일시적 네트워크 오류는 넘기고 계속 기다린다
        if r.status_code == 200:
            return True
        # 404 면 아직 없다. 생성 중인지 시작도 안 했는지 서버가 알려준다.
        try:
            state = r.json().get('status')
        except ValueError:
            state = None
        if state == 'not_started':
            log('[AI report] 서버에 생성 기록이 없습니다. 중단합니다.')
            return False
    return False


def request_ai_report(server, create_response, log=print):
    """업로드된 실험의 AI 리포트를 미리 생성해 둔다.

    server : '180.83.245.145:8000' 형태
    create_response : POST /api/v1/exp/ 의 응답 객체

    성공하면 True. 실패해도 예외를 던지지 않는다.
    """
    pk = extract_pk(create_response)
    if pk is None:
        log('[AI report] 응답에서 실험 번호를 찾지 못해 생성을 건너뜁니다. '
            '(서버가 구버전이면 pk 를 돌려주지 않습니다)')
        return False

    log('[AI report] Experiment {} Report Generating... — It takes 3~4 minutes'.format(pk))
    started = time.time()
    try:
        r = requests.post(_report_url(server, pk), timeout=POST_TIMEOUT_SEC)
        if r.status_code == 200:
            log('[AI report] Complete! ({:.0f}s)'.format(time.time() - started))
            return True
        # 서버가 명확히 거절한 경우다. 폴링해도 결과가 생기지 않는다.
        if r.status_code in (400, 401, 403, 404):
            log('[AI report] 서버 거절 {} — {}'.format(r.status_code, r.text[:200]))
            return False
        log('[AI report] 서버 응답 {} — 결과를 확인해 봅니다.'.format(r.status_code))
    except requests.RequestException as e:
        # 연결이 끊겨도 서버는 계속 만들고 있다. 여기서 포기하지 않는다.
        log('[AI report] 연결 끊김({}) — 결과를 기다립니다.'.format(type(e).__name__))

    if _poll(server, pk, log):
        log('[AI report] 완료 ({:.0f}초, 회수)'.format(time.time() - started))
        return True

    log('[AI report] 생성하지 못했습니다. 측정 데이터는 정상 업로드됐으니 '
        '웹에서 리포트를 다시 만들 수 있습니다.')
    return False

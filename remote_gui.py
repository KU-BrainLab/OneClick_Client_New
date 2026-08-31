# -*- coding:utf-8 -*-
"""측정 PC 용 원격 제출 GUI.

CSV 와 피험자 정보를 분석 서버(또는 Django 중계)로 보내고, 분석 로그를
실시간으로 창에 띄운다. remote_submit.py 의 GUI 판 — requests 와 tkinter
만 있으면 돌아가고, mne/torch 같은 분석 환경은 필요 없다.

    python remote_gui.py

한국어·영어·일본어를 지원한다. 창에서 고른 언어는 다음 실행 때도 유지된다.
서버 주소는 화면에 띄우지 않는다 — 아래 SERVER 또는 환경변수
ONECLICK_ANALYSIS_SERVER 로 정한다.
"""
import json
import os
import queue
import re
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import requests

# 설문 척도 (키, 화면 표기). 설명 문구는 아래 STRINGS 의 'scale_<키>' 에 있다.
#
# 이 파일은 측정 PC 에 단독으로 배포된다 — 분석 코드는 넘기지 않는다. 그래서
# utils.ai_report 를 import 하지 않고 여기에 둔다. 두 정의가 어긋나면 저장소
# 테스트가 잡는다.
QUESTIONNAIRE_SCALES = (
    ('IRLS', 'IRLS'),
    ('PSQI', 'PSQI'),
    ('ISI', 'ISI'),
    ('ESS', 'ESS'),
    ('COMPASS31', 'COMPASS31'),
    ('BAI', 'BAI'),
    ('BDI2', 'BDI-II'),
)

# ── 설정 (코드에서 수정) ─────────────────────────────────────────────
# 외부망(웹서버 중계): 'https://180.83.245.145:8443/api/v1/exp/analysis'
# 연구실 내부망(직접) : '<연구실서버IP>:8500' (스킴 없으면 http 로 해석)
# 환경변수 ONECLICK_ANALYSIS_SERVER 와 설정 파일의 server 키가 이 값을 덮는다.
# 개인정보가 인터넷 구간을 지나므로 외부 경로는 https 가 의무다.
SERVER = 'https://180.83.245.145:8443/api/v1/exp/analysis'
DEFAULT_SLEEP_MODEL = 'neuronet'
DEFAULT_CROP = True

POLL_INTERVAL_SEC = 15
POLL_LIMIT_SEC = 3600

# 언어 선택은 사용자별 설정으로 남긴다. 저장소 안에 두면 git 상태를
# 더럽히고, 측정 PC 마다 담당자가 다를 수 있다.
CONFIG_PATH = os.path.join(os.path.expanduser('~'), '.oneclick_remote_gui.json')


# ── 문구 ─────────────────────────────────────────────────────────────
LANGUAGES = (('ko', '한국어'), ('en', 'English'), ('ja', '日本語'))

STRINGS = {
    'ko': {
        'title': 'OneClick 원격 제출',
        'language': '언어',
        'file': '측정 CSV',
        'browse': '찾아보기',
        'file_dialog': '측정 CSV 선택',
        'filetype_csv': 'CSV 파일',
        'filetype_all': '모든 파일',
        'subject': '피험자 정보',
        'name': '이름',
        'age': '나이',
        'birth': '생년월일',
        'birth_hint': '(YYYY-MM-DD)',
        'sex': '성별',
        'sex_male': '남성',
        'sex_female': '여성',
        'mdate': '측정일시',
        'mdate_hint': '(YYYY-MM-DD HH:MM)',
        'stimulus': '자극 정보',
        'optional': '(선택)',
        'survey': '설문 점수 (모르는 항목은 비워 두세요)',
        'sleep_model': '수면 모델',
        'crop': '노이즈 크롭',
        'submit': '전송',
        'idle': '대기',
        'scale_IRLS': '하지불안 (0~40)',
        'scale_PSQI': '수면의 질 (0~21)',
        'scale_ISI': '불면 심각도 (0~28)',
        'scale_ESS': '주간 졸림 (0~24)',
        'scale_COMPASS31': '자율신경 증상 (0~100)',
        'scale_BAI': '불안 (0~63)',
        'scale_BDI2': '우울 (0~63)',
        'err_title': '입력 확인',
        'err_file': '측정 CSV 파일을 선택하세요.',
        'err_name': '이름을 입력하세요.',
        'err_age': '나이는 숫자로 입력하세요.',
        'err_birth': '생년월일은 YYYY-MM-DD 형식입니다.',
        'err_mdate': '측정일시는 "YYYY-MM-DD HH:MM" 형식입니다.',
        'err_scale': '{key} 점수는 숫자로 입력하세요.',
        'err_server': '서버 주소가 설정돼 있지 않습니다. 관리자에게 문의하세요.',
        'result_title': '원격 제출',
        'log_submit': '제출: {file} ({size:.1f}MB)\n',
        'log_accepted': '접수됐습니다. 분석을 기다립니다.\n',
        'st_upload': '업로드 중... {sent:.1f} / {total:.1f}MB',
        'st_job': '{state}',
        'st_retry': '연결 재시도 중',
        'state_queued': '대기열',
        'state_running': '분석 중',
        'state_done': '완료',
        'state_failed': '실패',
        'fin_done': '분석 완료 — 실험 번호 {pk}. 웹에서 확인하세요.',
        'fin_failed': '분석 실패: {error}',
        'fin_rejected': '제출 거부: {error}',
        'fin_badresp': '서버 응답을 해석할 수 없습니다 (HTTP {code}).',
        'fin_noconn': '서버에 연결할 수 없습니다 ({error}).',
        'fin_timeout': '제한 시간({min}분)을 넘겼습니다. 분석은 서버에서 계속될 수 '
                       '있습니다.',
        'fin_error': '오류: {error}',
        'fin_ssl': '서버 인증서를 확인하지 못했습니다. 프로그램과 함께 배포된 '
                   '인증서 파일이 빠졌거나 서버 설정이 바뀌었습니다 — 관리자에게 '
                   '문의하세요.',
        'srv_exit': '분석이 중단됐습니다 (종료 코드 {code}). 관리자에게 문의하세요.',
        'srv_restart': '서버가 다시 시작돼 중단됐습니다. 다시 전송해 주세요.',
        'srv_toobig': '파일이 너무 큽니다 ({mb}MB).',
        'srv_missing': '필수 항목이 빠졌습니다: {fields}',
        'srv_unreachable': '분석 서버가 응답하지 않습니다. 관리자에게 문의하세요.',
        'srv_notconfigured': '서버 설정이 끝나지 않았습니다. 관리자에게 문의하세요.',
    },
    'en': {
        'title': 'OneClick Remote Upload',
        'language': 'Language',
        'file': 'Recording CSV',
        'browse': 'Browse',
        'file_dialog': 'Select recording CSV',
        'filetype_csv': 'CSV files',
        'filetype_all': 'All files',
        'subject': 'Participant',
        'name': 'Name',
        'age': 'Age',
        'birth': 'Date of birth',
        'birth_hint': '(YYYY-MM-DD)',
        'sex': 'Sex',
        'sex_male': 'Male',
        'sex_female': 'Female',
        'mdate': 'Recorded at',
        'mdate_hint': '(YYYY-MM-DD HH:MM)',
        'stimulus': 'Stimulation',
        'optional': '(optional)',
        'survey': 'Questionnaire scores (leave blank if not administered)',
        'sleep_model': 'Sleep model',
        'crop': 'Crop noise',
        'submit': 'Send',
        'idle': 'Ready',
        'scale_IRLS': 'Restless legs (0-40)',
        'scale_PSQI': 'Sleep quality (0-21)',
        'scale_ISI': 'Insomnia severity (0-28)',
        'scale_ESS': 'Daytime sleepiness (0-24)',
        'scale_COMPASS31': 'Autonomic symptoms (0-100)',
        'scale_BAI': 'Anxiety (0-63)',
        'scale_BDI2': 'Depression (0-63)',
        'err_title': 'Check your entries',
        'err_file': 'Select the recording CSV file.',
        'err_name': 'Enter the name.',
        'err_age': 'Age must be a number.',
        'err_birth': 'Date of birth must be YYYY-MM-DD.',
        'err_mdate': 'Recorded at must be "YYYY-MM-DD HH:MM".',
        'err_scale': '{key} score must be a number.',
        'err_server': 'No server address is configured. Contact your administrator.',
        'result_title': 'Remote upload',
        'log_submit': 'Sending: {file} ({size:.1f}MB)\n',
        'log_accepted': 'Accepted. Waiting for the analysis.\n',
        'st_upload': 'Uploading... {sent:.1f} / {total:.1f}MB',
        'st_job': '{state}',
        'st_retry': 'Reconnecting',
        'state_queued': 'queued',
        'state_running': 'analyzing',
        'state_done': 'done',
        'state_failed': 'failed',
        'fin_done': 'Analysis complete - experiment {pk}. Open it on the web.',
        'fin_failed': 'Analysis failed: {error}',
        'fin_rejected': 'Upload rejected: {error}',
        'fin_badresp': 'Could not read the server response (HTTP {code}).',
        'fin_noconn': 'Could not reach the server ({error}).',
        'fin_timeout': 'Timed out after {min} minutes. The analysis may still be '
                       'running on the server.',
        'fin_error': 'Error: {error}',
        'fin_ssl': 'Could not verify the server certificate. The certificate '
                   'file shipped with this program is missing, or the server '
                   'changed - contact your administrator.',
        'srv_exit': 'The analysis stopped (exit code {code}). Contact your '
                    'administrator.',
        'srv_restart': 'The server restarted and interrupted the job. Please '
                       'send it again.',
        'srv_toobig': 'The file is too large ({mb}MB).',
        'srv_missing': 'Required fields are missing: {fields}',
        'srv_unreachable': 'The analysis server is not responding. Contact your '
                           'administrator.',
        'srv_notconfigured': 'The server is not fully configured. Contact your '
                             'administrator.',
    },
    'ja': {
        'title': 'OneClick リモート送信',
        'language': '言語',
        'file': '測定CSV',
        'browse': '参照',
        'file_dialog': '測定CSVを選択',
        'filetype_csv': 'CSVファイル',
        'filetype_all': 'すべてのファイル',
        'subject': '被験者情報',
        'name': '氏名',
        'age': '年齢',
        'birth': '生年月日',
        'birth_hint': '(YYYY-MM-DD)',
        'sex': '性別',
        'sex_male': '男性',
        'sex_female': '女性',
        'mdate': '測定日時',
        'mdate_hint': '(YYYY-MM-DD HH:MM)',
        'stimulus': '刺激情報',
        'optional': '(任意)',
        'survey': '質問票スコア（未実施の項目は空欄）',
        'sleep_model': '睡眠モデル',
        'crop': 'ノイズ除去',
        'submit': '送信',
        'idle': '待機中',
        'scale_IRLS': 'むずむず脚 (0～40)',
        'scale_PSQI': '睡眠の質 (0～21)',
        'scale_ISI': '不眠重症度 (0～28)',
        'scale_ESS': '日中の眠気 (0～24)',
        'scale_COMPASS31': '自律神経症状 (0～100)',
        'scale_BAI': '不安 (0～63)',
        'scale_BDI2': 'うつ (0～63)',
        'err_title': '入力の確認',
        'err_file': '測定CSVファイルを選択してください。',
        'err_name': '氏名を入力してください。',
        'err_age': '年齢は数字で入力してください。',
        'err_birth': '生年月日は YYYY-MM-DD 形式です。',
        'err_mdate': '測定日時は "YYYY-MM-DD HH:MM" 形式です。',
        'err_scale': '{key} のスコアは数字で入力してください。',
        'err_server': 'サーバーアドレスが設定されていません。管理者にご連絡ください。',
        'result_title': 'リモート送信',
        'log_submit': '送信: {file} ({size:.1f}MB)\n',
        'log_accepted': '受付が完了しました。解析をお待ちください。\n',
        'st_upload': 'アップロード中... {sent:.1f} / {total:.1f}MB',
        'st_job': '{state}',
        'st_retry': '再接続中',
        'state_queued': '待機列',
        'state_running': '解析中',
        'state_done': '完了',
        'state_failed': '失敗',
        'fin_done': '解析が完了しました — 実験番号 {pk}。ウェブでご確認ください。',
        'fin_failed': '解析に失敗しました: {error}',
        'fin_rejected': '送信が拒否されました: {error}',
        'fin_badresp': 'サーバーの応答を解釈できません (HTTP {code})。',
        'fin_noconn': 'サーバーに接続できません ({error})。',
        'fin_timeout': '制限時間({min}分)を超えました。解析はサーバー側で続いている'
                       '可能性があります。',
        'fin_error': 'エラー: {error}',
        'fin_ssl': 'サーバー証明書を確認できませんでした。プログラムに同梱の証明書'
                   'ファイルが見つからないか、サーバー設定が変わっています — 管理者に'
                   'ご連絡ください。',
        'srv_exit': '解析が中断されました（終了コード {code}）。管理者にご連絡ください。',
        'srv_restart': 'サーバーの再起動により中断されました。もう一度送信してください。',
        'srv_toobig': 'ファイルが大きすぎます（{mb}MB）。',
        'srv_missing': '必須項目が不足しています: {fields}',
        'srv_unreachable': '解析サーバーが応答しません。管理者にご連絡ください。',
        'srv_notconfigured': 'サーバーの設定が完了していません。管理者にご連絡ください。',
    },
}

# 서버가 보내는 문구는 한국어로 고정돼 있다. 담당자가 실제로 행동을 바꿔야
# 하는 것들만 골라 현재 언어로 옮긴다 (분석 로그 본문은 그대로 흘린다).
SERVER_MESSAGES = (
    (re.compile(r'파이프라인 종료 코드 (\d+)'), 'srv_exit', ('code',)),
    (re.compile(r'재시작으로 중단'), 'srv_restart', ()),
    (re.compile(r'파일이 너무 큽니다 \((\d+)MB\)'), 'srv_toobig', ('mb',)),
    (re.compile(r'필수 항목 누락: (.+)'), 'srv_missing', ('fields',)),
    (re.compile(r'분석 서버에 연결할 수 없습니다'), 'srv_unreachable', ()),
    (re.compile(r'중계가 설정돼 있지 않습니다'), 'srv_notconfigured', ()),
)


def server_message(text):
    """서버가 보낸 문구를 현재 언어로. 모르는 문구는 그대로 둔다."""
    if not text:
        return text
    for pattern, key, names in SERVER_MESSAGES:
        m = pattern.search(str(text))
        if m:
            return t(key, **dict(zip(names, m.groups())))
    return str(text)

_lang = 'ko'


def set_language(code):
    global _lang
    if code in STRINGS:
        _lang = code


def current_language():
    return _lang


def t(_key, **fmt):
    """현재 언어의 문구. 번역이 빠졌으면 한국어로 되돌린다.

    매개변수 이름 앞에 밑줄을 붙인 건 문구의 자리표시자와 이름이 겹치는 걸
    막기 위해서다 — t('err_scale', key='ISI') 가 TypeError 로 죽었었다.
    """
    text = STRINGS[_lang].get(_key) or STRINGS['ko'].get(_key, _key)
    return text.format(**fmt) if fmt else text


def resolve_server():
    """분석 서버 주소 — 환경변수 > 설정 파일 > 코드 기본값 순.

    exe 로 묶어 배포하면 주소가 실행 파일 안에 박힌다. 서버가 바뀌었을 때
    다시 빌드하지 않아도 되도록 설정 파일(server 키)로도 덮을 수 있게 한다.
    """
    raw = (os.environ.get('ONECLICK_ANALYSIS_SERVER')
           or load_config().get('server') or SERVER or '')
    return raw.strip().rstrip('/')


def _resource(name):
    """곁들여 배포되는 파일의 경로. exe 로 묶였을 때는 임시 해제 폴더에 있다."""
    base = getattr(sys, '_MEIPASS',
                   os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, name)


def _api_base(server):
    """'host:port' 나 'https://…' 를 스킴 있는 베이스 URL 로."""
    server = server.strip().rstrip('/')
    return server if '://' in server else 'http://' + server


def _tls_verify(base):
    """requests 의 verify 인자.

    서버가 자체 서명 인증서를 쓰므로, 함께 배포되는 연구실 CA 공개
    인증서(oneclick-ca.pem)가 있으면 그걸로 서버를 검증한다.
    """
    if base.startswith('https://'):
        ca = _resource('oneclick-ca.pem')
        if os.path.exists(ca):
            return ca
    return True


def load_config():
    try:
        with open(CONFIG_PATH, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_config(cfg):
    try:
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, ensure_ascii=False)
    except OSError:
        pass                     # 설정을 못 남겨도 실행에는 지장이 없다


# ── 제출·폴링 워커 (GUI 비의존 — 테스트에서 직접 부른다) ─────────────
class _ProgressFile:
    """read 할 때마다 진행률을 알리는 파일 래퍼. requests 가 본문으로 쓴다."""

    def __init__(self, path, notify):
        self._f = open(path, 'rb')
        self._notify = notify
        self._sent = 0
        self._reported = -1
        self.total = os.path.getsize(path)

    def read(self, size=-1):
        chunk = self._f.read(size)
        self._sent += len(chunk)
        # requests 가 16KB 씩 읽으므로 매번 알리면 큐가 수만 건으로 분다.
        # 1MB 경계와 마지막에만 알린다.
        mark = self._sent >> 20
        if mark != self._reported or self._sent == self.total:
            self._reported = mark
            self._notify(self._sent, self.total)
        return chunk

    def __len__(self):          # requests 가 Content-Length 를 정하는 근거
        return self.total

    def close(self):
        self._f.close()


def run_submission(server, file_path, params, q):
    """제출 → 폴링. 진행 상황을 q 에 넣는다. 스레드에서 돌린다.

    q 메시지: ('status', 문구) / ('log', 텍스트) / ('finish', 성공여부, 문구)
    서버 주소는 화면에 내보내지 않는다 — 담당자가 알 필요가 없고, 오류
    문구로 내부망 주소가 새는 것도 막는다.
    """
    try:
        size_mb = os.path.getsize(file_path) / 1e6

        def notify(sent, total):
            q.put(('status', t('st_upload', sent=sent / 1e6, total=total / 1e6)))

        q.put(('log', t('log_submit', file=os.path.basename(file_path),
                        size=size_mb)))
        base = _api_base(server)
        verify = _tls_verify(base)
        body = _ProgressFile(file_path, notify)
        try:
            r = requests.post('%s/jobs' % base, params=params,
                              data=body, timeout=600, verify=verify)
        finally:
            body.close()
        try:
            resp = r.json()
        except ValueError:
            q.put(('finish', False, t('fin_badresp', code=r.status_code)))
            return
        if r.status_code != 200 or 'job' not in resp:
            q.put(('finish', False, t('fin_rejected',
                                      error=server_message(resp.get('error'))
                                      or 'HTTP %d' % r.status_code)))
            return
        job_id = resp['job']
        q.put(('log', t('log_accepted', job=job_id)))
        q.put(('status', t('st_job', job=job_id, state=t('state_queued'))))

        deadline = time.time() + POLL_LIMIT_SEC
        seen = 0
        last_status = None
        while time.time() < deadline:
            try:
                st = requests.get('%s/jobs/%s' % (base, job_id),
                                  timeout=30, verify=verify).json()
            except (requests.RequestException, ValueError):
                q.put(('status', t('st_retry', job=job_id)))
                time.sleep(POLL_INTERVAL_SEC)
                continue
            # 중계를 거치면 서버 장애가 예외가 아니라 {'error':...} 로 온다.
            # seen 을 건드리지 않고 넘어가야 복구 후 로그가 중복되지 않는다.
            if st.get('status') is None:
                q.put(('status', t('st_retry', job=job_id)))
                time.sleep(POLL_INTERVAL_SEC)
                continue

            tail = st.get('log_tail') or ''
            if len(tail) > seen:
                q.put(('log', tail[seen:]))
                seen = len(tail)
            elif len(tail) < seen:      # 서버 쪽 tail 윈도가 밀렸다
                seen = len(tail)

            if st['status'] != last_status:
                last_status = st['status']
                q.put(('status', t('st_job', job=job_id,
                                   state=t('state_' + last_status)
                                   if 'state_' + last_status in STRINGS[_lang]
                                   else last_status)))

            if last_status == 'done':
                q.put(('finish', True, t('fin_done', pk=st.get('exp_pk'))))
                return
            if last_status == 'failed':
                q.put(('finish', False,
                       t('fin_failed', error=server_message(st.get('error')))))
                return
            time.sleep(POLL_INTERVAL_SEC)

        q.put(('finish', False, t('fin_timeout', min=POLL_LIMIT_SEC // 60,
                                  job=job_id)))
    except requests.exceptions.SSLError:
        q.put(('finish', False, t('fin_ssl')))
    except requests.RequestException as e:
        # 예외 문자열에는 서버 주소가 들어 있다. 종류만 보여준다.
        q.put(('finish', False, t('fin_noconn', error=type(e).__name__)))
    except Exception as e:      # 워커 스레드가 조용히 죽으면 버튼이 안 풀린다
        q.put(('finish', False, t('fin_error', error=e)))


# ── GUI ──────────────────────────────────────────────────────────────
class RemoteGui:
    def __init__(self, root):
        self.root = root
        set_language(load_config().get('language', 'ko'))
        root.minsize(660, 660)
        try:                    # 창·작업표시줄 아이콘 (없어도 동작엔 지장 없음)
            root.iconbitmap(_resource('remote_gui.ico'))
        except (tk.TclError, OSError):
            pass
        self.q = queue.Queue()
        self._tx = []           # 언어를 바꿀 때 다시 쓸 (위젯, 문구키) 목록
        self._build()
        self._retranslate()
        self._drain()

    # 문구 등록 --------------------------------------------------------
    def _tr(self, widget, key):
        """이 위젯의 text 를 문구키에 묶는다. 언어 변경 시 함께 바뀐다."""
        self._tx.append((widget, key))
        return widget

    def _retranslate(self):
        self.root.title(t('title'))
        for widget, key in self._tx:
            widget.configure(text=t(key))
        # 성별 콤보는 값이 아니라 표시만 번역한다 (서버로는 male/female).
        idx = self.sex_box.current()
        self.sex_box.configure(values=(t('sex_male'), t('sex_female')))
        self.sex_box.current(max(0, idx))
        if not self._busy:
            self.var_status.set(t('idle'))

    # 위젯 구성 ---------------------------------------------------------
    def _build(self):
        self._busy = False
        pad = {'padx': 6, 'pady': 3}
        frm = ttk.Frame(self.root, padding=10)
        frm.pack(fill='both', expand=True)
        frm.columnconfigure(1, weight=1)
        row = 0

        # 언어
        top = ttk.Frame(frm)
        top.grid(row=row, column=0, columnspan=4, sticky='ew')
        top.columnconfigure(0, weight=1)
        self._tr(ttk.Label(top, text=''), 'language').grid(
            row=0, column=1, sticky='e', **pad)
        self.var_lang = tk.StringVar(
            value=dict(LANGUAGES).get(current_language(), '한국어'))
        lang_box = ttk.Combobox(top, textvariable=self.var_lang,
                                state='readonly', width=10,
                                values=[n for _c, n in LANGUAGES])
        lang_box.grid(row=0, column=2, sticky='e', **pad)
        lang_box.bind('<<ComboboxSelected>>', self._change_language)
        row += 1

        # 측정 파일
        self._tr(ttk.Label(frm, text=''), 'file').grid(
            row=row, column=0, sticky='w', **pad)
        self.var_file = tk.StringVar()
        ttk.Entry(frm, textvariable=self.var_file).grid(
            row=row, column=1, columnspan=2, sticky='ew', **pad)
        self._tr(ttk.Button(frm, text='', command=self._pick_file),
                 'browse').grid(row=row, column=3, sticky='w', **pad)
        row += 1

        # 피험자 정보
        sub = self._tr(ttk.LabelFrame(frm, text='', padding=6), 'subject')
        sub.grid(row=row, column=0, columnspan=4, sticky='ew', **pad)
        sub.columnconfigure(1, weight=1)
        sub.columnconfigure(3, weight=1)
        self.var_name = tk.StringVar()
        self.var_age = tk.StringVar()
        self.var_birth = tk.StringVar()
        self.var_sex = tk.StringVar()
        self.var_mdate = tk.StringVar()
        self.var_stim = tk.StringVar()
        self._tr(ttk.Label(sub, text=''), 'name').grid(
            row=0, column=0, sticky='w', **pad)
        ttk.Entry(sub, textvariable=self.var_name).grid(
            row=0, column=1, sticky='ew', **pad)
        self._tr(ttk.Label(sub, text=''), 'age').grid(
            row=0, column=2, sticky='w', **pad)
        ttk.Entry(sub, textvariable=self.var_age, width=8).grid(
            row=0, column=3, sticky='w', **pad)
        self._tr(ttk.Label(sub, text=''), 'birth').grid(
            row=1, column=0, sticky='w', **pad)
        ttk.Entry(sub, textvariable=self.var_birth).grid(
            row=1, column=1, sticky='ew', **pad)
        self._tr(ttk.Label(sub, text=''), 'birth_hint').grid(
            row=1, column=2, columnspan=2, sticky='w', **pad)
        self._tr(ttk.Label(sub, text=''), 'sex').grid(
            row=2, column=0, sticky='w', **pad)
        self.sex_box = ttk.Combobox(sub, textvariable=self.var_sex,
                                    state='readonly', width=10)
        self.sex_box.grid(row=2, column=1, sticky='w', **pad)
        self.sex_box.configure(values=(t('sex_male'), t('sex_female')))
        self.sex_box.current(0)
        self._tr(ttk.Label(sub, text=''), 'mdate').grid(
            row=3, column=0, sticky='w', **pad)
        ttk.Entry(sub, textvariable=self.var_mdate).grid(
            row=3, column=1, sticky='ew', **pad)
        self._tr(ttk.Label(sub, text=''), 'mdate_hint').grid(
            row=3, column=2, columnspan=2, sticky='w', **pad)
        self._tr(ttk.Label(sub, text=''), 'stimulus').grid(
            row=4, column=0, sticky='w', **pad)
        ttk.Entry(sub, textvariable=self.var_stim).grid(
            row=4, column=1, sticky='ew', **pad)
        self._tr(ttk.Label(sub, text=''), 'optional').grid(
            row=4, column=2, sticky='w', **pad)
        row += 1

        # 설문 — 빈 칸은 '미실시'로 보낸다 (0 은 실제 0점으로 저장된다)
        qf = self._tr(ttk.LabelFrame(frm, text='', padding=6), 'survey')
        qf.grid(row=row, column=0, columnspan=4, sticky='ew', **pad)
        self.var_scales = {}
        for i, (key, label) in enumerate(QUESTIONNAIRE_SCALES):
            r, c = divmod(i, 2)
            holder = ttk.Frame(qf)
            holder.grid(row=r, column=c * 2, sticky='w', **pad)
            ttk.Label(holder, text=label + '  ').pack(side='left')
            self._tr(ttk.Label(holder, text=''), 'scale_' + key).pack(side='left')
            var = tk.StringVar()
            self.var_scales[key] = var
            ttk.Entry(qf, textvariable=var, width=8).grid(
                row=r, column=c * 2 + 1, sticky='w', **pad)
        row += 1

        # 옵션
        opt = ttk.Frame(frm)
        opt.grid(row=row, column=0, columnspan=4, sticky='ew', **pad)
        self._tr(ttk.Label(opt, text=''), 'sleep_model').grid(
            row=0, column=0, sticky='w', **pad)
        self.var_model = tk.StringVar(value=DEFAULT_SLEEP_MODEL)
        ttk.Combobox(opt, textvariable=self.var_model, state='readonly',
                     values=('neuronet', 'synthsleepnet'), width=14).grid(
            row=0, column=1, sticky='w', **pad)
        self.var_crop = tk.BooleanVar(value=DEFAULT_CROP)
        self._tr(ttk.Checkbutton(opt, text='', variable=self.var_crop),
                 'crop').grid(row=0, column=2, sticky='w', **pad)
        row += 1

        # 제출 버튼 + 상태
        act = ttk.Frame(frm)
        act.grid(row=row, column=0, columnspan=4, sticky='ew', **pad)
        act.columnconfigure(1, weight=1)
        self.btn = self._tr(ttk.Button(act, text='', command=self._submit),
                            'submit')
        self.btn.grid(row=0, column=0, **pad)
        self.var_status = tk.StringVar()
        ttk.Label(act, textvariable=self.var_status).grid(
            row=0, column=1, sticky='w', **pad)
        row += 1

        # 로그
        self.log = scrolledtext.ScrolledText(frm, height=18, state='disabled',
                                             font=('Consolas', 9))
        self.log.grid(row=row, column=0, columnspan=4, sticky='nsew', **pad)
        frm.rowconfigure(row, weight=1)
        row += 1

        # 기관 로고 (연구실·학교·회사) — 파일이 없으면 그 자리만 생략된다.
        # PhotoImage 는 참조를 붙들어 두지 않으면 GC 로 사라져 빈 칸이 된다.
        self._logo_imgs = []
        logos = ttk.Frame(frm)
        logos.grid(row=row, column=0, columnspan=4, pady=(6, 0))
        for name in ('logo_lab.png', 'logo_univ.png', 'logo_company.png'):
            path = _resource(name)
            if not os.path.exists(path):
                continue
            try:
                img = tk.PhotoImage(file=path)
            except tk.TclError:
                continue
            self._logo_imgs.append(img)
            ttk.Label(logos, image=img).pack(side='left', padx=16)

    # 동작 --------------------------------------------------------------
    def _change_language(self, _event=None):
        name = self.var_lang.get()
        for code, disp in LANGUAGES:
            if disp == name:
                set_language(code)
                cfg = load_config()
                cfg['language'] = code
                save_config(cfg)
                break
        self._retranslate()

    def _pick_file(self):
        path = filedialog.askopenfilename(
            title=t('file_dialog'),
            filetypes=[(t('filetype_csv'), '*.csv'), (t('filetype_all'), '*.*')])
        if not path:
            return
        self.var_file.set(path)
        # 파일명이 측정 규칙(YYYY-MM-DD-HHMM)이면 측정일시를 자동으로 채운다
        m = re.match(r'(\d{4}-\d{2}-\d{2})-(\d{2})(\d{2})',
                     os.path.basename(path))
        if m and not self.var_mdate.get().strip():
            self.var_mdate.set('%s %s:%s' % (m.group(1), m.group(2), m.group(3)))

    def _append_log(self, text):
        self.log.configure(state='normal')
        self.log.insert('end', text)
        self.log.see('end')
        self.log.configure(state='disabled')

    def _validate(self):
        """입력 검증. 통과하면 (경로, params), 아니면 None."""
        def bad(_key, **fmt):
            messagebox.showerror(t('err_title'), t(_key, **fmt))
            return None

        path = self.var_file.get().strip()
        if not path or not os.path.exists(path):
            return bad('err_file')
        name = self.var_name.get().strip()
        age = self.var_age.get().strip()
        birth = self.var_birth.get().strip()
        mdate = self.var_mdate.get().strip()
        if not name:
            return bad('err_name')
        if not age.isdigit():
            return bad('err_age')
        if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', birth):
            return bad('err_birth')
        if not re.fullmatch(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}', mdate):
            return bad('err_mdate')
        params = {
            'name': name, 'age': age, 'birth': birth,
            # 콤보는 번역된 이름을 보여주지만 서버로는 항상 male/female 이다
            'sex': 'male' if self.sex_box.current() == 0 else 'female',
            'measurement_date': mdate,
        }
        if self.var_stim.get().strip():
            params['stimulus'] = self.var_stim.get().strip()
        if self.var_model.get() != DEFAULT_SLEEP_MODEL:
            params['sleep_model'] = self.var_model.get()
        if not self.var_crop.get():
            params['crop'] = 'false'
        for key, var in self.var_scales.items():
            v = var.get().strip()
            if not v:
                continue
            if not v.isdigit():
                return bad('err_scale', key=key)
            params[key] = v
        return path, params

    def _submit(self):
        server = resolve_server()
        if not server:
            messagebox.showerror(t('err_title'), t('err_server'))
            return
        checked = self._validate()
        if checked is None:
            return
        path, params = checked
        self._busy = True
        self.btn.configure(state='disabled')
        self.log.configure(state='normal')
        self.log.delete('1.0', 'end')
        self.log.configure(state='disabled')
        threading.Thread(target=run_submission,
                         args=(server, path, params, self.q),
                         daemon=True).start()

    def _drain(self):
        """워커 큐를 GUI 스레드에서 비운다. 위젯은 여기서만 만진다."""
        try:
            while True:
                kind, *rest = self.q.get_nowait()
                if kind == 'log':
                    self._append_log(rest[0])
                elif kind == 'status':
                    self.var_status.set(rest[0])
                elif kind == 'finish':
                    ok, msg = rest
                    self._busy = False
                    self.var_status.set(msg)
                    self._append_log('\n' + msg + '\n')
                    self.btn.configure(state='normal')
                    (messagebox.showinfo if ok else messagebox.showerror)(
                        t('result_title'), msg)
        except queue.Empty:
            pass
        self.root.after(200, self._drain)


def run_check(out_path):
    """배포 점검 — 창을 띄우지 않고 서버까지 닿는지 확인해 파일로 남긴다.

        OneClickRemote.exe --check 결과.txt

    측정 PC 에 실행 파일을 두고 나서 '이 PC 에서 서버가 보이는지'를 확인할
    때 쓴다. 실행 파일이 제대로 묶였는지도 이걸로 드러난다.
    """
    lines = ['OneClick 원격 제출 — 배포 점검',
             'python %s' % sys.version.split()[0],
             'requests %s' % requests.__version__,
             '실행 형태: %s' % ('실행 파일' if getattr(sys, 'frozen', False)
                                else '스크립트')]
    server = resolve_server()
    lines.append('서버: %s' % (server or '(설정 없음)'))
    ok = False
    if server:
        base = _api_base(server)
        verify = _tls_verify(base)
        lines.append('암호화: %s' % (
            'TLS + 연구실 CA 검증' if isinstance(verify, str)
            else ('TLS (시스템 신뢰 저장소)' if base.startswith('https://')
                  else '없음 — 내부망 직결용')))
        try:
            r = requests.get('%s/jobs/1' % base, timeout=10, verify=verify)
            # 없는 작업이라 404 가 정상이다 — 응답이 왔다는 게 핵심
            ok = r.status_code in (200, 404)
            lines.append('연결: HTTP %d — %s' % (r.status_code,
                                                 '정상' if ok else '확인 필요'))
            lines.append('응답: %s' % r.text[:200])
        except requests.RequestException as e:
            lines.append('연결 실패: %s' % type(e).__name__)
    lines.append('결과: %s' % ('OK' if ok else 'FAIL'))
    text = '\n'.join(lines) + '\n'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(text)
    return ok


if __name__ == '__main__':
    if '--check' in sys.argv:
        i = sys.argv.index('--check')
        target = (sys.argv[i + 1] if len(sys.argv) > i + 1
                  else os.path.join(os.path.dirname(os.path.abspath(
                      sys.executable if getattr(sys, 'frozen', False)
                      else __file__)), 'oneclick_check.txt'))
        sys.exit(0 if run_check(target) else 1)
    root = tk.Tk()
    RemoteGui(root)
    root.mainloop()

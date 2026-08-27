# -*- coding:utf-8 -*-
"""측정 PC 용 원격 제출 GUI.

CSV 와 피험자 정보를 분석 서버(또는 Django 중계)로 보내고, 분석 로그를
실시간으로 창에 띄운다. remote_submit.py 의 GUI 판 — requests 와 tkinter
만 있으면 돌아가고, mne/torch 같은 분석 환경은 필요 없다.

    python remote_gui.py

서버 주소·기본값은 아래 DEFAULTS 에서 고친다. gui.py(로컬 분석용)와는
별개의 창이다.
"""
import os
import queue
import re
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import requests

from utils.ai_report import QUESTIONNAIRE_SCALES

# ── 기본값 (코드에서 수정) ───────────────────────────────────────────
DEFAULTS = {
    # 외부망(중계): '180.83.245.145:8000/api/v1/exp/analysis'
    # 연구실 내부망(직접): '<연구실서버IP>:8500'
    'SERVER': os.environ.get('ONECLICK_ANALYSIS_SERVER',
                             '180.83.245.145:8000/api/v1/exp/analysis'),
    'SLEEP_MODEL': 'neuronet',
    'CROP': True,
}

POLL_INTERVAL_SEC = 15
POLL_LIMIT_SEC = 3600


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
    """
    try:
        size_mb = os.path.getsize(file_path) / 1e6

        def notify(sent, total):
            q.put(('status', '업로드 중... %.1f / %.1fMB'
                             % (sent / 1e6, total / 1e6)))

        q.put(('log', '제출: %s (%.1fMB) -> %s\n'
                      % (os.path.basename(file_path), size_mb, server)))
        body = _ProgressFile(file_path, notify)
        try:
            r = requests.post('http://%s/jobs' % server, params=params,
                              data=body, timeout=600)
        finally:
            body.close()
        try:
            resp = r.json()
        except ValueError:
            q.put(('finish', False, '서버 응답을 해석할 수 없습니다 (HTTP %d).'
                                    % r.status_code))
            return
        if r.status_code != 200 or 'job' not in resp:
            q.put(('finish', False, '제출 거부: %s'
                                    % resp.get('error', 'HTTP %d' % r.status_code)))
            return
        job_id = resp['job']
        q.put(('log', '접수됨 — 작업 %s. 분석을 기다립니다.\n' % job_id))
        q.put(('status', '작업 %s — 대기열' % job_id))

        deadline = time.time() + POLL_LIMIT_SEC
        seen = 0
        last_status = None
        while time.time() < deadline:
            try:
                st = requests.get('http://%s/jobs/%s' % (server, job_id),
                                  timeout=30).json()
            except (requests.RequestException, ValueError):
                q.put(('status', '작업 %s — 연결 재시도 중' % job_id))
                time.sleep(POLL_INTERVAL_SEC)
                continue
            # 중계를 거치면 서버 장애가 예외가 아니라 {'error':...} 로 온다.
            # seen 을 건드리지 않고 넘어가야 복구 후 로그가 중복되지 않는다.
            if st.get('status') is None:
                q.put(('status', '작업 %s — 중계 오류, 재시도 중' % job_id))
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
                label = {'queued': '대기열', 'running': '분석 중',
                         'done': '완료', 'failed': '실패'}.get(last_status,
                                                               last_status)
                q.put(('status', '작업 %s — %s' % (job_id, label)))

            if last_status == 'done':
                q.put(('finish', True,
                       '분석 완료 — 실험 번호 %s. 웹에서 확인하세요.'
                       % st.get('exp_pk')))
                return
            if last_status == 'failed':
                q.put(('finish', False, '분석 실패: %s' % st.get('error')))
                return
            time.sleep(POLL_INTERVAL_SEC)

        q.put(('finish', False,
               '제한 시간(%d분)을 넘겼습니다. 작업 %s 는 서버에서 계속될 수 '
               '있습니다.' % (POLL_LIMIT_SEC // 60, job_id)))
    except requests.RequestException as e:
        q.put(('finish', False, '서버에 연결할 수 없습니다: %s'
                                % type(e).__name__))
    except Exception as e:      # 워커 스레드가 조용히 죽으면 버튼이 안 풀린다
        q.put(('finish', False, '오류: %s' % e))


# ── GUI ──────────────────────────────────────────────────────────────
class RemoteGui:
    def __init__(self, root):
        self.root = root
        root.title('OneClick 원격 제출')
        root.minsize(640, 640)
        try:                    # 창·작업표시줄 아이콘 (없어도 동작엔 지장 없음)
            root.iconbitmap(os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'remote_gui.ico'))
        except tk.TclError:
            pass
        self.q = queue.Queue()
        self._build()
        self._drain()

    # 위젯 구성 ---------------------------------------------------------
    def _build(self):
        pad = {'padx': 6, 'pady': 3}
        frm = ttk.Frame(self.root, padding=10)
        frm.pack(fill='both', expand=True)
        frm.columnconfigure(1, weight=1)
        frm.columnconfigure(3, weight=1)
        row = 0

        # 측정 파일
        ttk.Label(frm, text='측정 CSV').grid(row=row, column=0, sticky='w', **pad)
        self.var_file = tk.StringVar()
        ttk.Entry(frm, textvariable=self.var_file).grid(
            row=row, column=1, columnspan=2, sticky='ew', **pad)
        ttk.Button(frm, text='찾아보기', command=self._pick_file).grid(
            row=row, column=3, sticky='w', **pad)
        row += 1

        # 피험자 정보
        sub = ttk.LabelFrame(frm, text='피험자 정보', padding=6)
        sub.grid(row=row, column=0, columnspan=4, sticky='ew', **pad)
        sub.columnconfigure(1, weight=1)
        sub.columnconfigure(3, weight=1)
        self.var_name = tk.StringVar()
        self.var_age = tk.StringVar()
        self.var_birth = tk.StringVar()
        self.var_sex = tk.StringVar(value='male')
        self.var_mdate = tk.StringVar()
        self.var_stim = tk.StringVar()
        ttk.Label(sub, text='이름').grid(row=0, column=0, sticky='w', **pad)
        ttk.Entry(sub, textvariable=self.var_name).grid(
            row=0, column=1, sticky='ew', **pad)
        ttk.Label(sub, text='나이').grid(row=0, column=2, sticky='w', **pad)
        ttk.Entry(sub, textvariable=self.var_age, width=8).grid(
            row=0, column=3, sticky='w', **pad)
        ttk.Label(sub, text='생년월일').grid(row=1, column=0, sticky='w', **pad)
        ttk.Entry(sub, textvariable=self.var_birth).grid(
            row=1, column=1, sticky='ew', **pad)
        ttk.Label(sub, text='(YYYY-MM-DD)').grid(row=1, column=2, columnspan=2,
                                                 sticky='w', **pad)
        ttk.Label(sub, text='성별').grid(row=2, column=0, sticky='w', **pad)
        ttk.Combobox(sub, textvariable=self.var_sex, state='readonly',
                     values=('male', 'female'), width=10).grid(
            row=2, column=1, sticky='w', **pad)
        ttk.Label(sub, text='측정일시').grid(row=3, column=0, sticky='w', **pad)
        ttk.Entry(sub, textvariable=self.var_mdate).grid(
            row=3, column=1, sticky='ew', **pad)
        ttk.Label(sub, text='(YYYY-MM-DD HH:MM)').grid(
            row=3, column=2, columnspan=2, sticky='w', **pad)
        ttk.Label(sub, text='자극 정보').grid(row=4, column=0, sticky='w', **pad)
        ttk.Entry(sub, textvariable=self.var_stim).grid(
            row=4, column=1, sticky='ew', **pad)
        ttk.Label(sub, text='(선택)').grid(row=4, column=2, sticky='w', **pad)
        row += 1

        # 설문 — 빈 칸은 '미실시'로 보낸다 (0 은 실제 0점으로 저장된다)
        qf = ttk.LabelFrame(frm, text='설문 점수 (모르면 비워두기)', padding=6)
        qf.grid(row=row, column=0, columnspan=4, sticky='ew', **pad)
        self.var_scales = {}
        for i, (key, label, desc) in enumerate(QUESTIONNAIRE_SCALES):
            r, c = divmod(i, 2)
            ttk.Label(qf, text='%s (%s)' % (key, label)).grid(
                row=r, column=c * 2, sticky='w', **pad)
            var = tk.StringVar()
            self.var_scales[key] = var
            ttk.Entry(qf, textvariable=var, width=8).grid(
                row=r, column=c * 2 + 1, sticky='w', **pad)
        row += 1

        # 옵션 + 서버
        opt = ttk.Frame(frm)
        opt.grid(row=row, column=0, columnspan=4, sticky='ew', **pad)
        opt.columnconfigure(4, weight=1)
        ttk.Label(opt, text='수면 모델').grid(row=0, column=0, sticky='w', **pad)
        self.var_model = tk.StringVar(value=DEFAULTS['SLEEP_MODEL'])
        ttk.Combobox(opt, textvariable=self.var_model, state='readonly',
                     values=('neuronet', 'synthsleepnet'), width=14).grid(
            row=0, column=1, sticky='w', **pad)
        self.var_crop = tk.BooleanVar(value=DEFAULTS['CROP'])
        ttk.Checkbutton(opt, text='노이즈 크롭', variable=self.var_crop).grid(
            row=0, column=2, sticky='w', **pad)
        ttk.Label(opt, text='서버').grid(row=0, column=3, sticky='e', **pad)
        self.var_server = tk.StringVar(value=DEFAULTS['SERVER'])
        ttk.Entry(opt, textvariable=self.var_server).grid(
            row=0, column=4, sticky='ew', **pad)
        row += 1

        # 제출 버튼 + 상태
        act = ttk.Frame(frm)
        act.grid(row=row, column=0, columnspan=4, sticky='ew', **pad)
        act.columnconfigure(1, weight=1)
        self.btn = ttk.Button(act, text='전송', command=self._submit)
        self.btn.grid(row=0, column=0, **pad)
        self.var_status = tk.StringVar(value='대기')
        ttk.Label(act, textvariable=self.var_status).grid(
            row=0, column=1, sticky='w', **pad)
        row += 1

        # 로그
        self.log = scrolledtext.ScrolledText(frm, height=18, state='disabled',
                                             font=('Consolas', 9))
        self.log.grid(row=row, column=0, columnspan=4, sticky='nsew', **pad)
        frm.rowconfigure(row, weight=1)

    # 동작 --------------------------------------------------------------
    def _pick_file(self):
        path = filedialog.askopenfilename(
            title='측정 CSV 선택',
            filetypes=[('CSV', '*.csv'), ('모든 파일', '*.*')])
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
        """입력 검증. 통과하면 (경로, 서버, params), 아니면 None."""
        path = self.var_file.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showerror('입력 확인', '측정 CSV 파일을 선택하세요.')
            return None
        name = self.var_name.get().strip()
        age = self.var_age.get().strip()
        birth = self.var_birth.get().strip()
        mdate = self.var_mdate.get().strip()
        if not name:
            messagebox.showerror('입력 확인', '이름을 입력하세요.')
            return None
        if not age.isdigit():
            messagebox.showerror('입력 확인', '나이는 숫자로 입력하세요.')
            return None
        if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', birth):
            messagebox.showerror('입력 확인', '생년월일은 YYYY-MM-DD 형식입니다.')
            return None
        if not re.fullmatch(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}', mdate):
            messagebox.showerror('입력 확인',
                                 '측정일시는 "YYYY-MM-DD HH:MM" 형식입니다.')
            return None
        params = {
            'name': name, 'age': age, 'birth': birth,
            'sex': self.var_sex.get(), 'measurement_date': mdate,
        }
        if self.var_stim.get().strip():
            params['stimulus'] = self.var_stim.get().strip()
        if self.var_model.get() != 'neuronet':
            params['sleep_model'] = self.var_model.get()
        if not self.var_crop.get():
            params['crop'] = 'false'
        for key, var in self.var_scales.items():
            v = var.get().strip()
            if not v:
                continue
            if not v.isdigit():
                messagebox.showerror('입력 확인',
                                     '%s 점수는 숫자로 입력하세요.' % key)
                return None
            params[key] = v
        server = self.var_server.get().strip().rstrip('/')
        server = re.sub(r'^https?://', '', server)
        if not server:
            messagebox.showerror('입력 확인', '서버 주소를 입력하세요.')
            return None
        return path, server, params

    def _submit(self):
        checked = self._validate()
        if checked is None:
            return
        path, server, params = checked
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
                    self.var_status.set(msg)
                    self._append_log('\n' + msg + '\n')
                    self.btn.configure(state='normal')
                    (messagebox.showinfo if ok else messagebox.showerror)(
                        '원격 제출', msg)
        except queue.Empty:
            pass
        self.root.after(200, self._drain)


if __name__ == '__main__':
    root = tk.Tk()
    RemoteGui(root)
    root.mainloop()

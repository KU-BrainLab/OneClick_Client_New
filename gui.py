import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as _plt
import os
import sys
import json
import threading
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import queue

class QueueStream:
    def __init__(self, q):
        self.q = q

    def write(self, text):
        self.q.put(text)

    def flush(self):
        pass


def run_analysis(args_dict, log_queue):
    import numpy as np
    import requests
    from utils.eeg.analysis import main_analysis as eeg_analysis
    from utils.ecg.clean_up import CleanUpECG
    from utils.ecg.feature_extraction import ECGFeatureExtractor
    from main import NpEncoder, eeg_content_bulk, eeg_diff_content_bulk

    file      = args_dict['FILE_NAME']
    name      = args_dict['NAME']
    age       = args_dict['AGE']
    mdate     = args_dict['MEASUREMENT_DATE']
    birth     = args_dict['BIRTH']
    sex       = args_dict['SEX']
    debug     = args_dict['DEBUG_MODE']

    data_path = os.path.abspath('data')
    save_path = os.path.abspath(os.path.join('data', 'clean'))

    mode = 'DEBUG' if debug else 'LIVE'
    log_queue.put(f'[{mode} MODE] Start analyze: {file}\n')
    log_queue.put('=' * 55 + '\n')

    # ECG
    try:
        ecg = CleanUpECG(data_path=os.path.join(data_path, file))
        if ecg.isValid:
            ecg.save_filtered_data(save_path=save_path)
            ext = ECGFeatureExtractor(
                data_path=os.path.join(save_path, file),
                save_path=save_path,
                sfreq=125, age=age, sex=sex
            )
            hrv_results, trigger = ext.extract()
            hrv_payload = json.dumps(hrv_results, cls=NpEncoder)
            del ext, hrv_results
        else:
            log_queue.put('[ECG] 신호 이상 — ECG 분석 건너뜀\n')
            hrv_payload = json.dumps('', cls=NpEncoder)
            trigger = []
    except Exception as e:
        log_queue.put(f'[ECG 오류] {e}\n')
        hrv_payload = json.dumps('', cls=NpEncoder)
        trigger = []

    # EEG
    try:
        eeg_results = eeg_analysis(os.path.join(data_path, file), trigger)
    except Exception as e:
        log_queue.put(f'[EEG 오류] {e}\n')
        log_queue.put('FAILED\n')
        return

    eeg_payload = json.dumps({
        'psd': eeg_results['psd_result'],
        'sleep_staging': eeg_results['sleep_stage'],
        'frontal_limbic': eeg_results['frontal_limbic'],
        'baseline':      eeg_content_bulk(eeg_results, 'baseline'),
        'stimulation1':  eeg_content_bulk(eeg_results, 'stimulation1'),
        'recovery1':     eeg_content_bulk(eeg_results, 'recovery1'),
        'stimulation2':  eeg_content_bulk(eeg_results, 'stimulation2'),
        'recovery2':     eeg_content_bulk(eeg_results, 'recovery2'),
        'diff1': eeg_diff_content_bulk(eeg_results, 'diff1'),
        'diff2': eeg_diff_content_bulk(eeg_results, 'diff2'),
        'diff3': eeg_diff_content_bulk(eeg_results, 'diff3'),
        'diff4': eeg_diff_content_bulk(eeg_results, 'diff4'),
        'faa': eeg_results['faa'],
        'psd_spectrogram': eeg_results['psd_spectrogram'],
    }, cls=NpEncoder)

    report = eeg_results['sleep_report']
    report_payload = json.dumps({
        'tib':  report['tib'],  'tst':  report['tst'],
        'twt':  report['twt'],  'waso': report['waso'],
        'sleep_latency': report['sleep_latency'],
        'rem_latency':   report['rem_latency'],
        'sleep_eff':     report['sleep_eff'],
        'sleep_n1_tst':   report['sleep_tst'][0],
        'sleep_n2_tst':   report['sleep_tst'][1],
        'sleep_n3_tst':   report['sleep_tst'][2],
        'sleep_nrem_tst': report['sleep_tst'][3],
        'sleep_rem_tst':  report['sleep_tst'][4],
        'sleep_n1_min':   report['sleep_min'][0],
        'sleep_n2_min':   report['sleep_min'][1],
        'sleep_n3_min':   report['sleep_min'][2],
        'sleep_nrem_min': report['sleep_min'][3],
        'sleep_rem_min':  report['sleep_min'][4],
    }, cls=NpEncoder)

    if not debug:
        ip = '180.83.245.145:8000'
        s_index = ['male', 'female']
        try:
            resp = requests.post(
                f'http://{ip}/api/v1/exp/',
                data=json.dumps({
                    'name': name, 'measurement_date': mdate,
                    'age': age,   'birth': birth,
                    'sex': s_index.index(sex),
                    'hrv': hrv_payload, 'eeg': eeg_payload,
                    'report': report_payload, 'trigger': trigger,
                }),
                headers={'Content-type': 'application/json', 'Accept': '*/*'},
            )
            log_queue.put(f'[Server Response] {resp.status_code} {resp.text[:200]}\n')
        except Exception as e:
            log_queue.put(f'[Server Error] {e}\n')
    else:
        log_queue.put('[DEBUG] Server Passes\n')

    log_queue.put('DONE\n')


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('OneClick Analysis')
        self.resizable(False, False)
        self._log_queue = queue.Queue()
        self._build_ui()
        self._poll_log()

    # ── UI 구성 ──────────────────────────────────────────
    def _build_ui(self):
        pad = {'padx': 10, 'pady': 5}

        # 헤더
        header = tk.Frame(self, bg='#1a1a2e')
        header.grid(row=0, column=0, columnspan=2, sticky='ew')
        tk.Label(
            header, text='OneClick Analysis', font=('Helvetica', 16, 'bold'),
            bg='#1a1a2e', fg='white', pady=10
        ).pack()

        # 파라미터 폼
        form = tk.LabelFrame(self, text=' Subject Information ', font=('Helvetica', 10, 'bold'),
                             padx=10, pady=10)
        form.grid(row=1, column=0, columnspan=2, padx=15, pady=(10, 5), sticky='ew')

        fields = [
            ('이름 (NAME)',             'name',  'entry', None),
            ('나이 (AGE)',              'age',   'entry', None),
            ('측정일 (MEASUREMENT_DATE)','mdate', 'entry', None),
            ('생년월일 (BIRTH)',         'birth', 'entry', None),
            ('성별 (SEX)',              'sex',   'combo', ['male', 'female']),
            ('파일명 (FILE_NAME)',       'fname', 'file',  None),
        ]

        self._vars = {}
        for row_i, (label, key, wtype, opts) in enumerate(fields):
            tk.Label(form, text=label, anchor='w', width=28).grid(
                row=row_i, column=0, sticky='w', pady=3)

            if wtype == 'entry':
                var = tk.StringVar()
                tk.Entry(form, textvariable=var, width=30).grid(
                    row=row_i, column=1, sticky='ew', pady=3)
            elif wtype == 'combo':
                var = tk.StringVar(value=opts[0])
                ttk.Combobox(form, textvariable=var, values=opts,
                             state='readonly', width=28).grid(
                    row=row_i, column=1, sticky='ew', pady=3)
            elif wtype == 'file':
                var = tk.StringVar()
                file_frame = tk.Frame(form)
                file_frame.grid(row=row_i, column=1, sticky='ew', pady=3)
                tk.Entry(file_frame, textvariable=var, width=22).pack(side='left')
                tk.Button(file_frame, text='찾기', command=self._browse_file,
                          width=5).pack(side='left', padx=(4, 0))

            self._vars[key] = var

        # 기본값 채우기
        self._vars['name'].set('TEST')
        self._vars['age'].set('60')
        self._vars['mdate'].set('2026-03-10-1509')
        self._vars['birth'].set('1965-06-10')
        self._vars['fname'].set('2026-03-10-1509.csv')

        # DEBUG MODE 체크박스
        self._debug_var = tk.BooleanVar(value=False)
        debug_frame = tk.Frame(self)
        debug_frame.grid(row=2, column=0, columnspan=2, pady=(0, 5))
        tk.Checkbutton(
            debug_frame, text='DEBUG MODE',
            variable=self._debug_var, font=('Helvetica', 9)
        ).pack()

        # 실행 버튼
        self._run_btn = tk.Button(
            self, text='▶  Start analyze', font=('Helvetica', 12, 'bold'),
            bg='#16213e', fg='white', activebackground='#0f3460',
            relief='flat', padx=20, pady=8,
            command=self._on_run
        )
        self._run_btn.grid(row=3, column=0, columnspan=2, pady=(0, 8))

        # 로그 출력창
        log_frame = tk.LabelFrame(self, text='Log', font=('Helvetica', 10, 'bold'),
                                  padx=5, pady=5)
        log_frame.grid(row=4, column=0, columnspan=2, padx=15, pady=(0, 15), sticky='nsew')

        self._log = scrolledtext.ScrolledText(
            log_frame, width=70, height=18,
            font=('Consolas', 9), bg='#0d0d0d', fg='#00ff88',
            state='disabled', wrap='word'
        )
        self._log.pack(fill='both', expand=True)

        # 우측 하단 크레딧
        tk.Label(
            self, text='Created by Youngseok Kim  |  KU BrainLab',
            font=('Helvetica', 8), fg='#888888', anchor='e'
        ).grid(row=5, column=0, columnspan=2, padx=15, pady=(0, 8), sticky='e')

        self._file_var = self._vars['fname']

    def _browse_file(self):
        path = filedialog.askopenfilename(
            initialdir=os.path.abspath('data'),
            title='데이터 파일 선택',
            filetypes=[('CSV files', '*.csv'), ('All files', '*.*')]
        )
        if path:
            self._file_var.set(os.path.basename(path))

    # ── 실행 ─────────────────────────────────────────────
    def _on_run(self):
        # 입력값 검증
        age_str = self._vars['age'].get().strip()
        if not age_str.isdigit():
            self._append_log('[오류] 나이는 숫자로 입력하세요.\n')
            return
        fname = self._vars['fname'].get().strip()
        if not fname:
            self._append_log('[오류] 파일명을 입력하세요.\n')
            return

        args_dict = {
            'NAME':             self._vars['name'].get().strip(),
            'AGE':              int(age_str),
            'MEASUREMENT_DATE': self._vars['mdate'].get().strip(),
            'BIRTH':            self._vars['birth'].get().strip(),
            'SEX':              self._vars['sex'].get(),
            'FILE_NAME':        fname,
            'DEBUG_MODE':       self._debug_var.get(),
        }

        self._run_btn.config(state='disabled', text='running...')
        self._log.config(state='normal')
        self._log.delete('1.0', 'end')
        self._log.config(state='disabled')

        # stdout 캡처 → 로그창으로
        sys.stdout = QueueStream(self._log_queue)

        thread = threading.Thread(
            target=self._worker, args=(args_dict,), daemon=True
        )
        thread.start()

    def _worker(self, args_dict):
        try:
            run_analysis(args_dict, self._log_queue)
        except Exception as e:
            self._log_queue.put(f'[예외] {e}\n')
            self._log_queue.put('FAILED\n')

    # ── 로그 폴링 ─────────────────────────────────────────
    def _poll_log(self):
        try:
            while True:
                msg = self._log_queue.get_nowait()
                if msg in ('DONE\n', 'FAILED\n'):
                    sys.stdout = sys.__stdout__
                    status = '완료' if msg == 'DONE\n' else '오류 발생'
                    self._append_log(f'\n{"=" * 55}\n분석 {status}.\n')
                    self._run_btn.config(state='normal', text='▶  분석 시작')
                else:
                    self._append_log(msg)
        except queue.Empty:
            pass
        self.after(100, self._poll_log)

    def _append_log(self, text):
        self._log.config(state='normal')
        self._log.insert('end', text)
        self._log.see('end')
        self._log.config(state='disabled')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    app = App()
    app.mainloop()

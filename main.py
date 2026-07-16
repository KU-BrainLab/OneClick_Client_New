# -*- coding:utf-8 -*-
import matplotlib
matplotlib.use('Agg')   # GUI 없는 파일 저장 전용 백엔드 — joblib 스레드 충돌 방지
import os
import json
import argparse
import numpy as np
import requests
from utils.eeg.analysis import main_analysis as eeg_analysis
from utils.eeg.eeg_analysis.auto_crop import auto_crop_csv
from utils.eeg.eeg_analysis.crop import FS, count_rows
from utils.ecg.clean_up import CleanUpECG
from utils.ecg.feature_extraction import ECGFeatureExtractor
import torch
import pickle

def get_args():
    ### Subject Informations ###
    parser = argparse.ArgumentParser()
    parser.add_argument('--NAME', default='김조셉', type=str)
    parser.add_argument('--AGE', default= 38, type=int)
    parser.add_argument('--MEASUREMENT_DATE', default='2026-07-09 10:59', type=str)
    parser.add_argument('--BIRTH', default='1988-05-05', type=str)
    parser.add_argument('--SEX', default='male', choices=['male', 'female'], type=str)
    parser.add_argument('--FILE_NAME', default='2026-07-09-1059.csv', type=str)
    parser.add_argument('--STIMULUS', default='General Sleep protocol', type=lambda s: s.replace('\\n', '\n'))

    ### DEBUG_MODE ###
    ### False일때만 서버로 전송됨 ###
    ### 확인 필수로 해주세요 ###
    parser.add_argument('--DEBUG_MODE', default=False, type=bool)
    return parser.parse_args()


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        dtypes = (np.datetime64, np.complexfloating)
        if isinstance(obj, dtypes):
            return str(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            if any([np.issubdtype(obj.dtype, i) for i in dtypes]):
                return obj.astype(str).tolist()
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    

def eeg_content_bulk(payload, name):
    return {
        'topography_delta': payload['topography'][name]['delta'],
        'topography_theta': payload['topography'][name]['theta'],
        'topography_alpha': payload['topography'][name]['alpha'],
        'topography_beta': payload['topography'][name]['beta'],
        'topography_gamma': payload['topography'][name]['gamma'],
        'topography_sigma': payload['topography'][name]['sigma'],
        'connectivity_delta': payload['connectivity'][name]['delta'],
        'connectivity_theta': payload['connectivity'][name]['theta'],
        'connectivity_alpha': payload['connectivity'][name]['alpha'],
        'connectivity_beta': payload['connectivity'][name]['beta'],
        'connectivity_gamma': payload['connectivity'][name]['gamma'],
        'connectivity_sigma': payload['connectivity'][name]['sigma'],
        'connectivity2_delta': payload['connectivity2'][name]['delta'],
        'connectivity2_theta': payload['connectivity2'][name]['theta'],
        'connectivity2_alpha': payload['connectivity2'][name]['alpha'],
        'connectivity2_beta': payload['connectivity2'][name]['beta'],
        'connectivity2_gamma': payload['connectivity2'][name]['gamma'],
        'connectivity2_sigma': payload['connectivity2'][name]['sigma']
    }


def eeg_diff_content_bulk(payload, name):
    return payload[name]


TEMP_FILE_NAME = 'temp.csv'   # EEG 전용 노이즈 크롭 결과


def finalize_trigger(trigger, n_rows):
    """main_analysis 가 trigger 에 하던 정규화를, 원본 타임라인에 그대로 적용한다.

    EEG 는 크롭된 temp.csv 로 돌기 때문에 main_analysis 는 크롭 기준 끝 분을
    append 한다. 하지만 서버/HRV 로 가는 trigger 의 마지막 원소는 "총 녹화 길이(분)"
    로 쓰이고(프론트 finalMaxX = intervals.last), HRV 는 원본 ECG 기준이므로
    반드시 원본 길이여야 한다. 크롭 길이를 넣으면 RMSSD 그래프가 잘린다.

    int(n_rows / 125 / 60) 은 main_analysis 의
    int(n_epochs * 30 / 60) (100 Hz 리샘플 후 30 초 에폭) 과 항상 일치한다.
    """
    trigger.append(int(n_rows / FS / 60))
    if len(trigger) == 1:   # ECG 실패 등으로 phase 마커가 없으면 1-phase
        trigger.insert(0, 0)
    return trigger


def check_not_temp_file(file):
    """크롭 결과를 data/temp.csv 에 쓰므로, 입력이 temp.csv 면 원본을 덮어쓰게 된다."""
    if os.path.basename(file).lower() == TEMP_FILE_NAME:
        raise ValueError(f"FILE_NAME 이 '{TEMP_FILE_NAME}' 일 수 없습니다 "
                         f"(크롭 임시파일과 충돌해 원본을 덮어씁니다). 파일명을 바꿔주세요.")


def analyze_eeg_with_crop(data_path, file, trigger, log=print):
    """ECG 이후 단계: 노이즈 크롭 -> temp.csv -> EEG 분석 -> 서버용 trigger 정규화.

    main.py 와 gui.py 가 공유한다. 배선이 한쪽에만 적용되는 일을 막기 위해
    두 진입점 모두 반드시 이 함수를 통해야 한다.

    `trigger` 는 원본 타임라인 기준으로 in-place 정규화된다(서버/HRV 용).
    반환값은 eeg_results.
    """
    src_path = os.path.join(data_path, file)
    temp_path = os.path.join(data_path, TEMP_FILE_NAME)

    raw_trigger = list(trigger)   # finalize_trigger 가 trigger 를 mutate 하기 전 사본
    eeg_path, eeg_trigger = src_path, list(trigger)
    try:
        spans, cropped_trigger = auto_crop_csv(src_path, temp_path, amp_threshold=200.0)
        eeg_path = temp_path

        if not trigger:
            # ECG 실패로 phase 마커가 없는 경우. main_analysis 의 1-phase fallback 을
            # 그대로 타도록 빈 리스트를 유지한다.
            eeg_trigger = []
        elif len(cropped_trigger) != len(trigger):
            log(f"[CROP] 경고: 트리거 개수가 다릅니다 "
                f"(원본 {len(trigger)}개 {trigger} / 크롭 {len(cropped_trigger)}개 "
                f"{cropped_trigger}). 트리거 보호가 정상이면 같아야 합니다. "
                f"원본 트리거로 폴백합니다.")
            eeg_path, eeg_trigger = src_path, list(trigger)
        else:
            eeg_trigger = list(cropped_trigger)
    except Exception as e:
        log(f"[CROP] 크롭 실패, 원본 csv 로 EEG 진행: {e}")
        eeg_path, eeg_trigger = src_path, list(trigger)

    # 서버로 가는 trigger 는 원본 타임라인을 유지해야 한다. main_analysis 는
    # eeg_trigger 만 mutate 하므로 여기서 같은 정규화를 원본 기준으로 직접 해준다.
    finalize_trigger(trigger, count_rows(src_path))

    try:
        return eeg_analysis(eeg_path, eeg_trigger)
    except Exception as e:
        if eeg_path == src_path:
            raise   # 크롭 탓이 아니다. 그대로 올려보낸다.
        # 크롭이 EEG 를 죽이는 경로를 원천 차단한다: 크롭본이 실패하면 원본으로 1회 재시도.
        log(f"[CROP] 크롭된 EEG 분석 실패, 원본 csv 로 재시도: {e}")
        return eeg_analysis(src_path, list(raw_trigger))


if __name__ == '__main__':
    args = get_args()

    file = args.FILE_NAME

    try:
        check_not_temp_file(file)
    except ValueError as e:
        raise SystemExit(f"[CROP] {e}")

    os.makedirs(os.path.join('image', 'pac'), exist_ok=True)
    os.makedirs(os.path.join('image', 'psd'), exist_ok=True)
    os.makedirs(os.path.join('image', 'psd_diff'), exist_ok=True)

    #get dir path
    data_path = os.path.abspath('data')
    save_path = os.path.join('data', 'clean')
    save_path = os.path.abspath(save_path)
    # data_path = r"C:\Users\tjd64\OneDrive\바탕 화면\Oneclick\data"
    # save_path = r"C:\Users\tjd64\OneDrive\바탕 화면\Oneclick\data\clean"

    if args.DEBUG_MODE:
        print("###########################################################")
        print("###########################################################")
        print("###########################################################")
        print("#######################  DEBUG MODE  ######################")
        print("#####Last Update : 2026.03.15 created by Youngseok Kim#####")
        print("###########################################################")
        print("###########################################################")
        print("###########################################################")

    else:
        print("###########################################################")
        print("###########################################################")
        print("###########################################################")
        print("#######################  LIVE MODE  #######################")
        print("#####Last Update : 2026.03.15 created by Youngseok Kim#####")
        print("###########################################################")
        print("###########################################################")
        print("###########################################################")

    # ECG    
    # 신호 이상시

    ecg = CleanUpECG(data_path=os.path.join(data_path, file))

    trigger = []
    if(ecg.isValid):
        try:
            cleaned_data = ecg.save_filtered_data(save_path=save_path)
            ext = ECGFeatureExtractor(data_path=os.path.join(save_path, file), save_path=save_path,
                                        sfreq=125, age=args.AGE, sex=args.SEX)
            hrv_results, trigger = ext.extract()
            hrv_payload = json.dumps(hrv_results, cls=NpEncoder)
            del cleaned_data, ext, hrv_results
        except Exception as e:
            print(f"[ECG] extract() 에러 발생, HRV 스킵: {e}")
            hrv_payload = json.dumps("", cls=NpEncoder)
    else:
        hrv_payload = json.dumps("", cls=NpEncoder)

    #trigger = [0]

    # EEG 전용 노이즈 크롭 — ECG/HRV 는 위에서 원본 csv 로 이미 처리했다.
    ## 신호 이상시
    eeg_results = analyze_eeg_with_crop(data_path, file, trigger)
    eeg_payload = json.dumps({
        'psd': eeg_results['psd_result'],
        'sleep_staging': eeg_results['sleep_stage'],
        'frontal_limbic': eeg_results['frontal_limbic'],
        'baseline': eeg_content_bulk(eeg_results, 'baseline'),
        'stimulation1': eeg_content_bulk(eeg_results, 'stimulation1'),
        'recovery1': eeg_content_bulk(eeg_results, 'recovery1'),
        'stimulation2': eeg_content_bulk(eeg_results, 'stimulation2'),
        'recovery2': eeg_content_bulk(eeg_results, 'recovery2'),
        'diff1': eeg_diff_content_bulk(eeg_results, 'diff1'),
        'diff2': eeg_diff_content_bulk(eeg_results, 'diff2'),
        'diff3': eeg_diff_content_bulk(eeg_results, 'diff3'),
        'diff4': eeg_diff_content_bulk(eeg_results, 'diff4'),
        'faa' : eeg_results['faa'],
        'psd_spectrogram': eeg_results['psd_spectrogram'],
        'spindle_coupling': eeg_results['spindle_coupling']
    }, cls=NpEncoder)

    report_payload = json.dumps({
        'tib' : eeg_results['sleep_report']['tib'],
        'tst' : eeg_results['sleep_report']['tst'],
        'twt' : eeg_results['sleep_report']['twt'],
        'waso' : eeg_results['sleep_report']['waso'],
        'sleep_latency' : eeg_results['sleep_report']['sleep_latency'],
        'rem_latency' : eeg_results['sleep_report']['rem_latency'],
        'sleep_eff' : eeg_results['sleep_report']['sleep_eff'],

        'sleep_n1_tst': eeg_results['sleep_report']['sleep_tst'][0],
        'sleep_n2_tst': eeg_results['sleep_report']['sleep_tst'][1],
        'sleep_n3_tst': eeg_results['sleep_report']['sleep_tst'][2],
        'sleep_nrem_tst': eeg_results['sleep_report']['sleep_tst'][3],
        'sleep_rem_tst': eeg_results['sleep_report']['sleep_tst'][4],

        'sleep_n1_min': eeg_results['sleep_report']['sleep_min'][0],
        'sleep_n2_min': eeg_results['sleep_report']['sleep_min'][1],
        'sleep_n3_min': eeg_results['sleep_report']['sleep_min'][2],
        'sleep_nrem_min': eeg_results['sleep_report']['sleep_min'][3],
        'sleep_rem_min': eeg_results['sleep_report']['sleep_min'][4]
    }, cls=NpEncoder)


    headers = {'Content-type': 'application/json', 'Accept': '*/*'}
    ip = '180.83.245.145:8000'
    s_index = ['male', 'female']


    empty = {}
    ttttt = json.dumps(empty)

    #if you want send data without specific datas: use "\"\"",
    if not args.DEBUG_MODE:
        oo = requests.post('http://{}/api/v1/exp/'.format(ip),
                        data=json.dumps({'name': args.NAME,
                                            'measurement_date': args.MEASUREMENT_DATE,
                                            'age': args.AGE,
                                            'birth': args.BIRTH,
                                            'sex': s_index.index(args.SEX),
                                            'hrv': hrv_payload,
                                            'eeg': eeg_payload,
                                            'report': report_payload,
                                            'trigger': trigger,
                                            'stimulus_info': args.STIMULUS}),
                        headers=headers)
        print(oo)
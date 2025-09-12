# -*- coding:utf-8 -*-
import os
import json
import argparse
import numpy as np
import requests
from utils.eeg.analysis import main_analysis as eeg_analysis
from utils.ecg.clean_up import CleanUpECG
from utils.ecg.feature_extraction import ECGFeatureExtractor
import torch
import pickle

def get_args():
    ### Subject Informations ###
    parser = argparse.ArgumentParser()
    parser.add_argument('--NAME', default='테스트', type=str)
    parser.add_argument('--AGE', default=34, type=int)
    parser.add_argument('--MEASUREMENT_DATE', default='2025-09-09 13:19', type=str)
    parser.add_argument('--BIRTH', default='2004-01-17', type=str)
    parser.add_argument('--SEX', default='female', choices=['male', 'female'], type=str)
    parser.add_argument('--FILE_NAME', default='2025-09-09-1221.csv', type=str)

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
        'connectivity_delta': payload['connectivity'][name]['delta'],
        'connectivity_theta': payload['connectivity'][name]['theta'],
        'connectivity_alpha': payload['connectivity'][name]['alpha'],
        'connectivity_beta': payload['connectivity'][name]['beta'],
        'connectivity_gamma': payload['connectivity'][name]['gamma'],
        'connectivity2_delta': payload['connectivity2'][name]['delta'],
        'connectivity2_theta': payload['connectivity2'][name]['theta'],
        'connectivity2_alpha': payload['connectivity2'][name]['alpha'],
        'connectivity2_beta': payload['connectivity2'][name]['beta'],
        'connectivity2_gamma': payload['connectivity2'][name]['gamma'],
    }


def eeg_diff_content_bulk(payload, name):
    return {
        'topography_delta': payload['topography_diff'][name]['delta'],
        'topography_theta': payload['topography_diff'][name]['theta'],
        'topography_alpha': payload['topography_diff'][name]['alpha'],
        'topography_beta': payload['topography_diff'][name]['beta'],
        'topography_gamma': payload['topography_diff'][name]['gamma'],
        'connectivity_delta': payload['connectivity_diff'][name]['delta'],
        'connectivity_theta': payload['connectivity_diff'][name]['theta'],
        'connectivity_alpha': payload['connectivity_diff'][name]['alpha'],
        'connectivity_beta': payload['connectivity_diff'][name]['beta'],
        'connectivity_gamma': payload['connectivity_diff'][name]['gamma'],
        'connectivity2_delta': payload['connectivity2_diff'][name]['delta'],
        'connectivity2_theta': payload['connectivity2_diff'][name]['theta'],
        'connectivity2_alpha': payload['connectivity2_diff'][name]['alpha'],
        'connectivity2_beta': payload['connectivity2_diff'][name]['beta'],
        'connectivity2_gamma': payload['connectivity2_diff'][name]['gamma'],
    }


if __name__ == '__main__':
    args = get_args()

    file = args.FILE_NAME

    #get dir path
    data_path = os.path.abspath('data')
    save_path = os.path.join('data', 'clean')
    save_path = os.path.abspath(save_path)
    # data_path = r"C:\Users\tjd64\OneDrive\바탕 화면\Oneclick\data"
    # save_path = r"C:\Users\tjd64\OneDrive\바탕 화면\Oneclick\data\clean"

    if args.DEBUG_MODE:
        print("#####################################################")
        print("####################  DEBUG MODE  ###################")
        print("#####################################################")
    else:
        print("#####################################################")
        print("####################  LIVE MODE  ####################")
        print("#####################################################")

    # ECG    
    # 신호 이상시
    ecg = CleanUpECG(data_path=os.path.join(data_path, file))

    if(ecg.isValid):
        cleaned_data = ecg.save_filtered_data(save_path=save_path)        
        ext = ECGFeatureExtractor(data_path=os.path.join(save_path, file), save_path=save_path,
                                    sfreq=125, age=args.AGE, sex=args.SEX)
        hrv_results = ext.extract()
        hrv_payload = json.dumps(hrv_results, cls=NpEncoder)

        del cleaned_data, ext, hrv_results
    else:
        hrv_payload = json.dumps("", cls=NpEncoder)

    ## 신호 이상시
    eeg_results = eeg_analysis(os.path.join(data_path, file))

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
        'faa' : eeg_results['faa']
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

    if not args.DEBUG_MODE:
        oo = requests.post('http://{}/api/v1/exp/'.format(ip),
                        data=json.dumps({'name': args.NAME,
                                            'measurement_date': args.MEASUREMENT_DATE,
                                            'age': args.AGE,
                                            'birth': args.BIRTH,
                                            'sex': s_index.index(args.SEX),
                                            'hrv': hrv_payload,
                                            'eeg': eeg_payload,
                                            'report': report_payload}),
                        headers=headers)
        print(oo)
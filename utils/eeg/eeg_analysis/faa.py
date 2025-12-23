from scipy.integrate import simpson
from scipy.signal import welch
import mne
import copy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import base64

def get_frontal_alpha_asymmetry(epoch_data, uuid):
    psd_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    files_dict = {name: None for name in psd_names}

    def get_psd_analysis_func(data_, sfreq, low, high):
        frequencies, psd = welch(data_, sfreq, axis=1)

        idx_band = np.logical_and(frequencies >= low, frequencies < high)
        bp = simpson(psd[:, idx_band], dx=frequencies[1] - frequencies[0])

        if True:
            sum_ = 0
            for level, (l, h) in enumerate([(0.5, 4), (4, 8), (8, 13), (13, 30)]):
                idx = np.logical_and(frequencies >= l, frequencies < h)
                if level == 0:
                    sum_ = simpson(psd[:, idx], dx=frequencies[1] - frequencies[0])
                else:
                    sum_ += simpson(psd[:, idx], dx=frequencies[1] - frequencies[0])
            bp /= sum_
        return bp

    def get_faa_func(data_, ch_names, sfreq):
        alpha_band = (8, 13)
        l_channels, r_channels = ['Fp1'], ['Fp2']
        l_indices, r_indices = [ch_names.index(ch) for ch in l_channels], [ch_names.index(ch) for ch in r_channels]

        data_ = data_.get_data()
        data_ = copy.deepcopy(data_)
        segment_l, segment_r = data_[:, l_indices, :], data_[:, r_indices, :]
        total_l_alpha, total_r_alpha = [], []
        for sl, sr in zip(segment_l, segment_r):
            l_alpha = get_psd_analysis_func(sl, sfreq, low=8, high=13)
            r_alpha = get_psd_analysis_func(sr, sfreq, low=8, high=13)
            total_l_alpha.append(l_alpha)
            total_r_alpha.append(r_alpha)

        total_l_alpha, total_r_alpha = np.array(total_l_alpha), np.array(total_r_alpha)
        total_l_alpha, total_r_alpha = np.mean(np.log(total_l_alpha + 1e-12)), np.mean(np.log(total_r_alpha + 1e-12))
        faa = total_r_alpha - total_l_alpha
        return faa

    def plot_faa_on_brain_fuc(faa_value, title):
        # 이미지 불러오기
        brain_img_path = './brain.png'
        img = mpimg.imread(brain_img_path)
        height, width, _ = img.shape

        # 좌/우 전두엽 위치 (이미지 좌표 기준으로 수동 지정)
        left_coord = (int(width * 0.35), int(height * 0.25))   # Fp1~F3 근처
        right_coord = (int(width * 0.62), int(height * 0.25))  # Fp2~F4 근처

        # 컬러 매핑 설정
        cmap = plt.get_cmap('bwr')
        norm = plt.Normalize(-0.5, 0.5)

        left_color = cmap(norm(-faa_value))  # FAA < 0: 좌측 우세
        right_color = cmap(norm(faa_value))  # FAA > 0: 우측 우세

        # 시각화
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img)
        ax.set_title(f"{title}: {faa_value:.2f}", fontsize=15)

        # 좌/우 영역 색상 표시
        ax.scatter(*left_coord, color=left_color, s=4500, label='Left frontal', alpha=0.45)
        ax.scatter(*right_coord, color=right_color, s=4500, label='Right frontal', alpha=0.45)

        # 색상바
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04)
        cbar.set_label('FAA (Right - Left)', fontsize=12)
        ax.axis('off')

        tmp_name = os.path.join('image', 'frontal_alpha_asymmetry', '{}_{}.jpg'.format(uuid, title))
        fig.savefig(tmp_name)
        with open(tmp_name, 'rb') as f:
            im_bytes = f.read()
        im_b64 = base64.b64encode(im_bytes).decode("utf8")
        files_dict[title] = im_b64
        plt.close('all')
        plt.clf()
        
    files = []
    eeg_info = epoch_data.info
    epoch_data = epoch_data.get_data()
    sample_size = epoch_data.shape[0]
    step = sample_size // 5
    for i in range(5):
        start, end = i * step, (i+1) * step
        sample = epoch_data[start: end, ...]
        raw = mne.EpochsArray(sample, info=eeg_info)
        faa = get_faa_func(raw, ch_names=eeg_info['ch_names'], sfreq=eeg_info['sfreq'])
        plot_faa_on_brain_fuc(faa, psd_names[i])    
    return files_dict
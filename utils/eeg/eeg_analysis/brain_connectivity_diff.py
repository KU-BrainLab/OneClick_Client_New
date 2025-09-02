import os
import matplotlib.pyplot as plt
import cv2
import copy
import base64
import mne
from mne_connectivity import spectral_connectivity_epochs
from mne_connectivity.viz import plot_connectivity_circle


def center_crop(img, dim):
    width, height = img.shape[1], img.shape[0]

    # process crop width and height for max available dimension
    crop_width = dim[0] if dim[0] < img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0]
    mid_x, mid_y = int(width / 2), int(height / 2)
    cw2, ch2 = int(crop_width / 2), int(crop_height / 2)
    crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    return crop_img


def get_diff_brain_connectivity(epoch_data, uuid, type):

    def connecitivy(sample, band_range, eeg_info, type):
        fc_method = type

        raw = mne.EpochsArray(sample, info=eeg_info)
        con = spectral_connectivity_epochs(data=raw, names=raw.info['ch_names'], sfreq=sfreq,
                                           mode="multitaper",
                                           method=fc_method, fmin=band_range[0], fmax=band_range[1],
                                           faverage=True,
                                           mt_adaptive=True, n_jobs=1)
        conmat = con.get_data(output="dense")[:, :, 0]
        return conmat

    if(type == 'coh'):
        dir = 'connectivity_diff'
    elif(type == 'plv'):
        dir = 'connectivity_diff2'
        
    epoch_data = copy.deepcopy(epoch_data)
    eeg_info = epoch_data.info
    sfreq = eeg_info['sfreq']
    epoch_data = epoch_data.get_data()
    sample_size = epoch_data.shape[0]
    step = sample_size // 5
    exp_names = ['diff1', 'diff2', 'diff3', 'diff4']
    bands = {'delta': [0, 4], 'theta': [4, 8], 'alpha': [8, 13], 'beta': [13, 30], 'gamma': [30, 40]}
    files = {exp_name: {band_name: None for band_name in bands.keys()} for exp_name in exp_names}

    epochs = []
    for i in range(5):
        start, end = i * step, (i+1) * step
        sample = epoch_data[start: end, ...]
        epoch = mne.EpochsArray(sample, info=eeg_info)
        epochs.append(epoch)

    for diff_i, (i_epoch, j_epoch) in enumerate(zip(epochs[:-1], epochs[1:])):
        for band_name, band_range in bands.items():
            conmat_1 = connecitivy(i_epoch, band_range, eeg_info, type)
            conmat_2 = connecitivy(j_epoch, band_range, eeg_info, type)
            conmat_d = conmat_2 - conmat_1
            tmp_name = os.path.join('image', dir, '{}_{}_{}.jpg'.format(uuid, exp_names[diff_i], band_name))
            tmp_name = os.path.abspath(tmp_name)

            fig, ax = plt.subplots(1, figsize=(5, 5), subplot_kw={'projection': 'polar'})
            plot_connectivity_circle(conmat_d, i_epoch.info['ch_names'], n_lines=50,
                                     facecolor='white', textcolor='black',
                                     colormap='gray_r',
                                     ax=ax,
                                     colorbar=False,
                                     show=False,
                                     node_linewidth=1)
            fig.savefig(tmp_name)
            image = cv2.imread(tmp_name)
            crop_img = center_crop(image, (370, 370))
            cv2.imwrite(tmp_name, crop_img)

            plt.close('all')
            plt.clf()

            with open(tmp_name, 'rb') as f:
                im_bytes = f.read()
            im_b64 = base64.b64encode(im_bytes).decode("utf8")
            files[exp_names[diff_i]][band_name] = im_b64
    return files

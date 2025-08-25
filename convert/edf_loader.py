
import os
import mne

if __name__ == '__main__':
    file_ = 'A_01_baseline.edf'
    data_path_ = r'D:\IN\Work\OneClink\data\육군'
    path_ = os.path.join(data_path_, file_)
    raw = mne.io.read_raw_edf(path_, preload=True)


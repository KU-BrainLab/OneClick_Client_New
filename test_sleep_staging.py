import sys, warnings
warnings.filterwarnings("ignore")
import mne, numpy as np

ch_list_actual = ["Fp1","F7","F3","T3","C3","Cz","P3","Fp2","F4","F8","C4","T4","P4"]
sfreq = 100
n_epochs = 10
n_times  = 3000

rng = np.random.default_rng(0)
data = rng.standard_normal((n_epochs, len(ch_list_actual), n_times)).astype("float32") * 1e-6

info = mne.create_info(ch_names=ch_list_actual, sfreq=sfreq, ch_types="eeg")
epoch_data = mne.EpochsArray(data, info=info)

sys.path.insert(0, "utils/eeg")
from eeg_analysis.sleep_staging import get_sleep_staging

result = get_sleep_staging(epoch_data, ch_list_actual)
print("sleep_stage:", result["sleep_stage"])
print("sleep_summary:", result["sleep_summary"])
print("TEST PASSED")

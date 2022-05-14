import wfdb
import numpy as np
import matplotlib.pylab as plt
import scipy as sp
import math
from scipy import signal
from sklearn.metrics import mean_squared_error
from wavelet_transform import show_patient_record
from wavelet_transform import choose_enlargement_of_signal
from wavelet_transform import add_indexes
from wavelet_transform import signal_to_noise_ratio


#denoise ecg signal using savitzky-golay filter for the best window, polynomial and mode params
def denoise_ecg_signal(ecg_signal_original, ecg_signal_denoised, index, window_len=27, polyorder=5, mode='interp'):
    filtered = signal.savgol_filter(ecg_signal_original, window_len, polyorder, mode=mode)
    RMSE = round(math.sqrt(mean_squared_error(ecg_signal_original, filtered)), 4)

    min_sample = 0
    max_sample = min_sample + len(ecg_signal)
    plt.figure(figsize=(10, 7), dpi=80)
    plt.plot(index[min_sample:max_sample], filtered[min_sample:max_sample-1], label=f"Denoised signal", color="red")
    plt.xlabel('time/sample')
    plt.ylabel('amplitude [mV]')
    plt.plot(index[min_sample:max_sample], ecg_signal_denoised[min_sample:max_sample-1], label=f"Original filtered signal")
    plt.title(f"Clean and denoised signal, RMSE = {RMSE}", fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    path_patient2 = r"C:\Users\micha\Documents\Programming\ML\ECG_denoising\ECG_signals\Person_02\rec_1"

    #show_patient_record(path_patient2, "Patient 2")
    ecg_signal = choose_enlargement_of_signal(path_patient2, 1000, channels=[0], should_plot_signal=False)
    ecg_signal_denoised = choose_enlargement_of_signal(path_patient2, 1000, channels=[1], should_plot_signal=False)
    ecg_signal = ecg_signal.reshape(1000,)
    data, index = add_indexes(ecg_signal)
    denoise_ecg_signal(ecg_signal, ecg_signal_denoised, index)
    
    #TODO: find best walue of window in savgol_filter, best order of polynomials, mode
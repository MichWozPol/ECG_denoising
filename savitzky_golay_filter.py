from re import S
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
def denoise_ecg_signal(ecg_signal_original, ecg_signal_denoised, index, window_len=27, polyorder=5, mode='interp', should_plot_signal=True):
    filtered = signal.savgol_filter(ecg_signal_original, window_len, polyorder, mode=mode)
    min_sample = 0
    max_sample = min_sample + len(ecg_signal)
    RMSE = round(math.sqrt(mean_squared_error(ecg_signal_denoised, filtered)), 4)
    SNR = round(signal_to_noise_ratio(ecg_signal_denoised[0:max_sample], filtered[0:max_sample]), 4)
    if (should_plot_signal):
        plt.figure(figsize=(10, 7), dpi=80)
        plt.plot(index[min_sample:max_sample], filtered[min_sample:max_sample-1], label=f"Denoised signal", color="red")
        plt.xlabel('time/sample')
        plt.ylabel('amplitude [mV]')
        plt.plot(index[min_sample:max_sample], ecg_signal_denoised[min_sample:max_sample-1], label=f"Original filtered signal")
        plt.title(f"Clean and denoised signal, RMSE = {RMSE}, SNR={SNR}, window length={window_len},\n polynomial order={polyorder}" , fontsize=16)
        plt.legend()
        plt.tight_layout()
        plt.show()
    return RMSE, SNR


#find optimal params for particular signal
def find_optimal_polynomial_order_and_frame_size(ecg_signal_original, ecg_signal_denoised, index):
    modes = ['mirror', 'constant', 'nearest', 'wrap', 'interp'] # the type of extension to use for the padded signal
    RMSE_max = 1000
    SNR_min = -1000
    optimal_window = 0
    optimal_order = 0
    optimal_mode = ""
    optimal_RMSE, optimal_SNR = 0, 0
    for window_width in range(9, 99, 2):
        for order in range(1,9):
            for mode in modes:
                RMSE, SNR = denoise_ecg_signal(ecg_signal_original, ecg_signal_denoised, index, window_len=window_width, polyorder=order, 
                                               mode=mode, should_plot_signal=False)
                if((SNR > SNR_min) & (RMSE < RMSE_max)):
                    optimal_window = window_width
                    optimal_order = order
                    optimal_mode = mode
                    optimal_RMSE, RMSE_max = RMSE, RMSE
                    optimal_SNR, SNR_min = SNR, SNR
                                
    print("RMSE: " + str(optimal_RMSE) + " SNR: " + str(optimal_SNR) + " window: " + str(optimal_window) + " polynomial order: " + 
          str(optimal_order) + " mode " + str(optimal_mode))
    return optimal_window, optimal_order, optimal_mode

if __name__ == "__main__":
    path_patient2 = r"C:\Users\micha\Documents\Programming\ML\ECG_denoising\ECG_signals\Person_01\rec_1"
    show_patient_record(path_patient2, "Patient 12")
    ecg_signal = choose_enlargement_of_signal(path_patient2, 1000, channels=[0], should_plot_signal=False)
    ecg_signal_denoised = choose_enlargement_of_signal(path_patient2, 1000, channels=[1], should_plot_signal=False)
    ecg_signal = ecg_signal.reshape(1000,)
    data, index = add_indexes(ecg_signal)
    order, window, mode = find_optimal_polynomial_order_and_frame_size(ecg_signal, ecg_signal_denoised, index)
    denoise_ecg_signal(ecg_signal, ecg_signal_denoised, index, window_len=17, polyorder=2, mode='interp')
    
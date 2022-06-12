import numpy as np
import pandas as pd
import wfdb
import matplotlib.pyplot as plt
import pywt
import math
from sklearn.metrics import mean_squared_error


#show the particular person record
def show_patient_record(record_path, patient):
    record = wfdb.rdrecord(record_path)
    wfdb.plot_wfdb(record=record, title=patient)
    
    
#show the ROI of the noised ECG signal
def choose_enlargement_of_signal(path, nb_of_sampels=1000, channels=[0], should_plot_signal=True):
    ecg_signal, fields = wfdb.rdsamp(path, sampto=nb_of_sampels, channels=channels)
    if (should_plot_signal):
        wfdb.plot_items(signal=ecg_signal, fs=fields['fs'], title='ECG noised')    
    return ecg_signal
        
        
#add indexes to a wavelet
def add_indexes(ecg_signal):   
    index = []
    data = []
    for i in range (len(ecg_signal)-1):
        X = float(i)
        Y = float(ecg_signal[i])
        index.append(X)
        data.append(Y)
    return data, index


#denoise the ECG signal using a wavelet transform and choosen parameters
def denoise_ecg_signal(data, index, wavelet_type='sym11', sub_coeff_of_decomp=2, decision_type='soft', should_plot_signal=True):
    wavelet_type = wavelet_type
    w = pywt.Wavelet(wavelet_type)
    maxlev = pywt.dwt_max_level(len(data), w.dec_len) - sub_coeff_of_decomp # maximum useful level of decomposition
    coeffs = pywt.wavedec(data, wavelet_type, level=maxlev) # wavelet decomposition of the signal

    for i in range(1, len(coeffs)):
        M = len(coeffs[i])
        lambda_val = math.sqrt(2*math.log(M)) # Threshold for filtering, SureShrink method
        coeffs[i] = pywt.threshold(coeffs[i], lambda_val, decision_type) # Filter the noise using a soft decision
        datarec = pywt.waverec(coeffs, wavelet_type) # Wavelet reconstruction of the signal

    if (should_plot_signal):
        min_sample = 0
        max_sample = min_sample + len(data) +1

        plt.figure(figsize=(8, 6), dpi=80)
        plt.subplot(2, 1, 1)
        plt.plot(index[min_sample:max_sample], data[min_sample:max_sample])
        plt.legend(fontsize=30)
        plt.xlabel('time/sample')
        plt.ylabel('amplitude [mV]')
        plt.title("Raw signal", fontsize=18)
        plt.subplot(2, 1, 2)
        plt.plot(index[min_sample:max_sample], datarec[min_sample:max_sample-1])
        plt.xlabel('time/sample')
        plt.ylabel('amplitude [mV]')
        plt.title(f"De-noised signal, dec_lvl = {maxlev}, wavelet type = {wavelet_type}", fontsize=18)
        plt.tight_layout()
        plt.show()       
    return datarec, maxlev


#calculate the SNR according to an article equation
def signal_to_noise_ratio(original, denoised):
    orig_sum_square = sum(i*i for i in original)
    pred_mins_orig = sum((y-x)**2 for x,y in zip(denoised, original))
    return 10*math.log10(orig_sum_square/pred_mins_orig)


#calculate metrics of a choosen signal
def calculate_metrics(ecg_signal_original, ecg_signal_denoised, index, wavelet_type, dec_lvl, should_plot_signal=True):
    min_sample = 0
    max_sample_index = min_sample + min(len(ecg_signal_original), len(ecg_signal_denoised)) - 1
    RMSE = round(math.sqrt(mean_squared_error(ecg_signal_original[0: max_sample_index], ecg_signal_denoised[0:max_sample_index])), 4)
    SNR = round(signal_to_noise_ratio(ecg_signal_original[0:max_sample_index], ecg_signal_denoised[0:max_sample_index]), 4)
    if(should_plot_signal):
        plt.figure(figsize=(10, 7), dpi=80)
        plt.plot(index[min_sample:max_sample_index+1], ecg_signal_denoised[min_sample:max_sample_index], label=f"Denoised signal", color="red")
        plt.xlabel('time/sample')
        plt.ylabel('amplitude [mV]')
        plt.plot(index[min_sample:max_sample_index+1], ecg_signal_original[min_sample:max_sample_index], label=f"Original filtered signal")
        plt.title(f"Clean and denoised signal, RMSE = {RMSE}, SNR = {SNR}\n wavelet type = {wavelet_type}, level of decomposition = {dec_lvl}", fontsize=16)
        plt.legend()
        plt.tight_layout()
        plt.show()
    return RMSE, SNR, dec_lvl


def calculate_metrics_for_different_wavelets_and_decomposition_levels(filtered_signal, data, index, sub_dec_levels, wavelets, RMSE_list, SNR_list, wavelet_type_list, dec_lvl_list):
    for level in sub_dec_levels:
      for wavelet in wavelets:
        reconstructed, dec_lvl = denoise_ecg_signal(data, index, wavelet, level, should_plot_signal=False)
        RMSE, SNR, dec_lvl = calculate_metrics(filtered_signal, reconstructed, index, wavelet, dec_lvl, should_plot_signal=False)
        RMSE_list.append(RMSE)
        SNR_list.append(SNR)
        wavelet_type_list.append(wavelet)
        dec_lvl_list.append(dec_lvl)
    return RMSE_list, SNR_list, wavelet_type_list, dec_lvl_list


#denoise the whole ROI using a wavelet transform
def ecg_wavelet_denoising(path, wavelet='sym13', sub_coeff_of_decomp=3, enlargement=1000):
    ecg_signal = choose_enlargement_of_signal(path, nb_of_sampels=enlargement, channels=[0], should_plot_signal=False)
    ecg_filtered_signal = choose_enlargement_of_signal(path, nb_of_sampels=enlargement, channels=[1], should_plot_signal=False)
    data, index = add_indexes(ecg_signal)
    reconstructed, dec_lvl = denoise_ecg_signal(data, index, sub_coeff_of_decomp = sub_coeff_of_decomp, should_plot_signal=False)
    #below in the reconstrucred signal param specific value should be substracted in order to change a basic line level
    RMSE, SNR, dec_lvl =  calculate_metrics(ecg_filtered_signal, reconstructed, index, wavelet, dec_lvl, should_plot_signal=True)
    print(f"RMSE for given signal: {RMSE}, SNR for given signal: {SNR}")
    

#print which wavelet and decomposition level are the best for a particular record
def choose_best_wavelet_and_decomposition_level(path):
    sub_dec_levels = [1,2,3] # define list of decomposition levels
    wavelets = ['sym10', 'sym11', 'sym12', 'sym13', 'db10', 'db11', 'db12', 'db13'] #define list of desired wavelets
     #initialize list of RMSE, SNR, wavelet_type, dec_lvl, used to check which decomposition level and wavelet type are the best 
    RMSE_list, SNR_list, wavelet_type_list, dec_lvl_list = [], [], [], []
    ecg_signal = choose_enlargement_of_signal(path, nb_of_sampels=2000, channels=[0], should_plot_signal=False)
    ecg_filtered_signal = choose_enlargement_of_signal(path, nb_of_sampels=2000, channels=[1], should_plot_signal=False)
    data, index = add_indexes(ecg_signal)
    RMSE_list, SNR_list, wavelet_type_list, dec_lvl_list = calculate_metrics_for_different_wavelets_and_decomposition_levels(ecg_filtered_signal, data, 
                                                                index, sub_dec_levels, wavelets, RMSE_list, SNR_list, wavelet_type_list, dec_lvl_list)
    min_val_RMSE = min(RMSE_list, key=abs)
    max_val_SNR = max(SNR_list, key=abs)
    min_RMSE_index = RMSE_list.index(min_val_RMSE)
    max_SNR_index = SNR_list.index(max_val_SNR)
    
    print(f"Min value found for RMSE: {min_val_RMSE}, wavelet type: {wavelet_type_list[min_RMSE_index]}, decomposition_level: {dec_lvl_list[min_RMSE_index]}")
    print(f"Max value found for SNR: {max_val_SNR}, wavelet type: {wavelet_type_list[max_SNR_index]}, decomposition_level: {dec_lvl_list[max_SNR_index]}")


if __name__ == "__main__":
    path_patient1 = r"C:\Users\micha\Documents\Programming\ML\ECG_denoising\ECG_signals\Person_01\rec_1"
    path_patient2 = r"C:\Users\micha\Documents\Programming\ML\ECG_denoising\ECG_signals\Person_02\rec_1"
    show_patient_record(path_patient1, "Patient 1")
    choose_enlargement_of_signal(path_patient1)
    choose_enlargement_of_signal(path_patient1, channels=[1])
    choose_best_wavelet_and_decomposition_level(path_patient1)
    ecg_wavelet_denoising(path_patient1, sub_coeff_of_decomp=2, wavelet='sym10', enlargement=1000)

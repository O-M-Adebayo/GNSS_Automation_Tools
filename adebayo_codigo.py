# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 12:36:54 2024

@author: user
"""

from pylab import *
from numpy import *
import pywt


from scipy.signal import convolve
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import zscore
from scipy.signal import savgol_filter, butter, filtfilt, stft, istft
import glob
import os
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable
#import ruptures as rpt
#import ruptures as rpt
# import scienceplots
# plt.style.use(['science', 'notebook', 'grid'])
from signal_alam import *
import pandas as pd
plt.close('all')

#%%
# This Python function calculates the distance between two points
# on Earth given their longitude and latitude coordinates. It uses
# the Haversine formula to calculate the distance (d1) and the
# Equirectangular approximation (d2) based on the coordinates.

def calculate_distance(lon1, lat1, lon2, lat2):
    earth_radius = 6371  # Radius of the Earth in kilometers

    delta_lat = np.radians(lat2) - np.radians(lat1)
    delta_lon = np.radians(lon2) - np.radians(lon1)

    a = np.sin(delta_lat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(delta_lon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    d1 = earth_radius * c  # Haversine formula: distance in kilometers
    x = delta_lon * np.cos((np.radians(lat1) + np.radians(lat2))/2)
    y = delta_lat

    d2 = earth_radius * np.sqrt(x**2 + y**2)  # Equirectangular approximation: distance in kilometers

    return d1, d2


#%%

# This Python function applies a data filtering technique to
# a given array of TEC (Total Electron Content) values.
# It calculates the time difference between two consecutive
# elements of the array (dt_rec), determines the number of
# elements for two different time windows (nm1 and nm2), and
# applies a convolution operation on the TEC array to obtain
# a filtered result (dtec). The function returns the filtered TEC values.

def filter_tec_data(tec, lower_bound, upper_bound):
    time_resolution = (time_of_study[1] - time_of_study[0]) * 3600  # Time resolution in seconds

    first_time_window = int(lower_bound * 60 / time_resolution)  # Number of elements for the first time window
    second_time_window = int(upper_bound * 60 / time_resolution)  # Number of elements for the second time window

    filter_1 = np.ones((first_time_window,)) / first_time_window  # Filter kernel for the first time window
    filter_2 = np.ones((second_time_window,)) / second_time_window  # Filter kernel for the second time window

    dtec = convolve(tec, filter_1, mode='same') - convolve(tec, filter_2, mode='same')

    return dtec
#%%
# This Python function finds the nearest value to a given value in an array.
# It converts the input array to a NumPy array, calculates the absolute
# difference between each element in the array and the given value, finds
# the index of the element with the minimum difference, and returns the
# corresponding value from the array.

def find_nearest_value(array, value):
    array = np.asarray(array)
    absolute_diff = np.abs(array - value)
    index = np.argmin(absolute_diff)
    nearest_value = array[index]
    return nearest_value

#%%
def axis_param(im,vm,str_c):
    divider = make_axes_locatable(ax);
    cax = divider.append_axes("right", size="2%", pad=0.05)
    cbar=colorbar(im,cax=cax);gca().set_title(str_c) 

    return()

#%%

def get_folder_names(directory_path):
    
    """
    Purpose: Retrieves folder names from a specified directory path.
    
    Parameters:
    directory_path: Path of the directory.
    
    Returns:
    folder_names: List of folder names sorted alphabetically.
    """
    
    try:
        # Get a list of all items in the directory
        items = os.listdir(directory_path)

        # Filter out only directories
        folder_names = [item for item in items if os.path.isdir(os.path.join(directory_path, item))]
        
        # Sort the folder names alphabetically
        folder_names.sort()

        return folder_names
    except OSError as e:
        print(f"Error: {e}")
        return []

#%%

def remove_spikes(sTEC, spike_threshold=2):
    """
    Remove spikes from an array of sTEC (slant Total Electron Content) data.

    Parameters:
    - sTEC: numpy array or list, input array containing sTEC data.
    - spike_threshold: float, optional, threshold for spike detection (default is 5).

    Returns:
    - cleaned_sTEC: numpy array, sTEC data with spikes removed and NaN values interpolated.
    """
    # Convert input to numpy array and ensure it's float type
    sTEC = np.array(sTEC, dtype=float)
    
    # Find spikes (values exceeding the threshold)
    spikes = np.abs(sTEC - np.median(sTEC)) > spike_threshold * np.std(sTEC)
    
    # Replace spikes with NaN
    cleaned_sTEC = sTEC.copy()
    cleaned_sTEC[spikes] = np.nan
    
    # Interpolate NaN values
    nan_indices = np.isnan(cleaned_sTEC)
    not_nan_indices = ~nan_indices
    x = np.arange(len(cleaned_sTEC))
    cleaned_sTEC[nan_indices] = np.interp(x[nan_indices], x[not_nan_indices], cleaned_sTEC[not_nan_indices])
    
    return cleaned_sTEC
#%%


# def bayesian_change_point_detection(dtec, time, start_time):
#     """
#     Perform Bayesian Change Point Detection on GNSS TEC data starting from a specified time.
    
#     Parameters:
#     dtec (np.array): Array of differential TEC values.
#     time (np.array): Array of corresponding time values (in hours).
#     start_time (float): The time in hours from which to start the change point detection.
    
#     Returns:
#     list: Indices of detected change points in the time array.
#     """
#     # Validate input
#     if len(dtec) != len(time):
#         raise ValueError("The dtec and time arrays must have the same length.")
    
#     # Filter the data to start from the specified time
#     start_index = np.searchsorted(time, start_time, side='left')
#     filtered_dtec = dtec[start_index:]
#     filtered_time = time[start_index:]
    
#     # Reshape dtec for the ruptures library
#     filtered_dtec = np.array(filtered_dtec).reshape(-1, 1)

#     # Use the ruptures library for change point detection
#     model = "rbf"  # Radial basis function model
#     algo = rpt.Binseg(model=model).fit(filtered_dtec)
#     n_bkps = 20  # Number of breakpoints to detect (tune this parameter as needed)
    
#     # Perform change point detection
#     change_points = algo.predict(n_bkps=n_bkps)
    
#     # Map the change points to time values
#     change_points_time = [filtered_time[cp - 1] for cp in change_points[:-1]]  # Exclude the last point which is the end of the series
    
#     return change_points_time

#%%

def detect_transient_events(stec_array, window_size=60, poly_order=1, threshold=0.5):
    """
    Detrend slight Total Electron Content (TEC) to highlight transient events.
    
    Parameters:
    stec_array (np.array): Array of TEC values.
    window_size (int): Window size for Savitzky-Golay filter (must be odd). Default is 60.
    poly_order (int): Polynomial order for Savitzky-Golay filter. Default is 1.
    threshold (float): Z-score threshold for detecting transient events. Default is 0.5.

    Returns:
    np.array: Array of indices where transient events are detected.
    np.array: Smoothed TEC data after applying Savitzky-Golay filter.
    np.array: Residual TEC data after removing the trend.
    """
    # Apply Savitzky-Golay filter for noise reduction
    smoothed_data = savgol_filter(stec_array, window_size, poly_order)
    
    # Calculate the residuals by subtracting the smoothed data from the original data
    resid = stec_array - smoothed_data
    
    # Compute z-scores on the smoothed data
    z_scores = zscore(smoothed_data)
    
    # Identify transient events where the z-score exceeds the threshold
    transient_events = np.where(np.abs(z_scores) > threshold)[0]
    
    return transient_events, smoothed_data, resid

# # Testing
# np.random.seed(0)
# time = np.linspace(0, 1, 1000)
# tec_data = np.sin(2 * np.pi * 10 * time) + 0.5 * np.random.randn(1000)

# transient_events, smoothed_data, resid = detect_transient_events(tec_data)

# print("Transient event indices:", transient_events)


#%%
def detect_stec_enhancements(stec_data, time, sg_window=59, sg_poly=3, hp_cutoff=0.005):
    """
    Filters sTEC data to detect sudden enhancements (e.g., due to earthquakes).
    
    Parameters:
    stec_data (array-like): Array of sTEC data (time series).
    time (array-like): Array of time values corresponding to the sTEC data.
    sg_window (int): Window size for Savitzky-Golay filter (default is 51).
    sg_poly (int): Polynomial order for Savitzky-Golay filter (default is 3).
    hp_cutoff (float): Cutoff frequency for high-pass filter (default is 0.01).
    
    Returns:
    filtered_stec (array-like): Filtered sTEC data highlighting sudden enhancements.
    fig: The resulting figure showing original and filtered data.
    """
    # Apply Savitzky-Golay filter for initial smoothing
    smoothed_stec = savgol_filter(stec_data, sg_window, sg_poly)
    
    # Design a high-pass filter
    nyquist = 0.5 * len(time) / (time[-1] - time[0])
    normal_cutoff = hp_cutoff / nyquist
    b, a = butter(1, normal_cutoff, btype='high', analog=False)
    
    # Apply the high-pass filter
    filtered_stec = filtfilt(b, a, smoothed_stec)
    
    # # Plotting
    # fig, ax = plt.subplots(figsize=(12, 6))
    # ax.plot(time, stec_data, label='Original sTEC Data')
    # ax.plot(time, filtered_stec, label='Filtered sTEC Data', linewidth=2)
    # ax.legend()
    # ax.set_title('Detection of Sudden Enhancements in sTEC Data')
    # ax.set_xlabel('Time')
    # ax.set_ylabel('sTEC')
    
    return filtered_stec

#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d

def detect_stec_enhancements_convolution(stec_data, time, kernel_size=60):
    """
    Filters sTEC data to detect sudden enhancements (e.g., due to earthquakes) using convolution.
    
    Parameters:
    stec_data (array-like): Array of sTEC data (time series).
    time (array-like): Array of time values corresponding to the sTEC data.
    kernel_size (int): Size of the convolution kernel (default is 5).
    
    Returns:
    convolved_data (array-like): Convolved sTEC data highlighting sudden enhancements.
    fig: The resulting figure showing original and filtered data.
    """
    # Design a convolution kernel to detect changes (like a derivative)
    kernel = np.array([-1] * (kernel_size // 2) + [0] + [1] * (kernel_size // 2))
    
    # Normalize the kernel
    kernel = kernel / np.sum(np.abs(kernel))
    
    # Apply convolution to the sTEC data
    convolved_data = convolve1d(stec_data, kernel)
    return convolved_data


#%%

from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

def high_pass_filter_stec(stec_data, time, low_cutoff=0.002, high_cutoff=0.01, dt=15):
    """
    Applies a high-pass filter to sTEC data for a specified frequency range.
    
    Parameters:
    stec_data (array-like): Array of sTEC data (time series).
    time (array-like): Array of time values corresponding to the sTEC data.
    low_cutoff (float): Lower cutoff frequency in Hz (default 0.002 Hz).
    high_cutoff (float): Higher cutoff frequency in Hz (default 0.01 Hz).
    fs (float): Sampling frequency (Hz). If None, it will be inferred from the time data.
    
    Returns:
    filtered_data (array-like): Filtered sTEC data.
    fig: The resulting figure showing original and filtered data.
    """
    
    fs = 1 / dt  # Sampling frequency

    # Define the high-pass filter
    nyquist = 0.5 * fs
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist

    # Design a bandpass Butterworth filter
    b, a = butter(4, [low, high], btype='band')

    # Apply the filter to the sTEC data using filtfilt (zero-phase filtering)
    filtered_data = filtfilt(b, a, stec_data)

    # # Plot original and filtered data
    # plt.figure(figsize=(10, 6))
    # plt.plot(time, stec_data, label='Original sTEC Data', color='gray', alpha=0.7)
    # plt.plot(time, filtered_data, label='Filtered sTEC Data (High-pass)', color='blue', linewidth=2)
    # plt.xlabel('Time (s)')
    # plt.ylabel('sTEC')
    # plt.legend()
    # plt.title('High-pass Filtered sTEC Data')
    # plt.grid(True)
    # plt.show()

    return filtered_data









#%%


def adjust_dtec(dtec, time, earthquake_time):
    """
    Adjusts the DTEC values based on the average before the earthquake time.
    
    Parameters:
    - dtec: np.ndarray, absolute DTEC values
    - time: np.ndarray, corresponding time values
    - earthquake_time: float, the time of the earthquake
    
    Returns:
    - new_dtec: np.ndarray, adjusted DTEC values
    """
    # Filter DTEC values before the earthquake time
    valid_indices = time < earthquake_time
    dtec_before_eq = dtec[valid_indices]
    
    # Calculate the average of DTEC values before the earthquake
    average_dtec = np.mean(dtec_before_eq)
    
    # Adjust the DTEC values
    new_dtec = dtec - average_dtec
    
    # Set negative values to zero
    new_dtec[new_dtec < 0] = 0
    
    return new_dtec


#%%



# def axis_param(im, vm, str_c):
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="2%", pad=0.05)
#     cbar = colorbar(im, cax=cax)
#     gca().set_title(str_c)
#     return()

def wavelet_analysis(t, dtec_av, wavelet='cmor', scales=None, sampling_rate=15, cbar_label='TECU'):
    """
    Perform Continuous Wavelet Transform (CWT) on the given tec data.

    Parameters:
    - t (array-like): Time data.
    - dtec_av (array-like): TEC data.
    - wavelet (str): Type of wavelet (default 'cmor' for Complex Morlet).
    - scales (array-like): Scales for the CWT (default range from 1 to 16).
    - sampling_rate (float): The sampling rate of the time data (default 15).
    - cbar_label (str): The label for the colorbar (default 'TECU').

    Returns:
    - None (the plot will be displayed).
    """
    # If the time and tec data are not arrays, convert them to arrays
    t_arr = array(t)
    dtec_arr = array(dtec_av)

    # Default scales if not provided
    if scales is None:
        scales = arange(1, 16)

    # Perform the Continuous Wavelet Transform (CWT)
    coefficients, frequencies = pywt.cwt(dtec_arr, scales, wavelet, sampling_rate)

    # Plot the result
    im = imshow(abs(coefficients), 
                extent=[t_arr[0], t_arr[-1], frequencies[-1]*1e3, frequencies[0]*1e3], 
                aspect='auto', cmap='jet')
    im.set_interpolation('spline16')

    
    
    #tight_layout()

    # Show the plot
show()

#%%

def stft_band_pass_filter(stec_data, time, lowcut=0.003, highcut=0.006, dt=15, window_length=128, overlap=32):
    """
    Applies a band-pass filter to sTEC data using Short-Time Fourier Transform (STFT) 
    to retain frequencies within the range [lowcut, highcut].
    
    Parameters:
    stec_data (array-like): Array of sTEC data (time series).
    time (array-like): Array of time values corresponding to the sTEC data.
    lowcut (float): The lower frequency cutoff in Hz (default is 0.003 Hz).
    highcut (float): The upper frequency cutoff in Hz (default is 0.006 Hz).
    dt (float): The time step between each data point (in seconds).
    window_length (int): The length of the STFT window (default is 128).
    overlap (int): The number of overlapping points for STFT (default is 64).
    
    Returns:
    filtered_data (array-like): The filtered sTEC data with the desired frequency range.
    fig: The resulting figure showing original and filtered data.
    """
    # Sampling frequency based on time step
    fs = 1 / dt
    
    # Compute the STFT of the data
    f, t, Zxx = stft(stec_data, fs=fs, nperseg=window_length, noverlap=overlap)
    
    # Define the band-pass filter frequencies
    low_idx = np.searchsorted(f, lowcut)
    high_idx = np.searchsorted(f, highcut)
    
    # Zero out frequencies outside the desired range
    Zxx_filtered = Zxx.copy()
    Zxx_filtered[:low_idx, :] = 0
    Zxx_filtered[high_idx:, :] = 0
    
    # Reconstruct the signal from the filtered STFT
    _, filtered_data = istft(Zxx_filtered, fs=fs)
    
    # # Plot the contour of the STFT magnitude
    # plt.figure(figsize=(12, 6))
    # plt.contourf(t, f, np.abs(Zxx_filtered), 100, cmap='rainbow')
    # plt.title("STFT Magnitude of Filtered Data")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Frequency (Hz)")
    # plt.colorbar(label="Magnitude")
    # plt.ylim(lowcut, highcut)
    # plt.show()
    
    return Zxx_filtered, f, t

#%%

def find_peaks_mike_k(arr, start_idx, end_idx, threshold, k):
    """
    Function to find the peaks in a given range of dTEC data.
    
    Parameters:
    - arr (list or numpy array): The dTEC data array.
    - start_idx (int): The index from where to start searching for peaks.
    - end_idx (int): The index at which to stop searching for peaks.
    - threshold (float): The threshold value to filter peaks by dTEC.
    - k (int): The local search area to define the peak.
    
    Returns:
    - List of indices of peaks that meet the given threshold condition.
    """
    peaks = []
    
    # Ensure valid start and end indices
    start_idx = max(0, start_idx)
    end_idx = min(len(arr), end_idx)
    
    for i in range(start_idx + k, end_idx - k):  # Skip k values from the edges for local area search
        if arr[i] >= threshold:  # First, check if the dTEC value is above the threshold
            local_max = True
            # Check if arr[i] is the local peak by comparing with neighbors within the window of size k
            for j in range(i - k, i + k + 1):
                if arr[j] > arr[i]:
                    local_max = False
                    break
            if local_max:
                peaks.append(i)
    
    return peaks

#%%

def find_peaks_within_range(arr, start_idx, end_idx, threshold):
    peaks_indices = []
    
    # Ensure start_idx and end_idx are within valid bounds
    start_idx = max(0, start_idx)
    end_idx = min(len(arr), end_idx)
    
    # Loop over the range with proper boundary checking
    for i in range(start_idx + 1, end_idx - 1):
        if i - 4 >= 0 and i + 4 < len(arr):  # Ensure that the indices are within bounds
            if arr[i] > arr[i - 4] and arr[i] > arr[i + 4]:
                if arr[i] > threshold:
                    peaks_indices.append(i)
                
    return peaks_indices


def find_peaks_within_range_obs(arr, start_idx, end_idx, threshold):
    peaks_indices = []
    
    # Ensure start_idx and end_idx are within valid bounds
    start_idx = max(0, start_idx)
    end_idx = min(len(arr), end_idx)
    
    # Loop over the range with proper boundary checking
    for i in range(start_idx + 1, end_idx - 1):
        if i - 5 >= 0 and i + 5 < len(arr):  # Ensure that the indices are within bounds
            if arr[i] > arr[i - 5] and arr[i] > arr[i + 5]:
                if arr[i] > threshold:
                    peaks_indices.append(i)
                
    return peaks_indices

def find_peaks_within_range_k(arr, start_idx, end_idx, threshold, k):
    peaks_indices = []
    
    # Ensure start_idx and end_idx are within valid bounds
    start_idx = max(0, start_idx)
    end_idx = min(len(arr), end_idx)
    
    # Loop over the range with proper boundary checking
    for i in range(start_idx + 1, end_idx - 1):
        if i - k >= 0 and i + k < len(arr):  # Ensure that the indices are within bounds
            if arr[i] > arr[i - k] and arr[i] > arr[i + k]:
                if arr[i] > threshold:
                    peaks_indices.append(i)
                
    return peaks_indices


def data_smooth_mike(t,data, win1, win2):
    
    dt=t[1]-t[0]
    n_smooth=int(ceil(win1/(dt*60.)))
    win=ones((n_smooth,))/n_smooth
    data=data-convolve_al(data,win)
    return convolve_al(data,ones((win2,))/win2)

#%%

def band_pass_filter(stec_data, time, lowcut=0.003, highcut=0.006, dt=15):
    """
    Applies a band-pass filter to sTEC data to retain frequencies within the range [lowcut, highcut].
    
    Parameters:
    stec_data (array-like): Array of sTEC data (time series).
    time (array-like): Array of time values corresponding to the sTEC data.
    lowcut (float): The lower frequency cutoff in Hz (default is 0.003 Hz).
    highcut (float): The upper frequency cutoff in Hz (default is 0.006 Hz).
    fs (float): The sampling frequency of the sTEC data (in Hz). If None, it is inferred from the time series.
    
    Returns:
    filtered_data (array-like): The filtered sTEC data with the desired frequency range.
    fig: The resulting figure showing original and filtered data.
    """
    # If the sampling frequency is not provided, infer it from the time data
    
    fs = 1 / dt
    
    # Design the band-pass filter
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(4, [low, high], btype='band')

    # Apply the filter to the data
    filtered_data = filtfilt(b, a, stec_data)
    
    # # Plot the original and filtered data
    # plt.figure(figsize=(12, 6))
    # plt.plot(time, stec_data, label="Original sTEC Data", alpha=0.6)
    # plt.plot(time, filtered_data, label="Filtered sTEC Data", linewidth=2)
    # plt.title("sTEC Data and Band-Pass Filtered Output")
    # plt.xlabel("Time (s)")
    # plt.ylabel("sTEC")
    # plt.legend()
    # plt.show()
    
    return filtered_data


#%%

def compute_periodogram(data, sampling_rate=1, ploting=True, max_period=None):
    """
    Computes the periodogram of the provided data using FFT.
    
    Parameters:
    - data (array-like): Input time series data.
    - sampling_rate (float): Sampling rate of the data (default is 1).
    - plot (bool): Whether to plot the periodogram (default is True).
    - max_period (float or None): Maximum period to show on the plot (optional).
    
    Returns:
    - frequency (array): Frequencies corresponding to the periodogram.
    - period (array): Period corresponding to the frequency range.
    - magnitude (array): Magnitude of the FFT for each frequency.
    """
    
    
    # Perform FFT
    ne_fft = fft(data)
    N = len(ne_fft)
    
    # Create frequency bins
    n = np.arange(N)
    T = N / sampling_rate
    frequency = n / T
    
    # Extract one-sided frequency range
    n_oneside = N // 2
    frequency_oneside = frequency[:n_oneside]
    
    # Calculate the period (in seconds)
    time_secs = 1 / frequency_oneside
    
    # Calculate the magnitude (normalized)
    magnitude = np.abs(ne_fft[:n_oneside]) / n_oneside
    
    # # Plot the periodogram if requested
    # if ploting:
    #     plot(time_secs, 10000*magnitude)
    #     # Optionally limit the x-axis to max_period
    #     if max_period:
    #         xlim(0, max_period)
        
    #     #grid()
    #     show()
    
    return time_secs, magnitude



#%%


from math import radians, cos, sin
from pyproj import Geod


# Initialize geod object (WGS84 ellipsoid)
geod = Geod(ellps='WGS84')

# Function to plot great circles with distance labels
def plot_great_circle(ax, lon_ep, lat_ep, radius_km, side, interval_km=50):
    """
    Plots circles on a map at intervals of `interval_km` around a given epicenter.
    :param ax: Axes object where the plot is drawn
    :param lon_ep: Longitude of the epicenter
    :param lat_ep: Latitude of the epicenter
    :param radius_km: Maximum radius for the circles in kilometers
    :param interval_km: Interval between circles in kilometers
    """
    earth_radius_km = 6371  # Earth's radius in kilometers

    for r in range(interval_km, radius_km + interval_km, interval_km):
        # Angular distance in radians
        angle = r / earth_radius_km
        
        # Generate points around the epicenter for the circle
        lons = []
        lats = []
        num_points = 100  # Number of points to approximate the circle

        for i in range(num_points):
            theta = 2 * np.pi * i / num_points
            lat_new = lat_ep + (angle * 180 / np.pi) * sin(theta)
            lon_new = lon_ep + (angle * 180 / np.pi) * cos(theta) / np.cos(np.radians(lat_ep))
            
            # Wrap around longitude
            lon_new = (lon_new + 180) % 360 - 180
            
            lats.append(lat_new)
            lons.append(lon_new)

        # Plot the circle
        ax.plot(lons, lats, color='grey', linestyle='--', linewidth=0.5)
        
        if side == 'right':
            # Add distance label (beside the circle to the east)
            label_lat = lat_ep  # Keep latitude the same
            label_lon = lon_ep + (angle * 180 / np.pi) / np.cos(np.radians(lat_ep))  # Offset longitude
            
            ax.text(label_lon, label_lat, f'{r} km', fontsize=10, fontweight='bold',
                    rotation=-90, ha='left', va='center', color='grey')
            
        elif side == 'left':
        
            # === Modify label to be on the LEFT side (west) along the great circle ===
            azimuth_deg = 270  # Due west
            label_lon, label_lat, _ = geod.fwd(lon_ep, lat_ep, azimuth_deg, r * 1000)  # r in meters
        
            ax.text(label_lon, label_lat, f'{r} km', fontsize=10, fontweight='bold',
                    rotation=-90, ha='left', va='center', color='grey')



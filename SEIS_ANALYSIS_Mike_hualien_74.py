#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 10:26:06 2023

@author: oluwasegun
"""

#%%

# ---------------------- AVERAGE
from numpy import linspace, array, save, mean, arange
import pandas as pds
from scipy.signal import find_peaks
from matplotlib.pyplot import *
import matplotlib
from scipy.fftpack import fft
import glob
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import obspy
from adebayo_codigo import stft_band_pass_filter, band_pass_filter, compute_periodogram
from obspy.core import *
from obspy.signal.invsim import *
import pywt
import datetime
close('all')


#import scienceplots
#matplotlib.pyplot.style.use(['science', 'notebook', 'grid'])
close('all')


params={'axes.labelweight':'bold'};rcParams.update(params)
font={'family':'serif','weight':'bold','size':11};matplotlib.rc('font', **font);


#%%

# def compute_periodogram(data, sampling_rate=1, ploting=True, max_period=None):
#     """
#     Computes the periodogram of the provided data using FFT.
    
#     Parameters:
#     - data (array-like): Input time series data.
#     - sampling_rate (float): Sampling rate of the data (default is 1).
#     - plot (bool): Whether to plot the periodogram (default is True).
#     - max_period (float or None): Maximum period to show on the plot (optional).
    
#     Returns:
#     - frequency (array): Frequencies corresponding to the periodogram.
#     - period (array): Period corresponding to the frequency range.
#     - magnitude (array): Magnitude of the FFT for each frequency.
#     """
    
    
#     # Perform FFT
#     ne_fft = fft(data)
#     N = len(ne_fft)
    
#     # Create frequency bins
#     n = np.arange(N)
#     T = N / sampling_rate
#     frequency = n / T
    
#     # Extract one-sided frequency range
#     n_oneside = N // 2
#     frequency_oneside = frequency[:n_oneside]
    
#     # Calculate the period (in seconds)
#     time_secs = 1 / frequency_oneside
    
#     # Calculate the magnitude (normalized)
#     magnitude = np.abs(ne_fft[:n_oneside]) / n_oneside
    
#     # Plot the periodogram if requested
#     if ploting:
#         plot(time_secs, 10000*magnitude)
#         #title('Periodogram (Seconds)')
#         # xlabel('Period (seconds)')
#         # ylabel('Magnitude')
        
#         # Optionally limit the x-axis to max_period
#         if max_period:
#             xlim(0, max_period)
        
#         #grid()
#         show()
    
#     return frequency_oneside, time_secs, magnitude


#%%




# Idaho: Constants
lat_ep = 23.832
lon_ep = 121.604
figure_size = (6, 3)
date = 'Aug. 14, 2021'
time_eq = '12:29:08 UT'
time_of_quake_hr = 23 + (58/60) + (12/3600)


# Initialize lists to store seismic data
t_sism = []
vel_sism = []
dist_ep = []
sta_name = []
lat_sta = []
lon_sta = []

# File list
file_list = sorted(glob.glob('./hualien/sismo/*093*.SAC'))
#file_list = sorted(glob.glob(r'.\meteor\*186*.SAC'))
#print(file_list)
# Process each file in the list
for file_name in file_list:
    stream = read(file_name)
    print(stream)
    trace = stream[0]
    #print(trace)

    # Extract relevant information from the trace
    station_lat = trace.stats.sac.stla
    station_lon = trace.stats.sac.stlo
    start_time = str(trace.stats.starttime)
    #print(start_time)
    end_time = str(trace.stats.endtime)
    #print(end_time)
    npts = trace.stats.npts

    # Calculate epicentral distance
    epicenter_distance = station_lat - lat_ep
    print('Epicentral Distance:', file_name[19:23], epicenter_distance)

    # Calculate time array
    start_time_float = float(start_time[11:13]) + float(start_time[14:16]) / 60. + float(start_time[17:19]) / 3600.
    #print(start_time_float)
    end_time_float = 24.0 + float(end_time[11:13]) + float(end_time[14:16]) / 60. + float(end_time[17:19]) / 3600.
    #print(end_time_float)
    time = linspace(start_time_float, end_time_float, npts)

    # Process seismic data
    amplitude = trace.data
    velocity = amplitude * 1.58997e-06 * 6.e+00 / (1. * 1500)  # m/s (V/counts) / (gain * sensitivity (V/m/s))
    nx = arange(int(npts / 100), int(npts / 1), 1)
    velocity = velocity[nx]
    time = time[nx]
    dlat = station_lat - lat_ep
    
    # Plot seismic data
    fig = figure(1, figsize=figure_size, facecolor='w', edgecolor='k')
    #ax = subplot(111)
    plot(time, velocity*100, label=file_name[19:23]) # , color='navy'
    #plot(time, velocity*100, label=file_name[3:6], color='grey')
    #plot(time, epicenter_distance + velocity, label=file_name[3:6], color='grey')
    legend(ncol=2)
    axvline(time_of_quake_hr, color='grey', lw=2)
    #xlim(14,16.5)
    #title('Ground Vertical Velocity, ' + date + ", " + time_eq)
    xlabel('Time (UT)', fontweight='bold', fontsize=12)
    ylabel('Ground uplift (cm/s)', fontweight='bold', fontsize=12)
    grid(linestyle='--', lw=0.4)
    #xlim(2.4, 3.0)
    
    # figure(2)
    # subplot(211)
    # plot(time, velocity*100, label=file_name[19:23])
    
    
    # subplot(212)
    # plot(time[::200], (velocity*100)[::200] + epicenter_distance, label=file_name[19:23])
    
    # t = (time - time_of_quake_hr)*3600
    
    # wav = 'cmor' #mexh, cmor, sym
    # scales = arange(1, 128)
    # sampling_rate = 15
    
    # cof, fre = pywt.cwt(velocity, scales, wav, sampling_rate)
    
    # fig = figure(3)
    # imshow(abs(cof), extent=[t[0]/60, t[-1]/60, fre[-1]*1e3, fre[0]*1e3], 
    #        aspect = 'auto', cmap = 'jet')
    
    # xlabel('Time (minutes)')
    # ylabel('Frequency (mHz)')
    # xlim(0, 60)
    # #ylim(0,5)
    
    # # Store data in lists
    # t_sism.append(time)
    # vel_sism.append(velocity)
    # dist_ep.append(epicenter_distance)
    # sta_name.append(file_name[19:23])
    # lon_sta.append(station_lon)
    # lat_sta.append(station_lat)

# # Calculate the average values for all parameters
# avg_time = mean(array(t_sism), axis=0)
# avg_velocity = mean(array(vel_sism), axis=0)
# avg_dist_ep = mean(array(dist_ep), axis=0)


# # Plot seismic data
# fig = figure(1, figsize=figure_size, facecolor='w', edgecolor='k')
# #plot(avg_time, avg_dist_ep + avg_velocity, label='Average')
# plot(avg_time, avg_velocity, label='Average')
# legend()
# title('Northern Peru (Mw 7.5), ' + time_eq + ', ' + date)
# xlabel('Time (hr)')
# ylabel('Ground Vertical Velocity ($m/s$)')
# grid(linestyle='--')
# axvline(time_of_quake_hr, color='grey')


# # Convert lists to arrays
# t_sism = np.array(t_sism)
# vel_sism = np.array(vel_sism)
# dist_ep = np.array(dist_ep)
# sta_name = np.array(sta_name)
# lon_sta = np.array(lon_sta)
# lat_sta = np.array(lat_sta)


# # Save processed data
# save(r".\frequency_dep_manuscript\precursor_CIDs\SISMO\t_sism.npy", array(t_sism))
# save(r".\frequency_dep_manuscript\precursor_CIDs\SISMO\vel_sism.npy", array(vel_sism))
# save(r".\frequency_dep_manuscript\precursor_CIDs\SISMO\dist_ep.npy", array(dist_ep))
# save(r".\frequency_dep_manuscript\precursor_CIDs\SISMO\sta_name.npy", array(sta_name))
# save(r".\frequency_dep_manuscript\precursor_CIDs\SISMO\lon_sism.npy", array(lon_sta))
# save(r".\frequency_dep_manuscript\precursor_CIDs\SISMO\lat_sism.npy", array(lat_sta))




# # # Save processed average data
# # save('./SISMO/avg_time.npy', avg_time)
# # save('./SISMO/avg_velocity.npy', avg_velocity)
# # save('./SISMO/avg_dist_ep.npy', avg_dist_ep)
# # # save('./SISMO/avg_sta_name.npy', 'Average')  # Save the label for the average station
# savefig(r'.\data_downloader_file\costa_rica_76\ground_uplift_jts.png',
#               dpi=350, bbox_inches='tight') 


show()





#%%

file_path = "./hualien/tsunami/data_52404.txt"

# Reading the data from the file
with open(file_path, 'r') as file:
    data = file.read().strip().split("\n")[2:]  # Skip header lines
    # Prepare data as a list of dictionaries
    parsed_data = []
    
    for line in data:
        parts = line.split()
        
        # Create a datetime object for the timestamp
        timestamp = datetime.datetime(
            year=int(parts[0]),
            month=int(parts[1]),
            day=int(parts[2]),
            hour=int(parts[3]),
            minute=int(parts[4]),
            second=int(parts[5])
        )

        # Only add data where T == 3 (assuming you meant T == 3 based on your example)
        if int(parts[6]) == 3:
            parsed_data.append({
                "Timestamp": timestamp,
                "Height": float(parts[7])
            })
            
# Sort the data by the 'Timestamp' field in ascending order
parsed_data.sort(key=lambda x: x["Timestamp"])

# Extracting time and height for plotting
times_52404 = [entry["Timestamp"] for entry in parsed_data]
heights_52404 = [entry["Height"] for entry in parsed_data]

figure(22, figsize=(8,12))
subplot(311)
plot(times_52404[::4], heights_52404[::4])

t_data_hr = []
for tim in range(len(times_52404)):
    times_hr = str(times_52404[tim])[11:19]
    times_hr = float(times_hr[0:2]) + float(times_hr[3:5])/60 + float(times_hr[6:8])/3600
    t_data_hr.append(times_hr)
    #print("-------------------", times_hr)

t_data_secs = (array(t_data_hr) - time_of_quake_hr) * 3600
dheights = band_pass_filter(heights_52404[0:1011:4], t_data_hr[0:1011:4], lowcut=0.0005, highcut=0.003, dt=60)

figure(23, figsize=(10,10))
subplot(321)
plot(times_52404[0:1011:4], dheights, 
     marker='.', linestyle='-', color='b', label='52404')
#xticks(rotation=45)
xticks([])
ylabel("Water Height (m)", fontweight = 'bold')
ylim(-0.2, 0.2)
legend(loc='upper right')



figure(23, figsize=(10,10))
ax=subplot(322)
data_ft, f, tt = stft_band_pass_filter(dheights, t_data_hr, lowcut=0.0005, highcut=0.003, window_length=100, overlap=90, dt=60)
# Use imshow for displaying the 2D data

#height, width = data_ft.shape  # dimensions of the 2D data

im1 = ax.imshow(np.abs(data_ft), aspect='auto',
                extent=[0, tt.max()/60, (f*1e3).min(), (f*1e3).max()],
                cmap='jet', origin='lower')

# im1 = ax.imshow(np.abs(data_ft), aspect='auto', extent=[tt.min(), tt.max(), (f*1e3).min(), (f*1e3).max()], 
#                 cmap='jet', origin='lower')
im1.set_interpolation("spline16")
text(220, 3.6, '52404', color='white')
#xlabel('Detection Time (mins)')
ylabel('Frequency (mHz)')
axhline(1.25, color='w', lw=1)
text(200, 1.27, "1.30 mHz", color='w')
axhline(2.77, color='w', lw=1)
text(200, 2.79, "2.80 mHz", color='w')
axhline(2.07, color='w', lw=1)
text(200, 2.09, "2.07 mHz", color='w')
ylim(0,4)


#%%

file_path = "./hualien/tsunami/data_52405.txt"

# Reading the data from the file
with open(file_path, 'r') as file:
    data = file.read().strip().split("\n")[2:]  # Skip header lines
    # Prepare data as a list of dictionaries
    parsed_data = []
    
    for line in data:
        parts = line.split()
        
        # Create a datetime object for the timestamp
        timestamp = datetime.datetime(
            year=int(parts[0]),
            month=int(parts[1]),
            day=int(parts[2]),
            hour=int(parts[3]),
            minute=int(parts[4]),
            second=int(parts[5])
        )

        # Only add data where T == 3 (assuming you meant T == 3 based on your example)
        if int(parts[6]) == 2:
            parsed_data.append({
                "Timestamp": timestamp,
                "Height": float(parts[7])
            })
            
# Sort the data by the 'Timestamp' field in ascending order
parsed_data.sort(key=lambda x: x["Timestamp"])

# Extracting time and height for plotting
times_52405 = [entry["Timestamp"] for entry in parsed_data]
heights_52405 = [entry["Height"] for entry in parsed_data]

figure(22, figsize=(8,12))
subplot(313)
plot(times_52405, heights_52405)

t_data_hr = []
for tim in range(len(times_52405)):
    times_hr = str(times_52405[tim])[11:19]
    times_hr = float(times_hr[0:2]) + float(times_hr[3:5])/60 + float(times_hr[6:8])/3600
    t_data_hr.append(times_hr)
    #print("-------------------", times_hr)

t_data_secs = (array(t_data_hr) - time_of_quake_hr) * 3600
dheights = band_pass_filter(heights_52405, t_data_hr, lowcut=0.0005, highcut=0.003, dt=60)

figure(23, figsize=(10,10))
subplot(325)
plot(times_52405[60:], 10*dheights[60:], 
     marker='.', linestyle='-', color='b', label='52405')
#xticks([])
xticks(rotation=45)
xlabel("Timestamp (UT)", fontweight = 'bold')
ylabel("Water Height (m)", fontweight = 'bold')
ylim(-0.2, 0.2)
legend(loc='upper right')


figure(23, figsize=(12,12))
ax=subplot(326)
data_ft, f, tt = stft_band_pass_filter(dheights[60:], (array(t_data_hr[60:])/3600), 
                                       lowcut=0.0005, highcut=0.003, window_length=100, overlap=40, dt=60)
# Use imshow for displaying the 2D data

#height, width = data_ft.shape  # dimensions of the 2D data

im1 = ax.imshow(np.abs(data_ft), aspect='auto',
                extent=[0, tt.max()/60, (f*1e3).min(), (f*1e3).max()],
                cmap='jet', origin='lower')

# im1 = ax.imshow(np.abs(data_ft), aspect='auto', extent=[tt.min(), tt.max(), (f*1e3).min(), (f*1e3).max()], 
#                 cmap='jet', origin='lower')
im1.set_interpolation("spline16")
#text(-138, 9, file[17:24], color='white')
xlabel('Detection Time (mins)')
ylabel('Frequency (mHz)')
text(205, 3.6, '52405', color='white')
axhline(1.0, color='w', lw=1)
axhline(2.05, color='w', lw=1)
axhline(2.60, color='w', lw=1)
text(180, 1.02, "1.0 mHz", color='w')
text(180, 2.07, "2.05 mHz", color='w')
text(180, 2.64, "2.62 mHz", color='w')
ylim(0,4)




#%%

#file_path = "./hualien/tsunami/data_52404.txt"
file_path = "./hualien/tsunami/data_21420.txt"

# Reading the data from the file
with open(file_path, 'r') as file:
    data = file.read().strip().split("\n")[2:]  # Skip header lines
    
    # Prepare data as a list of dictionaries
    parsed_data = []
    
    for line in data:
        parts = line.split()
        
        # Create a datetime object for the timestamp
        timestamp = datetime.datetime(
            year=int(parts[0]),
            month=int(parts[1]),
            day=int(parts[2]),
            hour=int(parts[3]),
            minute=int(parts[4]),
            second=int(parts[5])
        )
        
        # Only add data where T == 3 (assuming you meant T == 3 based on your example)
        if int(parts[6]) == 2:
            parsed_data.append({
                "Timestamp": timestamp,
                "Height": float(parts[7])
            })

# Sort the data by the 'Timestamp' field in ascending order
parsed_data.sort(key=lambda x: x["Timestamp"])

# Extracting time and height for plotting
times = [entry["Timestamp"] for entry in parsed_data]
heights = [entry["Height"] for entry in parsed_data]

t_data_hr = []
for tim in range(len(times)):
    times_hr = str(times[tim])[11:19]
    times_hr = float(times_hr[0:2]) + float(times_hr[3:5])/60 + float(times_hr[6:8])/3600
    t_data_hr.append(times_hr)
    #print("-------------------", times_hr)

t_data_secs = (array(t_data_hr) - time_of_quake_hr) * 3600
dheights = band_pass_filter(heights, t_data_secs, lowcut=0.0001, highcut=0.005, dt=60)

figure(22, figsize=(8,12))
subplot(312)
plot(times, heights)
#xlim(datetime.datetime("2024-04-02 23:00:00"), datetime.datetime("2024-04-03 04:00:00"))


# Normalize heights so that they range from -1 to 1
heights_centered = np.array(heights) - np.mean(heights)  # Center data around 0
#heights_normalized = heights_centered / np.max(np.abs(heights_centered))  # Scale to [-1, 1]


# Plotting the filtered data
fig = figure(6, figsize=(10, 6))
subplot(211)
plot(times[60:], dheights[60:], marker='.', linestyle='-', color='b')
xlabel("Timestamp (UT)", fontweight = 'bold')
ylabel("Water Height (m)", fontweight = 'bold')
xticks(rotation=45)
grid(True, lw = 0.4, linestyle = '-.')


# Plotting the filtered data
figure(23, figsize=(10,10))
subplot(323)
plot(times[60:], 14*dheights[60:], 
     marker='.', linestyle='-', color='b', label='21420')
ylabel("Water Height (m)", fontweight = 'bold')
#xticks(rotation=45)
xticks([])
ylim(-0.2, 0.2)

#grid(True, lw = 0.4, linestyle = '-.')
legend(loc='upper right')


figure(23, figsize=(10,10))
ax=subplot(324)
data_ft, f, tt = stft_band_pass_filter(dheights[160:], (array(t_data_hr[160:])/3600), 
                                       lowcut=0.0005, highcut=0.002, window_length=60, overlap=55, dt=60)

# data_ft, f, tt = stft_band_pass_filter(dheights[160:], (array(t_data_hr[160:])/3600), 
#                                         lowcut=0.0007, highcut=0.003, window_length=80, overlap=75, dt=60)


#data_ft, f, tt = stft_band_pass_filter(dheights, array(t_data_hr)/3600, lowcut=0.0005, highcut=0.003, window_length=100, overlap=90, dt=60)

# Use imshow for displaying the 2D data

height, width = data_ft.shape  # dimensions of the 2D data

im1 = ax.imshow(np.abs(data_ft), aspect='auto',
                extent=[0, tt.max()/60, (f*1e3).min(), (f*1e3).max()],
                cmap='jet', origin='lower')

# im1 = ax.imshow(np.abs(data_ft), aspect='auto', extent=[tt.min(), tt.max(), (f*1e3).min(), (f*1e3).max()], 
#                 cmap='jet', origin='lower')
im1.set_interpolation("spline16")
text(220, 3.6, '21420', color='white')
#xlabel('Detection Time (mins)')
ylabel('Frequency (mHz)')
axhline(0.7, color='w', lw=1)
text(200, 0.4, "0.70 mHz", color='w')
ylim(0,4)
tight_layout()



# Convert times to a numerical format (e.g., seconds since the start)
time_num = np.array([(time - times[0]).total_seconds() for time in times])

# Perform Fourier Transform on the normalized heights
fft_values = np.fft.fft(heights_centered)
fft_frequencies = np.fft.fftfreq(len(fft_values), time_num[1] - time_num[0])  # Sampling rate is the time difference between samples

# Only keep the positive frequencies (the FFT is symmetric)
positive_frequencies = fft_frequencies[:len(fft_frequencies)//2]
positive_fft_values = np.abs(fft_values[:len(fft_values)//2])  # Magnitude of the FFT


with np.errstate(divide='ignore', invalid='ignore'):
    periods = 1 / positive_frequencies
    




# fig = figure(6, figsize=(10, 6))
# subplot(212)
# time_secs, magnitude = compute_periodogram(dheights, sampling_rate=1/60)
# plot(time_secs, 10000 * magnitude, color='r')
# ylim(0,4)
# xlim(0, 1260)
# grid(True, lw = 0.4, linestyle = '-.')
# xlabel('Period (seconds)')
# ylabel('PSD')
# #text(1250, 5.1, '0.7 mHz', fontsize=7)
# # text(2430, 8.8, '0.3 mHz', fontsize=7)
# # text(4200, 14, '0.2 mHz', fontsize=7)
# tight_layout()




# figure(3)
# ax=subplot(111)

fig = figure(6, figsize=(10, 6))
ax=subplot(212)



data_ft, f, tt = stft_band_pass_filter(dheights, t_data_hr, lowcut=0.0005, highcut=0.003, window_length=100, overlap=90, dt=60)
# Use imshow for displaying the 2D data

height, width = data_ft.shape  # dimensions of the 2D data

im1 = ax.imshow(np.abs(data_ft), aspect='auto',
                extent=[0, width, (f*1e3).min(), (f*1e3).max()],
                cmap='jet', origin='lower')

# im1 = ax.imshow(np.abs(data_ft), aspect='auto', extent=[tt.min(), tt.max(), (f*1e3).min(), (f*1e3).max()], 
#                 cmap='jet', origin='lower')
im1.set_interpolation("spline16")
#text(-138, 9, file[17:24], color='white')
xlabel('Detection Time (mins)')
ylabel('Frequency (mHz)')
#xlim(t_data_hr[0], t_data_hr[-1])
ylim(0,4)
#text(45, 9, 'G27', color='w')
#axis_param(im1,'') 
#axis_param(im, 0.1, '')


tight_layout()
show()

# wav = 'cmor' #mexh, cmor, sym
# scales = arange(1, 28)
# sampling_rate = 15

# cof, fre = pywt.cwt(heights_centered, scales, wav, sampling_rate)

# fig = figure(6, figsize=(10, 6))
# subplot(212)
# im = imshow(abs(cof), extent=[t[0]/60, t[-1]/60, fre[-1]*1e3, fre[0]*1e3], 
#        aspect = 'auto', cmap = 'jet')
# im.set_interpolation('spline16')

# xlabel('Detection Time (minutes)', fontweight = 'bold')
# ylabel('Frequency (mHz)', fontweight = 'bold')
# xlim(t[0]/60, 10)
# #ylim(-1, 10)
# colorbar(pad=0.01, label='Magnitude')


# # # Plot by period
# # subplot(313)
# # plt.plot(periods, positive_fft_values)
# # plt.xlabel('Period (s)')
# # plt.ylabel('Magnitude')
# # plt.title('Period Spectrum')


# tight_layout()
# show()


#%%
############## ALASKA 
# file_path = "./hualien/tsunami/data_46402_alaska.txt"
# #file_path = "./hualien/tsunami/data_21420.txt"

# # Reading the data from the file
# with open(file_path, 'r') as file:
#     data = file.read().strip().split("\n")[2:]  # Skip header lines
    
#     # Prepare data as a list of dictionaries
#     parsed_data = []
    
#     for line in data:
#         parts = line.split()
        
#         # Create a datetime object for the timestamp
#         timestamp = datetime.datetime(
#             year=int(parts[0]),
#             month=int(parts[1]),
#             day=int(parts[2]),
#             hour=int(parts[3]),
#             minute=int(parts[4]),
#             second=int(parts[5])
#         )
        
#         # Only add data where T == 3 (assuming you meant T == 3 based on your example)
#         if int(parts[6]) == 2:
#             parsed_data.append({
#                 "Timestamp": timestamp,
#                 "Height": float(parts[7])
#             })

# # Sort the data by the 'Timestamp' field in ascending order
# parsed_data.sort(key=lambda x: x["Timestamp"])

# # Extracting time and height for plotting
# times = [entry["Timestamp"] for entry in parsed_data]
# heights = [entry["Height"] for entry in parsed_data]


# # Normalize heights so that they range from -1 to 1
# heights_centered = np.array(heights) - np.mean(heights)  # Center data around 0
# #heights_normalized = heights_centered / np.max(np.abs(heights_centered))  # Scale to [-1, 1]


# # Plotting the filtered data
# fig = figure(7, figsize=(10, 6))
# subplot(211)
# plot(times, heights, marker='.', linestyle='-', color='b')
# xlabel("Timestamp (April 03, 2024)", fontweight = 'bold')
# ylabel("Water Height (m)", fontweight = 'bold')
# xticks(rotation=45)
# grid(True, lw = 0.4, linestyle = '-.')



# # # Convert times to a numerical format (e.g., seconds since the start)
# # time_num = np.array([(time - times[0]).total_seconds() for time in times])

# # # Perform Fourier Transform on the normalized heights
# # fft_values = np.fft.fft(heights_centered)
# # fft_frequencies = np.fft.fftfreq(len(fft_values), time_num[1] - time_num[0])  # Sampling rate is the time difference between samples

# # # Only keep the positive frequencies (the FFT is symmetric)
# # positive_frequencies = fft_frequencies[:len(fft_frequencies)//2]
# # positive_fft_values = np.abs(fft_values[:len(fft_values)//2])  # Magnitude of the FFT


# # with np.errstate(divide='ignore', invalid='ignore'):
# #     periods = 1 / positive_frequencies
    









# wav = 'cmor' #mexh, cmor, sym
# scales = arange(1, 128)
# sampling_rate = 5

# cof, fre = pywt.cwt(heights_centered, scales, wav, sampling_rate)

# fig = figure(7, figsize=(10, 6))
# subplot(212)
# im = imshow(abs(cof), extent=[t[0]/60, t[-1]/60, fre[-1]*1e3, fre[0]*1e3], 
#        aspect = 'auto', cmap = 'jet')
# im.set_interpolation('spline16')

# xlabel('Detection Time (minutes)', fontweight = 'bold')
# ylabel('Frequency (mHz)', fontweight = 'bold')
# xlim(t[0]/60, 60)
# #colorbar()
# colorbar(pad=0.01, label='Magnitude')


# # # Plot by period
# # subplot(313)
# # plt.plot(periods, positive_fft_values)
# # plt.xlabel('Period (s)')
# # plt.ylabel('Magnitude')
# # plt.title('Period Spectrum')


# # ax = subplot(212)
# # data_ft, f, tt = stft_band_pass_filter(heights_centered, times, lowcut=0.001, highcut=0.03, window_length=100, overlap=90, dt=60)
# # # Use imshow for displaying the 2D data
# # im1 = ax.imshow(np.abs(data_ft*10), aspect='auto', extent=[0, tt.max() / 60, (f*1e3).min(), (f*1e3).max()], 
# #                cmap='jet', origin='lower')

# # im1.set_interpolation("spline16")



# # #text(-138, 9, file[17:24], color='white')
# # xlabel('Detection Time (mins)')
# # ylabel('Frequency (mHz)')
# # #xlim(-135,0)
# # #ylim(1,10)
# # #text(45, 9, 'G27', color='w')
# # #axis_param(im1,'') 
# # #axis_param(im, 0.1, '')


# tight_layout()
# show()





























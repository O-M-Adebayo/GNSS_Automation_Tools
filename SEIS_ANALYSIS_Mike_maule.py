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
from obspy.core import *
from obspy.signal.invsim import *
import pywt


#import scienceplots
#matplotlib.pyplot.style.use(['science', 'notebook', 'grid'])
close('all')


params={'axes.labelweight':'bold'};rcParams.update(params)
font={'family':'serif','weight':'bold','size':11};matplotlib.rc('font', **font);


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
    
    # Plot the periodogram if requested
    if ploting:
        plot(time_secs, magnitude)
        title('Periodogram (Seconds)')
        xlabel('Period (seconds)')
        ylabel('Magnitude / Frequency Bin')
        
        # Optionally limit the x-axis to max_period
        if max_period:
            xlim(0, max_period)
        
        grid()
        show()
    
    return frequency_oneside, time_secs, magnitude


#%%




# Idaho: Constants
lat_ep = -36.122
lon_ep = -72.898
figure_size = (6, 3)
date = 'Aug. 14, 2021'
time_eq = '12:29:08 UT'
time_of_quake_hr = 6 + (34/60) + (11/3600)


# Initialize lists to store seismic data
t_sism = []
vel_sism = []
dist_ep = []
sta_name = []
lat_sta = []
lon_sta = []

# File list
folder_path = r"/seismometer/maule_sismo/test"

#file_list = sorted(glob.glob('/home/oluwasegun-adebayo/Downloads/2010-02-27-mw88-near-coast-of-central-chile-5/*058*.SAC'))
#file_list = sorted(glob.glob(r'.\meteor\*186*.SAC'))
file_list = sorted(glob.glob('./seismometers/maule_sismo/test/*058*.SAC'))
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
    end_time_float = float(end_time[11:13]) + float(end_time[14:16]) / 60. + float(end_time[17:19]) / 3600.
    #print(end_time_float)
    time = linspace(start_time_float, end_time_float, npts)

    # Process seismic data
    amplitude = trace.data
    velocity = amplitude * 1.58997e-06 * 1.e+00 / (1. * 1500)  # m/s (V/counts) / (gain * sensitivity (V/m/s))
    nx = arange(int(npts / 100), int(npts / 1), 1)
    velocity = velocity[nx]
    time = time[nx]
    dlat = station_lat - lat_ep
    
    if file_name[35:39] == 'TRQA':
        
        # Plot seismic data
        fig = figure(1, figsize=figure_size, facecolor='w', edgecolor='k')
        #ax = subplot(111)
        plot(time, velocity, label=file_name[35:39]) # , color='navy'
        #plot(time, velocity*100, label=file_name[3:6], color='grey')
        #plot(time, epicenter_distance + velocity, label=file_name[3:6], color='grey')
        legend(ncol=2)
        axvline(time_of_quake_hr, color='grey', lw=2)
        #xlim(14,16.5)
        #title('Ground Vertical Velocity, ' + date + ", " + time_eq)
        xlabel('Time (UT)', fontweight='bold', fontsize=12)
        ylabel('Ground uplift (m/s)', fontweight='bold', fontsize=12)
        grid(linestyle='--', lw=0.4)
        #xlim(2.4, 3.0)
    
    figure(2, figsize=(10, 7))
    subplot(211)
    plot(time, velocity, label=file_name[35:39])
    
    
    subplot(212)
    plot(time[::200], (velocity*10)[::200] + epicenter_distance, label=file_name[19:23])
    
    ylim(-20, 20)
    
    #t = (time - time_of_quake_hr)*3600
    
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
    
    nam = file_name[35:39]
    # Store data in lists
    t_sism.append(time)
    vel_sism.append(velocity)
    dist_ep.append(epicenter_distance)
    sta_name.append(file_name[35:39])
    lon_sta.append(station_lon)
    lat_sta.append(station_lat)
    
    # Saving the processed files
    data_sismo = {
        't_sismo': time,
        'vel_sismo': velocity,
        'ep_dist_sismo': epicenter_distance,
        'sta_name_sismo': file_name[35:39],
        'lon_sismo': station_lon,
        'lat_sismo': station_lat
        }
    
    # if not os.path.exists('./seismometers/maule_sismo/processed'):
        
    #     os.mkdir('./seismometers/maule_sismo/processed')
    
    # file_n = f"{nam}.npy"
    # file_p = f"./seismometers/maule_sismo/processed/"


    # save(file_p + file_n, data_sismo)

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

# time_arr = array(t_sism).mean(0)
# vel_arr = array(vel_sism).mean(0)

# # Combine the data into a 2D array (each row corresponds to a set of data)
# data_sism = np.vstack((time_arr, vel_arr))

# # # Save the array to a .npy file
# # save('Maule_mike.npy', data_sism)

# figure(3)
# plot(time_arr, vel_arr)
# show()
































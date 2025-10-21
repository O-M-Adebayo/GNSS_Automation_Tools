#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 22:47:22 2025

@author: oluwasegun-adebayo
"""

from numpy import *
from matplotlib.pyplot import *
import glob
from datetime import timedelta, datetime
from obspy import UTCDateTime
import os
 
#%%

lat_ep = -36.122
lon_ep = -72.898
time_of_quake_hr = 6 + (34/60) + (11/3600)


file_path = sorted(glob.glob("/home/oluwasegun-adebayo/Downloads/20100227_8.8M_36.17S_73.14W_30.1KM/st*.npy"))

n_files = len(file_path)
ncols = 3
nrows = int(np.ceil(n_files / ncols))  # Ensures enough rows

figure(figsize=(15, 10))

for idx, file in enumerate(file_path):
    #print(file)
    #if idx == 0:
    data = load(file, allow_pickle=True).item()
    #print(data['station_code'])
    
    # #Check if it's an object array with a dictionary
    # if isinstance(data, np.ndarray) and data.dtype == object:
    #     data = data.item()  # Extract the dictionary
    
    # Assuming `data` is already loaded from your file
    acc_filtered_1 = data['acc_filtered_1']
    acc_filtered_2 = data['acc_filtered_2']
    acc_filtered_3 = data['acc_filtered_3']
    
    
    
        
    component_1 = data['component_1']
    component_2 = data['component_2']
    component_3 = data['component_3']
    
    dt = data['dt']
    
    # Convert acceleration (m/s²) to velocity (m/s)
    vel_1 = cumsum(acc_filtered_1) * dt
    vel_2 = cumsum(acc_filtered_2) * dt
    vel_3 = cumsum(acc_filtered_3) * dt
    
    
    
    

    

    start_time = data['starttime']
    station_lat = data['station_lat']
    station_lon = data['station_lon']
    station_code = data['station_code']
    velocity = vel_3
    epicenter_distance = station_lat - lat_ep
    
    # Create a time vector
    n_samples = len(acc_filtered_1)
    #time = np.arange(0, n_samples * dt, dt)
    #time = 6.4022 + np.arange(n_samples) * dt

    # Convert 'starttime' to datetime object
    start_time_str = data['starttime']  # e.g., '2010-02-27 06:24:08.000000Z'
    start_time_dt = datetime.strptime(start_time_str.replace("Z", ""), "%Y-%m-%d %H:%M:%S.%f")
    
    # Convert to fractional hours
    start_time_in_hours = start_time_dt.hour + start_time_dt.minute / 60 + start_time_dt.second / 3600 + start_time_dt.microsecond / 3_600_000_000
    
    # Build time vector in hours
    time = start_time_in_hours + np.arange(n_samples) * dt / 3600

   
    # #start_time = data['starttime']  
    # start_time_dt = start_time.datetime if hasattr(start_time, 'datetime') else start_time
    
    # time = [start_time_dt + timedelta(seconds=i * dt) for i in range(n_samples)]
    
    # subplot(nrows, ncols, idx + 1)
    # plot(time, vel_3, label=f'{component_3} Component')
    # xlabel('Time (UTC)')
    # ylabel('Vel. (m/s)')
    # title(f'{component_3} Filtered Velocity\n{station_code}')
    # grid(True)



    # Plotting
    #figure(figsize=(12, 8))
    
    # subplot(3, 1, 1)
    # plot(time, acc_filtered_1, label=f'{component_1} Component')
    # ylabel('Acceleration (m/s²)')
    # title(f'{component_1} Filtered Acceleration')
    # grid(True)
    
    # subplot(3, 1, 2)
    # plot(time, acc_filtered_2, label=f'{component_2} Component')
    # ylabel('Acceleration (m/s²)')
    # title(f'{component_2} Filtered Acceleration')
    # grid(True)
    
    #print(time[-1]-time[0])
    
    subplot(nrows, ncols, idx + 1)
    #subplot(5, int((len(file_path)-1)/5), idx+1)
    plot(time, vel_3, label=f'{component_3} Component')
    xlabel('Time (s)')
    ylabel('Vel. (m/s)')
    #axvline(time_of_quake_hr)
    title(f'{component_3} Filtered Velocity')
    grid(True)
    
    tight_layout()
    suptitle(f'Seismic Acceleration Time Series\nEvent Time: {start_time}', y=1.02)
    show()

    # Saving the processed files
    data_sismo = {
        't_sismo': time,
        'vel_sismo': vel_3,
        'ep_dist_sismo': epicenter_distance,
        'sta_name_sismo': station_code,
        'lon_sismo': station_lon,
        'lat_sismo': station_lat
        }
    
    # if not os.path.exists('./seismometers/maule_sismo/processed'):
        
    #     os.mkdir('./seismometers/maule_sismo/processed')
    
    # file_n = f"NEW_{station_code}.npy"
    # file_p = f"./seismometers/maule_sismo/processed/"


    # save(file_p + file_n, data_sismo)


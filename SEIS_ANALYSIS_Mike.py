#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 10:26:06 2023

@author: oluwasegun
"""

# from numpy import linspace, sqrt, arange, array, save, mean
# from scipy.signal import find_peaks
# from matplotlib.pyplot import *
# import matplotlib
# import glob
# import matplotlib.patches as patches
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import obspy
# from obspy.core import *
# from obspy.signal.invsim import *
# import scienceplots
# matplotlib.pyplot.style.use(['science', 'notebook', 'grid'])

# # Constants
# lat_ep = 35.7
# lon_ep = -117.6
# epicenter_distance_threshold = 5  # Threshold for epicentral distance
# figure_size = (12, 12)

# # Initialize lists to store seismic data
# t_sism = []
# vel_sism = []
# dist_ep = []
# sta_name = []

# # File list
# file_list = sorted(glob.glob('*185*.SAC'))

# # Process each file in the list
# for file_name in file_list:
#     stream = read(file_name)
#     trace = stream[0]

#     # Extract relevant information from the trace
#     station_lat = trace.stats.sac.stla
#     station_lon = trace.stats.sac.stlo
#     start_time = str(trace.stats.starttime)
#     end_time = str(trace.stats.endtime)
#     npts = trace.stats.npts

#     # Calculate epicentral distance
#     epicenter_distance = station_lat - lat_ep
#     print('Epicentral Distance:', epicenter_distance)

#     # Calculate time array
#     start_time_float = float(start_time[11:13]) + float(start_time[14:16]) / 60. + float(start_time[17:19]) / 3600.
#     end_time_float = float(end_time[11:13]) + float(end_time[14:16]) / 60. + float(end_time[17:19]) / 3600.
#     time = linspace(start_time_float, end_time_float, npts)

#     # Process seismic data
#     amplitude = trace.data
#     velocity = amplitude * 1.58997e-06 * 6.e+00 / (1. * 1500)  # m/s (V/counts) / (gain * sensitivity (V/m/s))
#     nx = arange(int(npts / 100), int(npts / 1), 1)
#     velocity = velocity[nx]
#     time = time[nx]
#     dlat = station_lat - 35.77

#     # Plot seismic data
#     fig = figure(1, figsize=figure_size, facecolor='w', edgecolor='k')
#     #ax = subplot(111)
#     plot(time, epicenter_distance + velocity, label=file_name[3:6])
#     legend()
#     title('Ground Vertical Velocity')
#     xlabel('Time (hr)')
#     ylabel('m/s')

#     # Store data in lists
#     t_sism.append(time)
#     vel_sism.append(velocity)
#     dist_ep.append(epicenter_distance)
#     sta_name.append(file_name[3:6])


# # # # Save processed data
# # save('./SISMO/t_sism.npy', array(t_sism))
# # save('./SISMO/vel_sism.npy', array(vel_sism))
# # save('./SISMO/dist_ep.npy', array(dist_ep))
# # save('./SISMO/sta_name.npy', array(sta_name))
# # savefig('ridgecrest_ground_velocity.tif', dpi=350, bbox_inches='tight')
# show()



# ---------------------- AVERAGE
from numpy import linspace, array, save, mean, arange
from scipy.signal import find_peaks
from matplotlib.pyplot import *
import matplotlib
import glob
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import obspy
from obspy.core import *
from obspy.signal.invsim import *


# Idaho: Constants
lat_ep = 23.832
lon_ep = 121.604
figure_size = (12, 8)




# Initialize lists to store seismic data
t_sism = []
vel_sism = []
dist_ep = []
sta_name = []

# File list
file_list = sorted(glob.glob('./hualien/sismo/*093*.SAC'))

# Process each file in the list
for file_name in file_list:
    stream = read(file_name)
    trace = stream[0]

    # Extract relevant information from the trace
    station_lat = trace.stats.sac.stla
    start_time = str(trace.stats.starttime)
    end_time = str(trace.stats.endtime)
    npts = trace.stats.npts

    # Calculate epicentral distance
    epicenter_distance = station_lat - lat_ep
    print('Epicentral Distance:', epicenter_distance)

    # Calculate time array
    start_time_float = float(start_time[11:13]) + float(start_time[14:16]) / 60. + float(start_time[17:19]) / 3600.
    end_time_float = float(end_time[11:13]) + float(end_time[14:16]) / 60. + float(end_time[17:19]) / 3600.
    time = linspace(start_time_float, end_time_float, npts)

    # Process seismic data
    amplitude = trace.data
    velocity = amplitude * 1.58997e-06 * 6.e+00 / (1. * 1500)  # m/s (V/counts) / (gain * sensitivity (V/m/s))
    nx = arange(int(npts / 100), int(npts / 1), 1)
    velocity = velocity[nx]
    time = time[nx]
    dlat = station_lat - lat_ep
    
    #Plot seismic data
    fig = figure(1, figsize=figure_size, facecolor='w', edgecolor='k')
    #ax = subplot(111)
    plot(time, velocity, label=file_name[3:6], color='grey')
    #plot(time, epicenter_distance + velocity, label=file_name[3:6], color='grey')
    legend()
    title('Ground Vertical Velocity')
    xlabel('Time (hr)')
    ylabel('m/s')

    # Store data in lists
    t_sism.append(time)
    vel_sism.append(velocity)
    dist_ep.append(epicenter_distance)
    sta_name.append(file_name[3:6])

# # Calculate the average values for all parameters
# avg_time = mean(array(t_sism), axis=0)
# avg_velocity = mean(array(vel_sism), axis=0)
# avg_dist_ep = mean(array(dist_ep))


# # Plot seismic data
# fig = figure(1, figsize=figure_size, facecolor='w', edgecolor='k')
# plot(avg_time, avg_dist_ep + avg_velocity, label='Average')
# legend()
# title('Average Ground Vertical Velocity')
# xlabel('Time (hr)')
# ylabel('m/s')

# # # Save processed data
# save('./SISMO/Ecuador_2021/t_sism.npy', array(t_sism))
# save('./SISMO/Ecuador_2021/vel_sism.npy', array(vel_sism))
# save('./SISMO/Ecuador_2021/dist_ep.npy', array(dist_ep))
# save('./SISMO/Ecuador_2021/sta_name.npy', array(sta_name))
#savefig('cuba_velocity.png', dpi=350, bbox_inches='tight')


# # Save processed average data
# save('./SISMO/avg_time.npy', avg_time)
# save('./SISMO/avg_velocity.npy', avg_velocity)
# save('./SISMO/avg_dist_ep.npy', avg_dist_ep)
# save('./SISMO/avg_sta_name.npy', 'Average')  # Save the label for the average station
# savefig('average_ground_velocity.tif', dpi=350, bbox_inches='tight')
show()



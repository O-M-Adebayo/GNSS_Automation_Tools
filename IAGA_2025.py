#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 07:12:23 2025

@author: oluwasegun-adebayo
"""

from pylab import *
from numpy import *
from scipy import *
from scipy.ndimage import *
from scipy.special import erf
from scipy.integrate import trapz
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import *
from signal_alam import find_peaks,wavelet,convolve_al
import matplotlib.ticker as ticker
from adebayo_codigo import calculate_distance, band_pass_filter, find_nearest_value, find_peaks_within_range_k, data_smooth_mike, compute_periodogram, stft_band_pass_filter
import glob
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cf
close('all')
#%%
params={'axes.labelweight':'bold'};rcParams.update(params)
font={'family':'serif','weight':'bold','size':11};matplotlib.rc('font', **font);

#%%
file_p = "/home/oluwasegun-adebayo/Desktop/IAGA2025/Ondrej Santolik-20250826T075932Z-1-001/Ondrej Santolik/data/whistlers_waveform_*.asc"

data = sorted(glob.glob(file_p))

n_files = len(data)
ncols = 3
nrows = int(np.ceil(n_files / ncols))  # Ensures enough rows


#%%



#%%

for idx,dt in enumerate(data):
    with open(dt, 'r') as f:
        wav = [float(line.strip()) for line in f.readlines()[11:] if line.strip()]

        #wav = f.readlines()[11:]
        
        #if idx == 6:
            #print(wav)
        #print(dt)
        sampl_rate = 4096
        num_dt_pts = 46080
        
        len_dt = len(wav)
        
        dt = 1/4096
        
        first_pt = 5 + 59/60 + 31/3600
        
        time_t = first_pt + np.arange(len_dt) * dt/3600
        
        figure(1)
        plot(time_t, wav)
        
        figure(2)
        t, mag = compute_periodogram(wav, sampling_rate=4096)
        
        plot(t, 1000*mag)
        
        figure(3)
        #ax=subplot(111)
        ax=subplot(nrows, ncols, idx + 1)
        data_ft, f, tt = stft_band_pass_filter(wav, time_t, lowcut=0, highcut=3000, dt=dt, window_length=512, overlap=128)
        #data_ft, f, tt = stft_band_pass_filter(wav, time_t, lowcut=0, highcut=3000, dt=dt, window_length=128, overlap=64)
        # Use imshow for displaying the 2D data
        im1 = ax.imshow(np.abs(data_ft), aspect='auto', extent=[0, tt.max(), (f).min(), (f).max()], 
                       cmap='jet', origin='lower')

        im1.set_interpolation("spline16")



        #text(-138, 9, file[17:24], color='white')
        #xlabel('Detection Time (mins)')
        ylabel('Frequency (Hz)')
        # xlim(0,50)
        #ylim(0,500)
        # text(45, 9, 'G27', color='w')
        # text(2.5, 8.8, '(b1)', color='w')
        # text(19, 4.9, "2.80 mHz", color='w')
        # text(11, 6.5, "3.82 mHz", color='w')
        #axis_param(im1,'') 
        #axis_param(im, 0.1, '')
        tight_layout()
           
show()
        
    
    
#%%



light_file = "/home/oluwasegun-adebayo/Desktop/IAGA2025/Ivana/*.loc"

#file_p = "/home/oluwasegun-adebayo/Desktop/IAGA2025/Ondrej Santolik-20250826T075932Z-1-001/Ondrej Santolik/data/whistlers_waveform_*.asc"

data_l = sorted(glob.glob(light_file))

# n_files = len(data_l)
# ncols = 3
# nrows = int(np.ceil(n_files / ncols))  # En


for l, light in enumerate(data_l):
    light_dt = pd.read_csv(light, header=None)
    #print(light_dt)
    #print(light_dt.head())
    
    day = light_dt[0]; tim = light_dt[1]; lat = light_dt[2]; lon = light_dt[3]; energy = light_dt[6]
    
        
    
    figure(5)
    scatter(lon, lat)
    axis((-130, 0, -20, 60))
    
    
show()
    
    
hur_p = "//home/oluwasegun-adebayo/Desktop/IAGA2025/Ivana/huricane_positions_harvey.txt"
#hur_file = sorted(glob.glob(hur_p))


# for h, hur_f in enumerate(hur_file):
#     with open(hur_f, 'r') as f:
#         hur_dt = [float(line.strip()) for line in f.readlines() if line.strip()]
hur_dt = pd.read_csv(hur_p, header=None)
lat_hur = hur_dt[1]; lon_hur = hur_dt[2]

figure(5)
scatter(lon_hur, lat_hur, c='r')


































 
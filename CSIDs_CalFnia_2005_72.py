#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  9 16:10:58 2025

@author: oluwasegun-adebayo
"""

#%%

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
from adebayo_codigo import calculate_distance, band_pass_filter, find_nearest_value
import glob
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cf
close('all')
#%%
params={'axes.labelweight':'bold'};rcParams.update(params)
font={'family':'serif','weight':'bold','size':11};matplotlib.rc('font', **font);

def find_nearest(array, value):
    array = asarray(array)
    idx = (abs(array - value)).argmin()
    return array[idx]

# Función para convertir tiempo decimal a horas y minutos
def decimal_a_horas_minutos(decimal_time):
    horas = int(decimal_time)
    minutos = int((decimal_time - horas) * 60)
    return f"{horas:2d}:{minutos:02d}"

#%%

def axis_param(im,str_c):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad = 0.05)
    cbar = colorbar(im, cax=cax);gca().set_title(str_c, fontweight='bold')
    return()

def data_smooth(t,data):
        dt=t[1]-t[0]
        n_smooth=int(ceil(10./(dt*60.)))
        win=ones((n_smooth,))/n_smooth
        data=data-convolve_al(data,win)
        return convolve_al(data,ones((5,))/5)
def data_smooth_sim(t,data):
        dt=t[1]-t[0]
        n_smooth=int(ceil(10./(dt*60.)))
        win=ones((n_smooth,))/n_smooth
        data=-data+convolve_al(data,win)
        return convolve_al(data,ones((5,))/5)  
    
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

lat_ep=41.4472;lon_ep=-125.5848
year,month,day=2005, 6, 15
hour, minute, second=2, 50, 55
t_ep=hour+minute/60+second/3600

#%%

dy=10;dx=5*dy;#dt=5*dy/2;                                                            #A resoluções espaciais no kilometros
y=arange(0,400+dy,dy);ny=len(y)  
                                         #A faixa de altitude
x=arange(0,2200+dx,dx);nx=len(x)                                        #A faixa de longitude
[y2,x2]=meshgrid(y,x)
y2_m=y2*1.e+03;x2_m=x2*1.e+03
                                             #As resoluções espaciais no metros
i_alt=abs(y-80).argmin()
yi=y[i_alt:]
it=0;nt=100;t=0;#t_f=nt*dt

t=load('./CalFornia_2005_72/time.npy')
t_original = t/60

t_hr=(t)/60.
[t2,yt2]=meshgrid(t,y)
data_amb=load('./CalFornia_2005_72/data_amb.npy');rho_amb=data_amb[:,0][0];sn=data_amb[:,1][0]
wx3=load('./CalFornia_2005_72/wx3.npy')
wy3=load('./CalFornia_2005_72/wy3.npy')
n3=1*load('./CalFornia_2005_72/n3.npy');dn3=1*(len(t)/2.)*gradient(n3)[0];
t3 = load('./CalFornia_2005_72/t3.npy'); v_new = load('./CalFornia_2005_72/v_new.npy')

#i_xo=abs(x-x.mean()).argmin()
i_xo=abs(x-0.).argmin()

#%%

fig=figure(6,figsize=(10,10),facecolor='w',edgecolor='k')
ax = subplot(221)
#
plot(sn,y,lw=2)
#text(245,280,'(A)')
grid('on', ls = '-.', lw=0.4)
xlabel('m/s')
ylabel('Altitude (km)')
title('Neutral sound speed')
ylim(0,300)

ax=subplot(222)
semilogx(t_hr[0]+1*n3[0,i_xo,:],yi,lw=2)
#text(1.23e10,280,'(B)')
title('Electron number density')
ylim(0,300)
xlabel('m$^{-3}$')
grid('on', ls = '-.', lw=0.4)

ax=subplot(223)
# v1=(1.e-03*3600.)*sn[:20].mean()
# plot(t_ep+(1000/3600.)*y/sn,y,'go',markersize=4)
data=wy3[:,i_xo,:].T
vm=data.max()
ext=[t_hr[0],t_hr[-1],y2[0,0],y2[0,-1]]
im=imshow(data,origin='lower',extent=ext,
          cmap=cm.seismic,vmin=-vm,vmax=vm)
im.set_interpolation('bilinear')
#text(6.37,380,'(C)')
axis('tight');
grid('on', ls = '-.', lw=0.4)
ylabel('Altitude (km)')
xlabel('Time (UT, hr)')
title('Acoustic-gravity wave amplitude')



#text(2.7,285,'(A)')
axis((t_hr[0],t_hr[-1],y2[0,0],y2[0,-1]))
ax.yaxis.set_ticks_position('both')
cbar=colorbar(im, cax =fig.add_axes([0.17, 0.45, 0.3, 0.01]),\
            orientation='horizontal')
gca().set_xlabel('u$_y$, m/s') 
cbar.set_ticks(linspace(-vm,vm,5))
#cbar.set_ticks(linspace(round(-vm,0),round(vm,0),5))

ax=subplot(224)
data=100*(1e+06/1.e+16)*(dn3[:,i_xo,:]/n3.max()).T
#vm=data.max()/2.
ext=[t_hr[0],t_hr[-1],y2[0,7],y2[0,-1]]
plot(t_hr[0]+0.35*n3[0,i_xo,:]/n3.max(),yi,'g',lw=2)
#text(6.37,380,'(D)')

for iy in range (7,len(yi),2):
    n2=1*n3[:,i_xo,iy]
    dt_hr=t_hr[1]-t_hr[0];n_f=int(15/(dt_hr*60.))
    win=ones((n_f,))/n_f
    dn2=+n2-convolve_al(n2,win)
    plot(t_hr,yi[iy]+0.4e+04*dn2/n3.max(),'r',lw=1)
    if iy == 11:
        plot(t_hr,yi[iy]+0.4e+04*dn2/n3.max(),'k',lw=2)
    

#axis('tight');
grid('on', ls = '-.', lw=0.4)
xlabel('Time (UT, hr)')
title('Ionospheric density disturbance')
#text(2.7,285,'(B)')
axis((t_hr[0],t_hr[-1],y2[0,0],y2[0,-1]))
ax.yaxis.set_label_position("left")
ax.yaxis.tick_left()
ax.yaxis.set_ticks_position('both')
tight_layout()




#%%


# Collect maximum amplitudes for each ix
max_amps_sism = []
x_vals_sism = []

fig=figure(33,figsize=(8,8),facecolor='w',edgecolor='k')


#ax =subplot(1,1,1)
for ix in range (0,len(x),1):
    # wy3[:,i_xo,0]
    #pcolormesh(t3[:,0,0],x,v_new[:,:,0].T)
    
    #v_new[100:,ix,0] = 0
    ax =subplot(1,2,1)
    x_dt = (t3[:,0,0]/3600)+2.85
    y_dt = (x[ix]) + (100*v_new[:,ix,0])

    plot(x_dt, y_dt, color='g', label=f'x={x[ix]:.2f}')
    
    # max_amps_sism.append(abs(v_new[:, ix, 0]).max())
    # x_vals_sism.append(x[ix])

    
    ax =subplot(1,2,2)
    #for ix in range (0,len(x),1):   
    # delta_h = 2.5e4  # 10 km in meters
    # scale_factor = delta_h / 1e16  # converts to TECU
    # tec = scale_factor * dn3[:, ix, 7:23].sum(1)
    tec=(1e+06/1.e+16)*dn3[:,ix,7:23].sum(1)
    dtec=data_smooth_sim(t_hr,tec)
    
    # max_amps_tec_1.append(abs(dtec).max())
    # x_vals_tec_1.append(x[ix])
    nm=1

    vm=dtec.max()
    
    #xlim(t_hr[0],t_hr[-1])
    im=scatter(t_hr,0.1*dtec+(x[ix]),c=0.6*dtec,s=20.,
                vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow)   

    # if ix==0:
    #     v1=(4/1)/(1./3600.)
    #     axline((t_ep+420/3600,0),slope=v1,c='k' ,label = '~4 km/s')
    #     v1=(1/1)/(1./3600.)
    #     axline((t_ep+420/3600,0),slope=v1,c='g' ,label = '~1 km/s')
    #     #legend()
    #     axvline(x=t_ep,color='grey',lw=1,linestyle='--') 












#%%

f = np.loadtxt("./igrfgridData.csv", skiprows=13, delimiter=',')
years = f[:,0]; lat_igr = f[:,1]; lon_igr = f[:,2]; incl = f[:,4]




#------------------
# Create the plot
fig = plt.figure(30, figsize=(8, 8))
ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
#ax.stock_img()
# Set the extent of the map (latitude and longitude range)
ax1.set_extent([-128, -111, 34, 50])
ax1.add_feature(cf.COASTLINE, lw = 2)
ax1.add_feature(cf.BORDERS, lw = 0.8)
ax1.add_feature(cf.LAND)
ax1.add_feature(cf.STATES)
#ax.add_feature(cf.OCEAN)
#ax.add_feature(cf.RIVERS)
#ax.add_feature(cf.LAKES)

# Add borders with different linewidth for Brazil
#ax.add_feature(cf.BORDERS.with_scale('10m'), linestyle='-', edgecolor='red', linewidth=2)

# Add features to the map
ax1.coastlines()
# gline = ax.gridlines(linewidth=0.001)
# gline.xlabels_bottom = True
# gline.ylabels_left = True


# Add gridlines with labels and proper formatting
gl = ax1.gridlines(draw_labels=True,
                  linewidth=0.01,
                  color='gray',
                  
                  linestyle='--')

gl.top_labels = False
gl.right_labels = False
# gl.xformatter = cticker.LongitudeFormatter()
# gl.yformatter = cticker.LatitudeFormatter()
# gl.xlabel_style = {'size': 9, 'color': 'black'}
# gl.ylabel_style = {'size': 9, 'color': 'black'}



# Plot the dip equator (assuming lon_igr, lat_igr, incl are defined)
ax1.tricontour(lon_igr, lat_igr, incl, [0], linestyles='dashed', colors='k')









#%%

sism_list=sorted(glob.glob('./CalFornia_2005_72/sismo_more/*.npy'));#0937# 0261#0205
for idx, sism in enumerate(sism_list):  
    data_sismo = load(sism, allow_pickle=True).item()
    t_sismo = data_sismo['t_sismo']; vel_sismo = data_sismo['vel_sismo'];
    sta_name_sismo = data_sismo['sta_name_sismo'];
    lon_sismo = data_sismo['lon_sismo']; lat_sismo = data_sismo['lat_sismo']
    
    # Calculating teleseismic distance
    [d1, d2] = calculate_distance(lon_ep, lat_ep, lon_sismo, lat_sismo)
    #dist_sismo=((d1+d2)[0]/2)/1.
    dist_sismo = (d1 + d2) / 2
    
    vel_sismo_res = vel_sismo[::50]
    t_sismo_res = t_sismo[::50]
    
    peak_mx_raw = find_peaks_within_range_k(abs(vel_sismo_res), 100, 700, 0.00001, 1)
    
    figure(91, figsize=(8, 6))
    subplot(211)
    # vel_sismo_res[vel_sismo_res > 0.0068] = 0
    # bar(dist_sismo+abs(vel_sismo_res[peak_mx_raw]), 100*abs(vel_sismo_res[peak_mx_raw]), 
    #     color='g', width = 10)
    if 535 < dist_sismo < 600:
        continue
    bar(dist_sismo, 100*(vel_sismo_res.max()), color='g', width = 10)
    # if dist_sismo <1500:
    #     text(dist_sismo, 100*(vel_sismo_res.max()), 
    #          s=sta_name_sismo, rotation=90, fontsize=6)
    xlim(0, 1500)
    
    #axis((t_gnss[dtec_peak_obs][0],t_gnss[dtec_peak_obs][-1],0.,2300))
    ylabel('PGV (m/s)')
    #xlabel('Epicentral Distance (km)')   
    title('Seismic Data', fontweight = 'bold')
    grid(linestyle='-.', lw=0.4)
    if sta_name_sismo == 'KRP.':
        text(1400, 0.67, '(A)')
    
    vel_sismo_filt = band_pass_filter(vel_sismo_res, t_sismo_res, lowcut=0.011, highcut=0.04, dt=1)
    
    vel_long_period = band_pass_filter(vel_sismo, t_sismo, lowcut=0.004, highcut=0.02, dt=1)
    
    # figure(41)
    # plot(vel)
  
    # #peak_mx = abs(10*vel_sismo).argmax()
    
    # sismo_peaks_mx.append(peak_mx)
    # sismo_dist.append(dist_sismo)
    
    #if sta_name_sismo not in ['LVC.', 'PB09', 'PB11', 'EFI.']:
        
    #print(t_sismo[1] - t_sismo[0])
    
    figure(4, figsize=(10,8))
    ax=subplot(121)
    # if dist_sismo >=590 and dist_sismo <= 600:
    #     continue
    # else:
    #if len(vel_sismo) < 
    
    #plot(t_sismo, dist_sismo + (9000*vel_sismo), color='grey')
    
    #vm=0.1
    vel_sismo_amp = band_pass_filter(vel_sismo, t_sismo, lowcut=0.004, highcut=0.02, dt=1)
   # plot(t_gnss,ep_dist+250*dtec_gnss,lw=2)
    if 535 < dist_sismo < 600:
        continue
    # im2=scatter(t_sismo[::100],dist_sismo+(5000*vel_sismo_amp[::100]),c=vel_sismo_amp[::100],s=20,
    #             marker='.',cmap=cm.rainbow)
    
    plot(t_sismo[::100],dist_sismo+(7000*vel_sismo_amp[::100]), lw=2, color='g')
    
    axis((2.7, 3.3, 0, 1500))
    axvline(t_ep)
    xlabel("Time (UT, hr)")
    ylabel("Epicentral Distance (km)")
    grid('on', ls = '-.', lw=0.4)
    title("Seismic Data", fontweight = 'bold')
    # if dist_sismo <= 1500:
    #     text(3.3, dist_sismo, sta_name_sismo)
        
    if sta_name_sismo == 'NV31':
        plot([2.859, 3.007], [111, 1483], 
             ls='-', lw=1, color='k', label='~2.60 km/s')
        
        legend(loc='lower right')
        #f = axis_param(im2, "m/s")
    
    figure(3)
    if sta_name_sismo in ['JCC.']:
        plot(t_sismo, vel_sismo, label=sta_name_sismo)
        #save('./CalFornia_2005_72/jcc.npy',array([t_sismo, vel_sismo]))
        legend()
    # if dist_sismo < 1500:
        
    #     text(3.3, dist_sismo, sta_name_sismo, fontsize=8)
        
        
    # if sta_name_sismo == 'EL1.':
    #     v1=(4/1)/(1./3600.)
    #     axline((t_ep+20/3600,0),slope=v1,c='k' ,label = '~4 km/s')
        
    #     #text(6.68, 1250, "v$_{R}$ = 3.38 km/s")
    
    tight_layout()
    
    
    
    # ---------------------AMP distributions
    vel_sismo_filt[vel_sismo_filt <= 0.0000035] = 0
    
    vel_p = find_peaks_within_range_k(abs(vel_sismo_filt), 0, len(vel_sismo_filt), 0.00003, 1)
    #vel_p = find_peaks_within_range_obs(vel_sismo_filt, 0, len(vel_sismo_filt), 0.00003)
    
    # 80, 250, 0.015
    
    # time_obs = (t_gnss[dtec_peak_obs] - 6.705)*3600
    # dist_obs = (lat_gnss[dtec_peak_obs] + 33.762)*110
    # vel_obs = (dist_obs) / (time_obs)
    
    
    
    # figure(91, figsize=(6, 8))
    # subplot(312)
    
    
    # # scatter(ep_dist+150*dtec_gnss[dtec_peak_obs], dtec_gnss[dtec_peak_obs], 
    # #         s=50*dtec_gnss[dtec_peak_obs], c=(t_gnss[dtec_peak_obs]-t_ep)*60)
    # # bar(dist_sismo+vel_sismo_filt[vel_p], 10*vel_sismo_filt[vel_p], 
    # #     color='g', width = 10)
    
    # bar(dist_sismo, 10*vel_sismo_filt.max(), color='g', width = 10)
    
    
    
    # xlim(0, 1500)
    
    # #axis((t_gnss[dtec_peak_obs][0],t_gnss[dtec_peak_obs][-1],0.,2300))
    # ylabel('PGV_Infra (m/s)')
    # #xlabel('Epicentral Distance (km)')   
    # #title('Observation', fontweight = 'bold')
    # grid(linestyle='-.', lw=0.4)
    
    # if sta_name_sismo == 'KRP.':
    #     text(1400, 0.035, '(B)')
    
    # if idx == 0:
        
    #     text(62, 0.125, "(B)", fontsize=15)
    
    # ax=subplot(212)
    # bar((t_gnss[dtec_peak_obs]-t_ep)*60, dtec_gnss[dtec_peak_obs], 
    #     color='red', width = 0.2)
    
    # ylabel('dTEC (TECU)')
    # xlabel('Detection time (mins)')   
    # #title('Observation', fontweight = 'bold')
    # grid(linestyle='-.', lw=0.4)
    # text(0.65, 0.96, "(b)", fontsize=15)
    
    #----------------------------
    
tight_layout()
#%%    
# import glob
# import numpy as np
# import matplotlib.pyplot as plt
# from numpy import load

# --- Load All Seismometer Data ---
sism_list = sorted(glob.glob('./CalFornia_2005_72/sismo_more/*.npy'))

stations = []

for sism in sism_list:  
    data_sismo = load(sism, allow_pickle=True).item()
    
    fig = plt.figure(30, figsize=(8, 8))
    ax1.plot(data_sismo['lon_sismo'], data_sismo['lat_sismo'], 
             marker= "^", markersize=5, color='grey', 
             transform=ccrs.PlateCarree())
    
    #ax1.plot([-119.3, -127.9], [41.4472, 41.4472], color='y', ls='--', marker='4', transform=ccrs.PlateCarree())
    ax1.arrow(-130.9, 41.4472, 13, 0, head_width=0.2, head_length=0.3, fc='red', ec='red')
    
    # Calculating teleseismic distance
    [d1, d2] = calculate_distance(lon_ep, lat_ep, data_sismo['lon_sismo'], data_sismo['lat_sismo'])
    #dist_sismo=((d1+d2)[0]/2)/1.
    dist_sismo = (d1 + d2) / 2
    
    station = {
        't': data_sismo['t_sismo'],
        'v': data_sismo['vel_sismo'],
        'name': data_sismo['sta_name_sismo'],
        'lon': data_sismo['lon_sismo'],
        'lat': data_sismo['lat_sismo'],
        'ep_dist_s': dist_sismo
    }
    stations.append(station)

    


# --- Determine Global Max Amplitude for Scaling ---
all_amplitudes = np.concatenate([np.abs(sta['v']) for sta in stations])
max_amp = np.max(all_amplitudes)

# --- Plot ---
# #fig, ax = plt.subplots(1, figsize=(12, 10))
# fig=figure(1,figsize=(10,6))


# scale_amp = 0.01  # vertical exaggeration
# scale_time = 1  # horizontal scaling

# for sta in stations:
#     t = sta['t']
#     v = sta['v']
#     lon = sta['lon']
#     lat = sta['lat']
#     dis = sta['ep_dist_s']
#     nm = sta['name']
    
#     # Normalize and scale
#     v_scaled = (v / max_amp) * scale_amp
#     t_scaled = (t - np.mean(t)) * scale_time
    
#     x = (lon-lon_ep) + t_scaled
#     y = (lat-lat_ep) + (200*v_scaled)
    
#     if dis <= 1500:
#         ax=subplot(111)
#         plot(x, y, color='black', linewidth=0.8)
#         #ax.plot(lon, lat, 'ro', markersize=2)
#         plot(0, 0, marker='*', color='red', markersize=15)
#         if (-1 < x[0] < 18) and (-10 < y[0] < 7.5):
#             ax.text(x[0], y[0] + 0.02, nm, fontsize=6, ha='center')

#         #text(x, y, )

# # --- Format Plot ---
# ax.set_xlabel("Epicentral Longitude ($^{0}$)")
# ax.set_ylabel("Epicentral Latitude ($^{0}$)")
# ax.set_title("Vertical Ground Velocity", fontweight = 'bold')
# ax.grid(ls = '-.', lw=0.4)
# #axis((-5, 15, -10, 10))
# tight_layout()







from math import radians, cos, sin

# # Function to plot great circles
# def plot_great_circle(ax, lon_ep, lat_ep, radius_km, interval_km=50):
#     """
#     Plots circles on a map at intervals of `interval_km` around a given epicenter.
#     :param ax: Axes object where the plot is drawn
#     :param lon_ep: Longitude of the epicenter
#     :param lat_ep: Latitude of the epicenter
#     :param radius_km: Maximum radius for the circles in kilometers
#     :param interval_km: Interval between circles in kilometers
#     """
#     for r in range(interval_km, radius_km + interval_km, interval_km):
#         # Approximate the circle based on the great circle distance
#         # Earth's radius in kilometers
#         earth_radius_km = 6371

#         # Angular distance in radians
#         angle = r / earth_radius_km

#         # Generate points around the epicenter for the circle
#         lons = []
#         lats = []
#         num_points = 100  # Number of points to approximate the circle

#         for i in range(num_points):
#             theta = 2 * np.pi * i / num_points
#             lat_new = lat_ep + (angle * 180 / np.pi) * sin(theta)
#             lon_new = lon_ep + (angle * 180 / np.pi) * cos(theta) / np.cos(np.radians(lat_ep))
            
#             # Wrap around longitude
#             lon_new = (lon_new + 180) % 360 - 180
            
#             lats.append(lat_new)
#             lons.append(lon_new)

#         ax.plot(lons, lats, color='blue', linestyle='--', linewidth=1)

# # Create figure
# fig = plt.figure(1, figsize=(10, 6))

# # Scale variables
# scale_amp = 0.01  # vertical exaggeration
# scale_time = 1  # horizontal scaling

# # Define epicenter longitude and latitude (replace with actual values)
# # lon_ep = 0  # Example epicenter longitude
# # lat_ep = 0  # Example epicenter latitude

# # Plot great circles with intervals
# radius_km = 2000  # Maximum radius for circles (500 km for example)
# plot_great_circle(plt, lon_ep, lat_ep, radius_km, interval_km=200)

# # Plot stations
# for sta in stations:
#     t = sta['t']
#     v = sta['v']
#     lon = sta['lon']
#     lat = sta['lat']
#     dis = sta['ep_dist_s']
#     nm = sta['name']

#     # Normalize and scale
#     v_scaled = (v / max_amp) * scale_amp
#     t_scaled = (t - np.mean(t)) * scale_time
    
#     # Calculate x and y for station location
#     x = lon + t_scaled
#     y = lat + (200 * v_scaled)
    
#     # x = (lon - lon_ep) + t_scaled
#     # y = (lat - lat_ep) + (200 * v_scaled)
    
#     if dis <= 1500:
#         ax = plt.subplot(111)
#         plt.plot(x, y, color='black', linewidth=0.8)
#         plt.plot(lon_ep, lat_ep, marker='*', color='red', markersize=15)

#         if (-1 < x[0] < 18) and (-10 < y[0] < 7.5):
#             ax.text(x[0], y[0] + 0.02, nm, fontsize=6, ha='center')

# # --- Format Plot ---
# ax.set_xlabel("Epicentral Longitude ($^{0}$)")
# ax.set_ylabel("Epicentral Latitude ($^{0}$)")
# ax.set_title("Vertical Ground Velocity", fontweight='bold')
# ax.grid(ls='-.', lw=0.4)
# ax.set_xlim(-127, -105)
# ax.set_ylim(30,45)

# plt.tight_layout()
# plt.show()




#--------------------------------------------------

from math import radians, cos, sin

# Function to plot great circles with distance labels
def plot_great_circle(ax, lon_ep, lat_ep, radius_km, interval_km=50):
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

        # Add distance label (beside the circle to the east)
        label_lat = lat_ep  # Keep latitude the same
        label_lon = lon_ep + (angle * 180 / np.pi) / np.cos(np.radians(lat_ep))  # Offset longitude
        
        ax.text(label_lon, label_lat, f'{r} km', fontsize=10, fontweight='bold',
                rotation=-90, ha='left', va='center', color='grey')


# Create figure
fig = plt.figure(1, figsize=(10, 6))

# Scale variables
scale_amp = 0.01  # vertical exaggeration
scale_time = 1  # horizontal scaling

# Plot great circles with intervals
radius_km = 1400  # Maximum radius for circles (e.g., 2000 km)
plot_great_circle(plt, lon_ep, lat_ep, radius_km, interval_km=200)




# Plot stations
for sta in stations:
    t = sta['t']
    v = sta['v']
    lon = sta['lon']
    lat = sta['lat']
    dis = sta['ep_dist_s']
    nm = sta['name']

    # Normalize and scale
    v_scaled = (v / max_amp) * scale_amp
    t_scaled = (t - np.mean(t)) * scale_time
    
    # Calculate x and y for station location
    x = lon + t_scaled
    y = lat + (200 * v_scaled)
    
    if dis <= 1500:
        ax = plt.subplot(111)
        plt.plot(x, y, color='g', linewidth=0.8)
        plt.plot(lon_ep, lat_ep, marker='*', color='red', markersize=15)

        if (-1 < x[0] < 18) and (-10 < y[0] < 7.5):
            ax.text(x[0], y[0] + 0.02, nm, fontsize=6, ha='center')

circle1 = plt.Circle((-118.39, 38.47), 1.5, fill=False, color='r')
circle2 = plt.Circle((-116.63, 33.62), 1.2, fill=False, color='r')
ax.add_patch(circle1)
ax.add_patch(circle2)
# --- Format Plot ---
ax.set_xlabel("GLON ($^{0}$)")
ax.set_ylabel("GLAT ($^{0}$)")
ax.set_title("Vertical Ground Velocity", fontweight='bold')
#ax.grid(ls='-.', lw=0.4)
ax.set_xlim(-127, -108.5)
ax.set_ylim(32, 45)


plt.tight_layout()
plt.show()




    
    
#%%

f_list=sorted(glob.glob('CalFornia_2005_72/tec/***.npy'));#0937# 0261#0205
for idx, f_n in enumerate(f_list):  
    
    data_gnss=load(f_n,allow_pickle=True).item()
    if idx ==0:
        print(data_gnss.keys())
    t_gnss=data_gnss['t_gnss'];tec_gnss=data_gnss['tec_gnss'];
    vtec_gnss = data_gnss['vtec_gnss']; ep_dist = data_gnss['ep_dist'];
    lon_gnss=data_gnss['lon_gnss'];lat_gnss=data_gnss['lat_gnss'];
    lon_rec = data_gnss['lon_rec']; lat_rec = data_gnss['lat_rec']
    elev_ang = data_gnss['elev_ang']; azim_ang = data_gnss['azim_ang']
    
    v_prox = find_nearest_value(t_gnss, t_ep)
    p_sip = where(t_gnss == v_prox)[0]
    
    # v_prox=find_nearest(t_gnss, t_ep)
    # p_sip=where(t_gnss==v_prox)[0]
    # ep_sign=1
    # if lon_gnss[p_sip]-lon_ep <0:
    #     ep_sign=1
    # if lat_gnss[p_sip]-lat_ep <0:
    #     ep_sign=-1
    # [d1,d2]=distkm(lon_ep,lat_ep,([lon_gnss[p_sip]]),([lat_gnss[p_sip]]))
    # ep_dist=((d1+d2)[0]/2)/1.
    dtec_gnss=data_smooth_mike(t_gnss,tec_gnss, 3, 6)
    
       
    #dtec_gnss=data_smooth(t_gnss,tec_gnss)
    if "G08" in f_n[22:33] and max(dtec_gnss[150:250]) > 0.008:
        #print(f_n[22:33])
        fig=figure(4,figsize=(10,8),facecolor='w',edgecolor='k')
        ax=subplot(122)
        vm=0.03
       # plot(t_gnss,ep_dist+250*dtec_gnss,lw=2)
        im=scatter(t_gnss,ep_dist+1*dtec_gnss,c=0.6*dtec_gnss,s=20,
                    vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow)
        
        # if ep_dist < 1500:
    
        #     text(3.25,ep_dist,f_n[22:29],size='small',color='k')
    
        axvline(t_ep)
        # v1=(3./111)/(1./3600.)
        # axline((t_ep+8/60,0),slope=v1,c='g' ,label = '~1 km/s')
        axis((2.7, 3.4,0.,1500))
       # f=  (axis_param(im,vm,4.4,'TECU'))
        #ylabel('Epicentral Distance (km)')
        xlabel('Time (UT, hr)')
        
        
        
            
    #dtec_gnss=data_smooth(t_gnss,tec_gnss)
    if "G10" in f_n[22:33]:
        #print(f_n[22:33])
        #fig=figure(4,figsize=(8,8),facecolor='w',edgecolor='k')
        fig=figure(4,figsize=(10,8),facecolor='w',edgecolor='k')
        ax=subplot(122)
        vm=0.05
       # plot(t_gnss,ep_dist+250*dtec_gnss,lw=2)
        im1=scatter(t_gnss,ep_dist+1*dtec_gnss,c=0.6*dtec_gnss,s=20,
                    vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow)
    
        # if ep_dist < 1500:
        #     text(3.25,ep_dist,f_n[22:29],size='small',color='k')
        #text(6.35,ep_dist,f_n[9:16],size='small',color='k')
    
        #axvline(t_ep,0,1,c='grey',linestyle='--')
        # v1=(3./111)/(1./3600.)
        # axline((t_ep+8/60,0),slope=v1,c='g' ,label = '~1 km/s')
        axis((2.7, 3.4,0.,1500))
       # f=  (axis_param(im,vm,4.4,'TECU'))
        #ylabel('Epicentral Distance (km)')
        xlabel('Time (UT, hr)')
        title("TEC Data", fontweight = 'bold')
        
        if 'p589' in f_n[22:33]:
            plot([3.001, 3.116], [191, 732], 
                   ls='--', lw=1, color='k', label='~1.32 km/s')
            plot([3.027, 3.089], [616, 1410], 
                   ls='-', lw=1, color='k', label='~2.55 km/s')
            legend(loc='lower right')
            f = axis_param(im1, "TECU")
        tight_layout()
        
            
    if ("G10" in f_n[22:33] or "G08" in f_n[22:33]) and ep_dist < 1500:
        #print(f_n[22:33])
        #fig=figure(4,figsize=(8,8),facecolor='w',edgecolor='k')
        figure(10, figsize=(10,10))
        subplot(122)
        vm=0.03
       # plot(t_gnss,ep_dist+250*dtec_gnss,lw=2)
        im1=scatter(t_gnss,ep_dist+dtec_gnss,c=0.6*dtec_gnss,s=20,
                    vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow)
    
        #text(6.35,ep_dist,f_n[9:16],size='small',color='k')
    
        axvline(t_ep,0,1,c='grey',linestyle='--')
        # v1=(3./111)/(1./3600.)
        # axline((t_ep+8/60,0),slope=v1,c='g' ,label = '~1 km/s')
        axis((2.7, 3.4,0.,1500))
       # f=  (axis_param(im,vm,4.4,'TECU'))
        ylabel('Epicentral Distance (km)')
        xlabel('Time (UT, hr)')
        title("Observation")
        
        # if idx == 1:
        #     f = axis_param(im1, "TECU")
            
        #------------------------------------------------------
        
        
        ###### SIMULATION #####
        
        
        
        #fig=figure(10,figsize=(10,8),facecolor='w',edgecolor='k')
        subplot(121)
        #plot(t_gnss,ep_dist+250*dtec_gnss,'r',lw=1)
        
        
        # i_ep = abs(x - ep_dist).argmin()
        # i_ep = min(i_ep, n3.shape[1] - 1)  # prevent out-of-bounds

        
        # tec=(1e+06/1.e+16)*dn3[:,ix,7:23].sum(1)
        # dtec=data_smooth_sim(t_hr,tec)
        
        # # max_amps_tec_1.append(abs(dtec).max())
        # # x_vals_tec_1.append(x[ix])
        # nm=1

        # vm=dtec.max()
        
        # #xlim(t_hr[0],t_hr[-1])
        # im=scatter(t_hr,0.1*dtec+(ep_dist),c=0.6*dtec,s=20.,
        #             vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow)   









        # delta_h = 2.5e4  # 10 km in meters
        # scale_factor = delta_h / 1e16  # converts to TECU
        # tec = scale_factor * n3[:, idx, 7:23].sum(1)
        # #tec=(1e+06/1.e+16)*dn3[:,i_ep,7:23].sum(1)
        # #dtec=data_smooth_sim(t_hr,tec)
        
        # #tec=(1e+06/1.e+16)*dn3[:,idx,7:23].sum(1)
        # dtec=data_smooth_sim(t_hr,tec)
        
        # # max_amps_tec_1.append(abs(dtec).max())
        # # x_vals_tec_1.append(x[ix])
        # nm=1

        # vm=dtec.max()/2
        
        # #xlim(t_hr[0],t_hr[-1])
        # im=scatter(t_hr,0.2*dtec+(ep_dist),c=0.6*dtec,s=20.,
        #             vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow)   

        
        # #%%


        # # dtec=data_smooth_sim(t_hr,tec)
        # # vm = dtec.max()/2 
        # # #plot(t_hr,ep_dist+50*(dtec),'k',lw=1.5)
        
        # # # filtered
        # # scatter(t_hr,ep_dist+(300*dtec),c=dtec,s=20,
        # #             vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow)
        
        ylim(0,1500)
        # #axis((2.7, 3.4,0.,1500))
        
       
      
    
    
    
    # END
    
    fig = plt.figure(30, figsize=(8, 8))
    # ax1.text(-55, -8, 'Brazil', fontweight='bold', fontsize=12)
    # ax1.text(-73.2, -41, 'Chile', fontweight='bold', fontsize=8,
    #        rotation = 85)

    # Seismometers
    #ax.plot(-79, 0.24, "^", markersize=8, color='green')   # OTAV
    # # Plot specific points
    ax1.plot(-125.5848, 41.4472, "*", markersize=20, color='red')   # Chile



    # Plot satellite trajectories


    #for i in range(traj_lon04.shape[0]):
    #if (idx%4) == 0:
        
    if (idx%4) == 0:
        
        ax1.plot(lon_gnss, lat_gnss, '.', markersize=1, color='magenta',
                transform=ccrs.PlateCarree())
        ax1.plot(lon_gnss[p_sip], lat_gnss[p_sip], 'D', markersize=3, color='cyan',
                transform=ccrs.PlateCarree())
        ax1.text(lon_gnss[0], lat_gnss[0], s = f_n[26:29], fontsize = 8,
                transform=ccrs.PlateCarree())
        
    # Plot receiver locations
    ax1.plot(lon_rec, lat_rec, 'o', markersize=5, color='blue', transform=ccrs.PlateCarree())
    
tight_layout()   
    
#%%

















#%%



f_list=sorted(glob.glob('CalFornia_2005_72/tec/***.npy'));#0937# 0261#0205
for idx, f_n in enumerate(f_list):  
    
    data_gnss=load(f_n,allow_pickle=True).item()
    
    t_gnss=data_gnss['t_gnss'];tec_gnss=data_gnss['tec_gnss'];
    vtec_gnss = data_gnss['vtec_gnss']; ep_dist = data_gnss['ep_dist'];
    lon_gnss=data_gnss['lon_gnss'];lat_gnss=data_gnss['lat_gnss'];
    lon_rec = data_gnss['lon_rec']; lat_rec = data_gnss['lat_rec']
    elev_ang = data_gnss['elev_ang']; azim_ang = data_gnss['azim_ang']
    
    v_prox = find_nearest_value(t_gnss, t_ep)
    p_sip = where(t_gnss == v_prox)[0]
    
    #dtec_gnss=band_pass_filter(tec_gnss, t_gnss)
    
    dtec_gnss=data_smooth_mike(t_gnss,tec_gnss, 3, 6)
    
    
    vm = 0.005
    # figure(19)
    # #dtec_gnss=data_smooth(t_gnss,tec_gnss)
    if "G10" in f_n[22:33] or "G08" in f_n[22:33]:
    #     #print(f_n[22:33])
    #     #fig=figure(4,figsize=(8,8),facecolor='w',edgecolor='k')
    #     #fig=figure(4,figsize=(10,8),facecolor='w',edgecolor='k')
    #     #ax=subplot(122)
    #     #print(len(dtec_gnss))
    #     #dtec_gnss_amp = dtec_gnss
        
        vm=0.06
        figure(99)
        subplot(211)
       # plot(t_gnss,ep_dist+250*dtec_gnss,lw=2)
        im1=scatter(t_gnss[150:200],ep_dist+100*dtec_gnss[150:200],c=0.6*dtec_gnss[150:200],s=20,
                    vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow)
    
        #text(6.35,ep_dist,f_n[9:16],size='small',color='k')
    
        axvline(t_ep,0,1,c='grey',linestyle='--')
        # v1=(3./111)/(1./3600.)
        # axline((t_ep+8/60,0),slope=v1,c='g' ,label = '~1 km/s')
        axis((2.7, 3.4,0.,1500))
       # f=  (axis_param(im,vm,4.4,'TECU'))
        ylabel('Epicentral Distance (km)')
        xlabel('Time (UT, hr)')
        title("Observation")
        
        if idx == 1:
            f = axis_param(im1, "TECU")
            
        
        #dtec_gnss[dtec_gnss < 0.0007] = 0
        #dtec_peaks = find_peaks_within_range_k(dtec_gnss, 150, 200, 0.01, 1)
        #dtec_peaks = argmax(abs(dtec_gnss[150:200]))
        
        # figure(1)
        # #vm1=0.01
        # subplot(212)
        # im = scatter((lon_gnss[dtec_peaks]-lon_ep),(lat_gnss[dtec_peaks]-lat_ep)+1*dtec_gnss[dtec_peaks],
        #              c=1*dtec_gnss[dtec_peaks],s=50, marker='o',cmap=cm.jet)
        
        # # im = scatter((lon_gnss[dtec_peaks]-lon_ep),(lat_gnss[dtec_peaks]-lat_ep)+1*dtec_gnss[dtec_peaks],
        # #              c=0.6*dtec_gnss[dtec_peaks],s=50, vmax=vm1,vmin=-vm1,marker='o',cmap=cm.jet)
        # plot(0, 0, "*", markersize = 15, color='r')
        # grid(ls = '-.', lw=0.4)
        # #axis((-10, 10, -15, 15))
        # title("TEC Data", fontweight='bold')
            
    #if "G10" in f_n[22:33]:
        #print(f_n[22:33])
        #fig=figure(4,figsize=(8,8),facecolor='w',edgecolor='k')
        # figure(10, figsize=(10,10))
        # subplot(122)
       #  vm=0.05
       # # plot(t_gnss,ep_dist+250*dtec_gnss,lw=2)
       #  im1=scatter(t_gnss,ep_dist+dtec_gnss,c=0.6*dtec_gnss,s=20,
       #              vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow)
    
       #  #text(6.35,ep_dist,f_n[9:16],size='small',color='k')
    
       #  axvline(t_ep,0,1,c='grey',linestyle='--')
       #  # v1=(3./111)/(1./3600.)
       #  # axline((t_ep+8/60,0),slope=v1,c='g' ,label = '~1 km/s')
       #  axis((2.7, 3.4,0.,1500))
       # # f=  (axis_param(im,vm,4.4,'TECU'))
       #  ylabel('Epicentral Distance (km)')
       #  xlabel('Time (UT, hr)')
       #  title("Observation")
        
       #  if idx == 1:
       #      f = axis_param(im1, "TECU")
            
    #     #------------------------------------------------------
    #     if ep_dist <= 1500: 
    #         dte = data_smooth(t_gnss, tec_gnss)
    #         dte[dte<0.02]=0
    #         fig=figure(1,figsize=(10,6),facecolor='w',edgecolor='k')
    #         ax=subplot(212)
    #         # plot(t_gnss,ep_dist+250*dtec_gnss,lw=2)
    #         scatter((lon_gnss-lon_ep),(lat_gnss-lat_ep)+5*dte,c=0.6*dte,s=20,
    #                     vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow)
    
    # axis((-5, 15, -10, 10))
    # tight_layout()
    
    # v_prox=find_nearest(t_gnss, t_ep)
    # p_sip=where(t_gnss==v_prox)[0]
    # ep_sign=1
    # if lon_gnss[p_sip]-lon_ep <0:
    #     ep_sign=1
    # if lat_gnss[p_sip]-lat_ep <0:
    #     ep_sign=-1
    # [d1,d2]=distkm(lon_ep,lat_ep,([lon_gnss[p_sip]]),([lat_gnss[p_sip]]))
    # ep_dist=((d1+d2)[0]/2)/1.
    
    dtec_gnss=data_smooth_mike(t_gnss,tec_gnss, 3, 6)
    #------------------ TEC AMPs
    
    #if "G10" in f_n[22:33]:
        
    dtec_gnss[dtec_gnss <= 0.005] = 0
    
    dtec_peak_obs = find_peaks_within_range_k(dtec_gnss, p_sip[0], p_sip[0] + 120, 0.015, 1)
    
    # 80, 250, 0.015
    
    # time_obs = (t_gnss[dtec_peak_obs] - 6.705)*3600
    # dist_obs = (lat_gnss[dtec_peak_obs] + 33.762)*110
    # vel_obs = (dist_obs) / (time_obs)
    
    
    
    figure(91, figsize=(8,6))
    subplot(212)
    
    dtec_gnss[dtec_gnss > 0.1] = 0
    ## scatter(ep_dist+150*dtec_gnss[dtec_peak_obs], dtec_gnss[dtec_peak_obs], 
    ##         s=50*dtec_gnss[dtec_peak_obs], c=(t_gnss[dtec_peak_obs]-t_ep)*60)
    bar(ep_dist+dtec_gnss[dtec_peak_obs], dtec_gnss[dtec_peak_obs], 
        color='red', width = 10)
    
    # bar(ep_dist, abs(dtec_gnss).max(), color='r', width = 10)
    # if dist_sismo <1500:
    #     text(ep_dist, abs(dtec_gnss).max(), 
    #          s=f_n[20:23], rotation=90, fontsize=6)
    #xlim(0, 1500)
    
    if 'p262G08' in f_n:
        text(1400, 0.09, '(B)')
    
    # elif "G08" in f_n[22:33]:
    #     dtec_gnss[dtec_gnss <= 0.005] = 0
        
    #     dtec_peak_obs = find_peaks_within_range_k(dtec_gnss, p_sip[0], p_sip[0] + 100, 0.008, 5)
        
    #     # 80, 250, 0.015
        
    #     # time_obs = (t_gnss[dtec_peak_obs] - 6.705)*3600
    #     # dist_obs = (lat_gnss[dtec_peak_obs] + 33.762)*110
    #     # vel_obs = (dist_obs) / (time_obs)
        
        
        
    #     fig=figure(91,figsize=(8,6),facecolor='w',edgecolor='k')
    #     ax=subplot(211)
        
        
    #     # scatter(ep_dist+150*dtec_gnss[dtec_peak_obs], dtec_gnss[dtec_peak_obs], 
    #     #         s=50*dtec_gnss[dtec_peak_obs], c=(t_gnss[dtec_peak_obs]-t_ep)*60)
    #     bar(ep_dist+dtec_gnss[dtec_peak_obs], dtec_gnss[dtec_peak_obs], 
    #         color='red', width = 10)
    
    xlim(0, 1500)
    
    #axis((t_gnss[dtec_peak_obs][0],t_gnss[dtec_peak_obs][-1],0.,2300))
    ylabel('dTEC (TECU)')
    xlabel('Epicentral Distance (km)')   
    title('TEC Data', fontweight = 'bold')
    grid(linestyle='-.', lw=0.4)
    
    # if idx == 0:
    #     text(43, 0.95, "(A)", fontsize=15)
    # -------------------
    
tight_layout()
#%%

















    
    
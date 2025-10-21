#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 19:09:43 2025

@author: oluwasegun-adebayo
"""

from pylab import *
from scipy import *
from obspy import read
from datetime import date
from signal_alam import wavelet
import glob
import pandas as pd
from adebayo_codigo import band_pass_filter

# metadata = pd.read_csv("./high_rate_gnss/metadata_kaikoura_2016.csv")

# f_list=sorted(glob.glob('./uplift_seismic_all/Kaikoura2016/*LXZ.mseed'))#,allow_pickle=True)


# i=1;ipl=0
# vmx=[];distmx=[]
# for ix,file_name in enumerate(f_list):
#     data=read(file_name)
#     #print (data[0].stats)
#     st_name=data[0].stats.station
#     t_st=data[0].stats.starttime
#     ti=t_st.hour+t_st.minute/60.+t_st.second/3600.
#     t_st=data[0].stats.endtime
#     tf=t_st.hour+t_st.minute/60.+t_st.second/3600.
#     data=data[0].data
#     data=2*data*1.e-06
#     nt=len(data)
#     t=linspace(ti, tf,nt)
    
#     dt=(t[1]-t[0])*3600    
#     v=data#integrate.cumtrapz(data, x=t)#(gradient(gradient(data))+2*data)*dt*1.e-07
#     v=v-convolve(v,ones((50,))/50,mode='same')
#     v=v-v.mean()
#     vlow=convolve(v,ones((20,))/20,mode='same')
#     imx=abs(abs(v)-abs(v).max()).argmin()
#     #dist=3.4*(t[imx]-t[0])*3600.-100
    
#     # iw=0
#     # f=wavelet(iw, t, v,32)
#     # pd=f[0];emd=f[2];fr=1.e+03/(pd*3600.)
#     # if iw==2:
#     #     fr=fr/5.
#     # if iw==0:
#     #     fr=fr/1.4
#     # if iw==1:
#     #     fr=3*fr
#     # #if st_name=='CNBA' or st_name=='MAUL':# or st_name=='SJVA' or st_name=='CONZ':
#     # ipl=ipl+1
    
#     # figure(3,(12,12))
#     # subplot(1,2,ipl)
#     # for ifr in range (31):
#     #     plot(t,fr[ifr]+5*emd[ifr,:])
#     # ylim(11,50)
#     # title(st_name)
#     # xlabel('Time, UT')
#     # ylabel('Frequency, mHz')
#     # ifr40=abs(fr-40).argmin()
#     # vlow40=emd[ifr40,:]    
#     # ifr9=abs(fr-14).argmin()
#     # vlow9=emd[ifr9,:]    
    
#     # vlow=emd[ifr40:ifr9,:].mean(0)
    
#     figure(1,(12,12))
#     ax1=subplot(111)
#     plot(t, metadata['epi. distance'][ix]+100*v)
#     #plot(t, metdata['epi. distance'][ix]+100*v,label=st_name+' PGV= '+str(around(abs(v).max(),decimals=2)))
#     #xlim(6.56,6.7)
#     title('U$_{SISM}$')
#     ylabel('Epicentral distance, km')  
#     # ax2=subplot(122)
#     # plot(t, dist+100*vlow,label=st_name+' PGV= '+str(around(abs(vlow).max(),decimals=2)))
#     # #xlim(6.56,6.7)
#     # title('U$_{SISM}$ in 11mHz-40mHz')
#     # xlabel('Time, UT')
#     # ylabel('Epicentral distance, km')    
    
#     # figure(2,(12,12))
#     # subplot(211)
#     # vlines(dist,0,abs(v).max(),colors='r')
#     # title('PGV')
#     # ylabel('m/s')
#     # subplot(212)
#     # vlines(dist, 0, abs(vlow).max(),colors='r')
#     # text(dist,abs(vlow).max(),st_name)
#     # title('PGV in 11mHz-40 mHz')
#     # xlabel('Epicentral distance, km')
#     # ylabel('m/s')
    
#     # vmx.append(abs(vlow).max())
#     # distmx.append(dist)
    
        
#     i=i+1
# #ax1.legend()
# #ax2.legend()

# # idx=argsort(array(distmx))
# # dist_sort=array(distmx)[idx]
# # vmx_sort=array(vmx)[idx]
# # figure(4,(12,12))
# # plot(dist_sort,vmx_sort,'-o')
# # xlabel('Time, UT')
# # ylabel('m/s')
# # title('PGV in 11mHz-40mHz')

#     #subplot(212)
#     #plot(t,v-v.mean())

# # f_list=glob.glob('*.npy')#,allow_pickle=True)
# # for file_name in f_list:
# #     f=load(file_name,allow_pickle=True)
# #     print (f)
# #     data=f.item()
# #     t=data['t_sismo']
# #     vel=data['vel_sismo']
# #     figure(1,(12,12))
# #     plot(t,vel)
# #     show()















metadata = pd.read_csv("./high_rate_gnss/metadata_IIIapel_2015.csv")


# Path to your folder with .mseed files
mseed_folder = sorted(glob.glob('./uplift_seismic_all/Illapel2015/*LXZ.mseed'))#,allow_pickle=True)

total_figure = len(mseed_folder)
# Compute number of subplot rows and columns
cols = 4  
rows = int(total_figure / cols)

# Kaikoura 2016
# lat_ep = -42.737
# lon_ep = 173.054
# time_of_quake_hr = 11 + (2/60) + (56/3600)



lat_ep = -31.562
lon_ep = -71.426
time_of_quake_hr = 23 + (18/60) + (41/3600)



for f_no, filename in enumerate(mseed_folder):
    # Read the file using ObsPy
    stream = read(filename)
    
    #print(f"\nFile: {filename}")
    #print(stream)  # Summary of traces
    
    #print(stream[0].stats)
    
    trace = stream[0]
    amplitude = trace.data
    amplitude = amplitude * 1.58997e-06 * 6.e+00 / (1. * 1500)  # m/s (V/counts) / (gain * sensitivity (V/m/s))
    station_code = trace.stats.station
    start_time = str(trace.stats.starttime)
    end_time = str(trace.stats.endtime)
    npts = trace.stats.npts
    
    # Calculate time array
    start_time_float = float(start_time[11:13]) + float(start_time[14:16]) / 60. + float(start_time[17:19]) / 3600.
    #print(start_time_float)
    end_time_float = float(end_time[11:13]) + float(end_time[14:16]) / 60. + float(end_time[17:19]) / 3600.
    #print(end_time_float)
    time_sis = linspace(start_time_float, end_time_float, npts)
    # print(station_lat)
    
    # Compute the time step
    dt = trace.stats.delta  # Time interval between samples
    
    # Compute velocity by differentiating displacement
    velocity = 1000*np.gradient(amplitude, dt)
    
    # Get peak ground velocity (absolute max)
    pgv = np.max(np.abs(velocity))
    
    print(f"Station: {station_code}, {f_no}, {metadata['epi. distance'][f_no]},PGV: {pgv:.6f} m/s")

    dist = metadata['epi. distance'][f_no]
    st_name = metadata['station'][f_no]
    
    # figure(34, figsize=(12,12))
    # subplot(rows, cols, f_no + 1)
    # #plot(time_sis, amplitude, label=station_code)
    
    # plot(time_sis, velocity, label=station_code)

    # legend()
    
    figure(22)
    plot(time_sis, metadata['epi. distance'][f_no]+100*velocity)
    text(11.20, metadata['epi. distance'][f_no], st_name)
    
    # velocity_filtered = band_pass_filter(velocity, time_sis, lowcut=0.020, highcut=0.040, dt=1)
    
    # nm=int(160./1.)
    # nmode=ones((nm,))/nm
    # nm=int(200./1.)
    # nmode1=ones((nm,))/nm
    # #velocity_filtered=np.convolve(velocity, nmode,mode='same')#-np.convolve(velocity, nmode1,mode='same')
    # # plot(time_sis,metadata['epi. distance'][f_no] +200*vel_conv)#,label=f"{station_code}")
    # #legend()
    
    # # figure(35, figsize=(10,12))
    # # #if f_no==18 or f_no==14:
    # # subplot(121)
    # # plot(time_sis, metadata['epi. distance'][f_no] + (100*velocity),label=f"{station_code}")
    # #     #legend()
        
    # #     # filtered
        
    # # subplot(122)
        
    # plot(time_sis, metadata['epi. distance'][f_no] + (200*velocity_filtered))
        
   
    
    figure(36,(6,4))
    subplot(111)
    bar(metadata['epi. distance'][f_no],velocity.max())
    title('GNSS')
    xlim(0,1000)
    # subplot(212)
    # bar(metadata['epi. distance'][f_no],velocity_filtered[50:-50].max())
    #xlim(0,2300)
    print (len(time_sis),f"{station_code}")
    
    # figure(37,(12,12))
    # subplot(212)
    # bar(metadata['epi. distance'][f_no],velocity_filtered.max())
    # title('GNSS')
    # xlim(0,1500)
    # # subplot(212)
    # # bar(metadata['epi. distance'][f_no],velocity_filtered[50:-50].max())
    # #xlim(0,2300)
    # print (len(time_sis),f"{station_code}")
    
    distt = metadata['epi. distance'][f_no]
    lon_h_gnss = metadata['longitude'][f_no]
    lat_h_gnss = metadata['latitude'][f_no]
    # Saving the processed files
    data_sismo_gnss = {
        't_sismo': time_sis,
        'vel_sismo': velocity,
        'ep_dist_sismo': distt,
        'sta_name_sismo': station_code,
        'lon_sismo': lon_h_gnss,
        'lat_sismo': lat_h_gnss
        }
    
    # figure(38, figsize=(12,12))
    # scatter(metadata['longitude'][f_no], metadata['latitude'][f_no], marker='^')
    # scatter(lon_ep, lat_ep, marker='*')
    # text(metadata['longitude'][f_no], metadata['latitude'][f_no], s=f"{station_code}")
    
    # figure(30)
    # # Plot receiver locations
    # ax1.plot(metadata['longitude'][f_no], metadata['latitude'][f_no], 
    #          marker='^', markersize=4, color='r', transform=ccrs.PlateCarree())
    
    # if station_code in ['SJAV', 'CNBA']:
    #     #figure(30)
    #     text(metadata['longitude'][f_no], metadata['latitude'][f_no],
    #          station_code, fontsize=10, transform=ccrs.PlateCarree())

    
    # if not os.path.exists('./high_rate_gnss/high_rate_processed'):
        
    #     os.mkdir('./high_rate_gnss/high_rate_processed')
    
    # file_n = f"{station_code}.npy"
    # file_p = f"./high_rate_gnss/high_rate_processed/"


    #save(file_p + file_n, data_sismo)
    
    # Plot each trace in the stream
    #stream.plot()
    
    # Optionally, you can also plot individual traces
    #for tracee in trace:
        #print(tracee)
        
        #trace.plot(type='relative', title=f"{trace.id} - {filename}")



# # Loop through all .mseed files in the folder
# for filename in os.listdir(mseed_folder):
#     if filename.endswith('.mseed'):
#         filepath = os.path.join(mseed_folder, filename)
tight_layout()        
        
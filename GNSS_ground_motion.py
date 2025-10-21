#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 13:53:39 2025

@author: alam
"""
from pylab import *
from scipy import *
from obspy import read
from datetime import date
from signal_alam import wavelet
import glob


f_list=glob.glob('./uplift_seismic_all/Ecuador2016/*LXZ.mseed')#,allow_pickle=True)
i=1;ipl=0
vmx=[];distmx=[]
for file_name in f_list:
    data=read(file_name)
    #print (data[0].stats)
    st_name=data[0].stats.station
    t_st=data[0].stats.starttime
    ti=t_st.hour+t_st.minute/60.+t_st.second/3600.
    t_st=data[0].stats.endtime
    tf=t_st.hour+t_st.minute/60.+t_st.second/3600.
    data=data[0].data
    data=2*data*1.e-06
    nt=len(data)
    t=linspace(ti, tf,nt)
    
    dt=(t[1]-t[0])*3600    
    v=data#integrate.cumtrapz(data, x=t)#(gradient(gradient(data))+2*data)*dt*1.e-07
    v=v-convolve(v,ones((50,))/50,mode='same')
    v=v-v.mean()
    vlow=convolve(v,ones((20,))/20,mode='same')
    imx=abs(abs(v)-abs(v).max()).argmin()
    dist=3.4*(t[imx]-t[0])*3600.-100
    
    # iw=0
    # f=wavelet(iw, t, v,32)
    # pd=f[0];emd=f[2];fr=1.e+03/(pd*3600.)
    # if iw==2:
    #     fr=fr/5.
    # if iw==0:
    #     fr=fr/1.4
    # if iw==1:
    #     fr=3*fr
    # #if st_name=='CNBA' or st_name=='MAUL':# or st_name=='SJVA' or st_name=='CONZ':
    # ipl=ipl+1
    
    # figure(3,(12,12))
    # subplot(1,2,ipl)
    # for ifr in range (31):
    #     plot(t,fr[ifr]+5*emd[ifr,:])
    # ylim(11,50)
    # title(st_name)
    # xlabel('Time, UT')
    # ylabel('Frequency, mHz')
    # ifr40=abs(fr-40).argmin()
    # vlow40=emd[ifr40,:]    
    # ifr9=abs(fr-14).argmin()
    # vlow9=emd[ifr9,:]    
    
    # vlow=emd[ifr40:ifr9,:].mean(0)
    
    figure(1,(12,12))
    ax1=subplot(121)
    plot(t, dist+100*v,label=st_name+' PGV= '+str(around(abs(v).max(),decimals=2)))
    #xlim(6.56,6.7)
    title('U$_{SISM}$')
    ylabel('Epicentral distance, km')  
    ax2=subplot(122)
    plot(t, dist+100*vlow,label=st_name+' PGV= '+str(around(abs(vlow).max(),decimals=2)))
    #xlim(6.56,6.7)
    title('U$_{SISM}$ in 11mHz-40mHz')
    xlabel('Time, UT')
    ylabel('Epicentral distance, km')    
    
    figure(2,(12,12))
    subplot(211)
    vlines(dist,0,abs(v).max(),colors='r')
    title('PGV')
    ylabel('m/s')
    subplot(212)
    vlines(dist, 0, abs(vlow).max(),colors='r')
    text(dist,abs(vlow).max(),st_name)
    title('PGV in 11mHz-40 mHz')
    xlabel('Epicentral distance, km')
    ylabel('m/s')
    
    vmx.append(abs(vlow).max())
    distmx.append(dist)
    
        
    i=i+1
#ax1.legend()
#ax2.legend()

idx=argsort(array(distmx))
dist_sort=array(distmx)[idx]
vmx_sort=array(vmx)[idx]
figure(4,(12,12))
plot(dist_sort,vmx_sort,'-o')
xlabel('Time, UT')
ylabel('m/s')
title('PGV in 11mHz-40mHz')

    #subplot(212)
    #plot(t,v-v.mean())

# f_list=glob.glob('*.npy')#,allow_pickle=True)
# for file_name in f_list:
#     f=load(file_name,allow_pickle=True)
#     print (f)
#     data=f.item()
#     t=data['t_sismo']
#     vel=data['vel_sismo']
#     figure(1,(12,12))
#     plot(t,vel)
#     show()
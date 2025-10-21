
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
from adebayo_codigo import *
import glob
import pandas as pd
import pywt
plt.close("all")

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
# This Python function calculates the distance between two points
# on Earth given their longitude and latitude coordinates. It uses
# the Haversine formula to calculate the distance (d1) and the
# Equirectangular approximation (d2) based on the coordinates.

def distkm(lon1,lat1,lon2,lat2):
    radius=6371
    deltaLat=radians(lat2)-radians(lat1)
    deltaLon=radians(lon2)-radians(lon1)
    a=sin((deltaLat)/2)**2+cos(radians(lat1))*cos(radians(lat2))*sin(deltaLon/2)**2
    c=2*arctan(sqrt(a)/(sqrt(1-a)))
    d1=radius*c  # d1km: distância em km com base na fórmula Haversine
    x=deltaLon*cos((radians(lat1)+radians(lat2))/2);
    y=deltaLat;
    d2= radius*sqrt(x**2. + y**2.); #d2km:Equirectangular approximation, distância em km com base no teorema de Pitágoras
    return d1,d2
def axis_param(im,str_c):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad = 0.05)
    cbar = colorbar(im, cax=cax);gca().set_title(str_c, fontweight='bold')
    return()

def data_smooth(t,data):
        dt=t[1]-t[0]
        n_smooth=int(ceil(7./(dt*60.)))
        win=ones((n_smooth,))/n_smooth
        data=data-convolve_al(data,win)
        return convolve_al(data,ones((10,))/10)
    
def data_smooth_mike(t,data, win1, win2):
        dt=t[1]-t[0]
        n_smooth=int(ceil(win1/(dt*60.)))
        win=ones((n_smooth,))/n_smooth
        data=data-convolve_al(data,win)
        return convolve_al(data,ones((win2,))/win2)
    
def data_smooth_sim(t,data):
        dt=t[1]-t[0]
        n_smooth=int(ceil(10./(dt*60.)))
        win=ones((n_smooth,))/n_smooth
        data=-data+convolve_al(data,win)
        return convolve_al(data,ones((2,))/2)  

#%%    

def find_peaks_within_range_1(arr, start_idx, end_idx, threshold, k):
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

# other method

def find_peaks_within_range(arr, start_idx, end_idx, threshold, k):
    start_idx = max(0, start_idx)
    end_idx = min(len(arr), end_idx)
    arr_sub = arr[start_idx:end_idx]
    
    # Find peaks in the range
    peaks, _ = find_peaks(arr_sub, distance=k, height=threshold)
    
    # Adjust indices to the full array range
    return [start_idx + peak for peak in peaks]


#%%



lat_ep=23.8322 ;lon_ep=121.6044
year,month,day=2024, 4, 3
hour, minute, second=0, 0, 30
t_ep=hour+minute/60+second/3600

dtec_16 = []
dtec_27 = []
dtec_31 = []                                       #As resoluções espaciais no metros

#%%%%

f_list=sorted(glob.glob('./hualien/data/***.npy'))
for f_n in f_list:  
    data_gnss=load(f_n,allow_pickle=True).item()
    t_gnss=data_gnss['t_gnss'];tec_gnss=data_gnss['tec_gnss'];
    lon_gnss=data_gnss['lon_gnss'];lat_gnss=data_gnss['lat_gnss'];
    lon_rec = data_gnss['lon_rec']; lat_rec = data_gnss['lat_rec']
    elev_ang = data_gnss['elev_ang']; azim_ang = data_gnss['azim_ang']
  
    v_prox=find_nearest(t_gnss, t_ep)
    p_sip=where(t_gnss==v_prox)[0]
    ep_sign=1
    if lon_gnss[p_sip]-lon_ep <0:
        ep_sign=1
    if lat_gnss[p_sip]-lat_ep <0:
        ep_sign=-1
    [d1,d2]=distkm(lon_ep,lat_ep,([lon_gnss[p_sip]]),([lat_gnss[p_sip]]))
    ep_dist=((d1+d2)[0]/2)/1.
    dtec_gnss=data_smooth_mike(t_gnss,tec_gnss, 7, 10)
    
    
    
    # if f_n[19:22] == 'G27' or f_n[19:22] == 'G16':
    
    #     fig=figure(1, figsize=(10, 8), facecolor='w',edgecolor='k')
    #     ax=subplot(121)
        
    #     p_zon = find_peaks_within_range_1(dtec_gnss, 
    #                                               start_idx= 30, end_idx= 90, 
    #                                               threshold= 0.06, 
    #                                               k=3)
        
       
    #     im=scatter(t_gnss*60,ep_dist+150*dtec_gnss,c=dtec_gnss,s=200*abs(dtec_gnss),
    #                 marker='o',cmap=cm.rainbow,alpha=1)
    
    #     #text(0,ep_dist,f_n[15:22], fontsize = 5, color='k')
    
    #     axvline(1.2,c='grey')
    #     xlim(0, 25)
        
    #     ylabel('Epicentral Distance (km)')
    #     xlabel('Detection time (minutes)')
    #     if f_n == "./hualien/data/ZUZHG27.npy":
            
    #         f=  axis_param(im,'TECU') 
        
    #     # ------------------------------------------------   
    #     ax=subplot(122)
    #     # ------------------------------------------------ 
       
    #     scatter(t_gnss[p_zon]*60, ep_dist+(dtec_gnss[p_zon]), c='b')
        
    #     #title("G16 and G27")
    #     grid(linestyle = '-.', lw=0.4)
    #     xlabel("Detection time (minutes)")
    #     ylabel("Epicentral Distance (km)")
        
    # tight_layout()
        
    #     #savefig("./hualien/figures/fig1.png", bbox_inches = 'tight')    
    #     # --------------------------------------------------------
        
    fig=figure(2, figsize=(8, 5), facecolor='w',edgecolor='k')
    if f_n[19:22] == 'G16' or f_n[19:22] == 'G27':# or f_n[19:22] == 'G31':
    
        
        ax=subplot(111)
        
        ep_lat = lat_gnss - lat_ep
        ep_lon = lon_gnss - lon_ep
        t_gnss_min = t_gnss*60
        
        scatter(t_gnss_min, dtec_gnss + ep_lat, c = dtec_gnss, 
                s =100 * abs(dtec_gnss), cmap = "rainbow"
                )
        #xlabel('Detection time (minutes)')
        ylabel('Epicentral Distance ($^{o}$)')
        xlabel('Detection time (minutes)')
        xlim(0, 50)
        
        
        if f_n == "./hualien/data/ZUZHG27.npy":
            #text(1.8, 3.5, '(a)')
            plot([6.84, 12.20], [0.87, -1.82], 'b--', lw=1)
            plot([11.23, 20.57], [0.91, -1.65], 'b--', lw=1)
            plot([17.07, 41.44], [1.07, -1.19], 'b--', lw=1)
            
            text(10.74, -1.98, '1.05 km/s')
            text(19.43, -1.81, '0.60 km/s')
            text(40.54, -1.36, '0.22 km/s')
            f=  axis_param(im,'TECU') 
        
        
        # ax=subplot(212)
        
        # scatter(t_gnss_min, dtec_gnss + ep_lon, c = dtec_gnss, 
        #         s =100 * abs(dtec_gnss), cmap = "rainbow")
        # xlabel('Detection time (minutes)')
        # ylabel('Epicentral Longitude ($^{o}$)')
        # xlim(0, 50)
        # ylim(-1.7, 1.6)
        
        
        # if f_n == "./hualien/data/ZUZHG27.npy":
        #     text(1.8, 1.4, '(b)')
        #     f=  axis_param(im,'TECU') 
            
    tight_layout()
        
        # for f in range(int(fre[0]), int(fre[-1]), 2):
        #     plot(t / 60, cof[f - int(fre[0]), :], label=f'{f} Hz')  # Plot specific frequency band
            
        #savefig("./hualien/figures/fig2.png", bbox_inches = 'tight')   
        # --------------------------------------------------------
        
        
        # Propagating speed of the CIDs
        
        
        # fig=figure(3, figsize=(10, 7), facecolor='w',edgecolor='k')
        
        # # Meridional phase speed
        # ax=subplot(121)
        
        # dtec_peak_obs = find_peaks_within_range_1(dtec_gnss, 
        #                                           start_idx= 30, end_idx= 90, 
        #                                           threshold= 0.06, 
        #                                           k=1)
        
        
        # time_o = (t_gnss[dtec_peak_obs] - 0.101)*3600
        # dist_o = (lat_gnss[dtec_peak_obs] - lat_ep)*110
        # vel_o = (dist_o) / (time_o) 
                
        # scatter(abs(dist_o), abs(vel_o), c='b')
        # ylim(0, 1.2)
        # grid(linestyle = '-.', lw=0.4)
        # xlabel("Epicentral Latitude (km)")
        # ylabel("Phase speed (km/s)")
        # ax.annotate('', xy=(222.5, 0.886),
        #      xytext=(85, 0.638),
        #      arrowprops=dict(arrowstyle= '->',
        #                      color='r',
        #                      lw=2
        #                      )
        #    )
        
        # ax.annotate('', xy=(265, 0.426),
        #      xytext=(100, 0.2),
        #      arrowprops=dict(arrowstyle= '->',
        #                      color='r',
        #                      lw=2
        #                      )
        #    )
        
        # if f_n == "./hualien/data/ZUZHG27.npy":
        
        #     text(98, 0.711, 'fast-mode', rotation = 37)
        #     text(126, 0.275, 'slow-mode', rotation = 32)
        
        
        # # Zonal phase speed
        # ax=subplot(122)
        
        # p_mer = find_peaks_within_range_1(dtec_gnss, 
        #                                   start_idx= 30, end_idx= 90, 
        #                                   threshold= 0.025, 
        #                                   k = 1)
        
        # time_o = (t_gnss[p_mer] - 0.101)*3600
        # dist_o = (lon_gnss[p_mer] - lon_ep)*110
        # vel_o = (dist_o) / (time_o)
        
        
                
        # scatter(abs(dist_o), abs(vel_o), c='b')
        # ylim(-0.05, 1.2)
        # grid(linestyle = '-.', lw=0.4)
        # xlabel("Epicentral Longitude (km)")
        # ylabel("Phase speed (km/s)")
        
        # ax.annotate('', xy=(82.7, 0.773),
        #      xytext=(25, 0.453),
        #      arrowprops=dict(arrowstyle= '->',
        #                      color='r',
        #                      lw=2
        #                      )
        #    )
        
        # ax.annotate('', xy=(150, 0.138),
        #      xytext=(32.7, 0),
        #      arrowprops=dict(arrowstyle= '->',
        #                      color='r',
        #                      lw=2
        #                      )
        #    )
        
        # if f_n == "./hualien/data/ZUZHG27.npy":
        
        #     text(20.2, 0.528, 'fast-mode', rotation = 70)
        #     text(57.7, 0, 'slow-mode', rotation = 30)
        
        
    #tight_layout()
        #savefig("./hualien/figures/fig3.png", bbox_inches = 'tight') 

#%%        
    # -------------------- TTD G16, G27 and G31 ------------------------------------   
    fig=figure(4, figsize=(10, 12), facecolor='w',edgecolor='k')   
    
    # Satellite 27
    if f_n[19:22] == 'G27':
        
        ax=subplot(321)
        im=scatter(t_gnss*60,ep_dist+150*dtec_gnss,c=dtec_gnss,
                   s=200*abs(dtec_gnss), marker='o',cmap=cm.rainbow,alpha=1)
        
        xlim(0, 30)
        ylabel('Epicentral\nDistance (km)')
        #xlabel('Detection time (minutes)')
        #axvline(1.2, color = 'grey')
        
        
        
        if f_n == "./hualien/data/ZUZHG27.npy":
            text(1.5, 300, "G27")
            #v1=(4/1)/(1./3600.)
            axline((7,0),(13.45, 347),c='k', label = '~1.05 km/s')
            axline((12.50,0),(22.86, 376),c='k', ls='--', label = '~0.60 km/s')
            legend()
            
            f=  axis_param(im,'TECU') 
    
    # Satellite 16
    if f_n[19:22] == 'G16':
        
        ax=subplot(323)
        im=scatter(t_gnss*60,ep_dist+150*dtec_gnss,c=dtec_gnss,s=200*abs(dtec_gnss),
                    marker='o',cmap=cm.rainbow,alpha=1)
        
        xlim(0, 30)
        ylabel('Epicentral\nDistance (km)')
        #xlabel('Detection time (minutes)')
        #axvline(1.2, color = 'grey')
        
        
        if f_n == "./hualien/data/ZUZHG16.npy":
            axline((6,0),(14.07, 438),c='k', label = '~0.85 km/s')
            axline((10,0),(20.43, 436),c='k', ls='--', label = '~0.62 km/s')
            legend()
            text(1.5, 375, "G16")
            f=  axis_param(im,'TECU') 
    
    # Satellite 31
    if f_n[19:22] == 'G31':
        
        ax=subplot(325)
        im=scatter(t_gnss*60,ep_dist+150*dtec_gnss,c=dtec_gnss,s=200*abs(dtec_gnss),
                    marker='o',cmap=cm.rainbow,alpha=1)
        
        xlim(0, 30)
        ylabel('Epicentral\nDistance (km)')
        xlabel('Detection time (minutes)')
        #axvline(1.2, color = 'grey')
        
        
        if f_n == "./hualien/data/ZUZHG31.npy":
            axline((7,0),(18.17, 350),c='k', ls='--', label = '~0.55 km/s')
            
            legend(loc='lower right')
            text(1.5, 300, "G31")
            f=  axis_param(im,'TECU') 
            
#%%

    # fig=figure(4, figsize=(10, 12), facecolor='w',edgecolor='k')   
    
    # # Satellite 27
    # if f_n[19:22] == 'G27':
        
    #     ax=subplot(322)
        
    #     dtec_gnss[dtec_gnss <= 0.035] = 0
        
    #     pp = find_peaks_within_range_1(ep_dist+dtec_gnss, 
    #                                    start_idx= 30, 
    #                                    end_idx= 80, 
    #                                    threshold= 0.12, 
    #                                    k=3)
        
    #     # elevation and azimuth
    #     elev_ang_27 = elev_ang[pp]
    #     azim_ang_27 = azim_ang[pp]
        
    #     velll = (ep_dist+dtec_gnss[pp]) /  (t_gnss[pp]*3600 - 420)
        
    #     scatter(t_gnss[pp]*60, ep_dist+dtec_gnss[pp], c='b')
    #     #scatter(ep_dist+dtec_gnss[pp], velll, c = 'red')
    #     #ylim(-0.05, 1.2)
    #     xlim(0, 30)
    #     grid(linestyle = '-.', lw=0.4)
    #     ylabel('Epicentral\nDistance (km)')
    #     #xlabel('Detection time (minutes)')
        
    #     if f_n == "./hualien/data/ZUZHG27.npy":
        
    #         text(6.51, 163.4, "1.13 km/s", rotation=75)
    #         text(21.78, 156.6, "0.54 km/s", rotation=75)
            
    #     ax.annotate('', xy=(9.95, 243.1),
    #          xytext=(8.02, 162.5),
    #          arrowprops=dict(arrowstyle= '->',
    #                          color='red',
    #                          lw=2
    #                          ))
        
    #     ax.annotate('', xy=(22.46, 237),
    #          xytext=(20.27, 154),
    #          arrowprops=dict(arrowstyle= '->',
    #                          color='red',
    #                          lw=2
    #                          ))
        
        
        
    # # Satellite 16
    # if f_n[19:22] == 'G16':
        
    #     ax=subplot(324)
    #     dtec_gnss[dtec_gnss <= 0.03] = 0
        
    #     pp = find_peaks_within_range_1(ep_dist+dtec_gnss, 
    #                                    start_idx= 20, 
    #                                    end_idx= 70, 
    #                                    threshold= 0.1, k=3)
        
    #     # elevation and azimuth
    #     elev_ang_16 = elev_ang[pp]
    #     azim_ang_16 = azim_ang[pp]
        
        
    #     velll = (ep_dist+dtec_gnss[pp]) /  (t_gnss[pp]*3600 - 395)
        
    #     scatter(t_gnss[pp]*60, ep_dist+dtec_gnss[pp], c='b')
    #     #scatter(ep_dist+dtec_gnss[pp], velll, c = 'red')
    #     #ylim(0, 1.2)
    #     xlim(0, 30)
    #     ylabel('Epicentral\nDistance (km)')
    #     #xlabel('Detection time (minutes)')
    #     grid(linestyle = '-.', lw=0.4)
        
    #     if f_n == "./hualien/data/ZUZHG16.npy":
    #         text(5.92, 281, "0.83 km/s", rotation = 65)
    #         text(18.43, 183, "0.62 km/s", rotation = 70)
        
    #     ax.annotate('', xy=(10.87, 384),
    #          xytext=(7.52, 252),
    #          arrowprops=dict(arrowstyle= '->',
    #                          color='red',
    #                          lw=2
    #                          ))
        
    #     ax.annotate('', xy=(19.60, 327),
    #          xytext=(16.66, 180),
    #          arrowprops=dict(arrowstyle= '->',
    #                          color='red',
    #                          lw=2
    #                          ))
        
    
    # # Satellite 31
    # if f_n[19:22] == 'G31':
        
    #     ax=subplot(326)
    #     dtec_gnss[dtec_gnss <= 0.03] = 0
        
    #     pp = find_peaks_within_range_1(dtec_gnss, 
    #                                    start_idx = 28, end_idx= 55, 
    #                                    threshold= 0.05, k=3)
        
    #     # elevation and azimuth
    #     elev_ang_31 = elev_ang[pp]
    #     azim_ang_31 = azim_ang[pp]
        
    #     velll = (ep_dist+dtec_gnss[pp]) /  (t_gnss[pp]*3600 - 500)
        
    #     scatter(t_gnss[pp]*60, ep_dist+dtec_gnss[pp], c='b')
        
    #     #scatter(ep_dist+dtec_gnss[pp], velll, c = 'red')
    #     #ylim(0, 0.7)
    #     xlim(0, 30)
    #     #xlabel('Epicentral Distance (km)')
    #     ylabel('Epicentral\nDistance (km)')
    #     xlabel('Detection time (minutes)')
    #     grid(linestyle = '-.', lw=0.4)
        
    #     if f_n == "./hualien/data/ZUZHG31.npy":
        
    #         text(5.50, 159.5, "0.53 km/s", rotation = 80)
        
    #     ax.annotate('', xy=(8.86, 225.4),
    #          xytext=(7.18, 143.6),
    #          arrowprops=dict(arrowstyle= '->',
    #                          color='red',
    #                          lw=2
    #                          ))
        
        
    # tight_layout() 
        

#%%
    dtec_gnss=data_smooth_mike(t_gnss,tec_gnss, 7, 10)
    
    #fig=figure(5, figsize=(8, 10))
    fig=figure(4, figsize=(10, 12), facecolor='w',edgecolor='k')   
    
    # All PRNs
    if f_n[19:22] == 'G27':
        
        ep_elev = elev_ang[1]
        ep_azm = azim_ang[1]
        
        ppp = find_peaks_within_range_1(dtec_gnss, 
                                        start_idx = 25, end_idx= 90, 
                                        threshold= 0.03, k=1)
        
        ax=subplot(322)
        
        plot(t_gnss*60, elev_ang, c='g', lw=0.5)
        
        
        title('G27')
        #xlabel("Time (minutes)")
        ylabel("Elevation angle ($^{o}$)")
        axvline(1.2, c='grey')
        grid(linestyle='-.', lw = 0.4)
        
        
        # ax=subplot(322)
        
        # plot(t_gnss*60, azim_ang, c='b')
        
        
        # title('G27')
        # xlabel("Time (minutes)")
        # ylabel("Azimuth angle ($^{o}$)")
        # axvline(1.2, c='grey')
        # grid(linestyle='-.', lw = 0.4)
        
        dtec_27.append([dtec_gnss, t_gnss])
        
    if f_n[19:22] == 'G16':
        
        ep_elev = elev_ang[1]
        ep_azm = azim_ang[1]
        
        ppp = find_peaks_within_range_1(ep_dist+dtec_gnss, 
                                       start_idx= 20, 
                                       end_idx= 90, 
                                       threshold= 0.05, k=1)
        
        
        ax=subplot(324)
        
        plot(t_gnss*60, elev_ang, c='g', lw=0.5)
        
        title('G16')
        #xlabel("Time (minutes)")
        ylabel("Elevation angle ($^{o}$)")
        axvline(1.2, c='grey')
        grid(linestyle='-.', lw = 0.4)
        
        
        # ax=subplot(324)
        
        # plot(t_gnss*60, azim_ang, c='b')
        
        # title('G16')
        # xlabel("Time (minutes)")
        # ylabel("Azimuth angle ($^{o}$)")
        # axvline(1.2, c='grey')
        # grid(linestyle='-.', lw = 0.4)
        
        dtec_16.append([dtec_gnss, t_gnss])
        
    if f_n[19:22] == 'G31':
        
        ep_elev = elev_ang[1]
        ep_azm = azim_ang[1]
        
        ppp = find_peaks_within_range_1(dtec_gnss, 
                                       start_idx = 28, end_idx= 68, 
                                       threshold= 0.01, k=1)
        
        ax=subplot(326)
        
        plot(t_gnss*60, elev_ang, c='g', lw=0.5)
        title('G31')
        xlabel("Time (minutes)")
        ylabel("Elevation angle ($^{o}$)")
        axvline(1.2, c='grey')
        grid(linestyle='-.', lw = 0.4)
        
        # ax=subplot(326)
        
        # plot(t_gnss*60, azim_ang, c='b')
        # title('G31')
        # xlabel("Time (minutes)")
        # ylabel("Azimuth angle ($^{o}$)")
        # axvline(1.2, c='grey')
        # grid(linestyle='-.', lw = 0.4)
        
        dtec_31.append([dtec_gnss, t_gnss])
        
    tight_layout()
    # TEC amplitude
        
    figure(6, figsize=(8, 10))
    
    # Satellite 27
    if f_n[19:22] == 'G27':
        
        ax=subplot(311)
        im=scatter(t_gnss*60,ep_dist+150*dtec_gnss,c=dtec_gnss,
                   s=200*abs(dtec_gnss), marker='o',cmap=cm.rainbow,alpha=1)
        
        xlim(0, 50)
        ylabel('Epicentral\nDistance (km)')
        #xlabel('Detection time (minutes)')
        #axvline(1.2, color = 'grey')
        
        if int(ep_dist % 10) == 0:
        
            text(0, ep_dist, f_n[15:19])
        
        
        if f_n == "./hualien/data/ZUZHG27.npy":
            
            text(1.5, 300, "G27")
            
            
            
            f=  axis_param(im,'TECU') 
            
        #if ep_dist > 190 and ep_dist < 220:
        if f_n[15:22] == 'ZENDG27':
            figure(61)
            plot(t_gnss, dtec_gnss, label=f_n[15:22])
            legend()
            
            
            figure(62)
            ax=subplot(111)
            data_ft, f, tt = stft_band_pass_filter(dtec_gnss, t_gnss*3600, lowcut=0.001, highcut=0.005, window_length=100, overlap=64)
            # Use imshow for displaying the 2D data
            im1 = ax.imshow(np.abs(data_ft*10), aspect='auto', extent=[0, tt.max() / 60, (f*1e3).min(), (f*1e3).max()], 
                           cmap='jet', origin='lower')

            im1.set_interpolation("spline16")



            #text(-138, 9, file[17:24], color='white')
            xlabel('Detection Time (mins)')
            ylabel('Frequency (mHz)')
            #xlim(-135,0)
            ylim(0.01,10)
            text(55, 9, 'G27', color='w')
            text(2.5, 8.8, '(c)', color='w')
            axis_param(im1,'') 

        
        # if f_n == "./hualien/data/ZUZHG27.npy":
        #     figure(20, figsize=(10,6))
        #     ax=subplot(222)

        #     time_secs, magnitude = compute_periodogram(dtec_gnss, sampling_rate=1/15)
        #     plot(time_secs, 1000*magnitude, color='r')
        #     ylabel("PSD")
        #     grid(ls = '-.', lw=0.4)
        #     text(905, 15, '1 mHz', fontsize=6)
        #     text(670, 30, '1.6 mHz', fontsize=6)
        #     text(0, 10, '3.7 mHz', fontsize=6)
        #     text(3398, 30, '(c)')
        
        
        # t = (t_gnss[0:100] - t_ep)*3600
        # ddt = dtec_gnss[0:100]
        
        # # t = (t_gnss[0:100] - t_ep)*3600
        # # ddt = data_smooth_mike(t, dtec_gnss[0:100], 7, 10)
        
        # wav = 'cmor' #mexh, cmor, sym
        # scales = arange(1, 32)
        # sampling_rate = 15
        
        # cof, fre = pywt.cwt(ddt, scales, wav, sampling_rate)
        
        # fig = figure(7, figsize=(7,4))
        # subplot(211)
        # im = imshow(abs(cof), extent=[t[0]/60, t[-1]/60, fre[-1]*1e3, fre[0]*1e3], 
        #             aspect = 'auto', cmap = 'jet')
        # im.set_interpolation('spline16')
        # ylabel('Frequency (mHz)')
        
        # if f_n == "./hualien/data/ZUZHG27.npy":
        
        #     colorbar(pad = 0.01, label = 'TECU')
        
    #text(0.1, 30, 'G27', color='white')
    
        
        
    # Satellite 16
    if f_n[19:22] == 'G16':
        figure(6, figsize=(8, 10))
        ax=subplot(312)
        im=scatter(t_gnss*60,ep_dist+150*dtec_gnss,c=dtec_gnss,s=200*abs(dtec_gnss),
                    marker='o',cmap=cm.rainbow,alpha=1)
        
        xlim(0, 50)
        ylabel('Epicentral\nDistance (km)')
        #xlabel('Detection time (minutes)')
        #axvline(1.2, color = 'grey')
        
        #text(0, ep_dist, f_n[15:19])
        
        if f_n == "./hualien/data/ZUZHG16.npy":
            text(1.5, 375, "G16")
            f=  axis_param(im,'TECU') 
            
            
        # if f_n == "./hualien/data/ZUZHG16.npy":
        #     figure(20, figsize=(10,6))
        #     ax=subplot(224)

            
        #     time_secs, magnitude = compute_periodogram(dtec_gnss, sampling_rate=1/15)
        #     plot(time_secs, 1000*magnitude, color='r')
        #     xlabel("Period (seconds)")
        #     ylabel("PSD")
        #     grid(ls = '-.', lw=0.4)
        #     text(1280, 18, '0.8 mHz', fontsize=6)
        #     text(604, 16.5, '1.6 mHz', fontsize=6)
        #     text(0, 16, '3.1 mHz', fontsize=6)
        #     text(3408, 17, '(d)')
            
        # t = (t_gnss[0:120] - t_ep)*3600
        # ddt = dtec_gnss[0:120]
        
        # wav = 'cmor' #mexh, cmor, sym
        # scales = arange(1, 25)
        # sampling_rate = 15
        
        # cof, fre = pywt.cwt(ddt, scales, wav, sampling_rate)
        
        # fig = figure(7)
        # subplot(212)
        
        # im=imshow(abs(cof), extent=[t[0]/60, t[-1]/60, fre[-1]*1e3, fre[0]*1e3], 
        #             aspect = 'auto', cmap = 'jet')
        # im.set_interpolation('bilinear')
        # xlabel('Detection time (minutes)')
        # ylabel('Frequency (mHz)')
        
        # if f_n == "./hualien/data/ZUZHG16.npy":
        
        #     colorbar(pad = 0.01, label = 'TECU')
        
    #text(0.2, 30, 'G16', color = 'white')
    
    # Satellite 31
    if f_n[19:22] == 'G31':
        
        figure(6, figsize=(8, 10))
        ax=subplot(313)
        im=scatter(t_gnss*60,ep_dist+150*dtec_gnss,c=dtec_gnss,s=200*abs(dtec_gnss),
                    marker='o',cmap=cm.rainbow,alpha=1)
        
        xlim(0, 50)
        ylabel('Epicentral\nDistance (km)')
        xlabel('Detection time (minutes)')
        #axvline(1.2, color = 'grey')
        
        #text(0, ep_dist, f_n[15:19])
        
        if f_n == "./hualien/data/ZUZHG31.npy":
            text(1.5, 300, "G31")
            
            f=  axis_param(im,'TECU') 
            
            
        #if ep_dist > 190 and ep_dist < 220:
        if f_n[15:22] == 'PANGG31':
            figure(20, figsize=(10,10))
            ax=subplot(325)
            plot(t_gnss*60, dtec_gnss, label=f_n[15:22])
            xlim(0,50)
            grid(ls = '-.', lw=0.4)
            ylabel("dTEC (TECU)")
            text(45, 0.15, 'G31')
            text(0, 0.15, '(a3)')
            xlabel('Detection Time (mins)')
            
            
            figure(20, figsize=(10,10))
            ax=subplot(326)
            data_ft, f, tt = stft_band_pass_filter(dtec_gnss, t_gnss*3600, lowcut=0.001, highcut=0.003, window_length=100, overlap=64)
            # Use imshow for displaying the 2D data
            im1 = ax.imshow(np.abs(data_ft*10), aspect='auto', extent=[0, tt.max() / 60, (f*1e3).min(), (f*1e3).max()], 
                           cmap='jet', origin='lower')

            im1.set_interpolation("spline16")



            #text(-138, 9, file[17:24], color='white')
            xlabel('Detection Time (mins)')
            ylabel('Frequency (mHz)')
            xlim(0,50)
            ylim(0.01,10)
            text(45, 9, 'G31', color='w')
            text(2.5, 8.8, '(b3)', color='w')
            axhline(2.62, color='w', lw=1)
            text(10, 0.8, "2.62 mHz", color='w')
            axis_param(im1,'') 

            
        # t = (t_gnss[0:100] - t_ep)*3600
        # ddt = dtec_gnss[0:100]
        
        # wav = 'cmor' #mexh, cmor, sym
        # scales = arange(1, 64)
        # sampling_rate = 15
        
        # cof, fre = pywt.cwt(ddt, scales, wav, sampling_rate)
        
        # fig = figure(7)
        # subplot(313)
        # imshow(abs(cof), extent=[t[0], t[-1], fre[-1], fre[0]], aspect = 'auto', cmap = 'jet')
            
tight_layout()
#savefig("./hualien/figures/filt_tec.png", bbox_inches = 'tight')       

#%%

#from matplotlib.ticker import FormatStrFormatter

#%%

dtec_16 = array(dtec_16)
dtec_27 = array(dtec_27)
dtec_31 = array(dtec_31)

#G16
dtec_16good = dtec_16[70:80,:,0:210]

avtec_16 = dtec_16good[:,0,:].mean(0)
avtime_16 = dtec_16good[:,1,:].mean(0)

# G27
# dtec_27good = dtec_27[52:54,:,0:210]
dtec_27good = dtec_27[52:54,:,0:210]

avtec_27 = dtec_27good[:,0,:].mean(0)
avtime_27 = dtec_27good[:,1,:].mean(0)

# G31
dtec_31good = dtec_31[:,:,0:210]
avtec_31 = dtec_31good[:,0,:].mean(0)
avtime_31 = dtec_31good[:,1,:].mean(0)



show()


#%%
figure(20, figsize=(10,10))
ax=subplot(324)


#dtec_gnss = band_pass_filter(dtec_gnss, (t_gnss - time_of_quake_hr)*3600, 0.0001, 0.003)

data_ft, f, tt = stft_band_pass_filter(avtec_16, avtime_16, lowcut=0.002, highcut=0.01, window_length=100, overlap=64) #100,64, 64, 16
#data_ft, f, tt = stft_band_pass_filter(avtec_16, avtime_16, lowcut=0.001, highcut=0.01, window_length=50, overlap=45)
# Use imshow for displaying the 2D data
im = ax.imshow(np.abs(data_ft*10), aspect='auto', extent=[0, tt.max() / 60, (f*1e3).min(), (f*1e3).max()], 
               cmap='jet', origin='lower')

im.set_interpolation("spline16")
ylim(0.01,10)
xlim(0,50)
ylabel('Frequency (mHz)')
text(45, 9, 'G16', color='w')
text(2.5, 8.8, "(b2)", color='w')
axhline(2.45, color='w', lw=1)
axhline(3.65, color='w', lw=1)
text(10, 1.3, "2.60 mHz", color='w')
text(10, 4.92, "3.84 mHz", color='w')

axis_param(im,'') 
#xlabel('Detection Time (mins)')


ax=subplot(322)
data_ft, f, tt = stft_band_pass_filter(avtec_27, avtime_27, lowcut=0.002, highcut=0.01, window_length=32, overlap=16)
# Use imshow for displaying the 2D data
im1 = ax.imshow(np.abs(data_ft*10), aspect='auto', extent=[0, tt.max() / 60, (f*1e3).min(), (f*1e3).max()], 
               cmap='jet', origin='lower')

im1.set_interpolation("spline16")



#text(-138, 9, file[17:24], color='white')
#xlabel('Detection Time (mins)')
ylabel('Frequency (mHz)')
xlim(0,50)
ylim(0.01,10)
text(45, 9, 'G27', color='w')
text(2.5, 8.8, '(b1)', color='w')
text(19, 4.9, "2.80 mHz", color='w')
text(11, 6.5, "3.82 mHz", color='w')
axis_param(im1,'') 
#axis_param(im, 0.1, '')
tight_layout()

    
    
figure(20, figsize=(10,10))
ax=subplot(321)

filtered_data_27 = high_pass_filter_stec(avtec_27, avtime_27) 
filtered_data_16 = high_pass_filter_stec(avtec_16, avtime_16)
filtered_data_31 = high_pass_filter_stec(avtec_31, avtime_31)

plot(avtime_27*60, 10*filtered_data_27)
grid(ls = '-.', lw=0.4)
ylabel("dTEC (TECU)")
text(45, 0.24, 'G27')
text(0, 0.24, '(a1)')
xlim(0,50)


ax=subplot(323)
plot(avtime_16*60, 10*filtered_data_16)
grid(ls = '-.', lw=0.4)
# xlabel('Detection Time (mins)')
ylabel("dTEC (TECU)")
text(45, 0.1, 'G16')
text(0, 0.1, '(a2)')
xlim(0,50)
tight_layout()



# figure(21)
# ax=subplot(111)
# data_ft, f, tt = stft_band_pass_filter(avtec_27, avtime_27, lowcut=0.0005, highcut=0.005, window_length=100, overlap=90, dt=60)
# # Use imshow for displaying the 2D data

# height, width = data_ft.shape  # dimensions of the 2D data

# im1 = ax.imshow(np.abs(data_ft), aspect='auto',
#                 extent=[0, width, (f*1e3).min(), (f*1e3).max()],
#                 cmap='jet', origin='lower')

# #extent=[0, width, (f*1e3).min(), (f*1e3).max()],
# # im1 = ax.imshow(np.abs(data_ft), aspect='auto', extent=[tt.min(), tt.max(), (f*1e3).min(), (f*1e3).max()], 
# #                 cmap='jet', origin='lower')
# im1.set_interpolation("spline16")
# #text(-138, 9, file[17:24], color='white')
# xlabel('Detection Time (mins)')
# ylabel('Frequency (mHz)')
# ylim(0,5)
#%%







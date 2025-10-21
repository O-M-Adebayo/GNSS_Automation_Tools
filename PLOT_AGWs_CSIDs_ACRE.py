
from pylab import *
from numpy import *
from scipy import *
from scipy.ndimage import *
from scipy.special import erf
from scipy.integrate import trapz
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import *
from signal_alam import find_peaks,wavelet,convolve_al
from adebayo_codigo import band_pass_filter
import matplotlib.ticker as ticker
import glob
import pandas as pd
close('all')

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
# def axis_param(im,vm,nm,str_c):
#     divider = make_axes_locatable(ax);
#     cax = divider.append_axes("right", size="2%", pad=0.05)
#     cbar=colorbar(im,cax=cax);gca().set_title(str_c,fontsize=14) 
#     cbar.set_ticks(linspace(-vm,vm,5))
#     vm_n=round(vm*nm,1)
#     cbar.set_ticklabels(linspace(-vm_n,vm_n,5)) 

#     return()

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
        n_smooth=int(ceil(9./(dt*60.)))
        win=ones((n_smooth,))/n_smooth
        data=-data+convolve_al(data,win)
        return convolve_al(data,ones((5,))/5)    
#%%



lat_ep=-7 ;lon_ep=-71
year,month,day=2014, 1, 20
hour, minute, second=21, 31, 5
t_ep=hour+minute/60+second/3600

#%%
dy=10;dx=5*dy/2;#dt=5*dy/2;                                                            #A resoluções espaciais no kilometros
y=arange(0,300+dy,dy);ny=len(y)                                           #A faixa de altitude
x=arange(0,1000+dx,dx);nx=len(x)                                        #A faixa de longitude
[y2,x2]=meshgrid(y,x)
y2_m=y2*1.e+03;x2_m=x2*1.e+03
                                             #As resoluções espaciais no metros
i_alt=abs(y-80).argmin()
yi=y[i_alt:]
it=0;nt=100;t=0;#t_f=nt*dt

t=load('/home/oluwasegun-adebayo/Desktop/Simulation Work/time.npy')

t_hr=t/60.
[t2,yt2]=meshgrid(t,y)
data_amb=load('/home/oluwasegun-adebayo/Desktop/Simulation Work/data_amb.npy');rho_amb=data_amb[:,0][0];sn=data_amb[:,1][0]
wx3=load('/home/oluwasegun-adebayo/Desktop/Simulation Work/wx3.npy')
wy3=load('/home/oluwasegun-adebayo/Desktop/Simulation Work/wy3.npy')
n3=1*load('/home/oluwasegun-adebayo/Desktop/Simulation Work/n3.npy');dn3=1*(len(t)/2.)*gradient(n3)[0];


i_xo=abs(x-x.mean()).argmin()
i_xo=abs(x-0.).argmin()

#%%

fig=figure(1,figsize=(8,12),facecolor='w',edgecolor='k')
ax = subplot(221)
#
plot(sn,y,lw=2)
#text(264,280,'(A)')
grid('on')
xlabel('m/s')
title('Neutral sound speed')
ylim(0,300)

ax=subplot(222)
semilogx(t_hr[0]+1*n3[0,i_xo,:],yi,lw=2)
#text(264,1e+06,'(B)')
title('Electron number density')
ylim(0,300)
xlabel('m$^{-3}$')
grid('on')

ax=subplot(223)
# v1=(1.e-03*3600.)*sn[:20].mean()
# plot(t_ep+(1000/3600.)*y/sn,y,'go',markersize=4)
data=-2*wy3[:,i_xo,:].T
vm=50
ext=[t_hr[0],t_hr[-1],y2[0,0],y2[0,-1]]
im=imshow(data,origin='lower',extent=ext,
          cmap=cm.rainbow,vmin=-vm,vmax=vm)
im.set_interpolation('spline16')
#text(6.5,380,'(C)')
axis('tight');
grid('on')
ylabel('Altitude, km')
xlabel('Time, UT')
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
#plot(t_hr[0]+0.35*n3[0,i_xo,:]/n3.max(),yi,'g',lw=2)
#text(6.5,380,'(D)')

for iy in range (7,len(yi),2):
    
    n2=3*n3[:,i_xo,iy]
    dt_hr=t_hr[1]-t_hr[0];n_f=int(5/(dt_hr*60.))
    win=ones((n_f,))/n_f
    dn2=n2-convolve_al(n2,win)
    plot(t_hr,yi[iy]+5e+04*(-dn2/n3.max()),'r',lw=1)
    
    if iy == 9:
        plot(t_hr,yi[iy]+5e+04*(-dn2/n3.max()),'k',lw=2)
    
    
    # n2=1*n3[:,i_xo,iy]
    # dt_hr=t_hr[1]-t_hr[0];n_f=int(4/(dt_hr*60.))
    # win=ones((n_f,))/n_f
    # dn2=+n2-convolve_al(n2,win)
    # plot(t_hr,yi[iy]+4e+04*(dn2/n3.max()),'r',lw=1)

axis('tight');
grid('on')
#ylabel('Altitude, km')
xlabel('Time, UT')
title('Ionospheric density disturbance')
#text(2.7,285,'(B)')
axis((t_hr[0],t_hr[-1],y2[0,0],y2[0,-1]))
ax.yaxis.set_label_position("left")
ax.yaxis.tick_left()
#ax.yaxis.set_ticks_position('both')

tight_layout()
#%%

fig=figure(3,figsize=(13,12),facecolor='w',edgecolor='k')

ax =subplot(1,2,1)
for ix in range (0,len(x),2):
    #wy3[:,i_xo,0]
    plot(t_hr[::1],(x[ix])+1e+02*wy3[:,ix,0][::1],'g')
    xlim(t_hr[0],t_hr[-1])
    ylabel('Epicentral Distance, km ')
    xlabel('Time, UT')
    if ix==0:
        v1=(3.5/1)/(1./3600.)
        axline((t_ep+100/3600,0),slope=v1,c='k' ,label = '~4km/s')

        legend()
        axvline(x=t_ep,color='grey',lw=1)  
  
ax =subplot(1,2,2)
for ix in range (0,len(x),2):   
    tec=(1e+06/1.e+16)*dn3[:,ix,7:23].sum(1)
    dtec=data_smooth_sim(t_hr,tec)
    nm=1

    vm=dtec.max()/2
    xlim(t_hr[0],t_hr[-1])
    im=scatter(t_hr,0.2*dtec+(x[ix]),c=dtec,s=20.,
                vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow,alpha=1)   

    if ix==0:
        v1=(4/1)/(1./3600.)
        axline((t_ep+420/3600,0),slope=v1,c='k' ,label = '~4 km/s')
        v1=(1/1)/(1./3600.)
        axline((t_ep+420/3600,0),slope=v1,c='g' ,label = '~1 km/s')
        legend()
        axvline(x=t_ep,color='grey',lw=1)    
        
#%%%%

f_list=sorted(glob.glob('brazil_tec/***.npy'));#0937# 0261#0205
for f_n in f_list:  
    data_gnss = load(f_n, allow_pickle=True).item()
    
    #data_gnss=load(f_n,allow_pickle=True)
    t_gnss=data_gnss['t_gnss'];tec_gnss=data_gnss['tec_gnss'];
    lon_gnss=data_gnss['lon_gnss'];lat_gnss=data_gnss['lat_gnss'];
    ep_dist = data_gnss['ep_dist']
  
    # v_prox=find_nearest(t_gnss, t_ep+8.467*60/3600)
    # p_sip=where(t_gnss==v_prox)[0]
    # ep_sign=1
    # if lon_gnss[p_sip]-lon_ep <0:ep_sign=1
    # if lat_gnss[p_sip]-lat_ep <0:ep_sign=-1
    # [d1,d2]=distkm(lon_ep,lat_ep,([lon_gnss[p_sip]]),([lat_gnss[p_sip]]))
    # ep_dist=((d1+d2)[0]/2)/1.
    #dtec_gnss=data_smooth(t_gnss,tec_gnss)
    if 'pove' not in f_n:
        
        dtec_gnss = band_pass_filter(tec_gnss, t_gnss, 0.002, 0.005)
        fig=figure(4,figsize=(10,8),facecolor='w',edgecolor='k')
        ax=subplot(121)
        vm=0.06
        # plot(t_gnss,ep_dist+250*dtec_gnss,lw=2)
        
        if ep_dist < 500:
            dtec_gnss[0:149] = band_pass_filter(dtec_gnss[0:149], t_gnss, lowcut=0.0045, highcut=0.006)
            
            im=scatter(t_gnss,ep_dist+500*dtec_gnss,c=0.6*dtec_gnss,s=20,
                        vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow,alpha=1)
            
        elif ep_dist > 500 and 'amte' in f_n:
            
            dtec_gnss[0:166] = band_pass_filter(dtec_gnss[0:166], t_gnss, lowcut=0.0035, highcut=0.006)
            
            im=scatter(t_gnss,ep_dist+500*dtec_gnss,c=0.6*dtec_gnss,s=20,
                        vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow,alpha=1)
            
        else:
            dtec_gnss[0:165] = band_pass_filter(dtec_gnss[0:165], t_gnss, lowcut=0.0035, highcut=0.006)
            
            im=scatter(t_gnss,ep_dist+500*dtec_gnss,c=0.6*dtec_gnss,s=20,
                        vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow,alpha=1)
        
        
        axvline(t_ep,0,1,c='grey',linestyle='--')
        grid(ls='-.', lw=0.4)
        axis((21.4,22.5,0.,1000))
       
        ylabel('Epicentral Distance, km')
        xlabel('Time, UT')
        title("Observation")
        text(21.9, ep_dist+20, f_n[11:18], fontsize=8)
        
        if 'cruz' in f_n:
            text(21.7, 500, 'Phase speed:\n 3.0 km/s')
            
            v1=(3.0/1)/(1./3600.)
            axline((21.62+100/3600,0),slope=v1,c='k', lw=1, label = '~4km/s')
            
            v1=(0.6/1)/(1./3600.)
            axline((21.62+100/3600,0),slope=v1,c='k', lw=1, label = '~4km/s')
            
            f = axis_param(im, 'TECU')
        
        
        ax=subplot(122)
        #plot(t_gnss,ep_dist+250*dtec_gnss,'r',lw=1)
        i_ep=abs(x-ep_dist).argmin()
        tec=((3e+06/1.e+16))*n3[:,i_ep,7:23].sum(1)
        xlabel('Time, UT')
    
    #%%

      
        dtec=data_smooth_sim(t_hr,tec)
        
        vm=0.06
        #plot(t_hr,ep_dist+50*(dtec),'k',lw=1.5)
        if ep_dist < 500:
            im=scatter(t_hr,ep_dist+400*(dtec),c=dtec,s=20,
                       
                       vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow,alpha=0.8)
            
        else:
            im=scatter(t_hr,ep_dist+700*(dtec),c=dtec,s=20,
                       vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow,alpha=0.8)
            
        
        #if i_ep==1:
            # v1=(2/1)/(1./3600.)
            # axline((t_ep+560/3600,0),slope=v1,c='k' ,label = '~4 km/s')
            # v1=(1/1)/(1./3600.)
            
            # axline((t_ep+420/3600,0),slope=v1,c='g' ,label = '~1 km/s')
            # legend()
            # axvline(x=t_ep,color='grey',lw=1)    
            # legend()
        axis((21.4,22.5,0.,1000))
      # xvline(t_ep,0,1,c='grey',linestyle='--')
        title("Simulation")
        grid(ls='-.', lw=0.4)
        
        
        
        if 'cruz' in f_n:
            
            text(21.7, 500, 'Phase speed:\n 3.0 km/s')
            v1=(3.0/1)/(1./3600.)
            axline((21.62+100/3600,0),slope=v1,c='k', lw=1, label = '~4km/s')
            
            f = axis_param(im, 'TECU')
        
        tight_layout()
        
        # Comparative plot
        fig=figure(5,figsize=(5,7),facecolor='w',edgecolor='k')
        ax=subplot(111)
        
        # vm=0.06
        # # plot(t_gnss,ep_dist+250*dtec_gnss,lw=2)
        # im=scatter(t_gnss,ep_dist+500*dtec_gnss,c=0.6*dtec_gnss,s=20,
        #             vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow,alpha=1)
    
        # #text(6.57,ep_dist,f_n[9:16],size='small',color='k')
    
        # axvline(t_ep,0,1,c='grey',linestyle='--')
        # # v1=(3./111)/(1./3600.)
        # # axline((t_ep+8/60,0),slope=v1,c='g' ,label = '~1 km/s')
        
        
        # vm=dtec.max()/2 
        # #plot(t_hr,ep_dist+50*(dtec),'k',lw=1.5)
        # im=scatter(t_hr,ep_dist+2.4*(dtec),c=dtec,s=20,
        #             vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow,alpha=0.8)
        
        #dtec_gnss[0:150] = abs(dtec_gnss[0:150]) - 0.03
        
        
        if ep_dist < 500:
            dtec_gnss[0:150] = band_pass_filter(dtec_gnss[0:150], t_gnss, lowcut=0.0035, highcut=0.006)
            plot(t_gnss,ep_dist+500*dtec_gnss, color='b', lw=2.5, label = 'Observation')
            
            plot(t_hr,ep_dist+500*(dtec), color='r', lw=2.5, label = 'Simulation')
            
        elif ep_dist > 500 and 'amte' in f_n:
            
            dtec_gnss[0:166] = band_pass_filter(dtec_gnss[0:166], t_gnss, lowcut=0.0035, highcut=0.006)
            
            plot(t_gnss,ep_dist+500*dtec_gnss, color='b', lw=2.5)
            
            plot(t_hr,ep_dist+1000*(dtec), color='r', lw=2.5)
            
        else:
            
            dtec_gnss[0:165] = band_pass_filter(dtec_gnss[0:165], t_gnss, lowcut=0.0035, highcut=0.006)
            
            plot(t_gnss,ep_dist+500*dtec_gnss, color='b', lw=2.5)
            
            plot(t_hr,ep_dist+1000*(dtec), color='r', lw=2.5)
            
        
        axvline(t_ep,0,1,c='grey',linestyle='--')
        
        # v1=(3./111)/(1./3600.)
        # axline((t_ep+8/60,0),slope=v1,c='g' ,label = '~1 km/s')
        ylabel('Epicentral Distance, km')
        xlabel('Time, UT')
        axis((21.4,22.0,0.,1000))
        grid(ls = '-.', lw=0.4)
        
        if 'cruz' in f_n:
            legend(loc='lower right')
            text(21.7, 500, 'Phase speed:\n 3.0 km/s')
            
            v1=(3.0/1)/(1./3600.)
            axline((21.62+100/3600,0),slope=v1,c='k', lw=1, label = '~4km/s')
        
    
tight_layout()

     

#%%

# def find_peaks_within_range(arr, start_idx, end_idx, threshold):
#     peaks_indices = []
    
#     # Ensure start_idx and end_idx are within valid bounds
#     start_idx = max(0, start_idx)
#     end_idx = min(len(arr), end_idx)
    
#     # Loop over the range with proper boundary checking
#     for i in range(start_idx + 1, end_idx - 1):
#         if i - 4 >= 0 and i + 4 < len(arr):  # Ensure that the indices are within bounds
#             if arr[i] > arr[i - 4] and arr[i] > arr[i + 4]:
#                 if arr[i] > threshold:
#                     peaks_indices.append(i)
                
#     return peaks_indices


# def find_peaks_within_range_obs(arr, start_idx, end_idx, threshold):
#     peaks_indices = []
    
#     # Ensure start_idx and end_idx are within valid bounds
#     start_idx = max(0, start_idx)
#     end_idx = min(len(arr), end_idx)
    
#     # Loop over the range with proper boundary checking
#     for i in range(start_idx + 1, end_idx - 1):
#         if i - 5 >= 0 and i + 5 < len(arr):  # Ensure that the indices are within bounds
#             if arr[i] > arr[i - 5] and arr[i] > arr[i + 5]:
#                 if arr[i] > threshold:
#                     peaks_indices.append(i)
                
#     return peaks_indices

# def find_peaks_within_range_k(arr, start_idx, end_idx, threshold, k):
#     peaks_indices = []
    
#     # Ensure start_idx and end_idx are within valid bounds
#     start_idx = max(0, start_idx)
#     end_idx = min(len(arr), end_idx)
    
#     # Loop over the range with proper boundary checking
#     for i in range(start_idx + 1, end_idx - 1):
#         if i - k >= 0 and i + k < len(arr):  # Ensure that the indices are within bounds
#             if arr[i] > arr[i - k] and arr[i] > arr[i + k]:
#                 if arr[i] > threshold:
#                     peaks_indices.append(i)
                
#     return peaks_indices


# #%%
# f_list=sorted(glob.glob('DATA_TEC/***.npy'));#0937# 0261#0205
# for f_n in f_list:  
#     data_gnss=load(f_n,allow_pickle=True)
#     t_gnss=data_gnss[0];tec_gnss=data_gnss[1];
#     lon_gnss=data_gnss[2];lat_gnss=data_gnss[3];
  
#     v_prox=find_nearest(t_gnss, t_ep+8.467*60/3600)
#     p_sip=where(t_gnss==v_prox)[0]
#     ep_sign=1
#     if lon_gnss[p_sip]-lon_ep <0:
#         ep_sign=1
#     if lat_gnss[p_sip]-lat_ep <0:
#         ep_sign=-1
#     [d1,d2]=distkm(lon_ep,lat_ep,([lon_gnss[p_sip]]),([lat_gnss[p_sip]]))
#     ep_dist=((d1+d2)[0]/2)/1.
#     dtec_gnss=data_smooth(t_gnss,tec_gnss)
    
#     dtec_gnss[dtec_gnss <= 0.1] = 0

    
#     dtec_peak_obs = find_peaks_within_range_k(dtec_gnss, 80, 280, 0.1, 3)
    
#     time_obs = (t_gnss[dtec_peak_obs] - t_ep)*3600
#     dist_obs = (lat_gnss[dtec_peak_obs] - lat_ep)*110
#     vel_obs = dist_obs / time_obs
    
#     fig=figure(6,figsize=(10,12),facecolor='w',edgecolor='k')
#     ax=subplot(221)
    
    
#     # im=scatter(t_gnss,ep_dist+150*dtec_gnss,c=0.6*dtec_gnss,s=1.8*t_gnss,
#     #             vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow,alpha=1)
    
#     #scatter(t_hr[dtec_peak_obs], dtec_gnss[dtec_peak_obs])
    
#     scatter(t_gnss[dtec_peak_obs], ep_dist+(dtec_gnss[dtec_peak_obs]), c='b')

#     #text(6.47,ep_dist,f_n[9:16],size='small',color='k')
    
#     # axvline(x=t_ep,color='grey',lw=2) 
#     # axhline(850, color = 'r', linestyle='--')
#     # # v1=(3./111)/(1./3600.)
#     # # axline((t_ep+8/60,0),slope=v1,c='g' ,label = '~1 km/s')
#     xlim(6.6, 7.4)
#     #axis((t_gnss[dtec_peak_obs][0],t_gnss[dtec_peak_obs][-1],0.,2300))
#     xlabel('Time (hr, UT)')
#     ylabel('Epicentral Distance (km)')   
#     title('Observation', fontweight = 'bold')
#     grid(linestyle='-.', lw=0.4)
# text(6.834, 1122, "Fast mode", rotation = 90)
# text(7.090, 1110, "Slow mode", rotation = 46)
# #text(6.61, 1810, '(a)', fontsize=15)
    
# #f=  axis_param(im,vm,4.4,'TECU')
    
    
    
  
# ax=subplot(222)    
# for ix in range (0,len(x),2):   
#     tec=(1e+06/1.e+16)*dn3[:,ix,7:23].sum(1)
#     dtec=data_smooth_sim(t_hr,tec)
#     nm=1
    
#     vm=dtec.max()/2
#     #dtec[dtec <= 0.005] = 0
    
#     dtec_peak_sim = find_peaks_within_range_k(dtec, 20, 250, 0.06, 1)
    
#     #scatter(t_hr[dtec_peak_sim], dtec[dtec_peak_sim])
    
#     scatter(t_hr[dtec_peak_sim], dtec[dtec_peak_sim]+(x[ix]), c='b', vmax=vm, vmin=-vm) 
    
    
#     xlim(6.6, 7.8)
#     xlabel('Time (hr, UT)')
#     #ylabel('Epicentral Distance, km')   
#     title('Simulation', fontweight = 'bold')
#     grid(linestyle='-.', lw=0.4)
# text(6.812, 1171, "Fast mode", rotation = 90)
# text(7.092, 722, "Slow mode", rotation = 45)
# #text(6.61, 2215, '(b)', fontsize=15)

# #%%
# # Velocity distribution

# f_list=sorted(glob.glob('DATA_TEC/***.npy'));#0937# 0261#0205
# for f_n in f_list:  
#     data_gnss=load(f_n,allow_pickle=True)
#     t_gnss=data_gnss[0];tec_gnss=data_gnss[1];
#     lon_gnss=data_gnss[2];lat_gnss=data_gnss[3];
  
#     v_prox=find_nearest(t_gnss, t_ep+8.467*60/3600)
#     p_sip=where(t_gnss==v_prox)[0]
#     ep_sign=1
#     if lon_gnss[p_sip]-lon_ep <0:
#         ep_sign=1
#     if lat_gnss[p_sip]-lat_ep <0:
#         ep_sign=-1
#     [d1,d2]=distkm(lon_ep,lat_ep,([lon_gnss[p_sip]]),([lat_gnss[p_sip]]))
#     ep_dist=((d1+d2)[0]/2)/1.
#     dtec_gnss=data_smooth(t_gnss,tec_gnss)
    
#     dtec_gnss[dtec_gnss <= 0.02] = 0

    
#     dtec_peak_obs = find_peaks_within_range_k(dtec_gnss, 80, 250, 0.05, 1)
    
#     # 80, 250, 0.015
    
#     time_obs = (t_gnss[dtec_peak_obs] - 6.74)*3600
#     dist_obs = (lat_gnss[dtec_peak_obs] + 33.762)*110
#     vel_obs = (ep_dist) / time_obs
#     #vel_obs = (dist_obs) / (time_obs)
    
#     fig=figure(6,figsize=(11,12),facecolor='w',edgecolor='k')
#     ax=subplot(223)
    
    
   
#     scatter(dist_obs, vel_obs, c='r')
    
    
#     ylim(0, 3.5)
    
#     #axis((t_gnss[dtec_peak_obs][0],t_gnss[dtec_peak_obs][-1],0.,2300))
#     ylabel('Propagating speed (km/s)')
#     xlabel('Epicentral Distance (km)')   
#     #title('Observation', fontweight = 'bold')
#     grid(linestyle='-.', lw=0.4)
# text(697, 2.14, "Fast mode", rotation = 50)
# text(1035, 0.75, "Slow mode", rotation = 0)
# text(-109, 3.30, '(c)', fontsize=15)   
# #f=  axis_param(im,vm,4.4,'TECU')
    

# ax=subplot(224)    
# for ix in range (0,len(x),2):   
#     tec=(1e+06/1.e+16)*dn3[:,ix,7:23].sum(1)
#     dtec=data_smooth_sim(t_hr,tec)
#     nm=1
    
    
    
#     dtec_peak_sim = find_peaks_within_range_k(dtec, 30, 150, 0.1, 1)
    
#     time_sim = (t_hr[dtec_peak_sim] - 6.69)*3600
    
#     dist_sim = dtec[dtec_peak_sim]+(x[ix])
#     vel_sim = (dist_sim) / time_sim 
    
#     # x1, y1 = [588, 2290], [0.567, 0.762]
#     # x2, y2 = [0, 2214], [0, 1.935]
#     # plt.plot(x1, y1, x2, y2, lw=0.4)
    
#     scatter(dist_sim, vel_sim, c='r')
    
    
#     ylim(-0.05, 3.5)
#     #plot(598, 0.598, 2226, 2.026)  
    
#     #ylabel('Propagating speed (km/s)')
#     xlabel('Epicentral Distance (km)')   
    
#     #title('Simulation', fontweight = 'bold')
#     grid(linestyle='-.', lw=0.4)
# text(819, 1.6, "Fast mode", rotation = 40)
# text(1092, 0.381, "Slow mode", rotation = 5)
# text(-30, 3.30, '(d)', fontsize=15)
# tight_layout() 
# #savefig("./figures_pub/velocity.png", bbox_inches = 'tight') 
 

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
import glob
import pandas as pd
plt.close("all")

params={'axes.labelweight':'bold'};rcParams.update(params)
font={'family':'sans-serif','weight':'bold','size':11};matplotlib.rc('font', **font);

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
        return convolve_al(data,ones((2,))/2)  
    
 
def plot_tec(dtec, latitude, longitude):
    # Create a 2D meshgrid of latitude and longitude
    lon, lat = np.meshgrid(longitude, latitude)
    
    # Check that the shapes match between the input arrays
    if dtec.shape != lat.shape or dtec.shape != lon.shape:
        raise ValueError("dtec, latitude, and longitude must have compatible shapes.")
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot the TEC using a pcolormesh or contour plot
    plt.pcolormesh(lon, lat, dtec, cmap='jet', shading='auto')
    plt.colorbar(label='TEC (Total Electron Content)')
    
    # Set axis labels
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    # Add title
    plt.title('Total Electron Content (TEC) as a function of Latitude and Longitude')
    
    # Show the plot
    plt.show()

# Example usage:
# dtec = np.random.random((100, 100))  # Replace with actual TEC data
# latitude = np.linspace(-90, 90, 100)  # Replace with actual latitude data
# longitude = np.linspace(-180, 180, 100)  # Replace with actual longitude data

# plot_tec(dtec, latitude, longitude)

#%%



lat_ep=-36.122 ;lon_ep=-72.898
year,month,day=2010, 2, 27
hour, minute, second=6, 34, 11
t_ep=hour+minute/60+second/3600


#%%
dy=10;dx=5*dy;#dt=5*dy/2;                                                            #A resoluções espaciais no kilometros
y=arange(0,400+dy,dy);ny=len(y)                                           #A faixa de altitude
x=arange(0,2200+dx,dx);nx=len(x)                                        #A faixa de longitude
[y2,x2]=meshgrid(y,x)
y2_m=y2*1.e+03;x2_m=x2*1.e+03
                                             #As resoluções espaciais no metros
i_alt=abs(y-80).argmin()
yi=y[i_alt:]
it=0;nt=100;t=0;#t_f=nt*dt

t=load('time.npy')

t_hr=t/60.
[t2,yt2]=meshgrid(t,y)
data_amb=load('data_amb.npy');rho_amb=data_amb[:,0][0];sn=data_amb[:,1][0]
wx3=load('wx3.npy')
wy3=load('wy3.npy')
n3=1*load('n3.npy');dn3=1*(len(t)/2.)*gradient(n3)[0];


i_xo=abs(x-x.mean()).argmin()
i_xo=abs(x-0.).argmin()

#%%

fig=figure(1,figsize=(8,12),facecolor='w',edgecolor='k')
ax = subplot(221)
#
plot(sn,y,lw=2)
text(264,280,'(A)')
grid('on', linestyle = '-.', lw = 0.4)
xlabel('m/s')
ylabel("Altitude (km)")
title('Neutral sound speed', fontsize = 9, fontweight = 'bold')
ylim(0,300)

ax=subplot(222)
semilogx(t_hr[0]+0.2*n3[0,i_xo,:],yi,lw=2)
title('Electron number density', fontsize = 9, fontweight = 'bold')
text(4e05,280,'(B)')
ylim(0,300)
xlabel('m$^{-3}$')
grid('on', linestyle = '-.', lw = 0.4)

ax=subplot(223)
# v1=(1.e-03*3600.)*sn[:20].mean()
# plot(t_ep+(1000/3600.)*y/sn,y,'go',markersize=4)
data=100*wy3[:,i_xo,:].T
vm=int(data.max())
ext=[t_hr[0],t_hr[-1],y2[0,0],y2[0,-1]]
im=imshow(data,origin='lower',extent=ext,
          cmap=cm.seismic,vmin=-vm,vmax=vm)
im.set_interpolation('bilinear')
text(6.46,380,'(C)')
axis('tight');
grid('on', linestyle = '-.', lw = 0.4)
ylabel("Altitude (km)")
xlabel('Time (hr, UT)')
axvline(t_ep, color = 'grey')
title('Acoustic-gravity wave amplitude', fontsize = 9, fontweight = 'bold')


#text(2.7,285,'(A)')
axis((t_hr[0],t_hr[-1],y2[0,0],y2[0,-1]))
ax.yaxis.set_ticks_position('both')
cbar=colorbar(im, cax =fig.add_axes([0.14, 0.45, 0.3, 0.01]),\
            orientation='horizontal')
gca().set_xlabel('u$_y$, m/s') 
cbar.set_ticks(linspace(-vm,vm,5))
#cbar.set_ticks(linspace(round(-vm,0),round(vm,0),5))

ax=subplot(224)
data=100*(1e+06/1.e+16)*(dn3[:,i_xo,:]/n3.max()).T
#vm=data.max()/2.
ext=[t_hr[0],t_hr[-1],y2[0,7],y2[0,-1]]
plot(t_hr[0]+0.35*n3[0,i_xo,:]/n3.max(),yi,'g',lw=2)
text(6.46,380,'(D)')
axvline(t_ep, color = 'grey')

for iy in range (7,len(yi),1):
    n2=1*n3[:,i_xo,iy]
    dt_hr=t_hr[1]-t_hr[0];n_f=int(4/(dt_hr*60.))
    win=ones((n_f,))/n_f
    dn2=+n2-convolve_al(n2,win)
    plot(t_hr,yi[iy]+1e+07*dn2/n3.max(),'r',lw=1)

axis('tight');
grid('on', linestyle = '-.', lw = 0.4)
#ylabel("Altitude (km)")
xlabel('Time (hr, UT)')
title('Ionospheric density disturbance', fontsize = 9, fontweight = 'bold')
axis((t_hr[0],t_hr[-1],y2[0,0],y2[0,-1]))
ax.yaxis.set_label_position("left")
ax.yaxis.tick_left()
ax.yaxis.set_ticks_position('both')
fig.tight_layout()

#savefig("./figures_pub/atmosp_parameters.png", bbox_inches = 'tight')
#%%

# detec_time = (t_hr - t_ep) * 3600

# fig=figure(3,figsize=(13,12),facecolor='w',edgecolor='k')

# ax =subplot(1,2,1)
# for ix in range (0,len(x),2):
#     #wy3[:,i_xo,0]
#     plot(t_hr[::1],(x[ix])+1e+04*wy3[:,ix,0][::1],'g')
#     #plot(detec_time[::1],(x[ix])+1e+04*wy3[:,ix,0][::1],'g')
#     xlim(t_hr[0],t_hr[-1])
#     ylabel('Epicentral Distance, km ')
#     xlabel('Time, UT')
#     if ix==0:
#         v1=(4/1)/(1./3600.)
#         axline((t_ep+100/3600,0),slope=v1,c='k' ,label = '~4km/s')
#         #axline((detec_time+100,0),slope=v1,c='k' ,label = '~4km/s')

#         legend()
#         axvline(x=t_ep,color='grey',lw=1)  
  
# ax =subplot(1,2,2)
# for ix in range (0,len(x),2):   
#     tec=(1e+06/1.e+16)*dn3[:,ix,7:23].sum(1)
#     dtec=data_smooth_sim(t_hr,tec)
#     nm=1

#     vm=dtec.max()/2
#     xlim(t_hr[0],t_hr[-1])
#     im=scatter(t_hr,1e+03*dtec+(x[ix]),c=dtec,s=20.,
#                 vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow,alpha=1)   

#     if ix==0:
#         v1=(4/1)/(1./3600.)
#         axline((t_ep+420/3600,0),slope=v1,c='k' ,label = '~4 km/s')
#         v1=(1/1)/(1./3600.)
#         axline((t_ep+420/3600,0),slope=v1,c='g' ,label = '~1 km/s')
#         legend()
#         axvline(x=t_ep,color='grey',lw=1)    
  

      
#%%%%

f_list=sorted(glob.glob('DATA_TEC/***.npy'));#0937# 0261#0205
for f_n in f_list:  
    data_gnss=load(f_n,allow_pickle=True)
    t_gnss=data_gnss[0];tec_gnss=data_gnss[1];
    lon_gnss=data_gnss[2];lat_gnss=data_gnss[3];
  
    v_prox=find_nearest(t_gnss, t_ep+8.467*60/3600)
    p_sip=where(t_gnss==v_prox)[0]
    ep_sign=1
    if lon_gnss[p_sip]-lon_ep <0:
        ep_sign=1
    if lat_gnss[p_sip]-lat_ep <0:
        ep_sign=-1
    [d1,d2]=distkm(lon_ep,lat_ep,([lon_gnss[p_sip]]),([lat_gnss[p_sip]]))
    ep_dist=((d1+d2)[0]/2)/1.
    dtec_gnss=data_smooth(t_gnss,tec_gnss)
    fig=figure(4,figsize=(11,12),facecolor='w',edgecolor='k')
    ax=subplot(121)
    vm=0.1
   # plot(t_gnss,ep_dist+250*dtec_gnss,lw=2)
    im=scatter(t_gnss,ep_dist+150*dtec_gnss,c=0.6*dtec_gnss,s=1.8*t_gnss,
                vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow,alpha=1)

    text(6.47,ep_dist,f_n[9:16],size='small',color='k')
    
    axvline(t_ep,0,1,c='grey',linestyle='--')
    # v1=(3./111)/(1./3600.)
    # axline((t_ep+8/60,0),slope=v1,c='g' ,label = '~1 km/s')
    axis((t_hr[0],t_hr[-1],0.,2300))
   # f=  (axis_param(im,vm,4.4,'TECU'))
    ylabel('Epicentral Distance, km')
    xlabel('Time, UT')
    ax=subplot(122)
    plot(t_gnss,ep_dist+250*dtec_gnss,'r',lw=1)
    i_ep=abs(x-ep_dist).argmin()
    tec=((5e+8/1.e+16))*n3[:,i_ep,7:23].sum(1)
    xlabel('Time (hr, UT)')
    #%%

        
    dtec=data_smooth_sim(t_hr,tec)
    plot(t_hr,ep_dist+250*(dtec),'k',lw=1.5)
    
    if i_ep==1:
        v1=(4/1)/(1./3600.)
        axline((t_ep+560/3600,0),slope=v1,c='k' ,label = '~4 km/s')
        v1=(1/1)/(1./3600.)
       
        axline((t_ep+420/3600,0),slope=v1,c='g' ,label = '~1 km/s')
        legend()
        axvline(x=t_ep,color='grey',lw=1)    
        legend()
    axis((t_hr[0],t_hr[-1],0.,2300))
  #  axvline(t_ep,0,1,c='grey',linestyle='--')
    
      


#%%
# Observation and Simulation
f_list=sorted(glob.glob('DATA_TEC/***.npy'));
for f_n in f_list:  
    data_gnss=load(f_n,allow_pickle=True)
    t_gnss=data_gnss[0];tec_gnss=data_gnss[1];
    lon_gnss=data_gnss[2];lat_gnss=data_gnss[3];
  
    v_prox=find_nearest(t_gnss, t_ep+8.467*60/3600)
    p_sip=where(t_gnss==v_prox)[0]
    ep_sign=1
    if lon_gnss[p_sip]-lon_ep <0:
        ep_sign=1
    if lat_gnss[p_sip]-lat_ep <0:
        ep_sign=-1
    [d1,d2]=distkm(lon_ep,lat_ep,([lon_gnss[p_sip]]),([lat_gnss[p_sip]]))
    ep_dist=((d1+d2)[0]/2)/1.
    dtec_gnss=data_smooth(t_gnss,tec_gnss)
    fig=figure(5,figsize=(10,8),facecolor='w',edgecolor='k')
    ax=subplot(121)
    vm=0.1
    # plot(t_gnss,ep_dist+250*dtec_gnss,lw=2)
    im=scatter(t_gnss,ep_dist+150*dtec_gnss,c=0.8*dtec_gnss,s=1.8*t_gnss,
                vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow,alpha=1)
    
    # if f_n == 'DATA_TEC/areqG11.npy':
    #     v1=(4/1)/(1./3600.)
    #     axline((t_ep+500/3600,0),slope=v1,c='k' ,label = 'Rayleigh ~2.5 km/s')
    #     v1=(1/1)/(1./3600.)
    #     axline((t_ep+500/3600,0),slope=v1, c='k', linestyle = '--',
    #             label = 'Acoustic ~1.5 km/s')
    #     legend()
        
    # else: continue
    
    
    # ##

    #text(6.47,ep_dist,f_n[9:16],size='small',color='k')
    
    axvline(x=t_ep,color='grey',lw=2) 
    # v1=(3./111)/(1./3600.)
    # axline((t_ep+8/60,0),slope=v1,c='g' ,label = '~1 km/s')
    axis((t_hr[0],t_hr[-1],0.,2300))
    xlabel('Time (hr, UT)')
    ylabel('Epicentral Distance (km)')   
    title('Observation', fontweight = 'bold')
    text(6.47,ep_dist,f_n[9:16],size='small',color='k')
    
    

f=  axis_param(im,'TECU')
    
    
    
    
ax=subplot(122)    
for ix in range (0,len(x),2):   
    tec=(1e+06/1.e+16)*dn3[:,ix,7:23].sum(1)
    dtec=data_smooth_sim(t_hr,tec)
    nm=1
    
    vm = 0.1
    #vm=dtec.max()/2
    xlim(t_hr[0],t_hr[-1])
    im2=scatter(t_hr,1.1e+03*dtec+(x[ix]),c=10*dtec,s=20.,
                vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow,alpha=1)   

    # if ix==0:
    #     v1=(4/1)/(1./3600.)
    #     axline((t_ep+420/3600,0),slope=v1,c='k' ,label = 'Rayleigh ~3.34 km/s')
    #     v1=(1/1)/(1./3600.)
    #     axline((t_ep+600/3600,0),slope=v1, c='k', linestyle = '--',
    #            label = 'Acoustic ~1.15 km/s')
    #     legend(loc = 'lower right')
    #     axvline(x=t_ep,color='grey',lw=2) 
     #   title('Simulation', fontweight = 'bold')

xlabel('Time (hr, UT)')
axvline(x=t_ep,color='grey',lw=2)         
title('Simulation', fontweight = 'bold')        
f=  axis_param(im2,'TECU')
tight_layout()
#savefig("./figures_pub/tec_obs_sim.png", bbox_inches = 'tight')




#%%
# Finding peaks from the dtec 

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
#%%
f_list=sorted(glob.glob('DATA_TEC/***.npy'));#0937# 0261#0205
for f_n in f_list:  
    data_gnss=load(f_n,allow_pickle=True)
    t_gnss=data_gnss[0];tec_gnss=data_gnss[1];
    lon_gnss=data_gnss[2];lat_gnss=data_gnss[3];
  
    v_prox=find_nearest(t_gnss, t_ep+8.467*60/3600)
    p_sip=where(t_gnss==v_prox)[0]
    ep_sign=1
    if lon_gnss[p_sip]-lon_ep <0:
        ep_sign=1
    if lat_gnss[p_sip]-lat_ep <0:
        ep_sign=-1
    [d1,d2]=distkm(lon_ep,lat_ep,([lon_gnss[p_sip]]),([lat_gnss[p_sip]]))
    ep_dist=((d1+d2)[0]/2)/1.
    dtec_gnss=data_smooth(t_gnss,tec_gnss)
    
    dtec_gnss[dtec_gnss <= 0.1] = 0

    
    dtec_peak_obs = find_peaks_within_range_obs(dtec_gnss, 80, 280, 0.065)
    
    time_obs = (t_gnss[dtec_peak_obs] - t_ep)*3600
    dist_obs = (lat_gnss[dtec_peak_obs] - lat_ep)*110
    vel_obs = dist_obs / time_obs
    
    fig=figure(6,figsize=(10,12),facecolor='w',edgecolor='k')
    ax=subplot(221)
    
    
    # im=scatter(t_gnss,ep_dist+150*dtec_gnss,c=0.6*dtec_gnss,s=1.8*t_gnss,
    #             vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow,alpha=1)
    
    #scatter(t_hr[dtec_peak_obs], dtec_gnss[dtec_peak_obs])
    
    scatter(t_gnss[dtec_peak_obs], ep_dist+(dtec_gnss[dtec_peak_obs]), c='b')

    #text(6.47,ep_dist,f_n[9:16],size='small',color='k')
    
    # axvline(x=t_ep,color='grey',lw=2) 
    # axhline(850, color = 'r', linestyle='--')
    # # v1=(3./111)/(1./3600.)
    # # axline((t_ep+8/60,0),slope=v1,c='g' ,label = '~1 km/s')
    xlim(6.6, 7.4)
    #axis((t_gnss[dtec_peak_obs][0],t_gnss[dtec_peak_obs][-1],0.,2300))
    xlabel('Time (hr, UT)')
    ylabel('Epicentral Distance (km)')   
    title('Observation', fontweight = 'bold')
    grid(linestyle='-.', lw=0.4)
text(6.834, 1122, "Fast mode", rotation = 90)
text(7.090, 1110, "Slow mode", rotation = 46)
text(6.61, 1810, '(a)', fontsize=15)
    
#f=  axis_param(im,vm,4.4,'TECU')
    
    
    
  
ax=subplot(222)    
for ix in range (0,len(x),2):   
    tec=(1e+06/1.e+16)*dn3[:,ix,7:23].sum(1)
    dtec=data_smooth_sim(t_hr,tec)
    nm=1
    
    
    dtec[dtec <= 0.005] = 0
    
    dtec_peak_sim = find_peaks_within_range(dtec, 40, 100, 0.003)
    
    #scatter(t_hr[dtec_peak_sim], dtec[dtec_peak_sim])
    
    scatter(t_hr[dtec_peak_sim], 1.1e+03*dtec[dtec_peak_sim]+(x[ix]), c='b') 
    
    
    xlim(6.6, 7.4)
    xlabel('Time (hr, UT)')
    #ylabel('Epicentral Distance, km')   
    title('Simulation', fontweight = 'bold')
    grid(linestyle='-.', lw=0.4)
text(6.812, 1171, "Fast mode", rotation = 90)
text(7.092, 722, "Slow mode", rotation = 45)
text(6.61, 2215, '(b)', fontsize=15)

#%%
# Velocity distribution

f_list=sorted(glob.glob('DATA_TEC/***.npy'));#0937# 0261#0205
for f_n in f_list:  
    data_gnss=load(f_n,allow_pickle=True)
    t_gnss=data_gnss[0];tec_gnss=data_gnss[1];
    lon_gnss=data_gnss[2];lat_gnss=data_gnss[3];
  
    v_prox=find_nearest(t_gnss, t_ep+8.467*60/3600)
    p_sip=where(t_gnss==v_prox)[0]
    ep_sign=1
    if lon_gnss[p_sip]-lon_ep <0:
        ep_sign=1
    if lat_gnss[p_sip]-lat_ep <0:
        ep_sign=-1
    [d1,d2]=distkm(lon_ep,lat_ep,([lon_gnss[p_sip]]),([lat_gnss[p_sip]]))
    ep_dist=((d1+d2)[0]/2)/1.
    dtec_gnss=data_smooth(t_gnss,tec_gnss)
    
    dtec_gnss[dtec_gnss <= 0.02] = 0

    
    dtec_peak_obs = find_peaks_within_range_k(dtec_gnss, 80, 250, 0.05, 1)
    
    # 80, 250, 0.015
    
    time_obs = (t_gnss[dtec_peak_obs] - 6.74)*3600
    dist_obs = (lat_gnss[dtec_peak_obs] + 33.762)*110
    vel_obs = (ep_dist) / time_obs
    #vel_obs = (dist_obs) / (time_obs)
    
    fig=figure(6,figsize=(11,12),facecolor='w',edgecolor='k')
    ax=subplot(223)
    
    
   
    scatter(dist_obs, vel_obs, c='r')
    
    
    ylim(0, 3.5)
    
    #axis((t_gnss[dtec_peak_obs][0],t_gnss[dtec_peak_obs][-1],0.,2300))
    ylabel('Propagating speed (km/s)')
    xlabel('Epicentral Distance (km)')   
    #title('Observation', fontweight = 'bold')
    grid(linestyle='-.', lw=0.4)
text(697, 2.14, "Fast mode", rotation = 50)
text(1035, 0.75, "Slow mode", rotation = 0)
text(-109, 3.30, '(c)', fontsize=15)   
#f=  axis_param(im,vm,4.4,'TECU')
    

ax=subplot(224)    
for ix in range (0,len(x),2):   
    tec=(1e+06/1.e+16)*dn3[:,ix,7:23].sum(1)
    dtec=data_smooth_sim(t_hr,tec)
    nm=1
    
    
    
    dtec_peak_sim = find_peaks_within_range(dtec, 40, 200, 0.0035)
    
    time_sim = (t_hr[dtec_peak_sim] - 6.67)*3600
    
    dist_sim = dtec[dtec_peak_sim]+(x[ix])
    vel_sim = (dist_sim) / time_sim 
    
    # x1, y1 = [588, 2290], [0.567, 0.762]
    # x2, y2 = [0, 2214], [0, 1.935]
    # plt.plot(x1, y1, x2, y2, lw=0.4)
    
    scatter(dist_sim, vel_sim, c='r')
    
    
    ylim(0.05, 3.5)
    #plot(598, 0.598, 2226, 2.026)  
    
    #ylabel('Propagating speed (km/s)')
    xlabel('Epicentral Distance (km)')   
    
    #title('Simulation', fontweight = 'bold')
    grid(linestyle='-.', lw=0.4)
text(819, 1.6, "Fast mode", rotation = 40)
text(1092, 0.381, "Slow mode", rotation = 5)
text(-30, 3.30, '(d)', fontsize=15)
tight_layout() 
#savefig("./figures_pub/velocity.png", bbox_inches = 'tight') 
#%%
# # Temporal distribution
# Velocity distribution

f_list=sorted(glob.glob('DATA_TEC/***.npy'));#0937# 0261#0205
for f_n in f_list:  
    data_gnss=load(f_n,allow_pickle=True)
    t_gnss=data_gnss[0];tec_gnss=data_gnss[1];
    lon_gnss=data_gnss[2];lat_gnss=data_gnss[3];
  
    v_prox=find_nearest(t_gnss, t_ep+8.467*60/3600)
    p_sip=where(t_gnss==v_prox)[0]
    ep_sign=1
    if lon_gnss[p_sip]-lon_ep <0:
        ep_sign=1
    if lat_gnss[p_sip]-lat_ep <0:
        ep_sign=-1
    [d1,d2]=distkm(lon_ep,lat_ep,([lon_gnss[p_sip]]),([lat_gnss[p_sip]]))
    ep_dist=((d1+d2)[0]/2)/1.
    dtec_gnss=data_smooth(t_gnss,tec_gnss)
    
    dtec_gnss[dtec_gnss <= 0.08] = 0
    
    dtec_peak_obs = find_peaks_within_range_obs(dtec_gnss, 80, 280, 0.03)
    
    # 80, 250, 0.015
    
    time_obs = (t_gnss[dtec_peak_obs] - 6.76)*3600
    dist_obs = (lat_gnss[dtec_peak_obs] + 36.762)*110
    vel_obs = (ep_dist) / (time_obs)
    
    fig=figure(7,figsize=(8,4),facecolor='w',edgecolor='k')
    ax=subplot(121)
    
    
   
    im = scatter((t_gnss[dtec_peak_obs]-t_ep)*60, dist_obs, c=vel_obs)
    
    
    xlim(-2, 60)
    
    #axis((t_gnss[dtec_peak_obs][0],t_gnss[dtec_peak_obs][-1],0.,2300))
    xlabel('Detection Time (mins)')
    ylabel('Epicentral Distance (km)')   
    title('Observation', fontweight = 'bold')
    grid(linestyle='-.', lw=0.4)
text(14, 1248, "Fast mode", rotation = 90)
text(31.5, 1189, "Slow mode", rotation = 50)   
f=  axis_param(im,'km/s')

  
  
#%%
ax=subplot(122)    
for ix in range (0,len(x),2):   
    tec=(1e+06/1.e+16)*dn3[:,ix,7:23].sum(1)
    dtec=data_smooth_sim(t_hr,tec)
    vm = 2
    
    
    
    dtec_peak_sim = find_peaks_within_range(dtec, 40, 200, 0.0035)
    
    time_sim = (t_hr[dtec_peak_sim] - 6.66)*3600
    
    dist_sim = dtec[dtec_peak_sim]+(x[ix] -26)
    vel_sim = (dist_sim) / (time_sim) 
    
    # x1, y1 = [588, 2290], [0.567, 0.762]
    # x2, y2 = [0, 2214], [0, 1.935]
    # plt.plot(x1, y1, x2, y2, lw=0.4)
    
    im = scatter((t_hr[dtec_peak_sim]-t_ep)*60, dist_sim, c=vel_sim)
    
    xlim(0, 60)
    
    #plot(598, 0.598, 2226, 2.026)  
    
    #ylabel('Propagating speed (km/s)')
    xlabel('Detection Time (mins)')  
    
    title('Simulation', fontweight = 'bold')
    grid(linestyle='-.', lw=0.4)
text(13, 1250, "Fast mode", rotation = 90)
text(41.5, 1250, "Slow mode", rotation = 60)

f=  axis_param(im,'km/s')

tight_layout()
#savefig("./figures_pub/vel_contour.png", bbox_inches = 'tight')


#%%
################ Meridional and Zonal Density of CIDs #######################

f_list=sorted(glob.glob('DATA_TEC/***.npy'));#0937# 0261#0205
for f_n in f_list:  
    data_gnss=load(f_n,allow_pickle=True)
    t_gnss=data_gnss[0];tec_gnss=data_gnss[1];
    lon_gnss=data_gnss[2];lat_gnss=data_gnss[3];
  
    v_prox=find_nearest(t_gnss, t_ep+8.467*60/3600)
    p_sip=where(t_gnss==v_prox)[0]
    ep_sign=1
    if lon_gnss[p_sip]-lon_ep <0:
        ep_sign=1
    if lat_gnss[p_sip]-lat_ep <0:
        ep_sign=-1
    [d1,d2]=distkm(lon_ep,lat_ep,([lon_gnss[p_sip]]),([lat_gnss[p_sip]]))
    ep_dist=((d1+d2)[0]/2)/1.
    dtec_gnss=data_smooth(t_gnss,tec_gnss)
    
    dtec_gnss[dtec_gnss <= 0.08] = 0
    
    dtec_peak_obs = find_peaks_within_range_obs(dtec_gnss, 80, 280, 0.03)
    
    # 80, 250, 0.015
    
    time_obs = (t_gnss[dtec_peak_obs] - 6.705)*3600
    dist_obs = (lat_gnss[dtec_peak_obs] + 36.762)*110
    vel_obs = (dist_obs) / (time_obs)
    lon_obs = (lon_gnss[dtec_peak_obs] - lon_ep)*110
    lat_obs = (lat_gnss[dtec_peak_obs] - lat_ep)*110
    dtec_obs = dtec_gnss[dtec_peak_obs]
    
    fig=figure(8,figsize=(10,10))
    ax=subplot(221)
    
    
    
   
    #im = scatter(time_obs/60, dist_obs, c=vel_obs)
    
    im = scatter((t_gnss[dtec_peak_obs] - t_ep)*60, dtec_obs + lat_obs, c=dtec_obs, s = 70*abs(dtec_obs),
                 marker = 'o', cmap='rainbow')
    
    #axis((t_gnss[dtec_peak_obs][0],t_gnss[dtec_peak_obs][-1],0.,2300))
    xlabel('Detection Time (mins)')
    ylabel('Epicentral Latitude (km)')   
    
    grid(linestyle='-.', lw=0.4)
text(16.8, 1152, "Fast mode", rotation = 90)
text(30.3, 1101, "Slow mode", rotation = 45)
text(10, 1995, '(a)', fontsize=15) 
    
f=  axis_param(im,'TECU')
    

#%%

ax=subplot(222)  
f_list=sorted(glob.glob('DATA_TEC/***.npy'));#0937# 0261#0205
for f_n in f_list:  
    data_gnss=load(f_n,allow_pickle=True)
    t_gnss=data_gnss[0];tec_gnss=data_gnss[1];
    lon_gnss=data_gnss[2];lat_gnss=data_gnss[3];
  
    v_prox=find_nearest(t_gnss, t_ep+8.467*60/3600)
    p_sip=where(t_gnss==v_prox)[0]
    ep_sign=1
    if lon_gnss[p_sip]-lon_ep <0:
        ep_sign=1
    if lat_gnss[p_sip]-lat_ep <0:
        ep_sign=-1
    [d1,d2]=distkm(lon_ep,lat_ep,([lon_gnss[p_sip]]),([lat_gnss[p_sip]]))
    ep_dist=((d1+d2)[0]/2)/1.
    dtec_gnss=data_smooth(t_gnss,tec_gnss)
    
    dtec_gnss[dtec_gnss <= 0.08] = 0
    
    dtec_peak_obs = find_peaks_within_range_obs(dtec_gnss, 80, 280, 0.06)
    
    # 80, 250, 0.015
    
    time_obs = (t_gnss[dtec_peak_obs] - 6.705)*3600
    dist_obs = (lat_gnss[dtec_peak_obs] + 36.762)*110
    vel_obs = (dist_obs) / (time_obs)
    lon_obs = (lon_gnss[dtec_peak_obs] - lon_ep)*110
    lat_obs = (lat_gnss[dtec_peak_obs] - lat_ep)*110
    dtec_obs = dtec_gnss[dtec_peak_obs]
    
    
    im = scatter((t_gnss[dtec_peak_obs] - t_ep)*60, dtec_obs + lon_obs, c=dtec_obs, s = 70*abs(dtec_obs),
                 marker = 'o', cmap='rainbow')
    
    #axis((t_gnss[dtec_peak_obs][0],t_gnss[dtec_peak_obs][-1],0.,2300))
    xlabel('Detection Time (mins)')
    ylabel('Epicentral Longitude (km)')   
    
    grid(linestyle='-.', lw=0.4)

  
text(11, 100, "Fast mode", rotation = 110)
text(38, 300, "Slow mode", rotation = -65) 
text(10, 445, '(b)', fontsize=15)    
f=  axis_param(im,'TECU')

tight_layout()


#%%

############# Meridional and Zonal Speed of CIDs ##########################

f_list=sorted(glob.glob('DATA_TEC/***.npy'));#0937# 0261#0205
for f_n in f_list:  
    data_gnss=load(f_n,allow_pickle=True)
    t_gnss=data_gnss[0];tec_gnss=data_gnss[1];
    lon_gnss=data_gnss[2];lat_gnss=data_gnss[3];
  
    v_prox=find_nearest(t_gnss, t_ep+8.467*60/3600)
    p_sip=where(t_gnss==v_prox)[0]
    ep_sign=1
    if lon_gnss[p_sip]-lon_ep <0:
        ep_sign=1
    if lat_gnss[p_sip]-lat_ep <0:
        ep_sign=-1
    [d1,d2]=distkm(lon_ep,lat_ep,([lon_gnss[p_sip]]),([lat_gnss[p_sip]]))
    ep_dist=((d1+d2)[0]/2)/1.
    dtec_gnss=data_smooth(t_gnss,tec_gnss)
    
    dtec_gnss[dtec_gnss <= 0.035] = 0
    
    dtec_peak_obs = find_peaks_within_range_k(dtec_gnss, 80, 280, 0.03, k=1)
    
    # 80, 250, 0.015
    
    time_obs = (t_gnss[dtec_peak_obs] - 6.76)*3600
    dist_obs = (lat_gnss[dtec_peak_obs] + 33.762)*110
    vel_obs = (dist_obs) / (time_obs)
    
    
    
    fig=figure(8,figsize=(11,12),facecolor='w',edgecolor='k')
    ax=subplot(223)
    
   
    scatter(dist_obs, vel_obs, c='g')
    
    
    ylim(-0.05, 3.5)
    
    #axis((t_gnss[dtec_peak_obs][0],t_gnss[dtec_peak_obs][-1],0.,2300))
    ylabel('Meridional speed (km/s)')
    xlabel('Epicentral Latitude (km)')   
    #title('Observation', fontweight = 'bold')
    grid(linestyle='-.', lw=0.4)
text(629, 2.241, "Fast mode", rotation = 45)
text(1035, 0.530, "Slow mode", rotation = 0)
text(-125, 3.30, '(c)', fontsize=15) 
#f=  axis_param(im,vm,4.4,'TECU')


#%%

ax=subplot(224)
f_list=sorted(glob.glob('DATA_TEC/***.npy'));#0937# 0261#0205
for f_n in f_list:  
    data_gnss=load(f_n,allow_pickle=True)
    t_gnss=data_gnss[0];tec_gnss=data_gnss[1];
    lon_gnss=data_gnss[2];lat_gnss=data_gnss[3];
  
    v_prox=find_nearest(t_gnss, t_ep+8.467*60/3600)
    p_sip=where(t_gnss==v_prox)[0]
    ep_sign=1
    if lon_gnss[p_sip]-lon_ep <0:
        ep_sign=1
    if lat_gnss[p_sip]-lat_ep <0:
        ep_sign=-1
    [d1,d2]=distkm(lon_ep,lat_ep,([lon_gnss[p_sip]]),([lat_gnss[p_sip]]))
    ep_dist=((d1+d2)[0]/2)/1.
    dtec_gnss=data_smooth(t_gnss,tec_gnss)
    
    dtec_gnss[dtec_gnss <= 0.01] = 0
    
    dtec_peak_obs = find_peaks_within_range_k(dtec_gnss, 80, 280, 0.03, k=1)
    
    # 80, 250, 0.015
    
    time_obs = (t_gnss[dtec_peak_obs] - 6.83)*3600
    dist_obs = (lon_gnss[dtec_peak_obs] + 74.5107)*110
    vel_obs = (dist_obs) / (time_obs)
    
   
    scatter(dist_obs, vel_obs, c='g')
    
    
    ylim(-0.05, 3.5)
    
    #axis((t_gnss[dtec_peak_obs][0],t_gnss[dtec_peak_obs][-1],0.,2300))
    ylabel('Zonal speed (km/s)')
    xlabel('Epicentral Longitude (km)')   
    #title('Observation', fontweight = 'bold')
    grid(linestyle='-.', lw=0.4)
text(318, 1.828, "Fast mode", rotation = 40)
text(445, 0.05, "Slow mode", rotation = 15)
text(167, 3.30, '(d)', fontsize=15) 
#f=  axis_param(im,vm,4.4,'TECU')  

#savefig("./figures_pub/meridional_zonal.png", bbox_inches = 'tight')  



#%%

f_list=sorted(glob.glob('DATA_TEC/***.npy'));#0937# 0261#0205
for f_n in f_list:  
    data_gnss=load(f_n,allow_pickle=True)
    t_gnss=data_gnss[0];tec_gnss=data_gnss[1];
    lon_gnss=data_gnss[2];lat_gnss=data_gnss[3];
  
    v_prox=find_nearest(t_gnss, t_ep+8.467*60/3600)
    p_sip=where(t_gnss==v_prox)[0]
    ep_sign=1
    if lon_gnss[p_sip]-lon_ep <0:
        ep_sign=1
    if lat_gnss[p_sip]-lat_ep <0:
        ep_sign=-1
    [d1,d2]=distkm(lon_ep,lat_ep,([lon_gnss[p_sip]]),([lat_gnss[p_sip]]))
    ep_dist=((d1+d2)[0]/2)/1.
    dtec_gnss=data_smooth(t_gnss,tec_gnss)
    
    dtec_gnss[dtec_gnss <= 0.035] = 0
    
    dtec_peak_obs = find_peaks_within_range_obs(dtec_gnss, 80, 280, 0.03)
    
    # 80, 250, 0.015
    
    time_obs = (t_gnss[dtec_peak_obs] - 6.705)*3600
    dist_obs = (lat_gnss[dtec_peak_obs] + 33.762)*110
    vel_obs = (dist_obs) / (time_obs)
    
    
    
    fig=figure(9,figsize=(8,6),facecolor='w',edgecolor='k')
    ax=subplot(211)
    
    
    # scatter(ep_dist+150*dtec_gnss[dtec_peak_obs], dtec_gnss[dtec_peak_obs], 
    #         s=50*dtec_gnss[dtec_peak_obs], c=(t_gnss[dtec_peak_obs]-t_ep)*60)
    bar(ep_dist+dtec_gnss[dtec_peak_obs], dtec_gnss[dtec_peak_obs], 
        color='red', width = 10)
    
    
    
    xlim(0, 2300)
    
    #axis((t_gnss[dtec_peak_obs][0],t_gnss[dtec_peak_obs][-1],0.,2300))
    ylabel('dTEC (TECU)')
    xlabel('Epicentral Distance (km)')   
    #title('Observation', fontweight = 'bold')
    grid(linestyle='-.', lw=0.4)
    text(43, 0.95, "(a)", fontsize=15)
    
    ax=subplot(212)
    bar((t_gnss[dtec_peak_obs]-t_ep)*60, dtec_gnss[dtec_peak_obs], 
        color='red', width = 0.2)
    
    ylabel('dTEC (TECU)')
    xlabel('Detection time (mins)')   
    #title('Observation', fontweight = 'bold')
    grid(linestyle='-.', lw=0.4)
    text(0.65, 0.96, "(b)", fontsize=15)

    
tight_layout()
#savefig("./figures_pub/spatiotemp.png", bbox_inches = 'tight') 

#%%

f_list=sorted(glob.glob('DATA_TEC/***.npy'));
for f_n in f_list:  
    data_gnss=load(f_n,allow_pickle=True)
    t_gnss=data_gnss[0];tec_gnss=data_gnss[1];
    lon_gnss=data_gnss[2];lat_gnss=data_gnss[3];
  
    v_prox=find_nearest(t_gnss, t_ep+8.467*60/3600)
    p_sip=where(t_gnss==v_prox)[0]
    ep_sign=1
    if lon_gnss[p_sip]-lon_ep <0:
        ep_sign=1
    if lat_gnss[p_sip]-lat_ep <0:
        ep_sign=-1
    [d1,d2]=distkm(lon_ep,lat_ep,([lon_gnss[p_sip]]),([lat_gnss[p_sip]]))
    ep_dist=((d1+d2)[0]/2)/1.
    dtec_gnss=data_smooth(t_gnss,tec_gnss)
    fig=figure(10,figsize=(10,8),facecolor='w',edgecolor='k')
    ax=subplot(111)
    vm=0.1
    # plot(t_gnss,ep_dist+250*dtec_gnss,lw=2)
    im=scatter((lon_gnss - lon_ep),(lat_gnss - lat_ep) + (2*dtec_gnss),c=dtec_gnss,
                marker='o',cmap=cm.rainbow,alpha=1)
    
    scatter(0, 0, marker='*', s=500, c='red')
    
    xlabel("Epicentral Longitude ($^{o}$)")
    ylabel("Epicentral Latitude ($^{o}$)")
    grid(linestyle='-.', lw=0.4)
    text((lon_gnss[200] - lon_ep),(lat_gnss[200] - lat_ep),f_n[9:16],fontsize=6,color='k')
    text(5.268, 21.62, 'NE', fontsize=15)
    ax.annotate('', xy=(-0.095, 3.01), xytext=(0,0),
                arrowprops=dict(facecolor = 'black', headwidth = 4, width=1),)
    ax.annotate('', xy=(1.350, 2.53), xytext=(0,0),
                arrowprops=dict(facecolor = 'black', headwidth = 4, width=1),)
    ax.annotate('', xy=(2.334, 3.21), xytext=(0,0),
                arrowprops=dict(facecolor = 'black', headwidth = 4, width=1),)
    ax.annotate('', xy=(0.297, 7.17), xytext=(0,0),
                arrowprops=dict(facecolor = 'black', headwidth = 4, width=1),)
    ax.annotate('', xy=(1.559, 5.94), xytext=(0,0),
                arrowprops=dict(facecolor = 'black', headwidth = 4, width=1),)
    ax.annotate('', xy=(2.299, 5.23), xytext=(0,0),
                arrowprops=dict(facecolor = 'black', headwidth = 4, width=1),)
    ax.annotate('', xy=(1.141, 10.28), xytext=(0,0),
                arrowprops=dict(facecolor = 'black', headwidth = 4, width=1),)
    ax.annotate('', xy=(0.540, 11.06), xytext=(0,0),
                arrowprops=dict(facecolor = 'black', headwidth = 4, width=1),)
    ax.annotate('', xy=(2.648, 10.83), xytext=(0,0),
                arrowprops=dict(facecolor = 'black', headwidth = 4, width=1),)
tight_layout()
#savefig("./figures_pub/directivity.png", bbox_inches = 'tight') 



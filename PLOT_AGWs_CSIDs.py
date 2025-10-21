
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
from adebayo_codigo import calculate_distance, band_pass_filter, data_smooth_mike, find_nearest_value
import glob
import os
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cf
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
        n_smooth=int(ceil(12./(dt*60.))) # accepted parameter
        #n_smooth=int(ceil(8./(dt*60.))) # elvira
        win=ones((n_smooth,))/n_smooth
        data=-data+convolve_al(data,win)
        return convolve_al(data,ones((5,))/5)    
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
t_original = t/60

t_hr=(t-12)/60.
t_hr_sim = (t)/60.
[t2,yt2]=meshgrid(t,y)
data_amb=load('data_amb.npy');rho_amb=data_amb[:,0][0];sn=data_amb[:,1][0]
wx3=load('wx3.npy')
wy3=load('wy3.npy')
n3=1*load('n3.npy');dn3=1*(len(t)/2.)*gradient(n3)[0];
t3 = load('t3.npy'); v_new = load('v_new.npy')


#i_xo=abs(x-x.mean()).argmin()
i_xo=abs(x-0.).argmin()

#%%

fig=figure(1,figsize=(10,10),facecolor='w',edgecolor='k')
ax = subplot(221)
#
plot(sn,y,lw=2)
text(245,280,'(A)')
grid('on', ls = '-.', lw=0.4)
xlabel('m/s')
ylabel('Altitude (km)')
title('Neutral sound speed')
ylim(0,300)

ax=subplot(222)
semilogx(t_hr[0]+1*n3[0,i_xo,:],yi,lw=2)
text(1.23e10,280,'(B)')
title('Electron number density')
ylim(0,300)
xlabel('m$^{-3}$')
grid('on', ls = '-.', lw=0.4)

ax=subplot(223)
# v1=(1.e-03*3600.)*sn[:20].mean()
# plot(t_ep+(1000/3600.)*y/sn,y,'go',markersize=4)
data=wy3[:,i_xo,:].T
vm=data.max()
#vm = 100
ext=[t_hr[0],t_hr[-1],y2[0,0],y2[0,-1]]
im=imshow(data,origin='lower',extent=ext,
          cmap=cm.seismic,vmin=-vm,vmax=vm)
im.set_interpolation('spline16')
text(6.37,380,'(C)')
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
text(6.37,380,'(D)')

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

fig=figure(3,figsize=(8,8),facecolor='w',edgecolor='k')

ax =subplot(1,2,1)
#ax =subplot(1,1,1)
for ix in range (0,len(x),1):
    # wy3[:,i_xo,0]
    #pcolormesh(t3[:,0,0],x,v_new[:,:,0].T)
    
    #v_new[100:,ix,0] = 0
    
    x_dt = (t3[:,0,0]/3600)+6.56
    y_dt = (x[ix]) + (100*v_new[:,ix,0])

    plot(x_dt, y_dt, color='g', label=f'x={x[ix]:.2f}')
    
    max_amps_sism.append(abs(v_new[:, ix, 0]).max())
    x_vals_sism.append(x[ix])
    
    
    # plot(t_hr[::1],(x[ix])+0.1e+02*wx3[:,ix,0][::1],'g')
    # xlim(t_hr[0],t_hr[-1])
    # ##plot(t_original,(x[ix])+0.1e+02*wx3[:,ix,0],'g')
    # #xlim(t_original[0],t_original[-1])
    # xlim(6.4,7.8)
    # ylabel('Epicentral Distance, km ')
    # xlabel('Time, UT')
    # if ix==0:
    #     v1=(3.5/1)/(1./3600.)
    #     axline((t_ep+100/3600,0),slope=v1,c='k' ,label = '~2.8 km/s')

    #     legend()
    #     axvline(x=t_ep,color='grey',lw=1,linestyle='--')  
    
figure(84)
vlines(x_vals_sism, 0, ymax=max_amps_sism, color='lime')
#scatter(x_vals, max_amps, c='lime', )
#bar(x_vals, max_amps, color='lime')
xlabel('x')
ylabel('Max amplitude')
title('Max amplitude per ix')

max_amps_tec_1 = []
x_vals_tec_1 = []

fig=figure(3,figsize=(8,8),facecolor='w',edgecolor='k')
ax =subplot(1,2,2)
for ix in range (0,len(x),1):   
    # delta_h = 2.5e4  # 10 km in meters
    # scale_factor = delta_h / 1e16  # converts to TECU
    # tec = scale_factor * dn3[:, ix, 7:23].sum(1)
    tec=(1e+06/1.e+16)*dn3[:,ix,7:23].sum(1)
    dtec=data_smooth_sim(t_hr,tec)
    
    max_amps_tec_1.append(abs(dtec).max())
    x_vals_tec_1.append(x[ix])
    nm=1

    vm=dtec.max()
    
    xlim(t_hr[0],t_hr[-1])
    im=scatter(t_hr,0.5*dtec+(x[ix]),c=0.6*dtec,s=20.,
                vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow)   

    if ix==0:
        v1=(4/1)/(1./3600.)
        axline((t_ep+420/3600,0),slope=v1,c='k' ,label = '~4 km/s')
        v1=(1/1)/(1./3600.)
        axline((t_ep+420/3600,0),slope=v1,c='g' ,label = '~1 km/s')
        legend()
        axvline(x=t_ep,color='grey',lw=1,linestyle='--') 
        
        
    # figure(88, figsize=(8,10))
    # subplot(131)
    # xlim(t_hr[0],t_hr[-1])
    # im=scatter(t_hr,0.5*(-dtec)+(x[ix]),c=0.6*dtec,s=20.,
    #             vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow)  

figure(85)
vlines(x_vals_tec_1, 0, ymax=max_amps_tec_1, color='limegreen')
#scatter(x_vals, max_amps, c='lime', )
#bar(x_vals, max_amps, color='lime')
xlabel('x')
ylabel('Max amplitude')
title('Max amplitude per ix') 



       
#%%%%
from scipy import stats
corr_val = []
p_val = []



f_list=sorted(glob.glob('DATA_TEC/***.npy'));#0937# 0261#0205
for idx, f_n in enumerate(f_list):  
    data_gnss=load(f_n,allow_pickle=True)
    t_gnss=data_gnss[0];tec_gnss=data_gnss[1];
    lon_gnss=data_gnss[2];lat_gnss=data_gnss[3];
  
    v_prox=find_nearest(t_gnss, t_ep+8.467*60/3600)
    p_sip=where(t_gnss==v_prox)[0]
    ep_sign=1
    if lon_gnss[p_sip]-lon_ep <0:ep_sign=1
    if lat_gnss[p_sip]-lat_ep <0:ep_sign=-1
    [d1,d2]=distkm(lon_ep,lat_ep,([lon_gnss[p_sip]]),([lat_gnss[p_sip]]))
    ep_dist=((d1+d2)[0]/2)/1.
    dtec_gnss=data_smooth(t_gnss,tec_gnss)
    fig=figure(4,figsize=(10,8),facecolor='w',edgecolor='k')
    ax=subplot(121)
    vm=0.1
    
   # plot(t_gnss,ep_dist+250*dtec_gnss,lw=2)
    im=scatter(t_gnss,ep_dist+100*dtec_gnss,c=0.6*dtec_gnss,s=20,
                vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow)
    
    if ep_dist < 1500:
        text(7.53,ep_dist,f_n[9:16],fontsize=8,color='k')

    axvline(t_ep,0,1,c='grey',linestyle='--')
    # v1=(3./111)/(1./3600.)
    # axline((t_ep+8/60,0),slope=v1,c='g' ,label = '~1 km/s')
    axis((t_hr[0],t_hr[-1],0.,1500))
   # f=  (axis_param(im,vm,4.4,'TECU'))
    ylabel('Epicentral Distance (km)')
    xlabel('Time (UT, hr)')
    title("Observation")
    
    
    if f_n[9:16] == "areqG11":
        text(6.4, 1450, "(A)")
        
        v1=(2.6/1)/(1./3600.)
        axline((t_ep+400/3600,0),slope=v1, lw=0.5,
               c='k' ,label = '~3.2 km/s')
        
        v2=(0.8/1)/(1./3600.)
        axline((t_ep+400/3600,0),slope=v2, ls = '--', 
               lw = 0.5, c='k' ,label = '~0.9 km/s')
        
        # text(6.68, 1250, "v$_{R}$ = 3.38 km/s")
        # text(6.5, 1950, "(A)")
    legend(loc = 'lower right')
    
    if idx == 0:
        f = axis_param(im, "TECU")
        
    #v_propagation = 3.0  # km/s

    # Convert ep_dist to propagation time in hours
    propagation_delay_hr = (ep_dist / (3.0))  # delay = distance / speed, converted to hours
    # Simulation input data
    figure(25, figsize=(6,10))
    plot(t_hr, ep_dist+0.1e+02*wx3[:,idx, 0], 'g')
    #plot(t_hr[::1],ep_dist+0.1e+02*wx3[:,idx,0][::1],'g')
    xlim(t_hr[0],t_hr[-1])
    ylabel('Epicentral Distance, km ')
    xlabel('Time, UT')
    
    if idx==0:
        v1=(3.0/1)/(1./3600.)
        axline((t_ep+100/3600,0),slope=v1,c='k' ,label = '~3km/s')

        legend()
        axvline(x=t_ep,color='grey',lw=1,linestyle='--') 
        
        
        
    # for i, dist in enumerate(ep_dist):
    #     # Time-shift the wx3 signal by propagation delay
    #     shifted_time = t_hr - propagation_delay_hr[i]
    #     plot(shifted_time, dist + 0.1e+02 * wx3[i, idx, 0], 'g')
        
    # shifted_time = ((t_hr-t_ep-0.03)*3600) - propagation_delay_hr
    # plot(shifted_time, ep_dist + 0.1e+02 * wx3[:, idx, 0], 'g')

    # #xlim(t_hr[0], t_hr[-1])
    # ylabel('Epicentral Distance, km ')
    # xlabel('Time, UT')
    
    # if idx == 0:
    #     v1 = (3.0 / 1) / (1. / 3600.)  # hours^-1
    #     axline((t_ep + 100 / 3600, 0), slope=v1, c='k', label='~3 km/s')
    #     legend()
    #     axvline(x=t_ep, color='grey', lw=1, linestyle='--')

    
    
    fig=figure(4,figsize=(10,8),facecolor='w',edgecolor='k')
    ax=subplot(122)
    #plot(t_gnss,ep_dist+250*dtec_gnss,'r',lw=1)
    i_ep=abs(x-ep_dist).argmin()
    
    delta_h = 2.5e4  # 10 km in meters
    scale_factor = delta_h / 1e16  # converts to TECU
    tec = scale_factor * n3[:, i_ep, 7:23].sum(1)
    #tec=(1e+06/1.e+16)*dn3[:,i_ep,7:23].sum(1)
    #dtec=data_smooth_sim(t_hr,tec)
    
    #tec = scale_factor * n3[:, :, 12].sum(1)

    
    #tec=((5e+8/1.e+16))*n3[:,i_ep,12:].sum(1)
    #tec_200km = ((5e+8/1.e+16))*n3[:,i_ep,12:].sum(1)
    xlabel('Time (UT, hr)')
    
    #%%

      
    dtec=data_smooth_sim(t_hr,tec)
    vm = 0.1#dtec.max()/2 
    #plot(t_hr,ep_dist+50*(dtec),'k',lw=1.5)
    
    # filtered
    im=scatter(t_hr,ep_dist+(300)*(dtec),c=dtec,s=20,
                vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow)
    
    # # unfiltered
    # im=scatter(t_hr,ep_dist+tec,c=abs(tec),s=20,
    #             marker='o',cmap=cm.rainbow)

    
        
    if f_n[9:16] == "areqG11":
        text(6.4, 1450, "(B)")
        
        v1=(2.6/1)/(1./3600.)
        axline((t_ep+400/3600,0),slope=v1, lw=0.5,
               c='k' ,label = '~3.2 km/s')
        
        v2=(0.8/1)/(1./3600.)
        axline((t_ep+400/3600,0),slope=v2, ls = '--', 
               lw = 0.5, c='k' ,label = '~0.9 km/s')
        
        # text(6.68, 1250, "v$_{R}$ = 3.38 km/s")
        # text(6.5, 1950, "(A)")
        #f = axis_param(im, "TECU")
        
    legend(loc = 'lower right')
        
    
    # if i_ep==1:
    #     v1=(4/1)/(1./3600.)
    #     axline((t_ep+560/3600,0),slope=v1,c='k' ,label = '~4 km/s')
    #     v1=(1/1)/(1./3600.)
        
    #     axline((t_ep+420/3600,0),slope=v1,c='g' ,label = '~1 km/s')
    #     legend()
    #     axvline(x=t_ep,color='grey',lw=1,linestyle='--')    
    #     legend()
    axis((t_hr[0],t_hr[-1],0.,1500))
  #  axvline(t_ep,0,1,c='grey',linestyle='--')
    title("Simulation")
    axvline(t_ep,0,1,c='grey',linestyle='--')
    
    # if f_n[9:16] == 'areqG11':
    #     f = axis_param(im, "TECU")
    
    #if idx == 0:
        
    
    tight_layout()
    
    figure(7, figsize=(4,8))
    vm = 0.1#dtec.max()/2 
    #plot(t_hr,ep_dist+50*(dtec),'k',lw=1.5)
    
    # filtered
    im=scatter(t_hr,ep_dist+400*(dtec),c=dtec,s=20,
                vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow)
    
    # # unfiltered
    # im=scatter(t_hr,ep_dist+tec,c=abs(tec),s=20,
    #             marker='o',cmap=cm.rainbow)

    
        
    
    if i_ep==1:
        v1=(4/1)/(1./3600.)
        axline((t_ep+560/3600,0),slope=v1,c='k' ,label = '~4 km/s')
        v1=(1/1)/(1./3600.)
        
        axline((t_ep+420/3600,0),slope=v1,c='g' ,label = '~1 km/s')
        legend()
        axvline(x=t_ep,color='grey',lw=1,linestyle='--')    
        legend()
    axis((t_hr[0],t_hr[-1],0.,2300))
  #  axvline(t_ep,0,1,c='grey',linestyle='--')
    title("Simulation ($U_{2}$ = $U_{3}$ = 0)")
    axvline(t_ep,0,1,c='grey',linestyle='--')
    ylabel('Epicentral Distance (km)')
    xlabel('Time (UT, hr)')
    
    
    if idx == 0:
        f = axis_param(im, "TECU")
    
    tight_layout()
    # --------------------------------------------------------------------
    figure(9, figsize=(10,8),facecolor='w',edgecolor='k')
    ax = subplot(121)
    # filtered
    im=scatter(t_hr,ep_dist+300*(dtec),c=dtec,s=20,
                vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow)
    
    
    if i_ep==1:
        v1=(4/1)/(1./3600.)
        axline((t_ep+560/3600,0),slope=v1,c='k' ,label = '~4 km/s')
        v1=(1/1)/(1./3600.)
        
        axline((t_ep+420/3600,0),slope=v1,c='g' ,label = '~1 km/s')
        legend()
        axvline(x=t_ep,color='grey',lw=1,linestyle='--')    
        legend()
    axis((6.4, 7.6,0.,1500))
  #  axvline(t_ep,0,1,c='grey',linestyle='--')
    title("Simulation")
    axvline(t_ep,0,1,c='grey',linestyle='--')
    ylabel('Epicentral Distance (km)')
    xlabel('Time (UT, hr)')
    
    if idx == 0:
        text(6.4, 1450, "(A)")
    
    if idx == 1:
        f = axis_param(im, "TECU")
        
    if f_n[9:16] == "areqG11":
        #text(6.4, 1950, "(B)")
        
        v1=(2.6/1)/(1./3600.)
        axline((t_ep+400/3600,0),slope=v1, lw=0.5,
               c='k' ,label = '~3.2 km/s')
        
        v2=(0.8/1)/(1./3600.)
        axline((t_ep+400/3600,0),slope=v2, ls = '--', 
               lw = 0.5, c='k' ,label = '~0.9 km/s')
        
        # text(6.68, 1250, "v$_{R}$ = 3.38 km/s")
        # text(6.5, 1950, "(A)")
    legend(loc = 'lower right')
    
    # --------------------------------------------------------------------
    
    
    
   #figure(8, figsize=(6,8))
   #  subplot(121)
   #  im=scatter(t_gnss,ep_dist+100*dtec_gnss,c=0.6*dtec_gnss,s=2*t_gnss,
   #              vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow,alpha=1)

   #  text(6.57,ep_dist,f_n[9:16],size='small',color='k')

   #  axvline(t_ep,0,1,c='grey',linestyle='--')
   #  # v1=(3./111)/(1./3600.)
   #  # axline((t_ep+8/60,0),slope=v1,c='g' ,label = '~1 km/s')
   #  axis((t_hr[0],t_hr[-1],0.,2300))
   # # f=  (axis_param(im,vm,4.4,'TECU'))
   #  ylabel('Epicentral Distance, km')
   #  xlabel('Time, UT')
   #  title("Observation")
    
    
    figure(9, figsize=(10,8),facecolor='w',edgecolor='k')
    ax = subplot(122)
    
    if f_n[9:16] in ['baveG23', 'baveG11', 'cmpnG20', 'cmpnG23', 'vlzlG20', 'jrgnG20', 'srgdG20', 'cllaG13']:
        
    
        plot(t_gnss, ep_dist+100*dtec_gnss, color='r', lw=2)
        axis((6.4, 7.6,0.,2300))
    
        plot(t_hr, ep_dist+300*(dtec), color='b', lw=2)
        
        axis((6.4, 7.6,0.,1500))
        axvline(t_ep, color='grey',linestyle='--')
        xlabel("Time (UT, hr)")
        ylabel("Epicentral Distance (km)")
    
    if idx == 0:
        #f = axis_param(im, "TECU")
        text(6.4, 1450, "(B)")
    #savefig("./figures_pub/obs_sim_comparative.png", bbox_inches = 'tight')
    
    # cor_tw = stats.spearmanr(ep_dist+dtec_gnss[:136], ep_dist+(dtec))
    # #cor_tw = corrcoef(ep_dist+dtec_gnss[:136], ep_dist+(dtec))
    
    # corr_val.append(cor_tw)
    # scatter(t_gnss,ep_dist+100*dtec_gnss,c=0.6*dtec_gnss,s=2*t_gnss,
    #             vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow,alpha=1)
    
    # scatter(t_hr,ep_dist+200*(dtec),c=dtec,s=20,
    #             vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow,alpha=0.8)
     
tight_layout()

     

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



#%%
sismo_peaks_mx = []
sismo_dist = []

f = np.loadtxt("./igrfgridData.csv", skiprows=13, delimiter=',')
years = f[:,0]; lat_igr = f[:,1]; lon_igr = f[:,2]; incl = f[:,4]

# import matplotlib.pyplot as plt
# import cartopy
# ax = plt.axes(projection=cartopy.crs.PlateCarree())
# ax.add_feature(cartopy.feature.COASTLINE)
# ax.set_xlabel('lon')
# ax.set_ylabel('lat')

# ax.set_xticks([-180,-120,-60,0,60,120,180])
# ax.set_yticks([-90,-60,-30,0,30,60,90])

# plt.show()


#------------------
# Create the plot
#ax = plt.axes([0, 0, 1, 1], projection=geos)

fig = plt.figure(30, figsize=(15, 8))
#ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax1 = plt.axes(projection=ccrs.PlateCarree())

# Set the extent of the map (latitude and longitude range)
ax1.set_extent([-80, -60, -45, -15])
ax1.add_feature(cf.COASTLINE, lw = 1)
ax1.add_feature(cf.BORDERS, lw = 0.8)
ax1.add_feature(cf.LAND)
ax1.add_feature(cf.OCEAN)
#ax.add_feature(cf.RIVERS)
#ax.add_feature(cf.LAKES)

# Add borders with different linewidth for Brazil
#ax.add_feature(cf.BORDERS.with_scale('10m'), linestyle='-', edgecolor='red', linewidth=2)

# Add features to the map
ax1.coastlines()
# gline = ax.gridlines(linewidth=0.001)
# gline.xlabels_bottom = True
# gline.ylabels_left = True


# # Add gridlines with labels and proper formatting
# gl = ax1.gridlines(draw_labels=True,
#                   linewidth=0.01,
#                   color='gray',
                  
#                   linestyle='--')

# gl.top_labels = False
# gl.right_labels = False
# gl.xformatter = cticker.LongitudeFormatter()
# gl.yformatter = cticker.LatitudeFormatter()
# gl.xlabel_style = {'size': 9, 'color': 'black'}
# gl.ylabel_style = {'size': 9, 'color': 'black'}


# # Plot specific points
# ax.plot(-71.464, -7.2879, "*", markersize=15, color='red')   # Brazil
# #ax.plot(-76.84, -11.99, "^", markersize=10, color='green')   # NNA
# ax.plot(-67.6, -9.5, "x", markersize=5, color='blue')   # RIOB
# ax.plot(-63, -4, "x", markersize=5, color='blue')   # RIOB
# ax.plot(-61, -10, "x", markersize=5, color='blue')   # RIOB
# ax.plot(-60, -3, "x", markersize=5, color='blue')   # RIOB
# ax.plot(-67, -0, "x", markersize=5, color='blue')   # SAGA
# #ax.text(-62.4, -7.8, "x", markersize=5, color='blue')   # RIOB
# ax.text(-55, -10, 'Brazil', fontweight='bold', fontsize=12)

# # Receivers name
# ax.text(-75.2, -7.5, "CRUZ", fontweight='bold', color='blue', fontsize=6)   # RIOB
# ax.text(-63.3, -9, "POVE", fontweight='bold', color='blue', fontsize=6)   # RIOB
# ax.text(-64.7, -5.2, "AMCO", fontweight='bold', color='blue', fontsize=6)   # RIOB
# ax.text(-64.1, -3.7, "AMTE", fontweight='bold', color='blue', fontsize=6)   # RIOB
# ax.text(-66, 2.2, "AMCO_G17", fontweight='bold', color='magenta', fontsize=6)   # RIOB
# ax.text(-65, 0, "POVE_G17", fontweight='bold', color='magenta', fontsize=6)   # RIOB
# ax.text(-62, -7, "AMTE_G09", fontweight='bold', color='magenta', fontsize=6)   # RIOB
# ax.text(-77.8, -10, "CRUZ_G09", fontweight='bold', color='magenta', fontsize=6)   # RIOB

# Plot the dip equator (assuming lon_igr, lat_igr, incl are defined)
ax1.tricontour(lon_igr, lat_igr, incl, [0], linestyles='dashed', colors='k')


#set_xlabel('GLAT ($^{o}$)')
# # Plot satellite trajectories
# for i in range(traj_lon04.shape[0]):
#     if i == 3:
#         continue
#     ax.plot(traj_lon04[i], traj_lat04[i], 'o', markersize=2, color='magenta',
#             transform=ccrs.PlateCarree())
#     ax.plot(traj_lon04[i][239], traj_lat04[i][239], 'D', markersize=4, color='cyan',
#             transform=ccrs.PlateCarree())
    
# # Plot receiver locations
# for i in range(len(rec_lat)):
#     if i == 3:
#         continue
#     ax.plot(rec_lon[i], rec_lat[i], 'o', markersize=5, color='blue', transform=ccrs.PlateCarree())
    

# # # SISMO Data
# # sismo_path = sorted(glob.glob(r".\brazil_data\sismo\processed/*.npy"))

# for idx, data in enumerate(sismo_path):
#     data_sismo = load(data, allow_pickle=True).item()
#     t_sismo = data_sismo['t_sismo']; vel_sismo = data_sismo['vel_sismo']
#     ep_dist_sismo = data_sismo['ep_dist_sismo']; 
#     sta_name_sismo = data_sismo['sta_name_sismo']; 
#     lon_sismo = data_sismo['lon_sismo']; lat_sismo = data_sismo['lat_sismo']
    
#     fig = plt.figure(30, figsize=(10, 5))

#     ax.plot(lon_sismo, lat_sismo, "^", markersize=6, color='green', transform=ccrs.PlateCarree())
#     ax.text(lon_sismo+0.5, lat_sismo, sta_name_sismo, fontsize=6, 
#          transform=ccrs.PlateCarree())


# # Add labels
# ax1.text(-1.02, 0.63, 'GLAT ($^{o}$)', va='bottom', ha='center',
#         rotation='vertical', rotation_mode='anchor',
#         transform=ax.transAxes, fontweight='bold')
# ax1.text(0.02, 0.001, 'GLON ($^{o}$)', va='bottom', ha='center',
#         rotation='horizontal', rotation_mode='anchor',
#         transform=ax.transAxes, fontweight='bold')

# #tight_layout()
# #plt.savefig('./brazil_data/figures/observatory.png', dpi=200, bbox_inches='tight')
# # Show the plot
# #plt.show()


#---------------------


sta04_tec=np.load('./SISMO/obsvertario_data/sta_name_24_28.npy')
traj_lon04 = np.load('./SISMO/obsvertario_data/lon_nearest_24_28.npy')
traj_lat04 = np.load('./SISMO/obsvertario_data/lat_nearest_24_28.npy')
rec_lat = np.load('./SISMO/obsvertario_data/recs_lat_24_28.npy')
rec_lon = np.load('./SISMO/obsvertario_data/recs_lon_24_28.npy')


fig = plt.figure(30, figsize=(15, 8))
# ax1.text(-55, -8, 'Brazil', fontweight='bold', fontsize=12)
# ax1.text(-73.2, -41, 'Chile', fontweight='bold', fontsize=8,
#        rotation = 85)

# Seismometers
#ax.plot(-79, 0.24, "^", markersize=8, color='green')   # OTAV
# # Plot specific points
ax1.plot(-72.9327, -36.1485, "*", markersize=20, color='red')   # Chile



# Plot satellite trajectories


for i in range(0,traj_lon04.shape[0],1):
    ax1.plot(traj_lon04[i], traj_lat04[i], '.', markersize=0.05, color='magenta',
            transform=ccrs.PlateCarree())
    ax1.plot(traj_lon04[i][240], traj_lat04[i][240], 'D', markersize=3, color='cyan',
            transform=ccrs.PlateCarree())
    # ax1.text(rec_lon[i], rec_lat[i], s = sta04_tec[i], fontsize = 8,
    #         transform=ccrs.PlateCarree())

    
# Plot receiver locations
ax1.plot(rec_lon, rec_lat, 'o', markersize=4, color='blue', transform=ccrs.PlateCarree())

# -------------------

sism_list=sorted(glob.glob('./seismometers/maule_sismo/processed/***.npy'));#0937# 0261#0205
for idx, sism in enumerate(sism_list):  
    data_sismo = load(sism, allow_pickle=True).item()
    t_sismo = data_sismo['t_sismo']; vel_sismo = data_sismo['vel_sismo'];
    ep_dist_sismo = data_sismo['ep_dist_sismo'];
    sta_name_sismo = data_sismo['sta_name_sismo'];
    lon_sismo = data_sismo['lon_sismo']; lat_sismo = data_sismo['lat_sismo']
    
    # Calculating teleseismic distance
    [d1, d2] = calculate_distance(lon_ep, lat_ep, lon_sismo, lat_sismo)
    #dist_sismo=((d1+d2)[0]/2)/1.
    dist_sismo = (d1 + d2) / 2
    
    
    
    if sta_name_sismo not in ['LVC.', 'PB09', 'PB11', 'EFI.']:
        
        
        figure(17, figsize=(10,8))
        subplot(121)
        
        # if sta_name_sismo in ['TRQA', 'EL1.']:
        #     continue
        # plot(t_sismo[::200], dist_sismo + (150*vel_sismo[::200]), color='g')
        # text(6.53, dist_sismo+10, sta_name_sismo, fontsize=6)
        
        axvline(t_ep, color='grey',linestyle='--')
        xlabel("Time (UT, hr)")
        ylabel("Epicentral Distance (km)")
        grid('on', ls = '-.', lw=0.4)
        title("(A) U$_{sism}$")
        
        
        # if dist_sismo < 2000:
            
        #     text(7.0, dist_sismo+4, sta_name_sismo, fontsize=8)
            
            
            
        if sta_name_sismo == 'PB10':
            v1=(3.5/1)/(1./3600.)
            axline((t_ep+20/3600,0),slope=v1, lw = 0.5,
                   c='k' ,label = '~3.2 km/s')
            
            #text(6.68, 1250, "v$_{R}$ = 2.6 km/s")
            #text(6.53, 1450, "(A)")
            
            
        legend(loc='lower right')
        
    #     tight_layout()
    
    
    
    # figure(17, figsize=(10,8))
    # subplot(121)
    # plot(t_sismo, dist_sismo + (200*vel_sismo), color='g')
    # axis((6.5, 7.25, 0, 2000))
    
    
    vel_sismo = vel_sismo[::200]
    t_sismo = t_sismo[::200]
    
    if sta_name_sismo == 'TRQA': 
        # -----------------------------
        # List of (distance, V_max) pairs
        stations = [
            (650, 0.063),
            (800, 0.105),
            (900, 0.035),
            (1100, 0.010)
        ]
    
        # Output directory
        #os.makedirs("synthetic_waveforms", exist_ok=True)
    
        for distance, amp in stations:
            
            # Maximum velocities
            V_max_A = max(vel_sismo)
            V_max_B = amp #0.063  # example value in m/s
    
            # Scale waveform
            v_B = vel_sismo * (V_max_B / V_max_A)
    
            # Plot
            figure(21)
            
            plot(t_sismo, vel_sismo, label='Station A')
            plot(t_sismo, v_B, label='Synthetic Station B')
            legend()
            
            dat_mod = {'t': t_sismo,
                       'd': v_B,
                       'ep_dis': distance}
            # # Step 5: Save waveform
            # filename = f"synthetic_waveforms/station_{distance}km.npy"
            # save(filename, dat_mod)
    
    
    
    
    vel_sismo_filt = band_pass_filter(vel_sismo, t_sismo, lowcut=0.002, highcut=0.010)
    
    peak_mx = find_peaks_within_range_k(abs(vel_sismo_filt), 0, 300, 0.01, 1)
    
    
  
    # #peak_mx = abs(10*vel_sismo).argmax()
    
    # sismo_peaks_mx.append(peak_mx)
    # sismo_dist.append(dist_sismo)
    
    
    
    if sta_name_sismo not in ['LVC.', 'PB09', 'PB11', 'EFI.']:
        
        
    
        fig=figure(6,figsize=(10,8),facecolor='w',edgecolor='k')
        ax=subplot(121)
        
        if sta_name_sismo in ['TRQA', 'EL1.']:
            continue
        plot(t_sismo, dist_sismo + (300*vel_sismo), color='g')
        axis((6.5, 7.25, 0, 2000))
        axvline(t_ep, color='grey',linestyle='--')
        xlabel("Time (UT, hr)")
        ylabel("Epicentral Distance (km)")
        grid('on', ls = '-.', lw=0.4)
        title("$U_{sism}$")
        
        # if dist_sismo < 2000:
            
        #     text(7.0, dist_sismo+4, sta_name_sismo, fontsize=8)
            
            
            
        if sta_name_sismo == 'PB10':
            v1=(3.2/1)/(1./3600.)
            axline((t_ep+20/3600,0),slope=v1, lw = 0.5,
                   c='k' ,label = '~2.8 km/s')
            
            #text(6.68, 1250, "v$_{R}$ = 2.6 km/s")
            text(6.5, 1950, "(A)")
            
            
        legend(loc='lower right')
        
        tight_layout()
        
        # Observatory
        
        fig = plt.figure(30, figsize=(15, 8))
        
        if sta_name_sismo == 'TRQA':
            continue
        
        

        #ax1.plot(lon_sismo, lat_sismo, "^", markersize=6, color='green', transform=ccrs.PlateCarree())
        
        
        ax1.set_ylabel('GLAT ($^{o}$)')
        ax1.set_xlabel('GLON ($^{o}$)')
        
        ax1.set_xticks(ax1.get_xticks())
        ax1.set_yticks(ax1.get_yticks())
        
        # ax1.text(0.1, 0.71, 'GLAT ($^{o}$)', va='bottom', ha='center',
        #         rotation='vertical', rotation_mode='anchor',
        #         transform=ax.transAxes, fontweight='bold')
        # ax1.text(1.4, 0.00001, 'GLON ($^{o}$)', va='bottom', ha='center',
        #         rotation='horizontal', rotation_mode='anchor',
        #         transform=ax.transAxes, fontweight='bold')
        
        #savefig('./SISMO/obs1.png', bbox_inches='tight')
        
        # if sta_name_sismo == 'PB10':
        #     text(lon_sismo+0.5, lat_sismo, sta_name_sismo, fontsize=10, 
        #          transform=ccrs.PlateCarree())
        
        
        vel_sismo[vel_sismo <= 0.0035] = 0
        
        vel_pp = find_peaks_within_range_obs(vel_sismo, 20, 300, 0.003)
        
        
        
        figure(36, figsize=(12,12))
        subplot(211)
        if sta_name_sismo == 'TRQA':
            bar(dist_sismo+vel_sismo[vel_pp]*0, vel_sismo[vel_pp]*0, 
                color='g', width = 10)
            
        # elif sta_name_sismo == 'PB10':
        #     bar(dist_sismo+(vel_sismo[vel_pp]-0.12), vel_sismo[vel_pp]-0.12, 
        #         color='g', width = 10)
            
        else:
            bar(dist_sismo+(vel_sismo[vel_pp]/2), vel_sismo[vel_pp]/2, 
                color='g', width = 10)
        
        title("SEISMOMETER")
        #title("Peaks, $U_{sism}$ (Unfiltered data)")
        ylabel('Ground uplift (m/s)')
        xlim(0,900)
        
        
        
        figure(93, figsize=(6,8))        
        subplot(311)
        if sta_name_sismo == 'TRQA':
            bar(dist_sismo+vel_sismo[vel_pp]*0, vel_sismo[vel_pp]*0, 
                color='g', width = 10)
            
        # elif sta_name_sismo == 'PB10':
        #     bar(dist_sismo+(vel_sismo[vel_pp]-0.12), vel_sismo[vel_pp]-0.12, 
        #         color='g', width = 10)
        
        
        else:
            x_data = dist_sismo + vel_sismo[vel_pp]
            y_data = vel_sismo[vel_pp] 
        
            #print("CHECKKKKKKKKK: ", dist_sismo)
            bar(x_data, y_data, color='g', width=10)
            title("PGV")
            #title("Peaks, $U_{sism}$ (Unfiltered data)")
            ylabel('Ground uplift (m/s)')
            xlim(0,2300)
            
            if sta_name_sismo == 'PB10':
                text(2120, 0.85, "(A)")
            
            # figure(82, (6,8))
            # subplot(311)
            # bar(dist_sismo + (0.7*vel_sismo[vel_pp]) , 0.7*vel_sismo[vel_pp] , 
            #     color='g', width=15)
            #xlim(0,2300)
            # grid(linestyle='-.', lw=0.4)
            
        
            # Save data
        

            
            
    # ---------------------------
    
        
        
    
    
        vel_sismo_filt[vel_sismo_filt <= 0.000035] = 0
        
        vel_p = find_peaks_within_range_obs(vel_sismo_filt, 20, 300, 0.0003)
        
        

        figure(37,(12,12)) 
        subplot(211)
        
        if sta_name_sismo == 'TRQA':
            bar(dist_sismo+vel_sismo_filt[vel_p]*0, vel_sismo_filt[vel_p]*0, 
                color='g', width = 10)
        else:
            bar(dist_sismo+vel_sismo_filt[vel_p], vel_sismo_filt[vel_p], 
                color='g', width = 10)
        
        title("PGV_infra")
        #title("Peaks, $U_{sism}$ (Filtered data)")
        xlim(0, 900)

        
        figure(93, figsize=(6,8))
        subplot(312)
        
        if sta_name_sismo == 'TRQA':
            bar(dist_sismo+vel_sismo_filt[vel_p]*0, vel_sismo_filt[vel_p]*0, 
                color='g', width = 10)
        else:
            bar(dist_sismo+vel_sismo_filt[vel_p], vel_sismo_filt[vel_p], 
                color='g', width = 10)
            title("PGV_infra")
            #title("Peaks, $U_{sism}$ (Filtered data)")
            xlim(0, 2300)
            
            #axis((t_gnss[dtec_peak_obs][0],t_gnss[dtec_peak_obs][-1],0.,2300))
            ylabel('Ground uplift (m/s)')
            #xlabel('Epicentral Distance (km)')   
            #title('Observation', fontweight = 'bold')
            grid(linestyle='-.', lw=0.4)
            
            if sta_name_sismo == 'PB10':
                
                text(2120, 0.062, "(B)")
            
            
            # figure(82, (6,8))
            # subplot(312)
            # bar(dist_sismo+vel_sismo_filt[vel_p], vel_sismo_filt[vel_p], 
            #     color='g', width = 15)
            
            # grid(linestyle='-.', lw=0.4)
            # xlim(0, 2300)
        
        
        
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
np.save('./seismometers/iris_unfilt_peaks_x.npy', x_data)
np.save('./seismometers/iris_unfilt_peaks_y.npy', y_data)


















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
    ep_dist = (d1+d2)/2
    #ep_dist=((d1+d2)[0]/2)/1.
    dtec_gnss=data_smooth(t_gnss,tec_gnss)
    
    
    # # Add the satellite-receiver pair to the observatory
    # fig = plt.figure(30, figsize=(12, 10))
    # ax1.text(lon_gnss[0], lat_gnss[0], s = f_n[9:16], fontsize = 8,
    #         transform=ccrs.PlateCarree())
    
    fig=figure(6,figsize=(10,8),facecolor='w',edgecolor='k')
    ax=subplot(122)
    vm=0.1
   # plot(t_gnss,ep_dist+250*dtec_gnss,lw=2)
    im=scatter(t_gnss,ep_dist+100*dtec_gnss,c=0.6*dtec_gnss,s=20,
                vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow)
    
    if ep_dist <= 2000:
        
        text(7.4,ep_dist,f_n[9:16],size=8,color='k')
    

    axvline(t_ep,0,1,c='grey',linestyle='--')
    
    #axvline(6.755, ymin=0.3, color='r')
    # v1=(3./111)/(1./3600.)
    # axline((t_ep+8/60,0),slope=v1,c='g' ,label = '~1 km/s')
    axis((6.4,7.6,0.,2000))
   # f=  (axis_param(im,vm,4.4,'TECU'))
    ylabel('Epicentral Distance (km)')
    xlabel('Time (UT, hr)')
    title("dTEC")
    
    
    
    
    
    if f_n[9:16] == "srgdG20":
        f = axis_param(im, "TECU")
        
    if f_n[9:16] == "areqG11":
        text(6.4, 1950, "(B)")
        
        v1=(2.6/1)/(1./3600.)
        axline((t_ep+400/3600,0),slope=v1, lw=0.5,
               c='k' ,label = '~2.8 km/s')
        
        v2=(0.8/1)/(1./3600.)
        axline((t_ep+400/3600,0),slope=v2, ls = '--', 
               lw = 0.5, c='k' ,label = '~0.9 km/s')
        
        # text(6.68, 1250, "v$_{R}$ = 3.38 km/s")
        # text(6.5, 1950, "(A)")
    legend(loc = 'lower right')
    
    
    
    
    
    dtec_gnss[dtec_gnss <= 0.1] = 0

    
    dtec_peak_obs = find_peaks_within_range_k(dtec_gnss, 80, 280, 0.1, 1)
    
    time_obs = (t_gnss[dtec_peak_obs] - t_ep)*3600
    dist_obs = (lat_gnss[dtec_peak_obs] - lat_ep)*110
    vel_obs = dist_obs / time_obs
    
    
    
    
    
    #ax=subplot(122)
    # #plot(t_gnss,ep_dist+250*dtec_gnss,'r',lw=1)
    # i_ep=abs(x-ep_dist).argmin()
    
    # delta_h = 2.5e4  # 10 km in meters
    # scale_factor = delta_h / 1e16  # converts to TECU
    # tec = scale_factor * n3[:, i_ep, 7:23].sum(1)
    # #tec=(1e+06/1.e+16)*dn3[:,i_ep,7:23].sum(1)
    # #dtec=data_smooth_sim(t_hr,tec)
    
    # #tec = scale_factor * n3[:, :, 12].sum(1)

    
    # #tec=((5e+8/1.e+16))*n3[:,i_ep,12:].sum(1)
    # #tec_200km = ((5e+8/1.e+16))*n3[:,i_ep,12:].sum(1)
    # xlabel('Time (UT, hr)')
    
    
    # im=scatter(t_gnss,ep_dist+150*dtec_gnss,c=0.6*dtec_gnss,s=1.8*t_gnss,
    #             vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow,alpha=1)
    
    #scatter(t_hr[dtec_peak_obs], dtec_gnss[dtec_peak_obs])
    
    #scatter(t_gnss[dtec_peak_obs], ep_dist+(dtec_gnss[dtec_peak_obs]), c='b')

    #text(6.47,ep_dist,f_n[9:16],size='small',color='k')
    
    # axvline(x=t_ep,color='grey',lw=2) 
    # axhline(850, color = 'r', linestyle='--')
    # # v1=(3./111)/(1./3600.)
    # # axline((t_ep+8/60,0),slope=v1,c='g' ,label = '~1 km/s')
    #xlim(6.6, 7.4)
    #axis((t_gnss[dtec_peak_obs][0],t_gnss[dtec_peak_obs][-1],0.,2300))
    #xlabel('Time (hr, UT)')
    #ylabel('Epicentral Distance (km)')   
    #title('Observation')
    #grid(linestyle='-.', lw=0.4)
# text(6.81, 1122, "Fast mode", rotation = 90, fontsize=15)
# text(7.090, 1110, "Slow mode", rotation = 70, fontsize=15)
#text(6.61, 1810, '(a)', fontsize=15)
tight_layout()






# # Add labels
# ax1.text(-0.12, 0.53, 'GLAT ($^{o}$)', va='bottom', ha='center',
#         rotation='vertical', rotation_mode='anchor',
#         transform=ax.transAxes, fontweight='bold')
# ax1.text(0.5, -0.11, 'GLON ($^{o}$)', va='bottom', ha='center',
#         rotation='horizontal', rotation_mode='anchor',
#         transform=ax.transAxes, fontweight='bold')






   
    # ax=subplot(221)
        
        
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # figure(11)
    # scatter(t_sismo[peak_mx], dist_sismo + vel_sismo[peak_mx], 
    #         s = 10)
    # axis((t_sismo.min(), t_sismo.max(), 0, 2500))
    
    # figure(12)
    # bar(dist_sismo, vel_sismo_filt)


show()



#%%

f_list=sorted(glob.glob('DATA_TEC/***.npy'));#0937# 0261#0205
for idx, f_n in enumerate(f_list):  
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
    
    figure(17, figsize=(10,8))
    ax=subplot(122)
    vm=0.1
   # plot(t_gnss,ep_dist+250*dtec_gnss,lw=2)
    im=scatter(t_gnss,ep_dist+100*dtec_gnss,c=0.6*dtec_gnss,s=20,
                vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow)
    
    if ep_dist <= 1500:
        text(7.4,ep_dist,f_n[9:16],size=8,color='k')
    

    axvline(t_ep,0,1,c='grey',linestyle='--')
    
    #axvline(6.755, ymin=0.3, color='r')
    # v1=(3./111)/(1./3600.)
    # axline((t_ep+8/60,0),slope=v1,c='g' ,label = '~1 km/s')
    axis((6.4,7.6,0.,1500))
   # f=  (axis_param(im,vm,4.4,'TECU'))
    ylabel('Epicentral Distance (km)')
    xlabel('Time (UT, hr)')
    title("(B) dTEC")
    
    if f_n[9:16] == "areqG11":
        
        v1=(2.6/1)/(1./3600.)
        axline((t_ep+400/3600,0),slope=v1, lw=0.5,
               c='k' ,label = '~3.2 km/s')
        
        v2=(0.8/1)/(1./3600.)
        axline((t_ep+400/3600,0),slope=v2, ls = '--', 
               lw = 0.5, c='k' ,label = '~0.9 km/s')
        legend(loc='lower right')
        
        #text(6.4, 1450, "(B)")
        
        f=  (axis_param(im,'TECU'))
    #tight_layout()
    
    if 'baveG23' in f_n:
        figure(14, figsize=(7,6))
        subplot(313)
        plot(t_gnss, dtec_gnss, color='r', label='bave_G23')
        grid(linestyle='-.', lw=0.4)
        ylabel('dTEC (TECU)')
        legend(loc='upper right')
        xlim(6.5, 7.25)
        axvline(t_ep, color='grey', ls='--')
        xlabel('Time (UT, hr)')
        title("(C) dTEC")
        
        
        subplot(312)
        plot(t_gnss, 5+tec_gnss, color='r', label='bave_G23')
        grid(linestyle='-.', lw=0.4)
        ylabel('sTEC (TECU)')
        legend(loc='upper right')
        xlim(6.5, 7.25)
        axvline(t_ep, color='grey', ls='--')
        title("(B) sTEC")
    tight_layout()
    
    if f_n[9:16] == "areqG11":
        #text(6.4, 1950, "(B)")
        
        v1=(2.6/1)/(1./3600.)
        axline((t_ep+400/3600,0),slope=v1, lw=0.5,
               c='k' ,label = '~2.8 km/s')
        
        v2=(0.8/1)/(1./3600.)
        axline((t_ep+400/3600,0),slope=v2, ls = '--', 
               lw = 0.5, c='k' ,label = '~0.9 km/s')
    
    dtec_gnss[dtec_gnss <= 0.035] = 0
    
    dtec_peak_obs = find_peaks_within_range_obs(dtec_gnss, 80, 280, 0.03)
    
    # 80, 250, 0.015
    
    time_obs = (t_gnss[dtec_peak_obs] - 6.705)*3600
    dist_obs = (lat_gnss[dtec_peak_obs] + 33.762)*110
    vel_obs = (dist_obs) / (time_obs)
    
    
    
    # fig=figure(91,figsize=(8,6),facecolor='w',edgecolor='k')
    # ax=subplot(211)
    
    
    # # scatter(ep_dist+150*dtec_gnss[dtec_peak_obs], dtec_gnss[dtec_peak_obs], 
    # #         s=50*dtec_gnss[dtec_peak_obs], c=(t_gnss[dtec_peak_obs]-t_ep)*60)
    # bar(ep_dist+dtec_gnss[dtec_peak_obs], dtec_gnss[dtec_peak_obs], 
    #     color='red', width = 10)
    
    
    
    # xlim(0, 2300)
    
    # #axis((t_gnss[dtec_peak_obs][0],t_gnss[dtec_peak_obs][-1],0.,2300))
    # ylabel('dTEC (TECU)')
    # xlabel('Epicentral Distance (km)')   
    # #title('Observation', fontweight = 'bold')
    # grid(linestyle='-.', lw=0.4)
    
    # if idx == 0:
    #     text(43, 0.95, "(A)", fontsize=15)
        
        
        
    figure(93, figsize=(6, 8))
    subplot(313)
    bar(ep_dist+dtec_gnss[dtec_peak_obs], dtec_gnss[dtec_peak_obs], 
        color='red', width = 10)
    
    
    
    xlim(0, 2300)
    
    #axis((t_gnss[dtec_peak_obs][0],t_gnss[dtec_peak_obs][-1],0.,2300))
    ylabel('dTEC (TECU)')
    xlabel('Epicentral Distance (km)')   
    #title('Observation', fontweight = 'bold')
    grid(linestyle='-.', lw=0.4)
    title("Peaks, CSID")
    
    if idx == 0:
        text(2120, 0.95, "(C)")
        
        
    figure(82, (6,8))
    subplot(313)
    
    vlines(ep_dist,0,abs(dtec_gnss).max(),colors='r', lw=2)
    
    if ep_dist <= 1500:
        text(ep_dist, abs(dtec_gnss).max(), f_n[9:16], 
             fontsize=6, rotation=90)
    
    # bar(ep_dist+dtec_gnss[dtec_peak_obs], dtec_gnss[dtec_peak_obs], 
    #     color='red', width = 15)
    
    xlim(0, 1500)
    ylim(0,1.5)
    #axis((t_gnss[dtec_peak_obs][0],t_gnss[dtec_peak_obs][-1],0.,2300))
    ylabel('dTEC (TECU)')
    xlabel('Epicentral Distance (km)')   
    #title('Observation', fontweight = 'bold')
    grid(linestyle='-.', lw=0.4)
    title("(C) Peaks, CSID")
    # if f_n[9:16] == 'baveG23':
    #     text(240, 0.4, 'baveG23', 
    #          rotation=90, fontsize=6)
    #     text(760, 0.72, 'cmpnG23', 
    #          rotation=90, fontsize=6)
    #     text(644, 0.6, 'cmpnG20', 
    #          rotation=90, fontsize=6)
    
    # if idx == 0:
    #     text(2120, 0.95, "(C)")
    
    # fig=figure(92,figsize=(8,6),facecolor='w',edgecolor='k')
    # ax=subplot(313)
    # bar(ep_dist+dtec_gnss[dtec_peak_obs], dtec_gnss[dtec_peak_obs], 
    #     color='red', width = 10)
    
    
    
    # xlim(0, 2300)
    
    #axis((t_gnss[dtec_peak_obs][0],t_gnss[dtec_peak_obs][-1],0.,2300))
    # ylabel('dTEC (TECU)')
    # xlabel('Epicentral Distance (km)')   
    #title('Observation', fontweight = 'bold')
    grid(linestyle='-.', lw=0.4)

    
tight_layout()























































#%%
sism_list_1=sorted(glob.glob('./seismometers/maule_sismo/processed/new_data/***.npy'));#0937# 0261#0205
for idx, sism in enumerate(sism_list_1):  
    data_sismo = load(sism, allow_pickle=True).item()
    t_sismo = data_sismo['t_sismo']+0.2; vel_sismo = data_sismo['vel_sismo'];
    ep_dist_sismo = data_sismo['ep_dist_sismo'];
    sta_name_sismo = data_sismo['sta_name_sismo'];
    lon_sismo = data_sismo['lon_sismo']; lat_sismo = data_sismo['lat_sismo']
    
    print("-------------------------------", sta_name_sismo)
    # Calculating teleseismic distance
    [d1, d2] = calculate_distance(lon_ep, lat_ep, lon_sismo, lat_sismo)
    #dist_sismo=((d1+d2)[0]/2)/1.
    dist_sismo = (d1 + d2) / 2
    
    #----------------------------------------------------------
    # import numpy as np
    # from numpy import load, save
    # from matplotlib.pyplot import *
    
    
   
    # if sta_name_sismo == 'Constitucion':
    #     # Duration to extend on the right side (in seconds)
    #     ext_duration_sec = 360 / 3600  # 6 minutes
    #     sampling_rate = 1 / np.median(np.diff(t_sismo))  # infer from time array
    #     #print("888888888888888888888888888888", sampling_rate)
    #     n_extra_samples = int(ext_duration_sec * sampling_rate)
    
    #     # Compute stats for noise generation
    #     right_edge = vel_sismo[-int(10 * sampling_rate):]  # last 10 seconds
    #     max_amp = np.max(np.abs(vel_sismo))  # maximum amplitude of original data
    
    #     # Use small noise amplitude (1% of max amplitude)
    #     noise_std = 0.01 * max_amp
    #     noise_mean = np.mean(right_edge)
    
    #     # Generate noise to append to right
    #     right_noise = np.random.normal(noise_mean, noise_std, n_extra_samples)
    
    #     # Extend signal and time
    #     vel_sismo_extended = np.concatenate([vel_sismo, right_noise])
    #     t_sismo_extended = np.linspace(t_sismo[0], t_sismo[-1] + ext_duration_sec, len(vel_sismo_extended))
    
    #     # Plot the extended signal
    #     figure(15, figsize=(7,6))
    #     subplot(111)
    #     plot(t_sismo_extended, vel_sismo_extended, color='g', label='Constitucion (Extended)')
    #     xlim(6.6, 7.2)
    #     grid(linestyle='-.', lw=0.4)
    #     ylabel('Ground uplift (m/s)')
    #     legend(loc='upper right')
    #     axvline(t_ep, color='grey', ls='--')




    
        # # Optionally, save the extended trace
        # data_sismo['t_sismo'] = new_t_sismo
        # data_sismo['vel_sismo'] = new_vel_sismo
        # save('./seismometers/maule_sismo/processed/new_data_extended/' + sta_name_sismo + '_extended.npy', data_sismo)

    
    #----------------------------------------------------------
    
    
    
    ttt = t_sismo-0.025
    # if sta_name_sismo in ['Constitucion', 'ValparaisoAlmendral', 'Llolleo'] :
        
    #     if sta_name_sismo == "Constitucion":
    #         ext_duration_sec = 2500/3600
    #         #ext_duration_sec = 0.5  # 6 minutes
    #         sampling_rate = 1 / np.median(np.diff(t_sismo))
    #         #sampling_rate = 1/5000000 #10000
    #         n_extra_samples = int(ext_duration_sec * sampling_rate)
        
    #         right_edge = vel_sismo[-int(10 * sampling_rate):]
    #         max_amp = np.max(np.abs(vel_sismo))
        
    #         noise_std = 0.02 * max_amp
    #         noise_mean = np.mean(right_edge)
        
    #         # Generate noise and apply exponential decay
    #         right_noise = np.random.normal(noise_mean, noise_std, n_extra_samples)
    #         decay_envelope = np.exp(-np.linspace(0, 20, n_extra_samples))  # exponential decay
    #         right_noise *= decay_envelope
        
        
    #         vel_sismo_extended_consti = np.concatenate([vel_sismo, right_noise])
    #         t_sismo_extended = np.linspace(t_sismo[0], t_sismo[-1] + ext_duration_sec, len(vel_sismo_extended_consti))
        
    #         figure(14, figsize=(7,6))
    #         subplot(311)
    #         plot(t_sismo_extended-0.02, 3.2*vel_sismo_extended_consti, color='g', label='Constitucion')
    #         xlim(6.5, 7.25)
    #         grid(linestyle='-.', lw=0.4)
    #         ylabel('Ground uplift (m/s)')
    #         legend(loc='upper right')
    #         axvline(t_ep, color='grey', ls='--')
            
    #         figure(17)
    #         subplot(121)
    #         tcut = t_sismo_extended-0.1
    #         plot(tcut, dist_sismo + (800*vel_sismo_extended_consti), color = 'g')
    #         #xlim(6.5, 7.25)
    #         text(7.0, dist_sismo+10, 'Constitucion')
            
    #     elif sta_name_sismo == 'ValparaisoAlmendral':
    #         ext_duration_sec = 2500/3600
    #         #ext_duration_sec = 0.5  # 6 minutes
    #         #sampling_rate = 1 / np.median(np.diff(t_sismo))
    #         sampling_rate = 1/5000000 #10000
    #         n_extra_samples = int(ext_duration_sec * sampling_rate)
        
    #         right_edge = vel_sismo[-int(10 * sampling_rate):]
    #         max_amp = np.max(np.abs(vel_sismo))
        
    #         noise_std = 0.02 * max_amp
    #         noise_mean = np.mean(right_edge)
        
    #         # Generate noise and apply exponential decay
    #         right_noise = np.random.normal(noise_mean, noise_std, n_extra_samples)
    #         decay_envelope = np.exp(-np.linspace(0, 20, n_extra_samples))  # exponential decay
    #         right_noise *= decay_envelope
            
    #         vel_sismo_extended_paraiso = np.concatenate([vel_sismo, right_noise])
    #         t_sismo_extended = np.linspace(t_sismo[0], t_sismo[-1] + ext_duration_sec, len(vel_sismo_extended_paraiso))
            
    #         figure(17)
    #         subplot(121)
    #         tcut = t_sismo_extended-0.2
    #         plot(tcut, dist_sismo + (500*vel_sismo_extended_paraiso), color = 'g')
    #         #xlim(6.5, 7.25)
    #         text(7.0, dist_sismo+10, 'Valparaiso')
            
    #         # figure(15, figsize=(7,6))
    #         # subplot(111)
    #         # plot(t_sismo_extended-0.1, 3.2*vel_sismo_extended_paraiso, color='g', label='Constitucion')
    #         # xlim(6.5, 7.25)
    #         # grid(linestyle='-.', lw=0.4)
    #         # ylabel('Ground uplift (m/s)')
    #         # legend(loc='upper right')
    #         # axvline(t_ep, color='grey', ls='--')
            
            
    #     elif sta_name_sismo == 'Llolleo':
    #         ext_duration_sec = 2500/3600
    #         #ext_duration_sec = 0.5  # 6 minutes
    #         #sampling_rate = 1 / np.median(np.diff(t_sismo))
    #         sampling_rate = 1/5000000 #10000
    #         n_extra_samples = int(ext_duration_sec * sampling_rate)
        
    #         right_edge = vel_sismo[-int(10 * sampling_rate):]
    #         max_amp = np.max(np.abs(vel_sismo))
        
    #         noise_std = 0.02 * max_amp
    #         noise_mean = np.mean(right_edge)
        
    #         # Generate noise and apply exponential decay
    #         right_noise = np.random.normal(noise_mean, noise_std, n_extra_samples)
    #         decay_envelope = np.exp(-np.linspace(0, 20, n_extra_samples))  # exponential decay
    #         right_noise *= decay_envelope
            
    #         vel_sismo_extended_llo = np.concatenate([vel_sismo, right_noise])
    #         t_sismo_extended = np.linspace(t_sismo[0], t_sismo[-1] + ext_duration_sec, len(vel_sismo_extended_llo))
            
    #         figure(17)
    #         subplot(121)
    #         tcut = t_sismo_extended-0.2
    #         plot(tcut, dist_sismo + (500*vel_sismo_extended_llo), color = 'g')
    #         #xlim(6.5, 7.25)
    #         text(7.0, dist_sismo+10, 'Llolleo')
            
    #         # figure(15, figsize=(7,6))
    #         # subplot(111)
    #         # plot(t_sismo_extended-0.1, 3.2*vel_sismo_extended_llo, color='g', label='Constitucion')
    #         # xlim(6.5, 7.25)
    #         # grid(linestyle='-.', lw=0.4)
    #         # ylabel('Ground uplift (m/s)')
    #         # legend(loc='upper right')
    #         # axvline(t_ep, color='grey', ls='--')
    # tight_layout

    
    
    
    
    #if sta_name_sismo == 'Constitucion':
        
    #     # Duration to extend on the right side (in seconds)
    #     ext_duration_sec = 360 / 3600  # 6 minutes
    #     sampling_rate = 1 / np.median(np.diff(t_sismo))  # infer from time array
    #     #print("888888888888888888888888888888", sampling_rate)
    #     n_extra_samples = int(ext_duration_sec * sampling_rate)
    
    #     # Compute stats for noise generation
    #     right_edge = vel_sismo[-int(10 * sampling_rate):]  # last 10 seconds
    #     max_amp = np.max(np.abs(vel_sismo))  # maximum amplitude of original data
    
    #     # Use small noise amplitude (1% of max amplitude)
    #     noise_std = 0.01 * max_amp
    #     noise_mean = np.mean(right_edge)
    
    #     # Generate noise to append to right
    #     right_noise = np.random.normal(noise_mean, noise_std, n_extra_samples)
    
    #     # Extend signal and time
    #     vel_sismo_extended = np.concatenate([vel_sismo, right_noise])
    #     t_sismo_extended = np.linspace(t_sismo[0], t_sismo[-1] + ext_duration_sec, len(vel_sismo_extended))
    
        
        
        # figure(14, figsize=(7,6))
        # subplot(311)
        # #plot(t_sismo, vel_sismo, color='g', label='Constitucion')
        # plot(ttt, 3.2*vel_sismo, color='g', label='Constitucion')
        # xlim(6.55, 6.8)
        # grid(linestyle='-.', lw=0.4)
        # ylabel('Ground uplift (m/s)')
        
        # legend(loc='upper right')
        # axvline(t_ep, color='grey', ls='--')
    # tight_layout
    
    
    #hhhhhhh
    # vel_sismo = vel_sismo[::5]
    # t_sismo = t_sismo[::5]
    
    # vel_sismo_filt = band_pass_filter(vel_sismo, t_sismo, lowcut=0.002, highcut=0.010)
    
    # peak_mx = find_peaks_within_range_k(abs(vel_sismo_filt), 0, len(vel_sismo_filt), 0.05, 1)
    
    #print(lon_sismo)
    
    # figure(17)
    # subplot(121)
    
    # #ax=subplot(121)
    
    # if sta_name_sismo in ['Constitucion', 'ValparaisoAlmendral', 'Llolleo']:
    #     if dist_sismo < 250:
            
    #         if sta_name_sismo == 'Constitucion':
    #             t_conc = t_sismo - 0.023
                
    #             plot(t_conc, dist_sismo + (800*vel_sismo), color='g')
    #             #plot(t_sismo_extended, dist_sismo + (800*vel))
    #             text(7.0, dist_sismo+10, 'Constitucion')
                
            
    #         elif sta_name_sismo == 'Llolleo':
    #             t_conc = t_sismo - 0.023
    #             #plot(t_conc, dist_sismo + (100*(3*ve)), color='g')
    #             plot(t_conc, dist_sismo + (500*vel_sismo), color='g')
    #             text(7.0, dist_sismo+10, 'Llolleo')
            
    #     else:
            
    #         t_conc = t_sismo - 0.005
    #         #plot(t_conc, dist_sismo + (100*(3*ve)), color='g')
    #         plot(t_conc, dist_sismo + (500*vel_sismo), color='g')
    #         text(7.0, dist_sismo+10, 'Valparaiso')
    #         #plot(t_sismo, dist_sismo + (300*vel_sismo), color='g')
        
    
    # # # #scatter(lon_sismo, lat_sismo)
    # # # #print("LENGTH------------------", len(vel_sismo))
    # # # plot(t_sismo, dist_sismo + (300*vel_sismo))
    # # # #text(t_sismo[-1], dist_sismo, sta_name_sismo)
    # # axis((6.5, 6.9, 0, 2000))
    
    
    fig = plt.figure(30, figsize=(15, 8))
    
    # if sta_name_sismo in ['Vallenar', 'Copiapo']:
    #     continue
    
    # else:

    #     ax1.plot(lon_sismo, lat_sismo, "^", markersize=6, color='green', transform=ccrs.PlateCarree())
    #     #text(lon_sismo, lat_sismo, sta_name_sismo)
        
    
    
    # if sta_name_sismo in ['Constitucion']:
        
    #     text(lon_sismo+0.5, lat_sismo, sta_name_sismo, fontsize=10, 
    #          transform=ccrs.PlateCarree())
    #     #plot(-71.3391, -29.9591, '^', markersize=6, color='green', transform=ccrs.PlateCarree())
        
    #     text(-71.3391, -29.9591, 'Coquimbo', fontsize=10, transform=ccrs.PlateCarree())
        
    
    #if sta_name_sismo == 'Constitucion':
    
    
    #axis((6.4, 7.6, 0, 2000))
    # plot(t_sismo[::10], 3*ve, color='g', label='Constitucion\n(2-10 mHz)')
    # xlim(6.6, 6.7)
    # grid(linestyle='-.', lw=0.4)
    # ylabel('Ground uplift (m/s)')
    # legend(loc='upper right')
    # #tight_layout()
    
    # Filter data
    if len(vel_sismo) > 0:
        
        
        ve = band_pass_filter(vel_sismo[::20], t_sismo[::20], lowcut=0.002, highcut=0.01, dt=1)
        fig=figure(6,figsize=(10,8),facecolor='w',edgecolor='k')
        ax=subplot(121)
        
        if sta_name_sismo in ['Constitucion', 'ValparaisoAlmendral', 'Llolleo']:
        #if dist_sismo < 250 and sta_name_sismo in ['Constitucion', 'ValparaisoAlmendral', 'Llolleo']:
            t_conc = t_sismo[::20] - 0.005
            plot(t_conc, dist_sismo + (600*(3*ve)), color='g')
            
            
            if sta_name_sismo == 'Constitucion':
                t_conc = t_sismo[::20] - 0.03
                plot(t_conc, dist_sismo + (500*(3*ve)), color='g')
            
                
            
            
                
            
        # else:
        
        #     plot(t_sismo[::20], dist_sismo + (100*(3*ve)), color='g')
        
        vel_sismo = vel_sismo[::100]
        t_sismo = t_sismo[::100]
        
        vel_sismo_filt = band_pass_filter(vel_sismo, t_sismo, lowcut=0.002, highcut=0.010, dt = 1)
        
        # fig=figure(6,figsize=(10,8),facecolor='w',edgecolor='k')
        # # ax=subplot(121)
        
        
    
        vel_sismo_filt[vel_sismo_filt <= 0.000001] = 0
    
        vel_p = find_peaks_within_range_k(vel_sismo, 0, len(vel_sismo), 0.00002, 5)
        #find_peaks_within_range_k(arr, start_idx, end_idx, threshold, k)
        
        
    
        # fig=figure(91,figsize=(8,6),facecolor='w',edgecolor='k')
        # ax=subplot(212)
        
        
        # # scatter(ep_dist+150*dtec_gnss[dtec_peak_obs], dtec_gnss[dtec_peak_obs], 
        # #         s=50*dtec_gnss[dtec_peak_obs], c=(t_gnss[dtec_peak_obs]-t_ep)*60)
        # bar(dist_sismo+vel_sismo_filt[vel_p], vel_sismo_filt[vel_p], 
        #     color='g', width = 10)
        
        # # bar(dist_sismo+vel_sismo[vel_pp], vel_sismo[vel_pp], 
        # #     color='g', width = 10)
        
        
        
        
        # fig=figure(92,figsize=(8,6),facecolor='w',edgecolor='k')
        # ax=subplot(212)
        
        
        # bar(dist_sismo+vel_sismo_filt[vel_p], vel_sismo_filt[vel_p], 
        #     color='g', width = 10)
        
        # bar(650, 0.033, color='g', width = 10) # illapel
        
        # bar(800, 0.065, color='g', width = 10) # coquimbo
        
        # bar(900, 0.015, color='g', width = 10) # copiapo
        
        # bar(1100, 0.005, color='g', width = 10) # copiapo
        
        
        
        
        figure(93, figsize=(6,8))
        subplot(312)
        
        if sta_name_sismo == 'Concepcion':
            
            bar(dist_sismo+(vel_sismo_filt[vel_p]+0.018), vel_sismo_filt[vel_p]+0.018, 
                color='g', width = 10)
            
        else:
            bar(dist_sismo+(3*vel_sismo_filt[vel_p]), 3*vel_sismo_filt[vel_p], 
                color='g', width = 10)
            
        
        # bar(dist_sismo+(3*vel_sismo_filt[vel_p]), 3*vel_sismo_filt[vel_p], 
        #     color='g', width = 10)
        
        #bar(650, 0.033, color='g', width = 10) # illapel
        
        bar(700, 0.065, color='g', width = 10) # coquimbo
        
        # bar(900, 0.015, color='g', width = 10) # copiapo
        
        # bar(1100, 0.005, color='g', width = 10) # copiapo
        
        
        
        # fig=figure(92,figsize=(8,6),facecolor='w',edgecolor='k')
        # ax=subplot(211)
        vel_sismo[vel_sismo <= 0.0035] = 0
        
        vel_pp = find_peaks_within_range_obs(vel_sismo, 20, 300, 0.004)
        
        # bar(dist_sismo+vel_sismo[vel_pp], vel_sismo[vel_pp], 
        #     color='g', width = 10)
        # bar(650, 0.063, color='g', width = 10) # illapel
        
        # bar(800, 0.105, color='g', width = 10) # coquimbo
        
        # bar(900, 0.035, color='g', width = 10) # copiapo
        
        # bar(1100, 0.01, color='g', width = 10) # copiapo
        
        # xlim(0, 2300)
        
        
        # ylabel('Ground uplift\n(m/s)')
        # grid(linestyle='-.', lw=0.4)
        
        
        # -------------------------------------------------------------
        figure(36, figsize=(12,12))
        subplot(211)
        # vel_sismo[vel_sismo <= 0.0035] = 0
        
        # vel_pp = find_peaks_within_range_obs(vel_sismo, 20, 300, 0.004)
        
        bar(dist_sismo+(3*vel_sismo[vel_pp]), 3*vel_sismo[vel_pp], 
            color='g', width = 10)
        #bar(650, 0.063, color='g', width = 10) # illapel
        
        bar(700, 0.205, color='g', width = 10) # coquimbo
        
        # bar(900, 0.035, color='g', width = 10) # copiapo
        
        # bar(1100, 0.01, color='g', width = 10) # copiapo
        
        xlim(0, 900)
        
        figure(37,(12,12))
        ax=subplot(211)
        
        if sta_name_sismo == 'Concepcion':
            
            bar(dist_sismo+(vel_sismo_filt[vel_p]/3), vel_sismo_filt[vel_p]/3, 
                color='g', width = 10)
            
        else:
            bar(dist_sismo+(3*vel_sismo_filt[vel_p]), 3*vel_sismo_filt[vel_p], 
                color='g', width = 10)
        bar(700, 0.065, color='g', width = 10) # coquimbo
            
        xlim(0,900)
            
        
        figure(93, figsize=(6,8))
        ax=subplot(311)
        # vel_sismo[vel_sismo <= 0.0035] = 0
        
        # vel_pp = find_peaks_within_range_obs(vel_sismo, 20, 300, 0.004)
        
        bar(dist_sismo+(3*vel_sismo[vel_pp]), 3*vel_sismo[vel_pp], 
            color='g', width = 10)
        #bar(650, 0.063, color='g', width = 10) # illapel
        
        bar(700, 0.205, color='g', width = 10) # coquimbo
        
        # bar(900, 0.035, color='g', width = 10) # copiapo
        
        # bar(1100, 0.01, color='g', width = 10) # copiapo
        
        xlim(0, 2300)
        grid(linestyle='-.', lw=0.4)
        
        figure(93, figsize=(6,8))
        ax=subplot(312)
        
        if sta_name_sismo == 'Concepcion':
            
            bar(dist_sismo+(vel_sismo_filt[vel_p]/3), vel_sismo_filt[vel_p]/3, 
                color='g', width = 10)
            
        else:
            bar(dist_sismo+(3*vel_sismo_filt[vel_p]), 3*vel_sismo_filt[vel_p], 
                color='g', width = 10)
            
            
        
        
        #bar(650, 0.033, color='g', width = 10) # illapel
        
        bar(700, 0.065, color='g', width = 10) # coquimbo
        
        # bar(900, 0.015, color='g', width = 10) # copiapo
        
        # bar(1100, 0.005, color='g', width = 10) # copiapo
        
        # if idx == 0:
        #     text(43, 0.95, "(A)", fontsize=15)
    #suptitle("Amplitude Distribution of the Filtered data: 2-10 mHz")
    
    
    #figure(92)
    tight_layout()    
show()
    
#%%


mod_data = sorted(glob.glob("./synthetic_waveforms/*.npy"))


for i, mod in enumerate(mod_data):
    mod_sism = load(mod, allow_pickle=True).item()
    t = mod_sism['t']; d = mod_sism['d']; ep_dis = mod_sism['ep_dis']
    
    fig=figure(6,figsize=(10,8),facecolor='w',edgecolor='k')
    ax=subplot(121)
    
    if ep_dis == 650:
        plot(t-0.03, 50 + ep_dis + (1800*d), color='g')
        
    # elif ep_dis == 800:
        
    #     plot(t-0.03, ep_dis + (2000*d), color='g')
        
    # else:
    
    #     plot(t, ep_dis + (2000*d), color='g')
    #plot(t, d)
    
    
    
show()


#%%

# sism_list=sorted(glob.glob('./seismometers/maule_sismo/processed/***.npy'));#0937# 0261#0205
# for idx, sism in enumerate(sism_list):  
#     data_sismo = load(sism, allow_pickle=True).item()
#     t_sismo = data_sismo['t_sismo']; vel_sismo = data_sismo['vel_sismo'];
#     ep_dist_sismo = data_sismo['ep_dist_sismo'];
#     sta_name_sismo = data_sismo['sta_name_sismo'];
#     lon_sismo = data_sismo['lon_sismo']; lat_sismo = data_sismo['lat_sismo']
    
#     # Calculating teleseismic distance
#     [d1, d2] = calculate_distance(lon_ep, lat_ep, lon_sismo, lat_sismo)
#     #dist_sismo=((d1+d2)[0]/2)/1.
#     dist_sismo = (d1 + d2) / 2
    
    
#     if sta_name_sismo == 'TRQA':
        
#         figure(17, figsize=(10,8))
#         subplot(121)
        
        
#         plot(t_sismo-0.02, (dist_sismo-300) + (200*vel_sismo), color='g')
#         # axis((6.5, 6.8, 0, 2000))
#         # axvline(t_ep, color='grey',linestyle='--')
#         # xlabel("Time (UT, hr)")
#         # ylabel("Epicentral Distance (km)")
#         # grid('on', ls = '-.', lw=0.4)
#         # title("U$_{0}$")
#         text(7.0, 715, 'Coquimbo')
#         axis((6.5, 7.25, 0, 2000))

#%%

# shakemap = pd.read_csv('finite_fault.csv')
# shakemap.columns

import json
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import MultiLineString, LineString
from shapely.ops import transform
import cartopy.io.shapereader as shpreader

# --- Load PGV GeoJSON file ---
with open("cont_pgv.json", "r") as f:
    pgv_data = json.load(f)

# --- Setup Cartopy Map ---
#fig = plt.figure(figsize=(12, 10))
fig = plt.figure(30, figsize=(15, 8))
#ax = plt.axes(projection=ccrs.PlateCarree())

# Optional: Zoom into Chile region
#ax.set_extent([-76, -60, -45, -15], crs=ccrs.PlateCarree())
# -82, -65, -45, -25


# # Add map features
# ax.add_feature(cfeature.COASTLINE)
# ax.add_feature(cfeature.BORDERS, linestyle=':')
# ax.add_feature(cfeature.LAND, facecolor='lightgray')
# ax.add_feature(cfeature.OCEAN, facecolor='azure')
# ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)

# --- Plot PGV Contours ---
for feature in pgv_data["features"]:
    value = feature["properties"]["value"]
    color = feature["properties"]["color"]
    coords_list = feature["geometry"]["coordinates"]

    for line_coords in coords_list:
        lons, lats = zip(*line_coords)
        plot(lons, lats, color=color, linewidth=2,
                label=f'{value} {feature["properties"]["units"]}' if value else None)

# # Optional: Add legend (only unique labels)
# handles, labels = ax.get_legend_handles_labels()
# unique = dict(zip(labels, handles))
# ax1.legend(unique.values(), unique.keys())
# #ax1.legend(unique.values(), unique.keys(), title="PGV Contours")

#plt.title("PGV Contour Map – Maule, Chile Earthquake (2010)")
# savefig("/home/oluwasegun-adebayo/Desktop/chile/simulationonthewaverayleighinducedcoseismicionospher/Pub Figures/obs2.png",
#         bbox_inches = 'tight')
show()


#%%

# metadata of the stations

metadata = pd.read_csv("./high_rate_gnss/coordinates.csv")

#print(metadata.head())
#print(metadata.info())


#%%


from obspy import read

# Path to your folder with .mseed files
mseed_folder = sorted(glob.glob("./high_rate_gnss/*.LXZ.mseed"))

total_figure = len(mseed_folder)
# Compute number of subplot rows and columns
cols = 4  
rows = int(total_figure / cols)

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

    
    # figure(34, figsize=(12,12))
    # subplot(rows, cols, f_no + 1)
    # #plot(time_sis, amplitude, label=station_code)
    
    # plot(time_sis, velocity, label=station_code)

    # legend()
    
    velocity_filtered = band_pass_filter(velocity, time_sis, lowcut=0.020, highcut=0.040, dt=1)
    
    nm=int(160./1.)
    nmode=ones((nm,))/nm
    nm=int(200./1.)
    nmode1=ones((nm,))/nm
    #velocity_filtered=np.convolve(velocity, nmode,mode='same')#-np.convolve(velocity, nmode1,mode='same')
    # plot(time_sis,metadata['epi. distance'][f_no] +200*vel_conv)#,label=f"{station_code}")
    #legend()
    
    # figure(35, figsize=(10,12))
    # #if f_no==18 or f_no==14:
    # subplot(121)
    # plot(time_sis, metadata['epi. distance'][f_no] + (100*velocity),label=f"{station_code}")
    #     #legend()
        
    #     # filtered
        
    # subplot(122)
        
    # plot(time_sis, metadata['epi. distance'][f_no] + (200*velocity_filtered))
        
   
    
    figure(36,(12,12))
    subplot(212)
    bar(metadata['epi. distance'][f_no],velocity.max())
    title('GNSS')
    xlim(0,900)
    # subplot(212)
    # bar(metadata['epi. distance'][f_no],velocity_filtered[50:-50].max())
    #xlim(0,2300)
    print (len(time_sis),f"{station_code}")
    
    figure(37,(12,12))
    subplot(212)
    bar(metadata['epi. distance'][f_no],velocity_filtered[50:-50].max())
    title('GNSS')
    xlim(0,900)
    # subplot(212)
    # bar(metadata['epi. distance'][f_no],velocity_filtered[50:-50].max())
    #xlim(0,2300)
    print (len(time_sis),f"{station_code}")
    
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
    
    figure(38, figsize=(12,12))
    scatter(metadata['longitude'][f_no], metadata['latitude'][f_no], marker='^')
    scatter(lon_ep, lat_ep, marker='*')
    text(metadata['longitude'][f_no], metadata['latitude'][f_no], s=f"{station_code}")
    
    figure(30)
    # Plot receiver locations
    ax1.plot(metadata['longitude'][f_no], metadata['latitude'][f_no], 
             marker='^', markersize=4, color='r', transform=ccrs.PlateCarree())
    
    if station_code in ['SJAV', 'CNBA']:
        #figure(30)
        text(metadata['longitude'][f_no], metadata['latitude'][f_no],
             station_code, fontsize=10, transform=ccrs.PlateCarree())

    
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
        







# # Process seismic data
# velocity = amplitude * 1.58997e-06 * 6.e+00 / (1. * 1500)  # m/s (V/counts) / (gain * sensitivity (V/m/s))
# nx = arange(int(npts / 100), int(npts / 1), 1)
# velocity = velocity[nx]
# time = time_sis[nx]





#%%



from obspy import read
from datetime import date



f_list=glob.glob('./high_rate_gnss/*.LXZ.mseed')#,allow_pickle=True)
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
    
    iw=0
    f=wavelet(iw, t, v,32)
    pd=f[0];emd=f[2];fr=1.e+03/(pd*3600.)
    if iw==2:
        fr=fr/5.
    if iw==0:
        fr=fr/1.4
    if iw==1:
        fr=3*fr
    if st_name=='CNBA' or st_name=='MAUL':# or st_name=='SJVA' or st_name=='CONZ':
        ipl=ipl+1
        
        figure(80,(12,12))
        subplot(1,2,ipl)
        for ifr in range (31):
            plot(t,fr[ifr]+5*emd[ifr,:])
        ylim(11,50)
        title(st_name)
        xlabel('Time, UT')
        ylabel('Frequency, mHz')
    ifr40=abs(fr-40).argmin()
    vlow40=emd[ifr40,:]    
    ifr9=abs(fr-14).argmin()
    vlow9=emd[ifr9,:]    
    
    vlow=emd[ifr40:ifr9,:].mean(0)
    
    if st_name == 'SJAV':
        figure(14, figsize=(7,6))
        subplot(311)
        #plot(t_sismo, vel_sismo, color='g', label='Constitucion')
        plot(t[:-20], v[:-20], color='k', label=st_name)
        xlim(6.55, 6.8)
        grid(linestyle='-.', lw=0.4)
        ylabel('Ground uplift\n(m/s)')
        title("(A) $U_{0}$")
        legend(loc='upper right')
        axvline(t_ep, color='grey', ls='--')
    tight_layout()
    
    figure(81,(10,8))
    ax1=subplot(121)
    plot(t, dist+50*v, color='g',
         label=st_name+' PGV= '+str(around(abs(v).max(),decimals=2)))
    xlim(6.56,6.7)
    title('U$_{SISM}$')
    ylabel('Epicentral distance, km')  
    xlabel('Time, UT')
    text(6.68, dist, st_name)
    
    ax2=subplot(122)
    plot(t, dist+100*vlow, color='g',
         label=st_name+' PGV= '+str(around(abs(vlow).max(),decimals=2)))
    xlim(6.56,6.7)
    title('U$_{SISM}$ in 11mHz-40mHz')
    xlabel('Time, UT')
    #ylabel('Epicentral distance, km')  
    text(6.68, dist, st_name)
    tight_layout()
    
    # plotting into the figure 17
    figure(17)
    ax=subplot(121)
    
    if st_name in ['EMAT', 'LSCH', 'CRZL', 'OVLL', 'BTON', 'PFRJ']:
        continue
    
    vm=0.3
   # plot(t_gnss,ep_dist+250*dtec_gnss,lw=2)
    im=scatter(t,dist+50*v,c=0.6*v,s=15,
                vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow)
    
    # plot(t, dist+50*v, color='k',
    #      label=st_name+' PGV= '+str(around(abs(v).max(),decimals=2)))
    xlim(6.55,6.7)
    #title('(A) U$_{sism}$')
    # ylabel('Epicentral distance, km')  
    # xlabel('Time, UT')
    text(6.55, dist, st_name, fontsize=6)
    ylim(0,1500)
    
    if st_name == 'SJAV':
        f=  (axis_param(im,'m/s'))
    
    tight_layout()
    
    figure(82,(6,8))
    subplot(311)
    if st_name in ['EMAT', 'LSCH', 'CRZL', 'OVLL', 'BTON', 'PFRJ']:
        continue
        
    vlines(dist,0,abs(v).max(),colors='k', lw=2)
    title('(A) PGV')
    ylabel('m/s')
    xlim(0,1500)
    ylim(0,1.4)
    grid(linestyle='-.', lw=0.4)
    text(dist, abs(v).max(), st_name, 
         fontsize=6, rotation=90)
    
    subplot(312)
    vlines(dist, 0, abs(vlow).max(),colors='k', lw=2)
    #text(dist,abs(vlow).max(),st_name)
    title('(B) PGV in 11mHz-40 mHz')
    #xlabel('Epicentral distance, km')
    ylabel('m/s')
    xlim(0,1500)
    ylim(0, 0.14)
    grid(linestyle='-.', lw=0.4)
    text(dist, abs(vlow).max(), st_name, 
         fontsize=6, rotation=90)
    
    
    vmx.append(abs(vlow).max())
    distmx.append(dist)
    
        
    i=i+1
#ax1.legend()
#ax2.legend()
tight_layout()

idx=argsort(array(distmx))
dist_sort=array(distmx)[idx]
vmx_sort=array(vmx)[idx]
figure(83,(12,12))
plot(dist_sort,vmx_sort,'-o')
xlabel('Time, UT')
ylabel('m/s')
title('PGV in 11mHz-40mHz')



#%%

simulation_t = load("./chile_simulation_cases/t3_case_22.npy")
simulation_v_new1 = load("./chile_simulation_cases/v_new_case_22.npy")
simulation_n31 = load("./chile_simulation_cases/n3_case_22.npy")

#simulation_t = load("./chile_simulation_cases/t3_case_1.npy")
simulation_v_new2 = load("./chile_simulation_cases/v_new_case_2.npy")
simulation_n32 = load("./chile_simulation_cases/n3_case_2.npy")

#simulation_t = load("./chile_simulation_cases/t3_case_1.npy")
simulation_v_new3 = load("./chile_simulation_cases/v_new_case_3.npy")
simulation_n33 = load("./chile_simulation_cases/n3_case_3.npy")


# Collect maximum amplitudes for each ix
max_amps_sism_1 = []
x_vals_sism_1 = []

max_amps_sism_2 = []
x_vals_sism_2 = []

max_amps_sism_3 = []
x_vals_sism_3 = []


fig=figure(86,figsize=(8,8),facecolor='w',edgecolor='k')

ax =subplot(1,2,1)
#ax =subplot(1,1,1)
for ix in range (0,len(x),1):
    # wy3[:,i_xo,0]
    #pcolormesh(t3[:,0,0],x,v_new[:,:,0].T)
    
    #v_new[100:,ix,0] = 0
    
    x_dt = (simulation_t[:,0,0]/3600)+6.56
    y_dt = (x[ix]) + (100*simulation_v_new1[:,ix,0])

    plot(x_dt, y_dt, color='g', label=f'x={x[ix]:.2f}')
    
    max_amps_sism_1.append(abs(simulation_v_new1[:, ix, 0]/15).max())
    x_vals_sism_1.append(x[ix])
    
    
    x_dt = (simulation_t[:,0,0]/3600)+6.56
    y_dt = (x[ix]) + (100*simulation_v_new2[:,ix,0])

    #plot(x_dt, y_dt, color='g', label=f'x={x[ix]:.2f}')
    
    max_amps_sism_2.append(abs(simulation_v_new2[:, ix, 0]/10).max())
    x_vals_sism_2.append(x[ix])
    
    
    x_dt = (simulation_t[:,0,0]/3600)+6.56
    y_dt = (x[ix]) + (100*simulation_v_new3[:,ix,0])

    #plot(x_dt, y_dt, color='g', label=f'x={x[ix]:.2f}')
    
    max_amps_sism_3.append(abs(simulation_v_new3[:, ix, 0]).max())
    x_vals_sism_3.append(x[ix])
    
    
    # plot(t_hr[::1],(x[ix])+0.1e+02*wx3[:,ix,0][::1],'g')
    # xlim(t_hr[0],t_hr[-1])
    # ##plot(t_original,(x[ix])+0.1e+02*wx3[:,ix,0],'g')
    # #xlim(t_original[0],t_original[-1])
    # xlim(6.4,7.8)
    # ylabel('Epicentral Distance, km ')
    # xlabel('Time, UT')
    # if ix==0:
    #     v1=(3.5/1)/(1./3600.)
    #     axline((t_ep+100/3600,0),slope=v1,c='k' ,label = '~2.8 km/s')

    #     legend()
    #     axvline(x=t_ep,color='grey',lw=1,linestyle='--')  
    
figure(87, figsize=(12, 6))
subplot(232)
vlines(x_vals_sism_1, 0, ymax=max_amps_sism_1, color='lime')
title("(B) Case-PGV-Infra")

subplot(233)
vlines(x_vals_sism_2, 0, ymax=max_amps_sism_2, color='lime')
title("(C) Case-TEC")

subplot(231)
vlines(x_vals_sism_3, 0, ymax=max_amps_sism_3, color='lime')
ylabel("PGV (m/s)")
title("(A) Case-PGV")

#scatter(x_vals, max_amps, c='lime', )
#bar(x_vals, max_amps, color='lime')
# xlabel('x')
# ylabel('Max amplitude')
# title('Max amplitude per ix')

max_amps_tec_1 = []
x_vals_tec_1 = []

max_amps_tec_2 = []
x_vals_tec_2 = []

max_amps_tec_3 = []
x_vals_tec_3 = []

dn3_1=1*(len(t)/2.)*gradient(simulation_n31)[0];
dn3_2=1*(len(t)/2.)*gradient(simulation_n32)[0];
dn3_3=1*(len(t)/2.)*gradient(simulation_n33)[0];


fig=figure(86,figsize=(8,8),facecolor='w',edgecolor='k')
ax =subplot(1,2,2)
for ix in range (0,len(x),1):   
    # delta_h = 2.5e4  # 10 km in meters
    # scale_factor = delta_h / 1e16  # converts to TECU
    # tec = scale_factor * dn3[:, ix, 7:23].sum(1)
    tec=(1e+06/1.e+16)*dn3_1[:,ix,7:23].sum(1)
    dtec=data_smooth_sim(t_hr,tec)
    
    max_amps_tec_1.append(abs(dtec/900).max())
    x_vals_tec_1.append(x[ix])
    nm=1

    vm=dtec.max()
    
    xlim(t_hr[0],t_hr[-1])
    im=scatter(t_hr,0.5*dtec+(x[ix]),c=0.6*dtec,s=20.,
                vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow)   
    
    
    tec=(1e+06/1.e+16)*dn3_2[:,ix,7:23].sum(1)
    dtec=data_smooth_sim(t_hr,tec)
    
    max_amps_tec_2.append(abs(dtec/800).max())
    x_vals_tec_2.append(x[ix])
    
    tec=(1e+06/1.e+16)*dn3_3[:,ix,7:23].sum(1)
    dtec=data_smooth_sim(t_hr,tec)
    
    max_amps_tec_3.append(abs(dtec/250).max())
    x_vals_tec_3.append(x[ix])

    # if ix==0:
    #     v1=(4/1)/(1./3600.)
    #     axline((t_ep+420/3600,0),slope=v1,c='k' ,label = '~4 km/s')
    #     v1=(1/1)/(1./3600.)
    #     axline((t_ep+420/3600,0),slope=v1,c='g' ,label = '~1 km/s')
    #     legend()
    #     axvline(x=t_ep,color='grey',lw=1,linestyle='--')    
        

figure(87, figsize=(12,6))
subplot(235)
vlines(x_vals_tec_1, 0, ymax=max_amps_tec_1, color='r')
xlabel("Epicentral Distance (km)")

subplot(236)
vlines(x_vals_tec_2, 0, ymax=max_amps_tec_2, color='r')
xlabel("Epicentral Distance (km)")
title("")

subplot(234)
vlines(x_vals_tec_3, 0, ymax=max_amps_tec_3, color='r')
ylabel("dTEC (TECU)")
xlabel("Epicentral Distance (km)")
tight_layout()

# #scatter(x_vals, max_amps, c='lime', )
# #bar(x_vals, max_amps, color='lime')
# xlabel('x')
# ylabel('Max amplitude')
# title('Max amplitude per ix') 


#%%

# fig=figure(3,figsize=(8,8),facecolor='w',edgecolor='k')
# ax =subplot(1,2,2)
for ix in range (0,len(x),2):   
    # delta_h = 2.5e4  # 10 km in meters
    # scale_factor = delta_h / 1e16  # converts to TECU
    # tec = scale_factor * dn3[:, ix, 7:23].sum(1)
    tec=(1e+06/1.e+16)*dn3_1[:,ix,7:23].sum(1)
    dtec=data_smooth_sim(t_hr,tec)
    
    # max_amps_tec_1.append(abs(dtec).max())
    # x_vals_tec_1.append(x[ix])
    nm=1

    vm=dtec.max()
        
    figure(88, figsize=(12,8))
    subplot(132)
    xlim(t_hr[0],t_hr[-1])
    im=scatter(t_hr,0.1*(-dtec)+(x[ix]),c=0.6*-dtec,s=20.,
                vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow)  
    
    
for ix in range (0,len(x),2):  
    tec=(1e+06/1.e+16)*dn3_2[:,ix,7:23].sum(1)
    dtec=data_smooth_sim(t_hr,tec)
    
    # max_amps_tec_1.append(abs(dtec).max())
    # x_vals_tec_1.append(x[ix])
    nm=1

    vm=dtec.max()
        
    figure(88, figsize=(12,8))
    subplot(133)
    xlim(t_hr[0],t_hr[-1])
    im=scatter(t_hr,0.1*(-dtec)+(x[ix]),c=0.6*-dtec,s=20.,
                vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow)  
    
    
for ix in range (0,len(x),2):  
    tec=(1e+06/1.e+16)*dn3_3[:,ix,7:23].sum(1)
    dtec=data_smooth_sim(t_hr,tec)
    
    # max_amps_tec_1.append(abs(dtec).max())
    # x_vals_tec_1.append(x[ix])
    nm=1

    vm=dtec.max()
        
    figure(88, figsize=(12,8))
    subplot(131)
    xlim(t_hr[0],t_hr[-1])
    im=scatter(t_hr,0.1*(-dtec)+(x[ix]),c=0.6*-dtec,s=20.,
                vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow)  

tight_layout()



#%%

maule_more_tec = sorted(glob.glob("./maule/*.npy"))

recei_lon = []
recei_lat = []

# Set up the Cartopy figure only once

fig = plt.figure(102, figsize=(10, 10))
#fig = plt.figure(30, figsize=(15, 8))
#ax_c = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax_c = plt.axes(projection=ccrs.PlateCarree())

# Set the extent of the map (latitude and longitude range)
ax_c.set_extent([-80, -60, -45, -15])
ax_c.add_feature(cf.COASTLINE, lw = 1)
ax_c.add_feature(cf.BORDERS, lw = 0.8)
ax_c.add_feature(cf.LAND)
ax_c.add_feature(cf.OCEAN)
#ax.add_feature(cf.RIVERS)
#ax.add_feature(cf.LAKES)

# Add borders with different linewidth for Brazil
#ax.add_feature(cf.BORDERS.with_scale('10m'), linestyle='-', edgecolor='red', linewidth=2)

# Add features to the map
ax_c.coastlines()


for data_idx, data in enumerate(maule_more_tec):
    data_gnss=load(data,allow_pickle=True).item()
    # if idx ==0:
    #     print(data_gnss.keys())
    t_gnss=data_gnss['t_gnss'];tec_gnss=data_gnss['tec_gnss'];
    vtec_gnss = data_gnss['vtec_gnss']; ep_dist = data_gnss['ep_dist'];
    lon_gnss=data_gnss['lon_gnss'];lat_gnss=data_gnss['lat_gnss'];
    lon_rec = data_gnss['lon_rec']; lat_rec = data_gnss['lat_rec']
    elev_ang = data_gnss['elev_ang']; azim_ang = data_gnss['azim_ang']
    
    v_prox = find_nearest(t_gnss, t_ep)
    p_sip = where(t_gnss == v_prox)[0]
    
    
    if data[8:15] == 'baveG23':
        figure(30)
        text(lon_rec, lat_rec, data[8:15])
    
   
    dtec_gnss=data_smooth_mike(t_gnss,tec_gnss, 10, 5)
    vm=0.07
    fig=figure(100,figsize=(12,5),facecolor='w',edgecolor='k')
    
    
    if "G23" in data:
        ax=subplot(141)
        #vm=0.07
       # plot(t_gnss,ep_dist+250*dtec_gnss,lw=2)
        im=scatter(t_gnss,ep_dist+100*dtec_gnss,c=0.6*dtec_gnss,s=20*abs(dtec_gnss),
                    vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow)
        
        # if ep_dist < 1500:
    
        #     text(3.25,ep_dist,f_n[22:29],size='small',color='k')
    
        axvline(t_ep)
        # v1=(3./111)/(1./3600.)
        # axline((t_ep+8/60,0),slope=v1,c='g' ,label = '~1 km/s')
        #axis((2.7, 3.4,0.,1500))
       # f=  (axis_param(im,vm,4.4,'TECU'))
        ylabel('Epicentral Distance (km)')
        xlabel('Time (UT, hr)')
        xlim(6.4, 7.8)
        title("PRN 23")
        
        #text(7.5, ep_dist, data[8:15])
        
        
        recei_lat.append(lat_rec); recei_lon.append(lon_rec)
        
    elif "G20" in data:
        #fig=figure(100,figsize=(10,8),facecolor='w',edgecolor='k')
        ax=subplot(142)
        #vm=0.07
       # plot(t_gnss,ep_dist+250*dtec_gnss,lw=2)
        im=scatter(t_gnss,ep_dist+(150*dtec_gnss),c=dtec_gnss,s=20*abs(dtec_gnss),
                    vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow)
        
        # if ep_dist < 1500:
    
        #     text(3.25,ep_dist,f_n[22:29],size='small',color='k')
    
        axvline(t_ep)
        # v1=(3./111)/(1./3600.)
        # axline((t_ep+8/60,0),slope=v1,c='g' ,label = '~1 km/s')
        #axis((2.7, 3.4,0.,1500))
       # f=  (axis_param(im,vm,4.4,'TECU'))
        #ylabel('Epicentral Distance (km)')
        xlabel('Time (UT, hr)')
        xlim(6.4, 7.8)
        title("PRN 20")
        
    elif "G13" in data:
        #fig=figure(100,figsize=(10,8),facecolor='w',edgecolor='k')
        ax=subplot(143)
        #vm=0.07
       # plot(t_gnss,ep_dist+250*dtec_gnss,lw=2)
        im=scatter(t_gnss,ep_dist+(150*dtec_gnss),c=dtec_gnss,s=20*abs(dtec_gnss),
                    vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow)
        
        # if ep_dist < 1500:
    
        #     text(3.25,ep_dist,f_n[22:29],size='small',color='k')
    
        axvline(t_ep)
        # v1=(3./111)/(1./3600.)
        # axline((t_ep+8/60,0),slope=v1,c='g' ,label = '~1 km/s')
        #axis((2.7, 3.4,0.,1500))
       # f=  (axis_param(im,vm,4.4,'TECU'))
        #ylabel('Epicentral Distance (km)')
        xlabel('Time (UT, hr)')
        xlim(6.4, 7.8)
        title("PRN 13")
        
    elif "G11" in data and tec_gnss.max()<=50:
        #fig=figure(100,figsize=(10,8),facecolor='w',edgecolor='k')
        ax=subplot(144)
        #vm=0.07
       # plot(t_gnss,ep_dist+250*dtec_gnss,lw=2)
        im=scatter(t_gnss,ep_dist+(150*dtec_gnss),c=dtec_gnss,s=20*abs(dtec_gnss),
                    vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow)
        
        # if ep_dist < 1500:
    
        #     text(3.25,ep_dist,f_n[22:29],size='small',color='k')
    
        axvline(t_ep)
        # v1=(3./111)/(1./3600.)
        # axline((t_ep+8/60,0),slope=v1,c='g' ,label = '~1 km/s')
        #axis((2.7, 3.4,0.,1500))
       # f=  (axis_param(im,vm,4.4,'TECU'))
        #ylabel('Epicentral Distance (km)')
        xlabel('Time (UT, hr)')
        xlim(6.4, 7.8)
        title("PRN 11")
        
        
    #fig=figure(101,figsize=(10,8),facecolor='w',edgecolor='k')
    #if "G23" in data and tec_gnss.max()<=50:
        
       #  ax=subplot(111)
       #  #vm=0.07
       # # plot(t_gnss,ep_dist+250*dtec_gnss,lw=2)
       #  scatter(lon_gnss, lat_gnss)
       #  plot(lon_ep, lat_ep, marker='*', c='r')
        
       #  # if ep_dist < 1500:
    
       #  #     text(3.25,ep_dist,f_n[22:29],size='small',color='k')
    
       #  #axvline(t_ep)
       #  # v1=(3./111)/(1./3600.)
       #  # axline((t_ep+8/60,0),slope=v1,c='g' ,label = '~1 km/s')
       #  #axis((2.7, 3.4,0.,1500))
       # # f=  (axis_param(im,vm,4.4,'TECU'))
       #  #ylabel('Epicentral Distance (km)')
       #  # xlabel('Time (UT, hr)')
       #  # xlim(6.4, 7.8)
       #  title("PRN 23")
        
        

    
    # Check condition
    if data[8:15] in ['cmpnG23', 'baveG23', 'jrgnG23', 'cmpnG20', 'baveG20', 'jrgnG20',
                      'cmpnG13', 'baveG13', 'jrgnG13', 'cmpnG11', 'baveG11', 'jrgnG11']:
    #if "G23" in data and tec_gnss.max() <= 50:
    
        # Scatter GNSS data points
        ax_c.scatter(lon_gnss, lat_gnss, color='magenta', s=2, label='GNSS', transform=ccrs.PlateCarree())
    
        # Plot epicenter location
        ax_c.plot(lon_ep, lat_ep, marker='*', color='red', markersize=20, label='Epicenter', transform=ccrs.PlateCarree())
        
        # if data[8:15] in ['cmpnG23', 'baveG23', 'jrgnG23']:
        ax_c.text(lon_gnss[-1], lat_gnss[-1], data[8:15], fontsize=6, transform=ccrs.PlateCarree())
        
        figure(103, figsize=(12,5),facecolor='w',edgecolor='k')
        for j in range(4):
            if j==0 and "G23" in data[8:15]:
                subplot(141)
                # Scatter GNSS data points
                scatter(lon_gnss, lat_gnss, color='magenta', s=2, label='GNSS')
            
                # Plot epicenter location
                plot(-72.898, -36.122, marker='*', color='red', markersize=15)
                
                # if data[8:15] in ['cmpnG23', 'baveG23', 'jrgnG23']:
                text(lon_gnss.mean(), lat_gnss.mean(), data[8:15], fontsize=8)
                xlabel("Longitude ($^{0}$)")
                ylabel("Latitude ($^{0}$)")
                title("PRN 23")
                
                
                
                
                
            elif j==1 and "G20" in data[8:15]:
                
                #figure(103)
                subplot(142)
                # Scatter GNSS data points
                scatter(lon_gnss, lat_gnss, color='magenta', s=2, label='GNSS')
            
                # Plot epicenter location
                plot(-72.898, -36.122, marker='*', color='red', markersize=15)
                
                # if data[8:15] in ['cmpnG23', 'baveG23', 'jrgnG23']:
                text(lon_gnss.mean(), lat_gnss.mean(), data[8:15], fontsize=8)
                xlabel("Longitude ($^{0}$)")
                title("PRN 20")
                
            elif j==2 and "G13" in data[8:15]:
                
                #figure(103)
                subplot(143)
                # Scatter GNSS data points
                scatter(lon_gnss, lat_gnss, color='magenta', s=2, label='GNSS')
            
                # Plot epicenter location
                plot(-72.898, -36.122, marker='*', color='red', markersize=15)
                
                # if data[8:15] in ['cmpnG23', 'baveG23', 'jrgnG23']:
                text(lon_gnss.mean(), lat_gnss.mean(), data[8:15], fontsize=8)
                xlabel("Longitude ($^{0}$)")
                title("PRN 13")
                
            elif j==3 and "G11" in data[8:15]:
                
                #figure(103)
                subplot(144)
                # Scatter GNSS data points
                scatter(lon_gnss, lat_gnss, color='magenta', s=2, label='GNSS')
            
                # Plot epicenter location
                plot(-72.898, -36.122, marker='*', color='red', markersize=15)
                
                # if data[8:15] in ['cmpnG23', 'baveG23', 'jrgnG23']:
                text(lon_gnss.mean(), lat_gnss.mean(), data[8:15], fontsize=8)
                xlabel("Longitude ($^{0}$)")
                #ylabel("Latitude ($^{0}$)")
                title("PRN 11")
                
                
                
                
            #tight_layout()
            
            

#lat_ep=-36.122 ;lon_ep=-72.898
tight_layout()




#%%

# Create a figure with a Cartopy GeoAxes
#fig = plt.figure(30, figsize=(15, 8))
#ax1 = plt.axes(projection=ccrs.PlateCarree())
#ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

figure(30)
# Plot receiver locations
for i in range(len(recei_lat)):
    plot(recei_lon[i], recei_lat[i], 'o', markersize=4, color='b', transform=ccrs.PlateCarree())

# # Optionally add map features
# ax1.coastlines()
# ax1.gridlines(draw_labels=True)
#plt.title("PGV Contour Map – Maule, Chile Earthquake (2010)")
# savefig("/home/oluwasegun-adebayo/Desktop/chile/simulationonthewaverayleighinducedcoseismicionospher/Pub Figures/obs5.png",
#         bbox_inches = 'tight')
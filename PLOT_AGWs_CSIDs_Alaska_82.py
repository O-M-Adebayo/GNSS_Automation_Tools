
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
from adebayo_codigo import calculate_distance, band_pass_filter, find_nearest_value, find_peaks_within_range_k, data_smooth_mike
import glob
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
        n_smooth=int(ceil(6./(dt*60.)))
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

lat_ep = 55.3635
lon_ep = -157.8876
year,month,day=2021, 7, 29
hour, minute, second=6, 15, 49
t_ep=hour+minute/60+second/3600


#%%

f = np.loadtxt("./igrfgridData.csv", skiprows=13, delimiter=',')
years = f[:,0]; lat_igr = f[:,1]; lon_igr = f[:,2]; incl = f[:,4]




#------------------
# Create the plot
fig = plt.figure(30, figsize=(12, 10))
ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
#ax.stock_img()
# Set the extent of the map (latitude and longitude range)
ax1.set_extent([-95.08, -70.84, 27.35, 5.95])
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
sism_list_1=sorted(glob.glob('./alaska_82/sismo/processed/*.npy'));#0937# 0261#0205
for idx, sism in enumerate(sism_list_1):  
    data_sismo = load(sism, allow_pickle=True).item()
    t_sismo = data_sismo['t_sismo']; vel_sismo = 13*data_sismo['vel_sismo'];
    ep_dist_sismo = data_sismo['ep_dist_sismo'];
    sta_name_sismo = data_sismo['sta_name_sismo'];
    lon_sismo = data_sismo['lon_sismo']; lat_sismo = data_sismo['lat_sismo']
    
    # Calculating teleseismic distance
    [d1, d2] = calculate_distance(lon_ep, lat_ep, lon_sismo, lat_sismo)
    #dist_sismo=((d1+d2)[0]/2)/1.
    dist_sismo = (d1 + d2) / 2
    
    samp_rate = 1/((t_sismo[1]-t_sismo[0])*3600)
    
    #print("Sampling rate: ", samp_rate)
    
    if dist_sismo <= 2000:
        figure(1, figsize=(6, 10))
        t_sismo_15s = t_sismo[::10]; vel_sismo_15s = vel_sismo[::10]
        
        #plot(t_sismo_15s, dist_sismo + (1000*vel_sismo_15s), color='grey')
        plot(t_sismo, dist_sismo+(100*vel_sismo))
        #print(max(abs(vel_sismo_15s)))
        axis((6.1, 7.0, 0, 1500))
        
        vel_sismo_abs = abs(vel_sismo_15s)
        #pgv_idx = find_peaks_within_range_k(vel_sismo_abs, 200, 850, 0.00001, 5)
        
        
        figure(3)
        plot(t_sismo_15s, (lat_ep - lat_sismo) + (vel_sismo_15s))
        
        figure(4)
        
        vlines(dist_sismo, ymin=0, ymax=vel_sismo_abs.max())
        
        # bar(dist_sismo+vel_sismo_abs[pgv_idx], vel_sismo_abs[pgv_idx], 
        #     color='grey', width = 10)
        # xlim(0, 2000)
        # grid(ls = '-.', lw=0.4)
    
    
    
    if dist_sismo <= 2000:
        figure(2, figsize=(6, 10))
        vel_sismo_filt = band_pass_filter(vel_sismo, t_sismo, lowcut=0.003, highcut=0.02, dt=1)
        t_sismo_filt_15s = t_sismo[::200]; vel_sismo_filt_15s = vel_sismo_filt[::200]
        
        vel_sismo_filt_15s_abs = abs(vel_sismo_filt_15s)
        
        plot(t_sismo, dist_sismo + (100*vel_sismo_filt), color='grey')
        axis((6.1, 7.0, 0, 1500))
        
        figure(5, figsize=(8,6))
        subplot(211)
        # pgv_idx_filt = find_peaks_within_range_k(vel_sismo_filt_15s_abs, 200, 850, 0.001, 1)
        # bar(dist_sismo+vel_sismo_filt_15s_abs[pgv_idx_filt], vel_sismo_filt_15s_abs[pgv_idx_filt], 
        #     color='grey', width = 10)
        if (dist_sismo > 1000 and vel_sismo_filt_15s_abs.max() > 0.10) or (dist_sismo < 500 and vel_sismo_filt_15s_abs.max() > 0.15):
            continue
            
        vlines(dist_sismo, ymin=0, ymax=vel_sismo_filt_15s_abs.max(), color='g', lw=2)
        axis((0,1500, 0,0.15))
        grid(ls = '-.', lw=0.4)
        #xlabel("Epicentral Distance (km)")
        ylabel("PGV (m/s)")
        tight_layout()
#%%

# # --- Load All Seismometer Data ---
# sism_list = sorted(glob.glob('./honduras/sismo/processed/*.npy'))

# stations = []

# for sism in sism_list:  
#     data_sismo = load(sism, allow_pickle=True).item()
    
#     # fig = plt.figure(30, figsize=(12, 10))
#     # ax1.plot(data_sismo['lon_sismo'], data_sismo['lat_sismo'], 
#     #          marker= "^", markersize=5, color='grey', 
#     #          transform=ccrs.PlateCarree())
    
#     # Calculating teleseismic distance
#     [d1, d2] = calculate_distance(lon_ep, lat_ep, data_sismo['lon_sismo'], data_sismo['lat_sismo'])
#     #dist_sismo=((d1+d2)[0]/2)/1.
#     dist_sismo = (d1 + d2) / 2
    
#     station = {
#         't': data_sismo['t_sismo'],
#         'v': data_sismo['vel_sismo'],
#         'name': data_sismo['sta_name_sismo'],
#         'lon': data_sismo['lon_sismo'],
#         'lat': data_sismo['lat_sismo'],
#         'ep_dist_s': dist_sismo
#     }
#     stations.append(station)

#     # fig=figure(1,figsize=(9,12))
#     # ax=subplot(611)
#     # ax.text(data_sismo['lon_sismo'], data_sismo['lat_sismo'], data_sismo['sta_name_sismo'])


# # --- Determine Global Max Amplitude for Scaling ---
# all_amplitudes = np.concatenate([np.abs(sta['v']) for sta in stations])
# max_amp = np.max(all_amplitudes)

# # --- Plot ---
# #fig, ax = plt.subplots(1, figsize=(12, 10))



# scale_amp = 0.03  # vertical exaggeration
# scale_time = 3  # horizontal scaling

# for sta in stations:
#     t = sta['t']
#     v = sta['v']
#     lon = sta['lon']
#     lat = sta['lat']
#     dis = sta['ep_dist_s']
#     sta_name = sta['name']
    
#     # Normalize and scale
#     v_scaled = (v / max_amp) * scale_amp
#     t_scaled = (t - np.mean(t)) * scale_time
    
#     x = (lon-lon_ep) + t_scaled
#     y = (lat-lat_ep) + (150*v_scaled)
    
#     if dis <= 2000:
#         fig=figure(1,figsize=(9,12))
#         ax=subplot(411)
#         if sta_name == 'SJG.':
#             continue
#         plot(x, y, color='black', linewidth=0.8)
#         #ax.plot(lon, lat, 'ro', markersize=2)
#         plot(0, 0, "*", markersize = 15, color='k')
#         #text(x[0], y[0], sta_name)
#         grid(ls = '-.', lw=0.4)
#         axis((-10, 10, -15, 15))
#         xlabel('Ep. Lon. ($^{o}$)')
#         ylabel('Ep. Lat. ($^{o}$)')
        
#         fig=figure(7,figsize=(8,7))
#         ax=subplot(211)
#         if sta_name == 'SJG.':
#             continue
#         plot(x, y, color='black', linewidth=0.8)
#         #ax.plot(lon, lat, 'ro', markersize=2)
#         plot(0, 0, "*", markersize = 15, color='k')
#         #text(x[0], y[0], sta_name)
#         grid(ls = '-.', lw=0.4)
#         axis((-10, 10, -15, 15))
#         #xlabel('Ep. Lon. ($^{o}$)')
#         ylabel('Ep. Lat. ($^{o}$)')
        
        
        
        
#%%

f_list=sorted(glob.glob('./alaska_82/tec/*.npy'));#0937# 0261#0205
for idx, f_n in enumerate(f_list):  
    
    data_gnss=load(f_n,allow_pickle=True).item()
    # if idx ==0:
    #     print(data_gnss.keys())
    t_gnss=data_gnss['t_gnss'];tec_gnss=data_gnss['tec_gnss'];
    vtec_gnss = data_gnss['vtec_gnss']; ep_dist = data_gnss['ep_dist'];
    lon_gnss=data_gnss['lon_gnss'];lat_gnss=data_gnss['lat_gnss'];
    lon_rec = data_gnss['lon_rec']; lat_rec = data_gnss['lat_rec']
    elev_ang = data_gnss['elev_ang']; azim_ang = data_gnss['azim_ang']
    
    v_prox = find_nearest_value(t_gnss, t_ep)
    p_sip = where(t_gnss == v_prox)[0]
    
    dtec_gnss=data_smooth(t_gnss,tec_gnss)
    
    if "G04" in f_n:
        figure(5)
        subplot(212)
        bar(ep_dist, dtec_gnss[150:200].max(), color='r', width = 10)
        axis((0, 1500, 0, 1.0))
        grid(ls = '-.', lw=0.4)
        xlabel("Epicentral Distance (km)")
        ylabel("dTEC (TECU)")
    
    figure(6, figsize=(10,15))
    vm = 0.08
    if "G04" in f_n:
        #dtec_gnss=data_smooth(t_gnss,tec_gnss)
        subplot(221)
        
        im=scatter(t_gnss,ep_dist+(dtec_gnss),c=0.6*dtec_gnss,s=20,
                    vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow)
        axvline(t_ep)
        #text(6.9, ep_dist, f_n[16:23])
        axis((6.1, 7.0, 0, 1500))
        
    
    #figure(7, figsize=(10,12))
    
    if "G06" in f_n:
        vm = 0.08
        #dtec_gnss=data_smooth(t_gnss,tec_gnss)
        subplot(222)
        
        im=scatter(t_gnss,ep_dist+dtec_gnss,c=0.6*dtec_gnss,s=20,
                    vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow)
        axvline(t_ep)
        #text(6.9, ep_dist, f_n[16:23])
        axis((6.1, 7.0, 0, 1500))
        
        v1 = (3.5/1)/(1/3600)
        axline((t_ep + 500/3600, 0), slope=v1, lw=0.5, c='k', label='3.5 km/s')
        
        v2 = (2.2/1)/(1/3600)
        axline((t_ep + 700/3600, 0), slope=v2, lw=0.5, c='k', label='2.4 km/s')
        
        v3 = (1.5/1)/(1/3600)
        axline((t_ep + 1200/3600, 0), slope=v3, lw=0.5, c='k', label='1.5 km/s')
        
        
        
        if ep_dist > 800:
            subplot(221)
            
            im=scatter(t_gnss,ep_dist+(dtec_gnss),c=0.6*dtec_gnss,s=20,
                        vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow)
            
            v1 = (3.5/1)/(1/3600)
            axline((t_ep + 500/3600, 0), slope=v1, lw=0.5, c='k', label='3.0 km/s')
            
            v2 = (2.2/1)/(1/3600)
            axline((t_ep + 700/3600, 0), slope=v2, lw=0.5, c='k', label='3.0 km/s')
            
            if len(dtec_gnss) > 200:
                figure(5)
                subplot(212)
                
                
                if ep_dist > 1250 and dtec_gnss.max() > 1.0:
                    continue
                #dtec_gnss[dtec_gnss > 1.0] = 0.2
                bar(ep_dist, dtec_gnss[150:220].max(), color='r', width = 10)
                axis((0, 1500, 0, 1.0))
                grid(ls = '-.', lw=0.4)
                # xlabel("Epicentral Distance (km)")
                # ylabel("dTEC (TECU)")
        tight_layout()
    #figure(8, figsize=(10,12))
    vm = 0.08
    if "G09" in f_n:
        dtec_gnss=data_smooth_mike(t_gnss, tec_gnss, 4.5, 5)
        subplot(223)
        
        im=scatter(t_gnss,ep_dist+dtec_gnss,c=0.6*dtec_gnss,s=20,
                    vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow)
        axvline(t_ep)
        #text(6.9, ep_dist, f_n[16:23])
        axis((6.1, 7.0, 0, 1500))
    

    
        
        
    #figure(10, figsize=(10,12))
    vm = 0.08
    if "G16" in f_n:
        #dtec_gnss=data_smooth(t_gnss,tec_gnss)
        subplot(224)
        
        im=scatter(t_gnss,ep_dist+dtec_gnss,c=0.6*dtec_gnss,s=20,
                    vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow)
        axvline(t_ep)
        axis((6.1, 7.0, 0, 1500))
        
    
        
#%%        
    figure(9, figsize=(10,12))
    vm = 0.04
    if "G07" in f_n:
        dtec_gnss=data_smooth_mike(t_gnss, tec_gnss, 5, 5)
        subplot(111)
        
        im=scatter(t_gnss,ep_dist+dtec_gnss,c=0.6*dtec_gnss,s=20,
                    vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow)
        axvline(t_ep)
        axis((6.1, 7.0, 0, 800))

tight_layout()

#%%    
    
    
#%%

# f_list=sorted(glob.glob('./honduras/tec/*.npy'));#0937# 0261#0205
# for idx, f_n in enumerate(f_list):  
    
#     data_gnss=load(f_n,allow_pickle=True).item()
#     # if idx ==0:
#     #     print(data_gnss.keys())
#     t_gnss=data_gnss['t_gnss'];tec_gnss=data_gnss['tec_gnss'];
#     vtec_gnss = data_gnss['vtec_gnss']; ep_dist = data_gnss['ep_dist'];
#     lon_gnss=data_gnss['lon_gnss'];lat_gnss=data_gnss['lat_gnss'];
#     lon_rec = data_gnss['lon_rec']; lat_rec = data_gnss['lat_rec']
#     elev_ang = data_gnss['elev_ang']; azim_ang = data_gnss['azim_ang']
    
#     v_prox = find_nearest_value(t_gnss, t_ep)
#     p_sip = where(t_gnss == v_prox)[0]
    
    
# #     figure(3)
# #     if "G06" in f_n:
# #         dtec_gnss=data_smooth(t_gnss,tec_gnss)
# #         dtec_gnss[dtec_gnss < 0.03] = 0
# #         dtec_peaks = find_peaks_within_range_k(dtec_gnss, 79, 160, 0.055, 1)
        
# #         scatter(t_gnss[dtec_peaks], ep_dist + dtec_gnss[dtec_peaks], c=dtec_gnss[dtec_peaks])
# #         axis((min(t_gnss), max(t_gnss), 0, 1300))
# #         axvline(t_ep)
        
# #     if "G11" in f_n:
# #         dtec_gnss=data_smooth(t_gnss,tec_gnss)
# #         figure(4)
# #         dtec_gnss[dtec_gnss < 0.03] = 0
# #         dtec_peaks = find_peaks_within_range_k(dtec_gnss, 83, 160, 0.058, 1)
        
# #         scatter(t_gnss[dtec_peaks], ep_dist + dtec_gnss[dtec_peaks], c=dtec_gnss[dtec_peaks])
# #         axis((min(t_gnss), max(t_gnss), 0, 1300))
# #         axvline(t_ep)
        
# #     if "G19" in f_n:
# #         dtec_gnss=data_smooth(t_gnss,tec_gnss)
# #         figure(5)
# #         dtec_gnss[dtec_gnss < 0.01] = 0
# #         dtec_peaks = find_peaks_within_range_k(dtec_gnss, 83, 150, 0.1, 1)
        
# #         scatter(t_gnss[dtec_peaks], ep_dist + dtec_gnss[dtec_peaks], c=dtec_gnss[dtec_peaks])
# #         axis((min(t_gnss), max(t_gnss), 0, 1300))
# #         axvline(t_ep)
        
# #     if "G17" in f_n:
# #         dtec_gnss=data_smooth_mike(t_gnss,tec_gnss, 3, 5)
# #         figure(6)
# #         # dtec_17 = data_smooth_mike(t_gnss, data_gnss, 7, 5)
        
# #         # dtec_17[dtec_17 < 0.001] = 0
# #         # dtec_peaks = find_peaks_within_range_k(dtec_17, 83, 150, 0.003, 1)
        
# #         # scatter(t_gnss[dtec_peaks], ep_dist + dtec_17[dtec_peaks], c=dtec_17[dtec_peaks])
# #         # axis((min(t_gnss), max(t_gnss), 0, 2000))
# #         # axvline(t_ep)
# #         #dtec_gnss = data_smooth_mike(t, data, win1, win2)
# #         dtec_gnss[dtec_gnss < 0.05] = 0
# #         dtec_peaks = find_peaks_within_range_k(dtec_gnss, 95, 120, 0.06, 1)
        
# #         scatter(t_gnss[dtec_peaks], ep_dist + dtec_gnss[dtec_peaks], c=dtec_gnss[dtec_peaks])
# #         axis((min(t_gnss), max(t_gnss), 0, 2000))
# #         axvline(t_ep)
# # show()


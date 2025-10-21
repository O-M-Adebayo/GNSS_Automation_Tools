
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
from adebayo_codigo import calculate_distance, band_pass_filter, find_nearest_value, find_peaks_within_range_k, data_smooth_mike, plot_great_circle
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

lat_ep = 17.7016
lon_ep = -82.4564
year,month,day=2025, 2, 8
hour, minute, second=23, 23, 14
t_ep=hour+minute/60+second/3600


#%%

f = np.loadtxt("./igrfgridData.csv", skiprows=13, delimiter=',')
years = f[:,0]; lat_igr = f[:,1]; lon_igr = f[:,2]; incl = f[:,4]




#------------------
# Create the plot
fig = plt.figure(30, figsize=(8, 8))
ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
#ax.stock_img()
# Set the extent of the map (latitude and longitude range)
ax1.set_extent([-91.08, -75, 24, 5.95])
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
sism_list_1=sorted(glob.glob('./honduras/sismo/processed/*.npy'));#0937# 0261#0205
for idx, sism in enumerate(sism_list_1):  
    data_sismo = load(sism, allow_pickle=True).item()
    t_sismo = data_sismo['t_sismo']; vel_sismo = data_sismo['vel_sismo'];
    ep_dist_sismo = data_sismo['ep_dist_sismo'];
    sta_name_sismo = data_sismo['sta_name_sismo'];
    lon_sismo = data_sismo['lon_sismo']; lat_sismo = data_sismo['lat_sismo']
    
    # Calculating teleseismic distance
    [d1, d2] = calculate_distance(lon_ep, lat_ep, lon_sismo, lat_sismo)
    #dist_sismo=((d1+d2)[0]/2)/1.
    dist_sismo = (d1 + d2) / 2
    
    #hhhhhhh
    # vel_sismo = vel_sismo[::5]
    # t_sismo = t_sismo[::5]
    
    # vel_sismo_filt = band_pass_filter(vel_sismo, t_sismo, lowcut=0.002, highcut=0.010)
    
    # peak_mx = find_peaks_within_range_k(abs(vel_sismo_filt), 0, len(vel_sismo_filt), 0.05, 1)
    
    #print(lon_sismo)
    
    figure(17, figsize=(8, 8))
    ax=subplot(121)
    #scatter(lon_sismo, lat_sismo)
    #print("LENGTH------------------", len(vel_sismo))
    
    if sta_name_sismo == 'SJG.':
        continue
    
    plot(t_sismo, dist_sismo + (5000*vel_sismo), color='grey')
    axvline(t_ep)
    #ylim(0, 2000)
    #grid(ls = '-.', lw=0.4)
    xlabel("Time (UT, hr)")
    ylabel("Epicentral Distance (km)")
    axis((23.3, 23.9, 0, 2000))
    
    # # # Short-Period RWs
    # # vel_sismo_SP = band_pass_filter(vel_sismo, t_sismo, lowcut=20, highcut=50, dt=0.05)
    
    # # Long-Period RWs
    # vel_sismo_LP = band_pass_filter(vel_sismo, t_sismo, lowcut=0.004, highcut=20, dt=0.05)
    
    # figure(99)
    # subplot(121)
    # plot(t_sismo, dist_sismo + (5000*vel_sismo_LP), color='grey')
    # subplot(122)
    # plot(t_sismo, dist_sismo + (5000*vel_sismo_LP), color='grey')
    
    
    
    figure(17, figsize=(8, 8))
    ax=subplot(122)
    
    plot(t_sismo, dist_sismo + (5000*vel_sismo), color='grey')
    axvline(t_ep)
    ylim(0, 2000)
    #grid(ls = '-.', lw=0.4)
    xlim(23.3, 23.9)
    xlabel("Time (UT, hr)")
    ylabel("Epicentral Distance (km)")
    title("Seismic Data")
    
    if dist_sismo < 2000:
        text(23.8, dist_sismo, sta_name_sismo)
    
    if sta_name_sismo == 'AMTX':
        plot([23.421, 23.527], [495, 2000], 
               ls='-', lw=1, color='k', label='~3.60 km/s')
        legend(loc='lower right')
    
    vel_ds = vel_sismo[::100]
    t_ds = t_sismo[::100]
    
    if len(vel_ds) > 27:
        vel_sismo_filt = band_pass_filter(vel_ds, t_ds, lowcut=0.002, highcut=0.01, dt=10)
        # plotting code here
        figure(6, figsize=(10, 8))
        ax=subplot(121)
        
        if sta_name_sismo == 'SJG.':
            continue
        plot(t_ds, dist_sismo + (15000*vel_sismo_filt), color='g')
        #ylim(0, 2000)
        axvline(t_ep)
        #grid(ls = '-.', lw=0.4)
        xlabel("Time (UT, hr)")
        ylabel("Epicentral Distance (km)")
        axis((23.3, 23.9, 0, 2000))
        title("Seismic Data", fontweight='bold')
        
        if sta_name_sismo == 'AMTX':
            plot([23.421, 23.527], [402, 2000], 
                   ls='-', lw=1, color='k', label='~3.60 km/s')
            legend(loc='lower right')
        
        
        tight_layout()
        
        # figure(6, figsize=(10, 12))
        # subplot(122)
        
        # if sta_name_sismo == 'SJG.':
        #     continue
        # plot(t_ds, dist_sismo + (30000*vel_sismo_filt), color='grey')
        # ylim(0, 2000)
        # axvline(t_ep)
        # grid(ls = '-.', lw=0.4)
        
        # uuuuuuu
        vel_sismo_filt[vel_sismo_filt <= 0.0000001] = 0

        vel_p = find_peaks_within_range_k(vel_sismo_filt, 0, len(vel_sismo_filt), 0.000002, 1)
        
        # figure(19)
        # bar(dist_sismo+vel_sismo_filt[vel_p], vel_sismo_filt[vel_p], 
        #     color='g', width = 10)
        # xlim(0, 2000)
        
        figure(3, figsize=(9,12))
        subplot(611)
        if sta_name_sismo == 'SJG.':
            continue
        bar(dist_sismo+vel_sismo_filt[vel_p], vel_sismo_filt[vel_p], 
            color='grey', width = 10)
        xlim(0, 2000)
        grid(ls = '-.', lw=0.4)
        
        
        figure(8, figsize=(8,7))
        subplot(211)
        if sta_name_sismo == 'SJG.':
            continue
        bar(dist_sismo+vel_sismo_filt[vel_p], 10*vel_sismo_filt[vel_p], 
            color='g', width = 10, label='Vertical Uplift')
        #xlim(0, 2000)
        axis((0,2000, 0,0.08))
        grid(ls = '-.', lw=0.4)
        ylabel('[m/s]')
        title('Seismic Data', fontweight='bold')
        
        
    else:
        print("Filtered signal too short for band-pass filtering.")

    
    # if len(vel_sismo) > 0:
        
    #     #t_sismo = t_sismo[::100]
        
    #     #vel_sismo = vel_sismo[::100]
        
    #     vel_sismo_filt = band_pass_filter(vel_sismo[::200], t_sismo[::200], lowcut=0.002, highcut=0.01, dt=10)
        
    #     figure(17, figsize=(8, 10))
    #     ax=subplot(122)
    #     plot(t_sismo[::200], dist_sismo + (100000*vel_sismo_filt))
    #     ylim(0, 2000)
    #     axvline(t_ep)
        
        
    
    
    # figure(18)
    # x = (lon_sismo-lon_ep)+t_sismo
    # y = (lat_sismo-lat_ep) + 300*vel_sismo
    # plot(x, y)
    
    
    # fig = plt.figure(30, figsize=(12, 10))
    # ax=subplot(111)
    # ax.plot(lon_sismo, lat_sismo, "^", markersize=6, color='green', transform=ccrs.PlateCarree())
    # savefig("/home/oluwasegun-adebayo/Desktop/chile/simulationonthewaverayleighinducedcoseismicionospher/Pub Figures/obs.png",
    #         bbox_inches = 'tight')
    
    
    
    
    
    # Filter data
    # if len(vel_sismo) > 0:
        
    #     vel_sismo = vel_sismo[::100]
    #     t_sismo = t_sismo[::100]
        
    #     vel_sismo_filt = band_pass_filter(vel_sismo, t_sismo, lowcut=0.002, highcut=0.010, dt = 1)
        
    #     # fig=figure(6,figsize=(10,8),facecolor='w',edgecolor='k')
    #     # ax=subplot(121)
        
        # plot(t_sismo, dist_sismo + (3000*vel_sismo_filt), color='g')
    
    
        # vel_sismo_filt[vel_sismo_filt <= 0.000001] = 0
    
        # vel_p = find_peaks_within_range_k(vel_sismo, 0, len(vel_sismo), 0.00002, 5)
        #find_peaks_within_range_k(arr, start_idx, end_idx, threshold, k)
        
        
    
    #     fig=figure(91,figsize=(8,6),facecolor='w',edgecolor='k')
    #     ax=subplot(212)
        
        
    #     # scatter(ep_dist+150*dtec_gnss[dtec_peak_obs], dtec_gnss[dtec_peak_obs], 
    #     #         s=50*dtec_gnss[dtec_peak_obs], c=(t_gnss[dtec_peak_obs]-t_ep)*60)
    #     bar(dist_sismo+vel_sismo_filt[vel_p], vel_sismo_filt[vel_p], 
    #         color='g', width = 10)
    # suptitle("Amplitude Distribution of the Filtered data: 2-10 mHz")
    tight_layout()    
show()
    

#%%

# --- Load All Seismometer Data ---
sism_list = sorted(glob.glob('./honduras/sismo/processed/*.npy'))

stations = []

for sism in sism_list:  
    data_sismo = load(sism, allow_pickle=True).item()
    
    # fig = plt.figure(30, figsize=(12, 10))
    # ax1.plot(data_sismo['lon_sismo'], data_sismo['lat_sismo'], 
    #          marker= "^", markersize=5, color='grey', 
    #          transform=ccrs.PlateCarree())
    
    ax1.arrow(-87.456, 22.702, 10, -10, head_width=0.2, 
              head_length=0.3, length_includes_head = True, fc='red', ec='red')
    
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

    # fig=figure(1,figsize=(9,12))
    # ax=subplot(611)
    # ax.text(data_sismo['lon_sismo'], data_sismo['lat_sismo'], data_sismo['sta_name_sismo'])


# --- Determine Global Max Amplitude for Scaling ---
all_amplitudes = np.concatenate([np.abs(sta['v']) for sta in stations])
max_amp = np.max(all_amplitudes)

# --- Plot ---
fig=figure(77,figsize=(10,6))
ax=subplot(111)

# Plot great circles with intervals
radius_km = 2000  # Maximum radius for circles (e.g., 2000 km)
plot_great_circle(ax, lon_ep, lat_ep, radius_km, side='left', interval_km=200)

scale_amp = 0.01  # vertical exaggeration
scale_time = 2  # horizontal scaling

for sta in stations:
    t = sta['t']
    v = sta['v']
    lon = sta['lon']
    lat = sta['lat']
    dis = sta['ep_dist_s']
    sta_name = sta['name']
    
    # Normalize and scale
    v_scaled = (v / max_amp) * scale_amp
    t_scaled = (t - np.mean(t)) * scale_time
    
    x = lon + t_scaled
    y = lat + (200 * v_scaled)
    
    # x = (lon-lon_ep) + t_scaled
    # y = (lat-lat_ep) + (200*v_scaled)
    
    if dis <= 2000:
        fig=figure(1,figsize=(9,12))
        ax=subplot(411)
        if sta_name == 'SJG.':
            continue
        plot(x, y, color='black', linewidth=0.8)
        #ax.plot(lon, lat, 'ro', markersize=2)
        #plot(0, 0, "*", markersize = 15, color='k')
        #text(x[0], y[0], sta_name)
        grid(ls = '-.', lw=0.4)
        #axis((-10, 10, -15, 15))
        xlabel('Ep. Lon. ($^{o}$)')
        ylabel('Ep. Lat. ($^{o}$)')
        
        fig=figure(77,figsize=(10,6))
        ax=subplot(111)
        if sta_name == 'SJG.':
            continue
        plot(x, y, color='g', linewidth=0.8)
        #ax.plot(lon, lat, 'ro', markersize=2)
        plot(lon_ep, lat_ep, "*", markersize = 15, color='r')
        if (x[0] > -10 and x[0] < 10) and (y[0] > -15 and y[0] < 15):
        
            text(x[0], y[0]+0.5, sta_name, fontsize=6)
        grid(ls = '-.', lw=0.4)
        axis((-103, -70, 8, 23))
        xlabel('GLON ($^{o}$)')
        ylabel('GLAT ($^{o}$)')
        title("Seismic Data", fontweight='bold')
        tight_layout()
        
circle1 = plt.Circle((-87.41, 14.10), 1.7, fill=False, color='r')
circle2 = plt.Circle((-99.11, 19.31), 1.2, fill=False, color='r')
ax.add_patch(circle1)
ax.add_patch(circle2)
        
#%%

f_list=sorted(glob.glob('./honduras/tec/*.npy'));#0937# 0261#0205
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
    
    # figure(6, figsize=(10,12))
    # vm = 0.1
    # if "G06" in f_n or "G17" in f_n:
    #     dtec_gnss=data_smooth(t_gnss,tec_gnss)
    #     subplot(111)
        
    #     im=scatter(t_gnss,ep_dist+dtec_gnss,c=0.6*dtec_gnss,s=20,
    #                 vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow)
    #     axvline(t_ep)
    
    #f_n[15:22] == 'cn35G19':
    
    figure(9, figsize=(6,8))
    if "G19" in f_n:
        subplot(311)
        plot(t_gnss, elev_ang)
        axvline(t_ep)
        xlabel("Time (UT, hr)")
        ylabel("Elevation\nangle ($^{0}$)")
        #text(23.8, 42, f_n[15:22])
        if f_n[15:22] == 'cn35G19':
            text(23.8, 42, f_n[19:22])
        
        
    elif "G06" in f_n:
        subplot(312)
        plot(t_gnss, elev_ang)
        axvline(t_ep)
        xlabel("Time (UT, hr)")
        ylabel("Elevation\nangle ($^{0}$)")
        if f_n[15:22] == 'cn20G06':
            text(23.8, 42, f_n[19:22])
        
    elif "G17" in f_n:
        subplot(313)
        plot(t_gnss, elev_ang)
        axvline(t_ep)
        xlabel("Time (UT, hr)")
        ylabel("Elevation\nangle ($^{0}$)")
        if f_n[15:22] == 'cn23G17':
            text(23.8, 42, f_n[19:22])
    
            
    if "G19" in f_n or "G06" in f_n:
    #if "G19" in f_n and ep_dist < 2000:# or "G17" in f_n or "G06" in f_n or "G12" in f_n or 'G11':
            
        fig = plt.figure(30, figsize=(8, 8))
        # ax1.text(-55, -8, 'Brazil', fontweight='bold', fontsize=12)
        # ax1.text(-73.2, -41, 'Chile', fontweight='bold', fontsize=8,
        #        rotation = 85)
        
        

        # Seismometers
        #ax.plot(-79, 0.24, "^", markersize=8, color='green')   # OTAV
        # # Plot specific points
        ax1.plot(lon_ep, lat_ep, "*", markersize=20, color='red')   # Chile
        
        if (idx%2) == 0:
            
            ax1.plot(lon_gnss, lat_gnss, '.', markersize=1, color='magenta',
                    transform=ccrs.PlateCarree())
            ax1.plot(lon_gnss[p_sip], lat_gnss[p_sip], 'D', markersize=3, color='cyan',
                    transform=ccrs.PlateCarree())
            ax1.text(lon_gnss[0], lat_gnss[0], s = f_n[19:22], fontsize = 8,
                    transform=ccrs.PlateCarree())
            
        # Plot receiver locations
        ax1.plot(lon_rec, lat_rec, 'o', markersize=5, color='blue', transform=ccrs.PlateCarree())
    tight_layout()
    
    figure(2, figsize=(8,12))
    vm = 0.1
    if "G06" in f_n:
        dtec_gnss=data_smooth(t_gnss,tec_gnss)
        ax=subplot(221)
        
        im=scatter(t_gnss,ep_dist+(70*dtec_gnss),c=0.6*dtec_gnss,s=20,
                    vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow)
        axvline(t_ep)
        ylim(0, 2000)
        #xlabel('Time (UT, hr)')
        ylabel('Epicentral Distance (km)')
        
        
        if f_n[15:22] == 'cn55G06':
            text(23.2, 1900, f_n[19:22])
            
            axline(xy1=(23.652, 352), xy2=(23.856, 1282), 
                   ls='--', lw=1, color='k', label='~1.0 km/s')
            axline(xy1=(23.584, 604), xy2=(23.658, 1279), 
                   ls='-', lw=1, color='k', label='~2.72 km/s')
            legend(loc='lower right')
            
            #f = axis_param(im, 'TECU')
        
        
    
    if "G11" in f_n:
        dtec_gnss=data_smooth(t_gnss,tec_gnss)
        figure(2, figsize=(10,12))
        vm = 0.1
        ax=subplot(222)
        im=scatter(t_gnss,ep_dist+(70*dtec_gnss),c=0.6*dtec_gnss,s=20,
                    vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow)
        
        axvline(t_ep)
        ylim(0, 2000)
        
        if f_n[15:22] == 'cn55G11':
            text(23.2, 1900, f_n[19:22])
            axline(xy1=(23.575, 452), xy2=(23.678, 1279), 
                   ls='-', lw=1, color='k', label='~2.70 km/s')
            legend(loc='lower right')
            
            f = axis_param(im, 'TECU')
        
        
    if "G19" in f_n:
        dtec_gnss=data_smooth(t_gnss,tec_gnss)
        figure(2, figsize=(10,12))
        vm = 0.1
        ax=subplot(223)
        im=scatter(t_gnss,ep_dist+(70*dtec_gnss),c=0.6*dtec_gnss,s=20,
                    vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow)
        axvline(t_ep)
        ylim(0, 2000)
        xlabel('Time (UT, hr)')
        ylabel('Epicentral Distance (km)')
        
        if f_n[15:22] == 'cn35G19':
            text(23.2, 1900, f_n[19:22])
            
            axline(xy1=(23.516, 352), xy2=(23.731, 1230), 
                   ls='--', lw=1, color='k', label='~1.0 km/s')
            legend(loc='lower right')
            
            #f = axis_param(im, 'TECU')
        
        
        figure(6, figsize=(10,10))
        vm = 0.15
        ax=subplot(122)
        
        im=scatter(t_gnss,ep_dist+(1*dtec_gnss),c=0.6*dtec_gnss,s=20,
                    vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow)
        axvline(t_ep)
        #text(23.82, ep_dist, f_n[15:22], fontsize=7)
        axis((23.3, 23.9, 0, 2000))
        
        if f_n[15:22] == 'cn35G19':
            plot([23.516, 23.731], [352, 1168], 
                   ls='--', lw=1, color='k', label='~1.0 km/s')
            legend(loc='lower right')
            
            f=  axis_param(im,'TECU') 
            
        
    if "G17" in f_n:
        dtec_gnss=data_smooth_mike(t_gnss,tec_gnss, 4, 5)
        figure(2, figsize=(10,12))
        vm = 0.1
        ax=subplot(224)
        im=scatter(t_gnss,ep_dist+(70*dtec_gnss),c=0.6*dtec_gnss,s=20,
                    vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow)
        axvline(t_ep)
        ylim(0, 2000)
        xlabel('Time (UT, hr)')
        #ylabel('Epicentral Distance (km)')
        
        
        if f_n[15:22] == 'cn55G17':
            text(23.2, 1900, f_n[19:22])
            axline(xy1=(23.475, 425), xy2=(23.750, 1300), 
                   ls='--', lw=1, color='k', label='~1.0 km/s')
            axline(xy1=(23.566, 411), xy2=(23.637, 2000), 
                   ls='-', lw=1, color='k', label='~3.50 km/s')
            legend(loc='lower right')
            
            f = axis_param(im, 'TECU')
        
        tight_layout()
        
        figure(6, figsize=(10,8))
        vm = 0.10
        subplot(122)
        
        if f_n[15:22] in ['cn13G17', 'cn60G17', 'cn55G17', 'cn15G17', 'jme2G17']:
            continue
        
        #dtec_gnss=data_smooth_mike(t_gnss,tec_gnss, 2, 5)
        im=scatter(t_gnss,ep_dist+(1*dtec_gnss),c=0.6*dtec_gnss,s=20,
                    vmax=vm,vmin=-vm,marker='o',cmap=cm.rainbow)
        axvline(t_ep)
        xlabel("Time (UT, hr)")
        title("TEC Data", fontweight='bold')
        #text(23.82, ep_dist, f_n[15:22], fontsize=7)
        
        if f_n[15:22] == 'alefG17':
            #text(23.2, 1900, f_n[19:22])
            plot([23.592, 23.637], [976, 2000], ls='-', lw=1, color='k', label='~3.50 km/s')
            legend(loc='lower right')
        
        
        
        tight_layout()
        
        
        
        
#%%


    
    vm = 0.2
    if "G06" in f_n:
        
        dtec_gnss=data_smooth(t_gnss,tec_gnss)
        
        fig=figure(1,figsize=(9,12))
        ax=subplot(412)
        # dtec_gnss[dtec_gnss < 0.01] = 0
        # dtec_peaks = find_peaks_within_range_k(dtec_gnss, 60, 170, 0.05, 1)
        
        dtec_gnss[dtec_gnss < 0.03] = 0
        dtec_peaks = find_peaks_within_range_k(dtec_gnss, 79, 160, 0.055, 1)
        
        im = scatter((lon_gnss[dtec_peaks]-lon_ep),(lat_gnss[dtec_peaks]-lat_ep)+5*dtec_gnss[dtec_peaks],c=0.6*dtec_gnss[dtec_peaks],s=20,
                    vmax=vm,vmin=-vm,marker='o',cmap=cm.jet)
        plot(0, 0, "*", markersize = 15, color='k')
        grid(ls = '-.', lw=0.4)
        axis((-10, 10, -15, 15))
        
        
        if 'acho' in f_n:
            f=  axis_param(im,'TECU') 
            
        figure(3, figsize=(9,12))
        subplot(612)
        bar(ep_dist+dtec_gnss[dtec_peaks], dtec_gnss[dtec_peaks], 
            color='g', width = 10)
        axis((0, 2000, 0, 2))
        grid(ls = '-.', lw=0.4)
        
            
    elif "G11" in f_n:
        
        dtec_gnss=data_smooth(t_gnss,tec_gnss)
        
        # fig=figure(1,figsize=(9,12))
        # ax=subplot(613)
        # # dtec_gnss[dtec_gnss < 0.01] = 0
        # # dtec_peaks = find_peaks_within_range_k(dtec_gnss, 60, 170, 0.05, 1)
        
        # dtec_gnss[dtec_gnss < 0.03] = 0
        # dtec_peaks = find_peaks_within_range_k(dtec_gnss, 83, 160, 0.058, 1)
        
        # # dtec_gnss[dtec_gnss < 0.03] = 0
        # # dtec_peaks = find_peaks_within_range_k(dtec_gnss, 79, 160, 0.055, 1)
        
        # im = scatter((lon_gnss[dtec_peaks]-lon_ep),(lat_gnss[dtec_peaks]-lat_ep)+5*dtec_gnss[dtec_peaks],c=0.6*dtec_gnss[dtec_peaks],s=20,
        #             vmax=vm,vmin=-vm,marker='o',cmap=cm.jet)
        # plot(0, 0, "*", markersize = 15, color='k')
        # grid(ls = '-.', lw=0.4)
        # axis((-10, 10, -15, 15))
        
        # if 'cn20' in f_n:
        #     f=  axis_param(im,'TECU') 
            
            
        figure(3, figsize=(9,12))
        subplot(613)
        bar(ep_dist+dtec_gnss[dtec_peaks], dtec_gnss[dtec_peaks], 
            color='g', width = 10)
        axis((0, 2000, 0, 2))
        grid(ls = '-.', lw=0.4)
            
    elif "G19" in f_n:
        
        dtec_gnss=data_smooth(t_gnss,tec_gnss)
        
        fig=figure(1,figsize=(9,12))
        ax=subplot(413)
        # dtec_gnss[dtec_gnss < 0.01] = 0
        # dtec_peaks = find_peaks_within_range_k(dtec_gnss, 60, 170, 0.05, 1)
        
        dtec_gnss[dtec_gnss < 0.01] = 0
        dtec_peaks = find_peaks_within_range_k(dtec_gnss, 83, 150, 0.1, 1)
        
        # dtec_gnss[dtec_gnss < 0.03] = 0
        # dtec_peaks = find_peaks_within_range_k(dtec_gnss, 79, 160, 0.055, 1)
        
        im = scatter((lon_gnss[dtec_peaks]-lon_ep),(lat_gnss[dtec_peaks]-lat_ep)+5*dtec_gnss[dtec_peaks],
                     c=0.6*dtec_gnss[dtec_peaks],s=20,
                    vmax=vm,vmin=-vm,marker='o',cmap=cm.jet)
        plot(0, 0, "*", markersize = 15, color='k')
        grid(ls = '-.', lw=0.4)
        axis((-10, 10, -15, 15))
        
        if 'cn21' in f_n:
            f=  axis_param(im,'TECU') 
            
            
        fig=figure(7,figsize=(8,5))
        vm1=0.4
        ax=subplot(111)
        im = scatter((lon_gnss[dtec_peaks]-lon_ep),(lat_gnss[dtec_peaks]-lat_ep)+5*dtec_gnss[dtec_peaks],
                     c=0.6*dtec_gnss[dtec_peaks],s=50, vmax=vm1,vmin=-vm1,marker='o',cmap=cm.jet)
        plot(0, 0, "*", markersize = 15, color='r')
        grid(ls = '-.', lw=0.4)
        #axis((-10, 10, -15, 15))
        title("TEC Data", fontweight='bold')
            
            
        figure(3, figsize=(9,12))
        subplot(614)
        bar(ep_dist+dtec_gnss[dtec_peaks], dtec_gnss[dtec_peaks], 
            color='g', width = 10)
        axis((0, 2000, 0, 2))
        grid(ls = '-.', lw=0.4)
        
        
        figure(8, figsize=(8,7))
        subplot(212)
        bar(ep_dist, dtec_gnss.max(), color='r', width = 10)
        
        # bar(ep_dist+dtec_gnss[dtec_peaks], dtec_gnss[dtec_peaks], 
        #     color='g', width = 10)
        # axis((0, 2000, 0, 1.5))
        grid(ls = '-.', lw=0.4)
        
            
    elif "G17" in f_n:
        
        dtec_gnss=data_smooth(t_gnss,tec_gnss)
        
        fig=figure(1,figsize=(9,12))
        ax=subplot(414)
        # dtec_gnss[dtec_gnss < 0.01] = 0
        # dtec_peaks = find_peaks_within_range_k(dtec_gnss, 60, 170, 0.05, 1)
        
        
        
        dtec_gnss[dtec_gnss < 0.03] = 0
        dtec_peaks = find_peaks_within_range_k(dtec_gnss, 79, 160, 0.055, 1)
        
        im = scatter((lon_gnss[dtec_peaks]-lon_ep),(lat_gnss[dtec_peaks]-lat_ep)+5*dtec_gnss[dtec_peaks],c=0.6*dtec_gnss[dtec_peaks],s=20,
                    vmax=vm,vmin=-vm,marker='o',cmap=cm.jet)
        plot(0, 0, "*", markersize = 15, color='k')
        grid(ls = '-.', lw=0.4)
        axis((-10, 10, -15, 15))
        
        if 'cn19' in f_n:
            f=  axis_param(im,'TECU') 
            
        figure(3, figsize=(9,12))
        subplot(615)
        bar(ep_dist+dtec_gnss[dtec_peaks], dtec_gnss[dtec_peaks], 
            color='g', width = 10)
        axis((0, 2000, 0, 2))
        grid(ls = '-.', lw=0.4)
            
    else:
        
        dtec_gnss=data_smooth(t_gnss,tec_gnss)
        
        # fig=figure(1,figsize=(9,12))
        # ax=subplot(616)
        # # dtec_gnss[dtec_gnss < 0.01] = 0
        # # dtec_peaks = find_peaks_within_range_k(dtec_gnss, 60, 170, 0.05, 1)
        
        # dtec_gnss[dtec_gnss < 0.03] = 0
        # dtec_peaks = find_peaks_within_range_k(dtec_gnss, 79, 160, 0.055, 1)
        
        # im = scatter((lon_gnss[dtec_peaks]-lon_ep),(lat_gnss[dtec_peaks]-lat_ep)+5*dtec_gnss[dtec_peaks],c=0.6*dtec_gnss[dtec_peaks],s=20,
        #             vmax=vm,vmin=-vm,marker='o',cmap=cm.jet)
        # plot(0, 0, "*", markersize = 15, color='k')
        # grid(ls = '-.', lw=0.4)
        # axis((-10, 10, -15, 15))
        
        # if 'cn12' in f_n:
        #     f=  axis_param(im,'TECU') 
        # tight_layout()  
        
        
        figure(3, figsize=(9,12))
        subplot(616)
        bar(ep_dist+dtec_gnss[dtec_peaks], dtec_gnss[dtec_peaks], 
            color='g', width = 10)
        axis((0, 2000, 0, 2))
        grid(ls = '-.', lw=0.4)
        
    
#%%

f_list=sorted(glob.glob('./honduras/tec/*.npy'));#0937# 0261#0205
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
    
    
#     figure(3)
#     if "G06" in f_n:
#         dtec_gnss=data_smooth(t_gnss,tec_gnss)
#         dtec_gnss[dtec_gnss < 0.03] = 0
#         dtec_peaks = find_peaks_within_range_k(dtec_gnss, 79, 160, 0.055, 1)
        
#         scatter(t_gnss[dtec_peaks], ep_dist + dtec_gnss[dtec_peaks], c=dtec_gnss[dtec_peaks])
#         axis((min(t_gnss), max(t_gnss), 0, 1300))
#         axvline(t_ep)
        
#     if "G11" in f_n:
#         dtec_gnss=data_smooth(t_gnss,tec_gnss)
#         figure(4)
#         dtec_gnss[dtec_gnss < 0.03] = 0
#         dtec_peaks = find_peaks_within_range_k(dtec_gnss, 83, 160, 0.058, 1)
        
#         scatter(t_gnss[dtec_peaks], ep_dist + dtec_gnss[dtec_peaks], c=dtec_gnss[dtec_peaks])
#         axis((min(t_gnss), max(t_gnss), 0, 1300))
#         axvline(t_ep)
        
#     if "G19" in f_n:
#         dtec_gnss=data_smooth(t_gnss,tec_gnss)
#         figure(5)
#         dtec_gnss[dtec_gnss < 0.01] = 0
#         dtec_peaks = find_peaks_within_range_k(dtec_gnss, 83, 150, 0.1, 1)
        
#         scatter(t_gnss[dtec_peaks], ep_dist + dtec_gnss[dtec_peaks], c=dtec_gnss[dtec_peaks])
#         axis((min(t_gnss), max(t_gnss), 0, 1300))
#         axvline(t_ep)
        
#     if "G17" in f_n:
#         dtec_gnss=data_smooth_mike(t_gnss,tec_gnss, 3, 5)
#         figure(6)
#         # dtec_17 = data_smooth_mike(t_gnss, data_gnss, 7, 5)
        
#         # dtec_17[dtec_17 < 0.001] = 0
#         # dtec_peaks = find_peaks_within_range_k(dtec_17, 83, 150, 0.003, 1)
        
#         # scatter(t_gnss[dtec_peaks], ep_dist + dtec_17[dtec_peaks], c=dtec_17[dtec_peaks])
#         # axis((min(t_gnss), max(t_gnss), 0, 2000))
#         # axvline(t_ep)
#         #dtec_gnss = data_smooth_mike(t, data, win1, win2)
#         dtec_gnss[dtec_gnss < 0.05] = 0
#         dtec_peaks = find_peaks_within_range_k(dtec_gnss, 95, 120, 0.06, 1)
        
#         scatter(t_gnss[dtec_peaks], ep_dist + dtec_gnss[dtec_peaks], c=dtec_gnss[dtec_peaks])
#         axis((min(t_gnss), max(t_gnss), 0, 2000))
#         axvline(t_ep)
# show()


#%%

f_list=sorted(glob.glob('./honduras/tec/prn06_17/*.npy'));#0937# 0261#0205
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
    
    
    if "G06" in f_n:
        dtec_gnss=data_smooth(t_gnss,tec_gnss)
        
        fig=figure(7,figsize=(8,5))
        
        ax=subplot(111)
        # dtec_gnss[dtec_gnss < 0.01] = 0
        # dtec_peaks = find_peaks_within_range_k(dtec_gnss, 60, 170, 0.05, 1)
        
        dtec_gnss[dtec_gnss < 0.03] = 0
        dtec_peaks = find_peaks_within_range_k(dtec_gnss, 79, 160, 0.055, 5)
        
        im = scatter((lon_gnss[dtec_peaks]-lon_ep),(lat_gnss[dtec_peaks]-lat_ep)+5*dtec_gnss[dtec_peaks],
                     c=0.6*dtec_gnss[dtec_peaks],s=50,
                    vmax=vm1,vmin=-vm1,marker='o',cmap=cm.jet)
        plot(0, 0, "*", markersize = 15, color='r')
        grid(ls = '-.', lw=0.4)
        axis((-10, 10, -15, 15))
        
        
        if 'acho' in f_n:
            f=  axis_param(im,'TECU') 
            
        figure(8, figsize=(8,7))
        subplot(212)
        
        
        
        bar(ep_dist, dtec_gnss[100:].max(), color='r', width = 10)
        
        # bar(ep_dist+dtec_gnss[dtec_peaks], dtec_gnss[dtec_peaks], 
        #     color='g', width = 10, label='CIDs')
        axis((0, 2000, 0, 1.4))
        grid(ls = '-.', lw=0.4)
        xlabel('Epicentral Distance (km)')
        ylabel('[TECU]')
        title("TEC Data", fontweight='bold')
        tight_layout()
            
    elif "G17" in f_n:
        
        dtec_gnss=data_smooth(t_gnss,tec_gnss)
        #         
        fig=figure(7,figsize=(8,5))
        ax=subplot(111)
        # dtec_gnss[dtec_gnss < 0.01] = 0
        # dtec_peaks = find_peaks_within_range_k(dtec_gnss, 60, 170, 0.05, 1)
        
        
        
        dtec_gnss[dtec_gnss < 0.03] = 0
        dtec_peaks = find_peaks_within_range_k(dtec_gnss, 79, 160, 0.055, 1)
        
        im = scatter((lon_gnss[dtec_peaks]-lon_ep),(lat_gnss[dtec_peaks]-lat_ep)+5*dtec_gnss[dtec_peaks],
                     c=0.6*dtec_gnss[dtec_peaks],s=50,
                    vmax=vm1,vmin=-vm1,marker='o',cmap=cm.jet)
        plot(0, 0, "*", markersize = 15, color='r')
        grid(ls = '-.', lw=0.4)
        axis((-10, 10, -15, 15))
        xlabel("Ep. Lon. ($^{o}$)")
        ylabel("Ep. Lat. ($^{o}$)")
        
        if 'cn19' in f_n:
            f=  axis_param(im,'TECU') 
            
        # figure(8, figsize=(9,12))
        # subplot(212)
        # bar(ep_dist+dtec_gnss[dtec_peaks], dtec_gnss[dtec_peaks], 
        #     color='g', width = 10)
        # axis((0, 2000, 0, 2))
        # grid(ls = '-.', lw=0.4)
        tight_layout()



        
    


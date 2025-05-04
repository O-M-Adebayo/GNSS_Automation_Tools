#ESTE CÓDIGO RESOLVE AS EQUAÇÕES DADA POR KHERANI ET AL (2012, DOI: 10.1111/j.1365-246X.2012.05617.x)..
#(1) EQUAÇÃO DA ONDA PARA AMPLITUDE (W):
#d2w/dt2=(1/rho)grad(1.4 pn div.W)-((grad pn)/rho)div.(rho W)+(1/rho)grad(W.div)p+d((mu/rho)div.div (W)/dt-d(W.div W)/dt
#(2) EQUAÇÃO DA DENSIDADE (rho)
#d rho/dt+div.(rho W)=0
#(3) EQUAÇÃO DA ENERGIA OU PRESSÃO (pn)
#d pn/dt+div. (pn W)+(1.4-1)div. W=0
#ESTE CODIGO NUMERICO É COM ERRO DE SEGUNDA ORDEM EM SPAÇO..
#PORTANTO, EXISTE POSIBILIDADE DE MELHORAR...
#TEMBÉM, ESTE CÓDIGO EMPREGA METODO GAUSS-SEIDEL A RESOLVER A EQUAÇÃO DE
#MATRIZ. ESTE MÉTODO É SUBJETIVO..
#O CÓDIGO PODE REPRODUZ OBSEERVAÇÕES ATE 70%-80% QUALITATIVAMENTE..

#O CODIGO USA MKS UNIT E RESOLVE AS EQUAÇÕES EM O PLANO (X-Y) 
#QUE RERESENTA (LONGITUDE-ALTITUDE) OU (LATITUDE-ALTITUDE)
#ESTE CÓDIGO USA FORÇANTE NO SOLO (Y=0 KM) PARA EXCITAR AS ONDAS
#ESTE FORÇANTE VARIA NO TEMPO E NO X COMO GAUSSINAO
#FORÇANTE É DA CARATER MECANICAL ISTO É DE FORMA DE VENTO VERTICAL..
#O CÓDIGO FUNCIONA BEM COM dt=dy e dy<=dx<=2.*dy e 5km<=dy<=10 km
#OS CONTORNOS DAS LONGITUDES OU LATITUDES DEVERIA SER MAIS AFASTADA DE LOCALIZAÇÃO DA FORÇANTE PARA EVITAR AS REFELXÕES DAS ONDAS..
#O CONTORNO SUPERIOR (YMAX) DEVERIA SER IQUAL OU MAIOR DO 400 KM. 
#=============================================================================#
#sigma_t é espressura de pacote Gaussiano de tempo
#t_o é tempo em que forçante atinge amplitude maior e deve ser mair do 2*sigma_t
#t_f é tempo final de simulação e deve ser mais de 2*t_o
#=============================================================================#

#%%
#==================================MAIN====================================== 
#(wx,wy) são amplitudes da AGWs na direções (x,y) ou seja Longitudinal e transverse 
#(rho_o,tn_o,pn_o) são densidade, temeratura e pressão atmosferica
#(wx_m,wy_m)=(wx(t-dt,x,y),wy(t-dt,x,y))
#(wx_o,wy_o)=(wx(t,x,y),wy(t,x,y)
#(rho_o,tn_o,pn_o)=(rho(t-dt,x,y),tn(t-dt,x,y),pn(t-dt,x,y))

#%%
#==============================================================================
#=Plano (X-Y) de simulação representa a plano  em que 
#(+X,+Y) representam oeste OU norte e vertical para cima (altitude) 
#respectivamente.
   
from pylab import *
from numpy import *
#from pyiri2016 import *
from nrlmsise_2000 import *
from scipy import *
from scipy.ndimage import *
from scipy.special import erf
from scipy.integrate import trapz
from signal_alam import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
close('all')

import time
start = time.time()

matplotlib.rc("mathtext",fontset="cm")        #computer modern font 
matplotlib.rc("font",family="serif",size=12)

gma=1.33
def d1_3(n2,n3,data):
    return repeat(repeat(data[newaxis,:],n3,axis=0)[newaxis,:,:],n2,axis=0)

def d1_2(n,data):
    return repeat(data[newaxis,:],n,axis=0)

def d1_23(n,data):
    return repeat(data[:,:,newaxis],n,axis=2)

#=============================================================================#
def pdf(x):
    return exp(-x**2/2)

def cdf(x):
    return (1 + erf(x/sqrt(2)))

def skew(x,mu,sig,a):
    t=(x-mu) / sig
    return pdf(t)*cdf(a*t)
#=============================================================================#
def ddf(x,mu,sig):
    val = np.zeros_like(x)
    val[(-(1/(2*sig))<=x-mu) & (x-mu<=(1/(2*sig)))] = 1
    return val

#=============================================================================#
def div_f(f0,f1):
    return gradient(f0)[0]/dx_m+gradient(f1)[1]/dy_m 

#%%
def sum_gr(ndim,ndata,data):
    data_n=0*data
    
    if ndim==0:
        for j in range (ndata):
            if j==0:
                data_n[j,:]=(data[j+1,:]+data[j,:])/2.
            elif j==ndata-1:
                data_n[j,:]=(data[j-1,:]+data[j,:])/2.
            else:
                data_n[j,:]=data[j+1,:]+data[j-1,:]
    
    if ndim==1:
        for j in range (ndata):
            if j==0:
                data_n[:,j]=(data[:,j+1]+data[:,j])/2.
            elif j==ndata-1:
                data_n[:,j]=(data[:,j-1]+data[:,j])/2.
            else:
                data_n[:,j]=data[:,j+1]+data[:,j-1]
    return data_n


#%%
def data_antes(dim,ndim,data):
    data_n=0*data
    if dim==1:
        data_n=0*data
        data_n[1:-1]=data[0:-2]
        data_n[0]=data_n[1];data_n[-1]=data_n[-2];
    else:
        if ndim==0:
            data_n[1:-1,:]=data[0:-2,:]
            data_n[0,:]=data_n[1,:];data_n[-1,:]=data_n[-2,:];
        if ndim==1:
            data_n[:,1:-1]=data[:,0:-2]
            data_n[:,0]=data_n[:,1];data_n[:,-1]=data_n[:,-2];
    return data_n

#%%
def data_proximo(dim,ndim,data):
    data_n=0*data
    if dim==1:
        data_n=0*data
        data_n[1:-1]=data[2:]
        data_n[0]=data_n[1];data_n[-1]=data_n[-2];
    else:
        if ndim==0:
            data_n[1:-1,:]=data[2:,:]
            data_n[0,:]=data_n[1,:];data_n[-1,:]=data_n[-2,:];

        if ndim==1:
            data_n[:,1:-1]=data[:,2:]
            data_n[:,0]=data_n[:,1];data_n[:,-1]=data_n[:,-2];
    return data_n

#%%
def ambiente_atmos(iw,x2,y2):
    global rho_amb,tn_amb,r_g,nu_nn,lambda_c
    global pn, sn
    if iw==0:
        
        lat_ep=-36.122;lon_ep=-72.898
        year,month,dom=2010,2,27
                
        d0 = datetime.date(year,1,1)
        d1 = datetime.date(year, month, dom)
        delta = d1 - d0
        doy=delta.days 
        ut=6+34/60;lt=ut+lon_ep/15.
        
        f107A,f107,ap=150,150,4 #300,300,400
        
        f=nrl_msis(doy,ut*3600.,lt,f107A,f107,ap,lat_ep,lon_ep,dy,y[0],ny)
        tn_msis=f[1];#tn_msis=0*tn_msis+tn_msis.mean()
        den_ox=f[2]*1.e+06;den_n=f[3]*1.e+06;den_o2=f[4]*1.e+06;den_n2=f[5]*1.e+06;
        n_msis=den_ox+den_n+den_o2+den_n2;
        rho_msis=f[6]*1.e+03
        mean_mass=rho_msis/n_msis
        
        b_c=1.38e-23; 
        rg_msis=b_c/mean_mass;
        pn_msis=rg_msis*rho_msis*tn_msis;
        sn_msis=sqrt(gma*pn_msis/rho_msis)
        
        nu_msis=pi*(7*5.6e-11)**2.*sn_msis*n_msis    
        visc_mu_1=3.563e-07*tn_msis**(0.71);
        visc_mu_2=1.3*pn_msis/nu_msis;
        lambda_msis=sn_msis**2./nu_msis                                                       #Conductividade termica
        
        # rho_msis=0*rho_msis+rho_msis[:20].mean()
        # pn_msis=0*pn_msis+pn_msis[:20].mean()
        # tn_msis=0*tn_msis+tn_msis[:20].mean()
        # rg_msis=0*rg_msis+rg_msis[:20].mean()
        sn_msis=sqrt(gma*pn_msis/rho_msis)
        
        rho_amb=d1_2(nx,rho_msis)
        tn_amb=d1_2(nx,tn_msis)
        sn=d1_2(nx,sn_msis)
        r_g=d1_2(nx,rg_msis)
        nu_nn=d1_2(nx,nu_msis)
        lambda_c=d1_2(nx,lambda_msis)
    if iw==1:
        a=0.25e+25*32*1.6e-27*1.e-08#*1.e+06; 
        c=-4.5;d=-19.5;c=-3.5
        yr=130.;r_y=y2/yr-1
        rho_amb=a*(exp(c*r_y)+exp(d*r_y))                                             #Densidade de massa (kg/m³)
        
        a1=310;b1=200;c1=-1.;d1=0.5
        a2=220;b2=250;c2=-0.9;d2=6.
        a3=220;b3=135;c3=-0.2;d3=7.
        a4=120;b4=250;c4=-1.5;d4=18.
        a5=220;b5=220;c5=-0.25;d5=5.
        y_1=10.;y_2=65.;y_3=130.;y_4=100.;y_5=230.;
        
        r_y1=y2/y_1-1;r_y2=y2/y_2-1;r_y3=y2/y_3-1
        r_y4=y2/y_4-1;r_y5=y2/y_5-1
        
        first=a1-b1/(exp(c1*r_y1)+exp(d1*r_y1))
        second=-a2+b2/(exp(c2*r_y2)+exp(d2*r_y2))
        third= a3-b3/(exp(c3*r_y3)+exp(d3*r_y3))
        fourth=-a4+b4/(exp(c4*r_y4)+exp(d4*r_y4))
        fifth=a5-b5/(exp(c5*r_y5)+exp(d5*r_y5))
        sixth=-720+1.25*(first+second+0*fourth)+8.5*third+2.*fifth 
        tn_amb=0.5*sixth                                                              #Temperatura atmosferica (K)
    
        r_g=150.*(1.+sqrt(y2*1.e-03+5.)/5.);                                        #constante Boltzman/massa
        
        b_c=1.38e-23;                                                               #Constante Boltzmann
        mean_mass=b_c/r_g                                                           #massa atmosferica
        nn=rho_amb/mean_mass                                                          #Densidade numerica
        pn=r_g*rho_amb*tn_amb                                                           #Pressão atmosferica
        sn=sqrt(1.33*pn/rho_amb)                                                       #velocidade de som
        
        nu_nn=pi*(7*5.6e-11)**2.*sn*nn                                              #frequencia da colisão
        lambda_c=sn**2./nu_nn                                                       #Conductividade termica
        
    return

#%%
def atmos_evolve(rho_o,tn_o,pn_o,wx,wy):
    div_w=div_f(wx,wy)
    div_flux=div_f(rho_o*wx,rho_o*wy)
    # div_flux_x=wx*gradient(rho_o)[0]/gradient(x2_m)[0]
    # div_flux_y=wy*gradient(rho_o)[1]/gradient(y2_m)[1]
    # div_flux=div_flux_x+div_flux_y
    rho=rho_o-0.5*dt*div_flux
    
    div_flux=div_f(tn_o*wx,tn_o*wy)
    # div_flux_x=wx*gradient(tn_o)[0]/gradient(x2_m)[0]
    # div_flux_y=wy*gradient(tn_o)[1]/gradient(y2_m)[1]
    # div_flux=div_flux_x+div_flux_y        
    
    #gma=1.33#rho_ho/rho
    tn=tn_o-0.5*dt*(div_flux+(gma-1.)*tn_o*div_w)
    
    div_flux=div_f(pn_o*wx,pn_o*wy)        
    # div_flux_x=wx*gradient(pn_o)[0]/gradient(x2_m)[0]
    # div_flux_y=wy*gradient(pn_o)[1]/gradient(y2_m)[1]
    # div_flux=div_flux_x+div_flux_y
    
    pn=pn_o-0.5*dt*(div_flux+(gma-1.)*pn_o*div_w)/1.
    
    return (rho,tn,pn)

#%%
def ambiente_iono(iw,x,yi):
    global no_y,n_o,nu_in,nu_0,gyro_i,b_o
    if iw==0:
        import iri2016.profile as iri
        from datetime import datetime, timedelta
        #yi=y2[0,i_ion:]
        time_start_stop = (datetime(2010, 2, 27, 6,34,0), datetime(2010, 2, 27,6,52,0))
        time_step = timedelta(minutes=5)
        alt_km_range = (yi[0], yi[-1], dy)
        glat = -36.122
        glon = -72.898
        
       
        data_iri=iri.IRI(datetime(2010,2,27,6,34,11), alt_km_range, glat, glon)
        ne_iri=data_iri.ne
        sim = iri.timeprofile(time_start_stop, time_step, alt_km_range, glat, glon)

        ax = figure().gca()
        ax.contourf(sim.time, sim.alt_km, sim.ne.T, shading="nearest")
        ax.set_title("Number Density")
        ax.set_xlabel("time [UTC]")
        ax.set_ylabel("altitude [km]")
     

        [n_o,x3]=meshgrid(ne_iri,x)#d1_2(nx,ne_iri)
        sc_h=30.
        nu_in=1.e+03*exp(-(yi-80.)/sc_h)                                            #A perfile (em altitude) da frequencia 
                                                                                    #da colisão (nu_in)..
  
                                                                            
        b_o=30.e-06                                                                 #o campo geomagnetico em Tesla
        q_c=1.6e-19;m_i=1.67e-27;z_i=16.
        gyro_i=q_c*b_o/(z_i*m_i)                                                    # a freuqnecia de giração da ions
    
        gyro_e=-gyro_i*1837.
        

    if iw==1:
        y2=yi
        np=1.e+12;yp=200.;r=y2/yp
        a=2;b=-10.
        n_of=2.*np/(exp(a*(r-1.))+exp(b*(r-1.)))                                    #Perfile (em altitude (y)) 
        
        np=1.e+11;a=5.;b=-60.
        n_oe=2.*np/(exp(a*(r-0.5))+exp(b*(r-0.5)))
        
        n_ef=5.e-01*np*(r+0.5)**2./5.
        n_o=n_of+n_oe+n_ef
        
        sc_h=30.
        nu_in=1.e+03*exp(-(y2-80.)/sc_h)                                            #A perfile (em altitude) da frequencia 
                                                                                    #da colisão (nu_in)..
    #    nu_0=nu_in
    #    i_500=abs(y-500.).argmin()
    #    for i in range (i_500,ny):
    #        nu_0[:,i]=nu_0[:,i_500]
                                                                            
        b_o=30.e-06                                                                 #o campo geomagnetico em Tesla
        q_c=1.6e-19;m_i=1.67e-27;z_i=16.
        gyro_i=q_c*b_o/(z_i*m_i)                                                    # a freuqnecia de giração da ions
    
        gyro_e=-gyro_i*1837.
        
    return ()   

#%%
def iono_evolve(n_o,vx,vy):
    fx_o=vx*gradient(n_o)[0]/2.        
    fy_o=vy*gradient(n_o)[1]/2.
    n=n_o
    vnx=dx_m/dt;vny=dy_m/dt
    for iter in range (11):                                                     #ITERATION LOOP PARA GAUSS-SEIDEL CONVERGENCE
        fx_g=vx*gradient(n)[0]/2.
        fy_g=vy*gradient(n)[1]/2.
        fx=(fx_o+fx_g)/2.                                                       #SEMI IMPLICIT CRANK-NICOLSON TIME INTEGRATION
        fy=(fy_o+fy_g)/2.
        n=n_o-0.5*(fx/vnx+fy/vny)                                                   #EQUAÇÃO NUMERICA DA DENSIDADE
#        n[:,0]=n[:,1];n[:,-1]=n[:,-2];                                          #condições contorno na ALTITUDE 
    return n
#%%
def vel(b_o,nu,gyro,wx,wy):
    global mu_p                                                                 
    # |vx| | mu_p mu_h | |Ex|
    # |  |=|           | |  |
    # |vy| |-mu_h mu_p | |Ey|
    kappa=gyro/nu
    mu_p=kappa/(b_o*(1.+kappa**2.))                                             #PEDERSON MOBILITY
    mu_h=kappa**2./(b_o*(1.+kappa**2.))                                         #HALL MOBILITY
    lat=radians(38)
    mag_m=8.e+15         #Tm^3
    r_ea=6.371e+06
    by=-2.*mag_m*sin(abs(lat))/r_ea**3.;
    bz=mag_m*cos(lat)/r_ea**3.;
    bx=0
    wz=0
    ey=wz*bx-wx*bz
    ex=wy*bz-wz*by
    ez=wx*by-wy*bx
#    ex=wy*b_o*cos(lat);ey=-wx*b_o*cos(lat)
    vx=mu_p*ex+mu_h*ey
    vy=mu_p*ey-mu_h*ex
    return (vx,vy)
#%%
def agw_dispersion(idim):
    if idim==0:
        delta=2.*dx_m;wkv=wk_x;wkh=wk_y
    if idim==1:
        delta=2.*dy_m;wkv=wk_y;wkh=wk_x
    
    #gma=1.33
    pn=pn_o#r_g*rho_o*tn_o;
    c_s=1.2*sqrt(gma*pn/rho_o);#c_s=0*c_s+c_s[:,:20].mean()
    gr_pn=gradient(pn)[idim];dy_m2=2.*dy_m
    zeta=(1./rho_o)*gr_pn/delta
    k0=zeta/c_s**2.
    k0_antes=roll(k0,-1,axis=idim)#data_antes(2,idim,k0)
    k0_proximo=roll(k0,1,axis=idim)#data_proximo(2,idim,k0)
    mu=0.5*(k0_proximo+k0_antes)*delta/2.
    mu=cumsum(mu,idim)/2.;#mu=9.*mu/abs(mu).max()
    omega_ac=k0*c_s/1.#1.45*gma*zeta/(2.*c_s)
    omega_c2=omega_ac**2.#c_s**2.*(gma*k0/2)**2.#(gma**2.*k0*c_s)**2./4.;
    omega_b2=((gma-1)*k0**2-(k0/c_s**2.)*gradient(c_s**2.)[idim]/delta)*c_s**2.
    omega_b2[omega_b2<0]=abs(omega_b2).min()
    omega_h2=wkh**2.*c_s**2.
    omega_2=0*omega_b2+wkv**2.*c_s**2.+0*omega_h2+0*omega_c2
    omega_mais=sqrt(omega_2+sqrt(omega_2**2.-0*4.*omega_h2*omega_b2))/sqrt(2)
    omega_menos=sqrt(omega_2-sqrt(omega_2**2.-0*4.*omega_h2*omega_b2))/sqrt(2)

    visc_mu=1.3*pn_amb/nu_nn;visc_ki=visc_mu/rho_amb
    nu_col=visc_ki*(-wkh**2-wkv**2.+k0**2.-gradient(k0)[idim]/delta)
    wx_mais=(omega_mais**2.+omega_h2-omega_2)/(wkv*wkh*c_s**2.)
    wx_menos=(omega_menos**2.+omega_h2-omega_2)/(wkv*wkh*c_s**2.)
    
    gamma_ad=(gma-1)*k0**2.
    gamma_e=(k0/c_s**2.)*gradient(c_s**2.)[idim]/delta
    
    wkv2=(omega_mais**2.-omega_c2)/c_s**2.
    wk_real=0*wkv2;wk_im=0*wkv2
    #wkv2[wkv2>0]=0*wkv2[wkv2>0]
    #wkv2[wkv2<0]=sqrt(abs(wkv2[wkv2<0]))
    
    wk_real[wkv2>0]=sqrt(abs(wkv2[wkv2>0]))
    wk_im[wkv2<0]=sqrt(abs(wkv2[wkv2<0]))
    
    return (mu,omega_mais,omega_menos,nu_col,wx_mais,wx_menos,omega_b2,\
            omega_c2,c_s,wk_real,wk_im)#ob2_im,gamma_ad,gamma_e,c_s)

#%%

def agw_propagator_v(omega,wk_y,wk_x):
    omega3=repeat(omega[newaxis,:,:],nt,axis=0)
    v_phase=omega3/wk_y;
    if abs(v_phase).min() !=0:
        t_phase=y3_m/v_phase
    else:
        t_phase=0+0*y3_m
        
    wy0_t=0*v_new
    sigma=1./omega
    for it0 in range (nt):
        f_frente=cos(-omega3*t_s[it0]+wk_y*y3_m+0*wk_x*abs(x3_m))
        f_prop=skew(t_s[it0]-t_phase,t3,sigma/2.,0)*f_frente
        wy0_t[it0,:,:]=(v_new*f_prop).sum(0)
    
        wy0_t[it0,:,0]=v_new[it0,:,0]+0*wy0_t[it0,:,0]
       # wy0_t[it0,:,:]=v_new[it0,:,:]*skew(y3[it0,:,:],20,10,0)+wy0_t[it0,:,:]
        
    sigma_x=0.1*dx_m*1.e-03;x_o=0.#Gaussian for Lamb wave 20.*dx_m*1.e-03
    f_lon=skew(x3,x_o,sigma_x,0)
    
    wy0=wy0_t#*(f_lon-f_lon[0,-1,0])
    return wy0

#%%
def agw_propagator_h(omega,wk_y,wk_x,wy0_x):
    omega3=repeat(omega[newaxis,:,:],nt,axis=0)
    v_phase=omega3/wk_x;
    if abs(v_phase).min() !=0:
        t_phase=abs(x3_m)/v_phase
    else:
        t_phase=0+0*x3_m
        
    wy_h=0*wy0_x
    sigma=1./omega
    for it0 in range (nt):
        f_frente=cos(-omega3*t_s[it0]+wk_x*abs(x3_m))
        f_prop=skew(t_s[it0]-t_phase,t3,sigma/2.,0)*f_frente
        wy_h[it0,:,:]=(wy0_x*f_prop).sum(0)
        #wy_h[it0,:,0]=0
    return wy_h

#%%
def agw_propagator_resonance(omega,wk_y,wr):
    omega3=repeat(omega[newaxis,:,:],nt,axis=0)
    v_phase=omega3/wk_y;
    if abs(v_phase).min() !=0:
        t_phase=y3_m/v_phase
    else:
        t_phase=0+0*y3_m
        
    wr_prop=0*wr
    sigma=1./omega
    for it0 in range (nt):
        f_frente=cos(-omega3*t_s[it0]+wk_y*y3_m)
        f_prop=skew(t_s[it0]-t_phase,t3,sigma/1.,0)*f_frente
        wr_prop[it0,:,:]=-(wr*f_prop).sum(0)    
    return wr_prop
#%%
global rho_o,tn_o,pn_o,rho_amb,tn_amb,pn_amb

data_sism= load('Maule_mike.npy')
t_s=3600*data_sism[0,:]
vel_s=0.5e-00*data_sism[1,:]
figure(4)
plot(t_s/3600, vel_s)

i_ep=abs(abs(vel_s)-abs(vel_s).max()).argmin()
i_start=abs(t_s-(t_s[i_ep]-500)).argmin()
i_last=abs(t_s-(t_s[i_ep]+4500)).argmin()
t_s=around(t_s[i_start:i_last:739]);vel_s=vel_s[i_start:i_last:739]
print (t_s[0]/3600,t_s[-1]/3600,len(t_s))
t_ss=t_s;t_s=t_s-t_s[0];nt=len(t_s)
dt=t_s[1]-t_s[0]
i_max=argmax(abs(vel_s))
vel_s[0:i_max-10][abs(vel_s[0:i_max-10])<3e-01]=0
# figure(2,(12,12))
# plot(t_s,vel_s,'r')
# vel_s=convolve(vel_s,ones((4,))/4,mode='same')
# plot(zoom(t_s[::5],5),zoom(vel_s[::5],5),'g')
# draw()
# t_s=zoom(t_s[::5],5)
# vel_s=zoom(vel_s[::5],5)
# t_ss=t_s;t_s=t_s-t_s[0];nt=len(t_s)
# dt=t_s[1]-t_s[0]
#print(t_s[1]-t_s[0])


dy=10;dx=5*dy/2;                                                            #A resoluções espaciais no kilometros
y=arange(0,400+dy,dy);ny=len(y)                                           #A faixa de altitude
x=arange(0,2200+dx,dx);nx=len(x)  #3000                                      #A faixa de longitude
i_alt=abs(y-80).argmin()
[y2,x2]=meshgrid(y,x)
y2_m=y2*1.e+03;x2_m=x2*1.e+03
dy_m=dy*1.e+03;dx_m=dx*1.e+03                                                 #As resoluções espaciais no metros

f=ambiente_atmos(0,x2,y2)
pn_amb=r_g*rho_amb*tn_amb    
 

[x3,v3,y3]=meshgrid(x,vel_s,y)
[x3,t3,y3]=meshgrid(x,t_s,y)
x3_m=x3*1.e+03;y3_m=y3*1.e+03


#f_source=skew(x3, 0, 5*dx, 0)
#v_new=v3*f_source

# def fonte_rayleigh():
#     v_new=0*v3
#     v0=3.e+00
#     t_phase=(abs(x3)-0)/v0;sigma=dt/2.
#     f_prop=skew(t3,t_phase,sigma,0)
#     for k in range (ny):
#         for j in range (nx):
#             v_new[:,j,k]=-convolve(f_prop[:,j,k],v3[:,j,k])[:nt]
#     # for it0 in range (nt):
#     #     f_prop=skew(t_s[it0]-t_phase,t3,sigma,0)
#     #     v_new[it0,:,:]=(v3*f_prop).sum(0)
#     return v_new
# v_new=fonte_rayleigh()

# ### Saul 1
# def fonte_rayleigh():
#     v_new = 0 * v3
#     v0 = v_new + 3.5  # km/s
#     # Primera perturbación
#     t_phase1 = (abs(x3) - 0) / v0
#     sigma = dt / 2.
#     t_phase1[v0 == 0] = 0
#     f_prop1 = skew(t3, 1 * t_phase1, sigma, 0)
   
#     # Segunda perturbación a 1000 km y 6.65 horas
#     x_center2 = 800  # km
#     t_center2 = 6.8 * 3600  # en segundos
#     t_phase2 = (abs(x3 - x_center2) - 0) / v0
#     f_prop2 = skew(t3, 1 * t_phase2, sigma, 0)
    
#     # Segunda perturbación a 1000 km y 6.65 horas
#     x_center3 = 100  # km
#     t_center3 = 6.8 * 3600  # en segundos
#     t_phase3 = (abs(x3 - x_center3) - 0) / v0
#     f_prop3 = skew(t3, 1 * t_phase3, sigma, 0)

#     sigma_x = 30 * dx_m * 1.e-03 + 0 * v0  # 25 km

#     for k in range(ny):
#         for j in range(nx):
#             #x0 = x3[:, j, k]

#             fonte_falha1 = skew(x3[:, j, k], 0, sigma_x[:, j, k], 0)
#             fonte_rl = 2.*skew(x3[:, j, k] - x_center2, 0, sigma_x[:, j, k], 0)
#             fonte_rl2 = 0.5*skew(x3[:, j, k] - x_center3, 0, sigma_x[:, j, k], 0)

#             v_new[:, j, k] = (fonte_falha1 * convolve(f_prop1[:, j, k], v3[:, j, k])[:nt] +
#                               fonte_rl * convolve(f_prop2[:, j, k], v3[:, j, k])[:nt] +
#                               fonte_rl2 * convolve(f_prop3[:, j, k], v3[:, j, k])[:nt])
#     return v_new
# v_new=fonte_rayleigh()



# Saul 2
def fonte_rayleigh():
    v_new = 0 * v3
    v0 = v_new + 2.8  # km/s
    
    # Primera perturbación
    t_phase1 = (abs(x3) - 0) / v0
    sigma = dt / 2.
    t_phase1[v0 == 0] = 0
    f_prop1 = 1*skew((t3-250), 1 * t_phase1, sigma, 0)
    sigma_x1 = 5 * dx_m * 1.e-03 + 0 * v0  # 25 km
    # Segunda perturbación a 1000 km en t_center2
    
    t_phase2 = (abs(x3) - 0) / v0
    f_prop2 = 1*skew((t3 - 250), 1 * t_phase2, sigma, 0)
    sigma_x2 = 15 * dx_m * 1.e-03 + 0 * v0  # 25 km
    
    # # #x_center3 = 300  # km
    # # #t_center3 = 300  # 
    t_phase3 = (abs(x3) + 500) / v0
    f_prop3 = skew((t3 - 250), 1 * t_phase3, sigma, 0)
    #sigma_x3 = 10 * dx_m * 1.e-03 + 0 * v0  # 25 km
   
    for k in range(ny):
        for j in range(nx):

            #fonte_falha = 11.1*skew(x3[:, j, k], 350, sigma_x2[:, j, k]/3, 15)
            fonte_falha = 12.1*skew(x3[:, j, k], 340, sigma_x2[:, j, k], 15)
            fonte_rl2 = skew(x3[:, j, k], 150, sigma_x1[:, j, k], 0)
            #fonte_rl3 = skew(x3[:, j, k], 400, sigma_x3[:, j, k], 15)
            fonte_rl3 = 1.5*skew(x3[:, j, k], 600, sigma_x2[:, j, k], 10)
            
            v_new[:, j, k] = ((fonte_falha)* convolve(f_prop2[:, j, k], v3[:, j, k])[:nt] +
                              fonte_rl2 * convolve(f_prop1[:, j, k], v3[:, j, k])[:nt] +
                              fonte_rl3 * convolve(f_prop3[:, j, k], v3[:, j, k])[:nt])

    return v_new
v_new=fonte_rayleigh()

# def fonte_rayleigh():
#     v_new=0*v3
#     v0=0*v_new+3.5# 4 km/s
#     # v0[x3<1250]=2.
#     t_phase=(abs(x3)-0)/v0;sigma=dt/2.
#     t_phase[v0==0]=0
#     f_prop=skew(t3,t_phase,sigma,0)
#     sigma_x=40.*dx_m*1.e-03+0*v0
   
#     peak_position_1 = 100  # Set this to the position of the first peak
#     peak_position_2 = 800  # Set this to the position of the second peak

    
#     x0=0
#     # sigma_x[x3<400]=0.3*sigma_x[x3<400]
#     for k in range (ny):
#         for j in range (nx):
#            # v_new[:,j,k]=skew(x3[:,j,k],0,sigma_x,0)*convolve(f_prop[:,j,k],v3[:,j,k])[:nt]
#            # x0=x3[:,j,k]
#            # x0[x0>500]=0
           
#            # fonte_falha=skew(x3[:,j,k],x0,sigma_x[:,j,k]/10.,0)
           
#            # Apply the adjusted widths in skew functions
#            fonte_rl = skew(x3[:, j, k], peak_position_1, sigma_x[:, j, k], 0)
#            fonte_rl_2 = skew(x3[:, j, k], peak_position_2, sigma_x[:, j, k], 0)
                       
#            # fonte_rl=skew(x3[:,j,k],0,sigma_x[:,j,k],0)
#            # fonte_rl_2=skew(x3[:,j,k],700,sigma_x[:,j,k],0)
#            v_new[:,j,k]=(fonte_rl+fonte_rl_2)*convolve(f_prop[:,j,k],v3[:,j,k])[:nt]
#     # for it0 in range (nt):
#     #     f_prop=skew(t_s[it0]-t_phase,t3,sigma,0)
#     #     v_new[it0,:,:]=(v3*f_prop).sum(0)
#     return v_new
# v_new=fonte_rayleigh()
# #v_new = v3







# ----------------------------


#%%==================================MAIN====================================== 
                                                   


global wk_x,wk_y

#%%
time=[];wx3=[];wy3=[];n3=[];data_arrival=[]
wave_all=[];data_amb=[];pr3=[];rho3=[];tn3=[];
o_br=[];omega_all=[];gr_ci3=[]

#%%============================================================================


f=ambiente_iono(1,x,y[i_alt:])   
data_amb.append((rho_amb[0,:],sn[0,:]))

#%%============================INTIALIZATION===================================

rho_o=rho_amb;tn_o=tn_amb;pn_o=pn_amb
rho_to=0*rho_o;rho_ho=rho_amb
                                                          
#%%SOLUCAO ANALYTICA

i_pl=1

wy_ana=zeros((nt,nx,ny));wx_ana=zeros((nt,nx,ny));
wy=zeros((nt,nx,ny));wx=zeros((nt,nx,ny));

lambda_y0=arange(2*dy_m,ny*dy_m/2.,2*dy_m)
for ik in range (len(lambda_y0)):
    lambda_y=lambda_y0[ik]
    #lambda_x0=arange(lambda_y,max(nx*dx_m/2.,ny*dy_m/2.),2*dx_m)
   # lambda_x0=arange(max((nx-1)*dx_m/2.,(ny-1)*dy_m/2.),max(nx*dx_m/2.,ny*dy_m/2.),2*dx_m)
    #lambda_x0=arange(lambda_y,nx*dx_m/20.,2*dx_m)
    lambda_x0=arange(2.1*lambda_y,2.2*lambda_y,1.0*dx_m)#
    for ikx in range (len(lambda_x0)):
        lambda_x=lambda_x0[ikx]#max(sigma_x,2*ikx*dx_m)
        wk_x=2.*pi/lambda_x;wk_y=2.*pi/lambda_y    
        
        f=agw_dispersion(1)
        muy=f[0];omega_mais=f[1];omega_menos=f[2];nu_col=f[3];
        wx_mais=f[4];wx_menos=f[5]
        omega_br=sqrt(f[6]);omega_ac=sqrt(f[7]);
        #omega_ci=sqrt(abs(f[8]));gamma_ad=f[9];gamma_e=f[10]
        c_s=f[8]
        
        f=agw_dispersion(0);
        mux=f[0];
        omega_mais=(omega_mais+f[1])/2.;
        omega_menos=(omega_menos+f[2])/2.
        
#        if ik==0 and ikx==ik and omega_ci.any()!=0:
#            print ('CONVECTIVELY UNSTABLE GWs')                

        wy_alt=exp(-muy)
        wx_alt=exp(-mux)
        w_alt=wx_alt*wy_alt
        w_alt3=repeat(w_alt[newaxis,:,:],nt,axis=0)
        nu_col3=repeat(nu_col[newaxis,:,:],nt,axis=0)
        lambda_c3=repeat(lambda_c[newaxis,:,:],nt,axis=0)
        n=2 # Non-linear = 2, linear = 1
        wy_damp=exp(2.*nu_col3*t3/(2.*n))*exp(-lambda_c3*t3*wk_y**2.)
        wy_growth=exp(0)#epx(omega_ci*t_s[-1]/(2.*pi))
        omega_aw=(1./n)*sqrt(1.-(n-1)**2./(4.*omega_mais.max()*t_s[-1])**2.)*omega_mais
        omega_gw=(1./n)*sqrt(1.-(n-1)**2./(4.*omega_menos.max()*t_s[-1])**2.)*omega_menos

        # if omega_aw.min()<omega_ac.max():
        #     break
    
        # if 1.e+03*omega_mais.max()/(2.*pi)>10:
        #     break          
        
        if 1.e+03*omega_mais.max()/(2.*pi)>1.e+03/dt:
            break
        
        for i_wv in range (1):
            if i_wv==0: 
                omega=omega_aw
                wx_amp=wx_mais/1.
            if i_wv==1: 
                omega=omega_gw
                wx_amp=wx_menos/1.
            
            wy0=agw_propagator_v(omega,wk_y,wk_x);
            wy_ground=wy0[:,:,0]
            wy_ondas=wy0*w_alt*wy_damp*wy_growth
            f_source=skew(x3, 150, 3*dx, 0)
            wy0_x=(repeat(wy_ondas.max(1)[:,newaxis,:],nx,axis=1)*f_source+wy_ondas)/2.   
            #wy0_x=(repeat(wy_ondas.max(1)[:,newaxis,:],nx,axis=1)+wy_ondas)/2.
            wy_x=agw_propagator_h(omega, wk_y, wk_x, wy0_x) 
            wy_ana=wy_ana+(wy_ondas+wy_x) # 0*wy_x gives rayleigh waves alone
            wy_ana[:,:,0]=wy_ground
            # wx_ana=wx_ana+wx_amp*(wy_ondas+wy_x)#wx_amp*gradient(wy_ana)[1]
            wx_ana=wx_ana+wx_amp*gradient(wy_ana)[1]
            
            
        wave_all.append([omega_aw[0,:],omega_gw[0,:],omega_aw[0,:],\
                         wk_x,wk_y,omega_br,omega_ac,c_s[0,:]])
        print (1.e+03/dt,1.e+03*omega_mais.max()/(2.*pi),(2.*pi/wk_y).max(),(omega_mais/wk_y).max())   
#print((2.*pi/wk_y).max())         
for it0 in range (nt):
    f = atmos_evolve(rho_o, tn_o, pn_o, wx_ana[it0, :, :], wy_ana[it0, :, :])
    rho = f[0]
    tn = f[1]
    pn = f[2]
    
    f=vel(b_o,nu_in,gyro_i,wx_ana[it0,:,i_alt:],wy_ana[it0,:,i_alt:])
    vx=f[0];vy=f[1]
   # vx=wx_ana[it0,:,i_alt:];vy=wy_ana[it0,:,i_alt:]
    n=iono_evolve(n_o,vx,vy)
   
#%%===========================ATUALIZAÇÃO EM TEMPO===================#
    rho_o=rho;tn_o=tn;pn_o=pn
    n_o=n
    
    itm=it0-1
    if it0==0:itm=it0
    itp=it0+1
    if it0==nt-1:itp=it0
    
    wy[:,:,0]=wy_ana[:,:,0]
    wx_0=2.*wx[it0,:,:]-wx[itm,:,:];
    wy_0=2.*wy[it0,:,:]-wy[itm,:,:];
    cs2=1.33*pn_o/rho_o#cs2=cs2.mean()+0*cs2
    
    delta_x=gradient(x2_m)[0];delta_y=gradient(y2_m)[1]
    div_w=gradient(wx[it0,:,:])[0]/delta_x+gradient(wy[it0,:,:])[1]/delta_y
    
    d2wx=gradient(div_w)[0]/delta_x
    wx[itp,:,:]=wy_damp[itp,:,:]*(wx_0+cs2*dt**2.*d2wx)
    
    d2wy=gradient(div_w)[1]/delta_y
    wy[itp,:,:]=wy_damp[itp,:,:]*(wy_0+cs2*dt**2.*d2wy)
    wy_num=w_alt3[it0,:,:]*wy[it0,:,:]
    wx_num=w_alt3[it0,:,:]*wx[it0,:,:]
    
    print ('Time, seconds=',dt,t_s[it0])
    print ('GROUND UPLIFT, m/s=',wy_ana[it0,:,0].max())
    print('AGWs amplitudes, m/s=',round(wy_ana[it0,:,:].max(),2),\
          round(wx_ana[it0,:,:].max(),2))
    print (d2wy.max())

    time.append(t_ss[it0]/60.)
    rho3.append(rho);tn3.append(tn)
    pr3.append(pn);
    wx3.append(wx_ana[it0,:,:])
    wy3.append(wy_ana[it0,:,:])
    #gr_ci3.append(omega_ci)
    n3.append(n)
    
    figure(21,(12,12))
    subplot(121)
    imshow(wy_ana[it0,:,:].T,origin='lower',vmax=0.1,vmin=-0.1,cmap=cm.seismic,interpolation='bilinear')
    axis('tight')
    subplot(122)
    imshow(wx_ana[it0,:,:].T,origin='lower',vmax=0.1,vmin=-0.1,cmap=cm.seismic,interpolation='bilinear')
    axis('tight')
    draw()
   # pause(0.1)
i_xo=abs(x-x.mean()).argmin()
#omega_all.append([omega_ac[i_xo,:],omega_br[i_xo,:],omega_ci[i_xo,:],\
#                  gamma_ad[i_xo,:],gamma_e[i_xo,:]])


#%%
save('time.npy',array(time))
save('data_amb.npy',array(data_amb))
save('wy3.npy',array(wy3))
save('wx3.npy',array(wx3))
#save('wave_all.npy',array(wave_all))
save('pr3.npy',array(pr3))
save('n3.npy',array(n3))


figure(11,(12,12))
pcolormesh(t3[:,0,0],x,v_new[:,:,0].T)

figure(12,(12,12))
dn=gradient(array(n3)[:,:,10])[0]
vm=abs(dn).max()/2
pcolormesh(t3[:,0,0],x,dn.T,cmap=cm.seismic,vmax=vm,vmin=-vm)

# figure(13,(12,12))
# dn=gradient(array(n3)[:,:,5:15].sum[2])[0]
# vm=abs(dn).max()/2
# pcolormesh(t3[:,0,0],x,dn.T,cmap=cm.seismic,vmax=vm,vmin=-vm)

show()

import time
end = time.time()
print(end - start)


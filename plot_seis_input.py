#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 11:48:21 2025

@author: oluwasegun-adebayo
"""

from pylab import *
from scipy.special import erf

def pdf(x):
    return exp(-x**2/2)

def cdf(x):
    return (1 + erf(x/sqrt(2)))

def skew(x,mu,sig,a):
    t=(x-mu) / sig
    return pdf(t)*cdf(a*t)

data_sism= load('Maule_mike.npy')
t_s=3600*data_sism[0,:]
vel_s=1.7e-00*data_sism[1,:] # accpeted parameter
#vel_s=4.0e-00*data_sism[1,:] # testing
#vel_s=1.7e-00*data_sism[1,:] # 90 degree
#vel_s=2.7e-00*data_sism[1,:] # elvira
figure(4)


i_ep=abs(abs(vel_s)-abs(vel_s).max()).argmin()
i_start=abs(t_s-(t_s[i_ep]-500)).argmin()
i_last=abs(t_s-(t_s[i_ep]+4500)).argmin()
#t_s=around(t_s[i_start:i_last:739]);vel_s=vel_s[i_start:i_last:739] # elvira
t_s=around(t_s[i_start:i_last:740]);vel_s=vel_s[i_start:i_last:740] # accepted parameter
#t_s=around(t_s[i_start:i_last:300]);vel_s=vel_s[i_start:i_last:300] # 15 seconds
print (t_s[0]/3600,t_s[-1]/3600,len(t_s))
t_ss=t_s;t_s=t_s-t_s[0];nt=len(t_s)
dt=t_s[1]-t_s[0]
i_max=argmax(abs(vel_s))


vel_s[0:i_max-10][abs(vel_s[0:i_max-10])<3e-01]=0

dy=10;dx=5*dy/2;                                                            #A resoluções espaciais no kilometros
y=arange(0,400+dy,dy);ny=len(y)                                           #A faixa de altitude
x=arange(0,2200+dx,dx);nx=len(x)  #3000                                      #A faixa de longitude
i_alt=abs(y-80).argmin()
[y2,x2]=meshgrid(y,x)
y2_m=y2*1.e+03;x2_m=x2*1.e+03
dy_m=dy*1.e+03;dx_m=dx*1.e+03                                                 #As resoluções espaciais no metros
  
 

[x3,v3,y3]=meshgrid(x,vel_s,y)
[x3,t3,y3]=meshgrid(x,t_s,y)
x3_m=x3*1.e+03;y3_m=y3*1.e+03


def fonte_rayleigh():
    v_new = 0 * v3
    #v0 = v_new + 0.0000001 # km/s 90 degree
    #v0 = v_new + 2.8 # km/s accepted parameter
    #v0 = v_new + 4.0 # 20 degree
    v0 = v_new + 2.6
    
    ##Primera perturbación
    t_phase1 = (abs(x3) - 0) / v0
    sigma = dt / 2.
    t_phase1[v0 == 0] = 0
    f_prop1 = 1*skew((t3-450), 1 * t_phase1, sigma, 0) # accepted parameter
    #f_prop1 = 1*skew((t3-450+200), 1 * t_phase1, sigma, 0)
    sigma_x1 = 15 * dx_m * 1.e-03 + 0 * v0  # 25 km
    # Segunda perturbación a 1000 km en t_center2
    
    t_phase2 = (abs(x3) -0) / v0
    # f_prop2 = 1*skew((t3 - 250), 1 * t_phase2, sigma, 0) # accepted parameter
    f_prop2 = 1*skew((t3 -0), 1 * t_phase2, sigma, 0)
    sigma_x2 = 15 * dx_m * 1.e-03 + 0 * v0  # 25 km
    
    # # #x_center3 = 300  # km
    # # #t_center3 = 300  # 
    t_phase3 = (abs(x3) - 0) / v0
    # f_prop3 = skew((t3 - 250), 1 * t_phase3, sigma, 0) #accepted parameter
    f_prop3 = skew((t3 - 250), 1 * t_phase3, sigma, 0)
    #sigma_x3 = 10 * dx_m * 1.e-03 + 0 * v0  # 25 km
   
    # for k in range(ny):
    #     for j in range(nx):

    #         fonte_falha = 0*6.0*skew(x3[:, j, k], 350, sigma_x2[:, j, k]/5, 5) # testing
    #         fonte_rl2 = 1#(3.0)*skew(x3[:, j, k], (70-70), sigma_x2[:, j, k]/(5.-4), 5-5) # testing
    #         fonte_rl3 = (0*1.0)*skew(x3[:, j, k], 600, sigma_x1[:, j, k]/1.5, 5) # testing
            
    #         v_new[:, j, k] = ((0*fonte_falha)* convolve(f_prop2[:, j, k], v3[:, j, k])[:nt] +
    #                           fonte_rl2 * convolve(f_prop2[:, j, k], v3[:, j, k])[:nt] +
    #                           0*fonte_rl3 * convolve(f_prop1[:, j, k], v3[:, j, k])[:nt])
    
    t_phase= (abs(x3) - 0) / v0
    sigma = dt/2
    for it0 in range (nt):
        f_prop=skew(t_s[it0]-t_phase,t3,sigma,0)
        v_new[it0,:,:]=(v3*f_prop).sum(0)
    
    vx=3.0*skew(x3,0,sigma_x2,0)
    return v_new
v_new=fonte_rayleigh()

figure(3,(12,12))
for ix in range (0,nx,2):
    subplot(121)
    plot(t_s/3600.,x[ix]+1.e+02*v_new[:,ix,0])
    subplot(122)
    vlines(x[ix],0,abs(v_new[:,ix,0]).max(),'b')
    
show()
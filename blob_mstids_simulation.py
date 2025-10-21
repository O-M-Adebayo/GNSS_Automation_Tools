#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 20:42:52 2025

@author: oluwasegun-adebayo
"""

# import numpy as np
# import matplotlib.pyplot as plt

# # -------------------------
# # Grid & Physical Parameters
# # -------------------------
# nx, ny = 101, 101
# dx = dy = 0.1
# D = 0.05                 # diffusion coefficient
# vx, vy = 0.5, 0.2        # wind/advection velocity (x, y)
# dt = 0.0008              # must satisfy CFL & diffusion stability
# steps = 600

# x = np.linspace(-5, 5, nx)
# y = np.linspace(-5, 5, ny)
# X, Y = np.meshgrid(x, y, indexing='ij')

# # -------------------------
# # Initial Condition (Gaussian blob)
# # -------------------------
# sigma = 0.5
# u = np.exp(-((X)**2 + (Y)**2) / (2 * sigma**2))

# # -------------------------
# # Time Integration
# # -------------------------
# for n in range(steps):
#     u_new = u.copy()

#     # Diffusion term (central)
#     lap = (u[2:,1:-1] + u[:-2,1:-1] + u[1:-1,2:] + u[1:-1,:-2] - 4*u[1:-1,1:-1])
#     u_new[1:-1,1:-1] += D * dt / dx**2 * lap

#     # Advection (upwind)
#     # x-derivative
#     if vx > 0:
#         dudx = (u[1:-1,1:-1] - u[:-2,1:-1]) / dx
#     else:
#         dudx = (u[2:,1:-1] - u[1:-1,1:-1]) / dx

#     # y-derivative
#     if vy > 0:
#         dudy = (u[1:-1,1:-1] - u[1:-1,:-2]) / dy
#     else:
#         dudy = (u[1:-1,2:] - u[1:-1,1:-1]) / dy

#     u_new[1:-1,1:-1] -= dt * (vx * dudx + vy * dudy)

#     # Update
#     u = u_new

# # -------------------------
# # Visualization
# # -------------------------
# plt.figure(figsize=(6,5))
# plt.contourf(X, Y, u, levels=40, cmap="rainbow")
# plt.colorbar(label="u (density)")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("2D Gaussian after advection + diffusion")
# plt.axis('equal')
# plt.show()




#%%

import numpy as np
import matplotlib.pyplot as plt

# ---- Grid & physical parameters ----
nx, ny = 151, 101     # grid points (x ~ longitude, y ~ latitude)
Lx, Ly = 1000.0, 600.0  # domain physical size in km (so dx in km)
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)

# diffusion & advection
D = 50.0            # diffusion coefficient in (km^2 / s) -- toy value
vx, vy = 50.0, 5.0  # advection velocities in km/s (toy values)
# choose stable dt by CFL and diffusion
dt_adv = 0.4 * min(dx/abs(vx) if abs(vx)>0 else np.inf, dy/abs(vy) if abs(vy)>0 else np.inf)
dt_diff = 0.25 * min(dx*dx/(4*D), dy*dy/(4*D))
dt = min(dt_adv, dt_diff, 1.0)  # limit to 1 s for safety in toy units
if dt == np.inf:
    dt = 0.1
steps = 600

# ---- Create grid ----
x = np.linspace(0, Lx, nx)
y = np.linspace(-Ly/2, Ly/2, ny)
X, Y = np.meshgrid(x, y, indexing='ij')  # X shape (nx, ny)

# ---- Initial Gaussian blob centered in domain ----
sigma_x = 60.0  # km
sigma_y = 40.0  # km
x0, y0 = Lx*0.25, 0.0  # initial blob center (quarter longitude, equator)
u0 = np.exp(-(((X - x0)**2) / (2*sigma_x**2) + ((Y - y0)**2) / (2*sigma_y**2)))

# ---- MSTID parameters (traveling sinusoid source) ----
A = 0.08            # source amplitude (adds into u over time)
wavelength = 200.0  # km
kx = 2*np.pi / wavelength
ky = 0.0            # mostly zonal MSTID (set small to include obliquity)
phase_speed = 150.0 # km/s (toy fast phase speed) - adjust as wanted

# prepare arrays
u_base = u0.copy()
u_mst = u0.copy()

# storage for center time-series
center_i = nx//4  # near initial blob center in x
center_j = ny//2
ts_base = []
ts_mst = []
time_array = []

# ---- time-stepping ----
for n in range(steps):
    t = n * dt
    time_array.append(t)
    # baseline step (no forcing)
    u = u_base
    u_new = u.copy()

    # periodic in x: we will use np.roll for neighbors in x
    up = np.roll(u, -1, axis=0)
    um = np.roll(u, 1, axis=0)
    vp = np.roll(u, -1, axis=1)
    vm = np.roll(u, 1, axis=1)

    # Laplacian (central differences)
    lap = (up + um - 2*u) / dx**2 + (vp + vm - 2*u) / dy**2
    u_new += D * dt * lap

    # upwind for advection in x
    dudx = np.zeros_like(u)
    # x-direction
    if vx >= 0:
        dudx[1:,:] = (u[1:,:] - u[:-1,:]) / dx
        dudx[0,:] = (u[0,:] - u[-1,:]) / dx  # periodic
    else:
        dudx[:-1,:] = (u[1:,:] - u[:-1,:]) / dx
        dudx[-1,:] = (u[0,:] - u[-1,:]) / dx
    # y-direction (use simple upwind with Neumann top/bottom)
    dudy = np.zeros_like(u)
    if vy >= 0:
        dudy[:,1:] = (u[:,1:] - u[:,:-1]) / dy
        dudy[:,0] = (u[:,0] - u[:,1]) / dy  # zero-gradient approx
    else:
        dudy[:,:-1] = (u[:,1:] - u[:,:-1]) / dy
        dudy[:,-1] = (u[:,-1] - u[:,-2]) / dy

    u_new -= dt * (vx * dudx + vy * dudy)

    u_base = u_new

    # record center
    ts_base.append(u_base[center_i, center_j])

    # ---- with MSTID forcing ----
    u = u_mst
    u_new = u.copy()

    up = np.roll(u, -1, axis=0)
    um = np.roll(u, 1, axis=0)
    vp = np.roll(u, -1, axis=1)
    vm = np.roll(u, 1, axis=1)

    lap = (up + um - 2*u) / dx**2 + (vp + vm - 2*u) / dy**2
    u_new += D * dt * lap

    # advection upwind
    dudx = np.zeros_like(u)
    if vx >= 0:
        dudx[1:,:] = (u[1:,:] - u[:-1,:]) / dx
        dudx[0,:] = (u[0,:] - u[-1,:]) / dx
    else:
        dudx[:-1,:] = (u[1:,:] - u[:-1,:]) / dx
        dudx[-1,:] = (u[0,:] - u[-1,:]) / dx
    dudy = np.zeros_like(u)
    if vy >= 0:
        dudy[:,1:] = (u[:,1:] - u[:,:-1]) / dy
        dudy[:,0] = (u[:,0] - u[:,1]) / dy
    else:
        dudy[:,:-1] = (u[:,1:] - u[:,:-1]) / dy
        dudy[:,-1] = (u[:,-1] - u[:,-2]) / dy

    u_new -= dt * (vx * dudx + vy * dudy)

    # MSTID source term (traveling wave)
    # S(x,y,t) = A * sin(kx*(x - c*t) + ky*y)
    phase = kx * (X - phase_speed * t) + ky * Y
    S = A * np.sin(phase)
    # add forcing as a source (scaled by dt)
    u_new += dt * S

    u_mst = u_new
    ts_mst.append(u_mst[center_i, center_j])

# ---- compute differential field ----
u_diff = u_mst - u_base

# ---- Visualization ----
plt.figure(figsize=(6,4))
plt.contourf(X, Y, u0, levels=40)
plt.colorbar(label='u (initial)')
plt.title('Initial Gaussian blob')
plt.xlabel('x (km)'); plt.ylabel('y (km)')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.contourf(X, Y, u_base, levels=40)
plt.colorbar(label='u (baseline final)')
plt.title('Baseline: advection + diffusion (final)')
plt.xlabel('x (km)'); plt.ylabel('y (km)')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.contourf(X, Y, u_mst, levels=40)
plt.colorbar(label='u (with MSTID final)')
plt.title('With MSTID forcing (final)')
plt.xlabel('x (km)'); plt.ylabel('y (km)')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
levels = np.linspace(np.min(u_diff), np.max(u_diff), 41)
plt.contourf(X, Y, u_diff, levels=levels)
plt.colorbar(label='Δu = u_MST - u_base')
plt.title('Differential field (MSTID effect)')
plt.xlabel('x (km)'); plt.ylabel('y (km)')
plt.tight_layout()
plt.show()

# time series at center
plt.figure(figsize=(6,3))
plt.plot(time_array, ts_base, label='baseline')
plt.plot(time_array, ts_mst, label='with MSTID')
plt.xlabel('time (s)'); plt.ylabel('u at center')
plt.title('Center time series')
plt.legend()
plt.tight_layout()
plt.show()

# show a zonal cut of differential at the central latitude
central_lat_idx = ny//2
plt.figure(figsize=(6,3))
plt.plot(x, u_diff[:, central_lat_idx])
plt.xlabel('x (km)'); plt.ylabel('Δu at central latitude')
plt.title('Zonal cut of Δu at y=0')
plt.tight_layout()
plt.show()


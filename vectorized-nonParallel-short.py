#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 23:51:34 2018

@author: ulphysics
"""

import numpy as np
import matplotlib.pyplot as plt

na       = 9
c        = np.array([[ 0, 0],[ 1, 0],[-1, 0],[ 0, 1],[ 0,-1],[ 1, 1],[-1,-1],[ 1,-1],[-1, 1]]) # Right to left
ai       = np.array([  0   ,   2   ,   1   ,   4   ,   3   ,   6   ,   5   ,   8   ,   7  ])
D        = 2

w0 = 4.0/9.0
w1 = 1.0/9.0
w2 = 1.0/36.0

w  =  np.array([w0,w1,w1,w1,w1,w2,w2,w2,w2])

dt = 1
dx = 1
S  = dx/dt
c1 =  1.0
c2 =  3.0/(S**2)
c3 =  9.0/(2.0*S**4)
c4 = -3.0/(2.0*S**2)

# Initialize the relaxation times
nu_f         = 0.1        # Viscosity
tau_f = nu_f * 3./(S*dt) + 0.5

nonVectorized = 0; vectorized=1
stream_opt     = nonVectorized
stream_opt     = vectorized

nt   =  100          # Number of time steps
nx   =   101          # X-axis size
nz   =   101          # Z-axis size


# new indexes for the vectorized streaming calculations
indexes = np.zeros((na,nx*nz),dtype=int)
for a in range(na):
    xArr = (np.arange(nx) - c[a][0] + nx)%nx
    zArr = (np.arange(nz) - c[a][1] + nz)%nz
    xInd,zInd=np.meshgrid(xArr,zArr)
    indTotal = zInd*nx + xInd
    indexes[a] = indTotal.reshape(nx*nz)
    
                   
# Initialize arrays

f        = np.zeros((na,nz,nx))
f_stream = np.zeros((na,nz,nx))
f_stream1 = np.zeros((na,nz,nx))
f_stream2 = np.zeros((na,nz,nx))
f_bounce = np.zeros((na,nz,nx))
f_eq     = np.zeros((na,nz,nx))
Delta_f  = np.zeros((na,nz,nx))
solid    = np.zeros((nz,nx),dtype=bool)
Solid    = np.zeros((na,nz,nx),dtype=int)
rho      = np.ones((nz,nx))
u        = np.zeros((D,nz,nx))
Pi       = np.zeros((D,nz,nx))
u2       = np.zeros((nz,nx))
xx       = np.arange(nx)
zz       = np.arange(nz)    
cu       = np.zeros((nz,nx))

# Initialize the density and the number densities
rho_0        = 1.0          # Density
rho     *= rho_0
rho[nz//2,3*nx//4] = 2*rho_0

# Initialize the density, velocity and solid boolean
#f[0:na] = 1
u2 = np.einsum('ijk,ijk->jk', u, u)
for a in np.arange(na):
    f[a] = rho * w[a]
    cu = np.einsum('i,ijk->jk', c[a], u)
    for d in np.arange(D):
        f[a] += w[a]*(c2*c[a][d]*u[d] + c3*cu**2 + c4*u2)

myimg = plt.imshow(u2);plt.colorbar()

for t in np.arange(nt+1):

    #periodic BC for f
    f[0:na,0:nz,0] = f[0:na,0:nz,-2]
    f[0:na,0:nz,-1] = f[0:na,0:nz,1]

    #   Streaming step with bounce back boundary conditions from solid surfaces
    if stream_opt == nonVectorized:
        for a in range(na):
            for x in range(1,nx-1):
                x_xa = (x - c[a][0] + nx)%nx
                for z in range(nz):
                    z_za = (z - c[a][1] + nz)%nz
                    if solid[z_za][x_xa]:
                        f_stream[a][z][x] = f[ai[a]][z][x]    # Bounce-back BC
                    else:
                        f_stream[a][z][x] = f[a][z_za][x_xa]  # Streaming step
                
    #correct but without BC
    elif stream_opt == vectorized:
        for a in np.arange(na):
            f_new = f[a].reshape(nx*nz)[indexes[a]]
            f_stream[a] = f_new.reshape(nz,nx)

    f = f_stream.copy()

#   Macroscopic properties
    rho = np.sum(f,axis=0)
    Pi = np.einsum('azx,ad->dzx',f,c)
    u[0:D]=Pi[0:D]/rho

#   Equilibrium distribution
    u2 = u[0]*u[0]+u[1]*u[1]#np.einsum('ijk,ijk->jk', u, u)#np.linalg.norm(u,axis=0)**2#
    for a in np.arange(na):
        cu = c[a][0]*u[0] + c[a][1]*u[1]#np.einsum('j,jkl->kl',c[a],u) #
        f_eq[a] = rho * w[a] * (c1 + c2*cu + c3*cu**2 + c4*u2)

#   Collision term
    Delta_f = (f_eq - f)/tau_f
    f += Delta_f

    myimg = plt.imshow(u2);plt.colorbar()
    plt.show()    

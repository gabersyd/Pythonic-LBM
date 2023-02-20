#import sys
from mpi4py import MPI
from os import remove
#import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import random
#from numpy.linalg import inv
#import cmath
#from numpy.random import randn
#from scipy.fftpack import fft, ifft
#import math
#from mpl_toolkits.mplot3d import axes3d, Axes3D
import time
#import timeit

#---------------------------------------------------------------------------------------
# Plot a field
#---------------------------------------------------------------------------------------

def plot_field(figure,title,filename,field,mean,clip_factor,gpow):

    if rank==0:
        dx = 1 ; dz = 1
        tmp = np.zeros((nz,nx))
        tmp = field
        tmp_max = np.max(tmp)
        tmp_min = np.min(tmp)
        if mean == 1: tmp     = tmp - tmp_min

        epsilon = 1.e-100
        abs_tmp = np.abs(tmp)
        max_val = np.max(abs_tmp)
        tmp     = tmp/(max_val+epsilon)
        abs_tmp = abs_tmp/(max_val+epsilon)

        tmp = max_val * tmp * np.power(abs_tmp,gpow)
        clip_val = clip_factor * max_val
        tmp = np.clip(tmp,-clip_val,clip_val)

        aspect_field = 1
        aspect_ratio = float(nx)/float(nz)
        if aspect_ratio < 0.75:
            plt.figure(figure, figsize=(3.5,6))   # 1x2
        elif aspect_ratio < 1.5:
            plt.figure(figure, figsize=(6,6))   # 1x1
        elif aspect_ratio < 2.5:
            plt.figure(figure, figsize=(6,3.5)) # 2x1
        elif aspect_ratio < 3.5:
            plt.figure(figure, figsize=(6,2.5)) # 3x1
        else:
            plt.figure(figure, figsize=(6,2))   # 4x1

        fig = plt.subplots_adjust(left=0.1, bottom=0.1, top=0.9, right=0.95)
        image = plt.imshow(tmp,extent=[0, (nx-1)*dx, (nz-1)*dz, 0],aspect=aspect_field,cmap=plt.get_cmap("jet"))
        plt.title(title)
#        cbar = plt.colorbar(image,orientation='vertical')
        plt.xlim(0, nx-1)
        plt.ylim(nz-1, 0)
        plt.xlabel('x', fontsize=16)
        plt.ylabel('z', fontsize=16)
        full_filename = filename + '.png'
        try: remove(full_filename)
        except OSError: pass
        plt.savefig(full_filename)
        plt.close()

#-------------------------------------------------------------------------------
# Put solid array from the master node to all processors
#-------------------------------------------------------------------------------

def put_solid():
    #tmp = np.zeros(nz)
    if rank == 0:
        for r in range(1,size):
            for x in range(mx[r]):
                tmp = SOLID[0:nz,(r-1)*nxi + x]                    
                comm.send(tmp,dest=r)
    if rank > 0 and rank < nr:
        for x in range(mx[rank]):
            tmp = comm.recv(source=0)
            solid[0:nz,x+1] = tmp            

#-------------------------------------------------------------------------------
# Put density in the left and right boundaries and initialize f
#-------------------------------------------------------------------------------

def put_rho_boundaries(rho_left,rho_right):
    if rank == 1:
        x = 0
        for a in range(na):
            f[a,0:nz,x+1] = rho_left  * w[a]
    elif rank == nr-1:
        x = mx[rank]
        for a in range(na):
            f[a,0:nz,x+1] = rho_right * w[a]

#-------------------------------------------------------------------------------
# Put density in the left and right boundaries and initialize f
#-------------------------------------------------------------------------------

def put_rho_boundaries_new(rho_left,rho_right):  # Has problems
    if rank == 1:
        x = 0
        f[0:na,0:nz,x+1] = rho_left  * w[0:na]
    elif rank == nr-1:
        x = mx[rank]
        f[0:na,0:nz,x+1] = rho_right * w[0:na]

#-------------------------------------------------------------------------------
# Initialization: put velocity and density from the master node to all processors
# this routine also initializes f in every node
#-------------------------------------------------------------------------------

def put_rho_u():
    if rank == 0:
        for r in np.arange(1,size): #iterate through the nodes
            x0=(r-1)*nxi #memory chunk

            # send densities to all nodes
            tmpRho = np.zeros((nz,mx[r]))
            tmpRho = RHO[0:nz,x0:x0+mx[r]]
            comm.send(tmpRho,dest=r)

            # send velocities to all nodes
            tmpU = np.zeros((nz,mx[r]))
            for d in np.arange(D):
                tmpU = U[d][0:nz,x0:x0+mx[r]]
                comm.send(tmpU,dest=r)
                
    if rank > 0 and rank < nr:
        # receive densities at the node "rank"
        tmpRho = comm.recv(source=0)
        rho[0:nz,1:mx[rank]+1]=tmpRho

        # receive velocities at the node "rank"
        for d in np.arange(D):
            tmpU = comm.recv(source=0)
            u[d][0:nz,1:mx[rank]+1]=tmpU

        # initialize the distribution function f
        u2 = np.einsum('ijk,ijk->jk', u, u)
        for a in np.arange(na):
            f[a] = rho * w[a] * c1
            cu = np.einsum('i,ijk->jk', c[a], u)
            for d in np.arange(D):
                f[a] += w[a]*(c2*c[a][d]*u[d] + c3*cu**2 + c4*u2)
   
#-------------------------------------------------------------------------------
# Put velocity from the master node to all processors and initialize f
#-------------------------------------------------------------------------------


def put_u():
    if rank == 0:
        for r in np.arange(1,size):
            tmp = np.zeros((nz,mx[r]))
            for d in np.arange(D):
                x0=(r-1)*nxi
                tmp = U[d][0:nz,x0:x0+mx[r]]
                comm.send(tmp,dest=r)
                
    if rank > 0 and rank < nr:
        for d in np.arange(D):
            tmp = comm.recv(source=0)
            u[d][0:nz,1:mx[rank]+1]=tmp

        u2 = np.einsum('ijk,ijk->jk', u, u)
        for a in np.arange(na):
            f[a] = rho * w[a] * c1
            cu = np.einsum('i,ijk->jk', c[a], u)
            for d in np.arange(D):
                f[a] += w[a]*(c2*c[a][d]*u[d] + c3*cu**2 + c4*u2)

#-------------------------------------------------------------------------------
# Put density from the master node to all processors and initialize f
#-------------------------------------------------------------------------------

def put_rho():
    if rank == 0:
#        tmp = np.zeros(nz)
        for r in range(1,size):
            tmp = np.zeros((nz,mx[r]))
            x0=(r-1)*nxi
            tmp = RHO[0:nz,x0:x0+mx[r]]
            comm.send(tmp,dest=r)
    if rank > 0 and rank < nr:
        tmp = comm.recv(source=0)
        rho[0:nz,1:mx[rank]+1]=tmp
        for a in range(na):
            f[a] = rho * w[a]

#-------------------------------------------------------------------------------
# Get density from regions on processors to a single array on the master node
#-------------------------------------------------------------------------------

def get_rho():
    if rank > 0 and rank < nr:
        print('getting rho at rank:',rank)
        comm.send(rho,dest=0)
    if rank == 0:
        print('getting rho at rank 0:',rank)
        for r in range(1,nr):
            tmp = comm.recv(source=r)
            x0=(r-1)*nxi
            RHO[0:nz,x0:x0+mx[r]] = tmp[0:nz,1:1+mx[r]]

#-------------------------------------------------------------------------------
# Get velocity from regions on processors to a single array on the master node
#-------------------------------------------------------------------------------

def get_u():
    if rank > 0 and rank < nr:
        comm.send(u,dest=0)
    if rank == 0:
        for r in range(1,nr):
            tmp = comm.recv(source=r)
            x0=(r-1)*nxi
            U[0:D,0:nz,x0:x0+mx[r]] = tmp[0:D,0:nz,1:1+mx[r]]
                        
#-------------------------------------------------------------------------------
# Get the edges to the regions in each node
#-------------------------------------------------------------------------------

def get_edges():

    if nr == 2:
        if rank == 1:
            f[0:na,0:nz,0      ] = f[0:na,0:nz,mx[1]]
            f[0:na,0:nz,mx[1]+1] = f[0:na,0:nz,   1 ]
    else:
        if rank > 0 and rank < nr:
            edgeR = np.zeros(nz)
            edgeL = np.zeros(nz)

            rr = rank+1 if rank<nr-1 else 1  #right block
            rl = rank-1 if rank>1 else nr-1  #left block
            for a in range(na):
                comm.send(f[a,0:nz,mx[rank]],dest=rr)
                comm.send(f[a,0:nz,1],dest=rl)
                
                edgeR = comm.recv(source=rr)
                f[a,0:nz,mx[rank]+1] = edgeR[0:nz]
                edgeL = comm.recv(source=rl)
                f[a,0:nz,0] = edgeL[0:nz]


#===============================================================================
# Main program
#===============================================================================

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
print('rank',rank,'active!')

na       = 9
c        = np.array([[ 0, 0],[ 1, 0],[-1, 0],[ 0, 1],[ 0,-1],[ 1, 1],[-1,-1],[ 1,-1],[-1, 1]]) # Right to left
# c       *= -1                                                                                # Left  to right
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

C_s = S/np.sqrt(3)
iCs2dt = 1/(C_s**2*dt)

#==============================================================
# Input below here

# Check for validity

run_number = 44           # Choose the run number identifier
if rank == 0:
    print("===================================================",flush=True)
    print("               run # ",run_number,flush=True)
    print("Running MPI version of the D2Q9 LBM",flush=True)
    print("===================================================",flush=True)
if size <= 1:
    print("",flush=True)
    print("ERROR: You must specify at least 2 nodes (ie. 1 for the master and at least 1 slave).",flush=True)
    print("",flush=True)
    exit()

timing = False            # Time a run
Timer  = False            # Time segments of the code
if timing: time_loops = 1 # How many time loops to do
else:      time_loops = 1

normal         = 0
gabrieleNoBC   = 4
gabrieleWithBC = 5
#stream_opt     = normal
stream_opt     = gabrieleWithBC

plots  = 0
files  = 1
none   = 2
output = plots        # Choose the output type

nt   =  2000          # Number of time steps
nx   =   1001          # X-axis size
nz   =   401          # Z-axis size
Inc  =   25          # Increment between output
Tinc =   25          # Increment between prints

# Plot options

plot_U   = True
plot_u   = True
plot_u_x = True
plot_u_z = True
plot_rho = True

# Plot scales

clip_T   = 0.1  ; gpow_T   = 0.25
clip_rho = 1.0  ; gpow_rho = 1.0
clip_u_i = 0.5  ; gpow_u_i = 1.0
clip_u   = 0.5  ; gpow_u   = 0.2
clip_U   = 0.25 ; gpow_U   = 0.1

File          = 2
point_source  = False     # Point source input
random_model  = True      # Random grain model input
void_grains   = False     # Add void grains
fill_line     = random_model # Fill lines that are one unit wide
grain_density = 0.9       # Grain density

fixed_edges   = False     # Fixed upper and lower boundaries

RHO_0        = 1          # Density
rho_left     = 1.5*RHO_0  # Density of left  boundary
rho_right    = 0.5*RHO_0  # Density of right boundary
nu_f         = 0.1        # Viscosity

# Input above here
#==============================================================

# Initialize x-size on slave nodes

nx_i = (nx+(size-1)*2)//(size-1)
while (nx_i-2)*(size-1)<nx: nx_i += 1
nxi = nx_i - 2 ; nr  = nx//nxi ; dnx = nx%nxi
if dnx > 0: nr += 1
nr += 1
if rank == 0:
    print ("----------------------------------------------------",flush=True)
    print ("nx_i = ",nx_i,", nxi = ",nxi,", nr  = ",nr,", dnx  = ",dnx,flush=True)

mx = np.zeros(size,dtype=int)
for r in range(nr): mx[r] = nxi
if dnx > 0: mx[nr-1] = dnx
if rank==0:
    print ("----------------------------------------------------",flush=True)
    print ("Node 0 = Master + ",nr-1," slave nodes out of ",size-1," total",flush=True)
    print ("----------------------------------------------------",flush=True)
    print ("Partitioning x axis of size nx = ",nx," as follows:",flush=True)
    print ("----------------------------------------------------",flush=True)
    x_width_total = 0
    for r in range(1,nr):
        print ("Node # ",r," : x_width = ",mx[r],flush=True)
        x_width_total += mx[r]
    print ("----------------------------------------------------",flush=True)
    print ("TOTAL = ",x_width_total,flush=True)
    print ("----------------------------------------------------",flush=True)

#   Set increment arrays
_x_xa_ = np.zeros((na,nx_i),dtype=int)
_z_za_ = np.zeros((na,nz  ),dtype=int)
for a in range(na):
    for x in range(nx_i):
        x_xa = (x - c[a][0] + nx_i)%nx_i
        _x_xa_[a][x] = x_xa
    for z in range(nz):
        z_za = (z - c[a][1] + nz)%nz
        _z_za_[a][z] = z_za

# new indexes for the vectorized streaming calculations
indexes = np.zeros((na,nx_i*nz),dtype=int)
for a in range(na):
    xArr = (np.arange(nx_i) - c[a][0] + nx_i)%nx_i
    zArr = (np.arange(nz) - c[a][1] + nz)%nz
    xInd,zInd=np.meshgrid(xArr,zArr)
    indTotal = zInd*nx_i + xInd
    indexes[a] = indTotal.reshape(nx_i*nz)
    
#indexes = np.zeros((na,nx_i*nz),dtype=int)
#for a in range(na):
#    for x in range(nx_i):
#        for z in range(nz):
#            ind=z*nx_i+x
#            x_xa = (x - c[a][0] + nx_i)%nx_i
#            z_za = (z - c[a][1] + nz)%nz
#            indexes[a,ind]=z_za*nx_i+x_xa
                   
# Initialize arrays

f        = np.zeros((na,nz,nx_i))
f_stream = np.zeros((na,nz,nx_i))
f_stream1 = np.zeros((na,nz,nx_i))
f_stream2 = np.zeros((na,nz,nx_i))
f_bounce = np.zeros((na,nz,nx_i))
f_eq     = np.zeros((na,nz,nx_i))
Delta_f  = np.zeros((na,nz,nx_i))
solid    = np.zeros((nz,nx_i),dtype=bool)
Solid    = np.zeros((na,nz,nx_i),dtype=int)
rho      = np.ones((nz,nx_i))
tau_f    = np.zeros((nz,nx_i))
u        = np.zeros((D,nz,nx_i))
Pi       = np.zeros((D,nz,nx_i))
u2       = np.zeros((nz,nx_i))
xx       = np.arange(nx_i)
zz       = np.arange(nz)    
cu       = np.zeros((nz,nx_i))

# Initialize the density and the number densities

if rank == 0:
    SOLID    = np.zeros((nz,nx))
    RHO      = np.ones((nz,nx))
    U        = np.zeros((D,nz,nx))
    RHO     *= RHO_0

    SOLID[0,:]=fixed_edges    # Solid upper boundary
    SOLID[nz-1,:]=fixed_edges # Solid lower boundary

    if point_source:
        RHO[nz//2][3*nx//4] = 2*RHO_0

    number = (nx//2+nz//2)//10
    nx_g = nx//number
    nz_g = nx//number
    if random_model == True:
        n_c = np.int(nx_g*nz_g)
        ix_c = np.zeros((n_c,D),np.int)
        ir_c = np.zeros(n_c,np.int)
        n_c = 0
        random.seed(0)
        for ix_g in np.arange(nx_g):
            for iz_g in np.arange(nz_g):
                ix_c[n_c][0] = 0.999999*(ix_g+random.random())*number
                ix_c[n_c][1] = 0.999999*(iz_g+random.random())*number
                ir_c[n_c]    = grain_density*number/2*(np.random.normal(1.0,0.25))
                n_c += 1
        print("Made random model",flush=True)
    else:
        n_c = 0

    for i_c in np.arange(n_c):
        print("Making grain ",i_c," of ",n_c," grains",flush=True)
        x_c = ix_c[i_c][0]
        z_c = ix_c[i_c][1]
        r_c = ir_c[i_c]
        for x in np.arange(max(0,np.int(x_c-3*nx_g)),min(nx,np.int(x_c+3*nx_g))):
            del_x = x-x_c
            for z in np.arange(max(0,np.int(z_c-3*nz_g)),min(nz,np.int(z_c+3*nz_g))):
                del_z = z-z_c
                if np.sqrt(del_x**2+del_z**2) < r_c:
                    SOLID[z][x] = True
    if random_model == True: print("Made random matrix",flush=True)
    if random_model == True and void_grains:
        r_max = number//2
#        if r_max>10: r_max = 10
#        if r_max>6: r_max = 6
        r_min = max(2,min(4,r_max//2))
        plot_field(6,"Solid matrix",str(run_number)  + "_solid_%02d" %(r_max+1),SOLID,1,1,1)
        for r in range(r_max,r_min,-1):
            print("Making void grains at r = ",r,flush=True)
            for x in range(nx):
                for z in range(nz):
                    if not SOLID[z][x]:
                        void = True
                        for xx in range(x-r_max,x+r_max+1):
                            xxx = (xx+nx)%nx
                            del_x = xx-x
                            for zz in range(z-r_max,z+r_max+1):
                                zzz = (zz+nz)%nz
                                del_z = zz-z
                                rr = np.sqrt(del_x**2+del_z**2)
                                if rr < r and SOLID[zzz][xxx]: void = False
                        if (void and r > min(r_max/2,r_min+2)) or (void and random.random() < 0.33):
                            print("    Void grain found at (x,z) = (",x,",",z,")",flush=True)
                            for xx in range(x-3*r//2,x+3*r//2+1):
                                xxx = (xx+nx)%nx
                                del_x = xx-x
                                for zz in range(z-3*r//2,z+3*r//2+1):
                                    zzz = (zz+nz)%nz
                                    del_z = zz-z
                                    rr = np.sqrt(del_x**2+del_z**2)
                                    if rr < r-2.5: SOLID[zzz][xxx] = True
            plot_field(7,"Solid matrix",str(run_number)  + "_solid_%02d" %(r),SOLID,1,1,1)
    if random_model == True and fill_line:
       for x in range(1,nx-1):
           for z in range(1,nz-1):
               if (not SOLID[z][x]) and ((SOLID[z-1][x] and SOLID [z+1][x]) or (SOLID[z][x-1] and SOLID[z][x+1])):
                   SOLID[z][x] = True
    if random_model == True:
        file = str(run_number)  + "_SOLID.txt"
        print ("Saving  SOLID file = ",file)
        np.savetxt(file,SOLID,fmt="%12.8f")
    elif random_model == File:
        file = str(run_number)  + "_SOLID.txt"
        print ("Loading SOLID file = ",file)
        SOLID = np.loadtxt(file)

# Initialize the density, velocity and solid boolean on the slave nodes

f[0:na] = 1
put_rho_u()
#put_rho()
#put_u()
put_solid()
for a in np.arange(na):
    for z in np.arange(nz):
        for x in range(nx_i):
            if solid[_z_za_[a][z]][_x_xa_[a][x]]:
                Solid[a][z][x] = 1
            else:
                Solid[a][z][x] = 0

# Initialize the relaxation times

tau_f = nu_f * iCs2dt + 0.5

if rank == 0:
    print("---------------------------------------------------",flush=True)
    print("Starting time steps of the LBM",flush=True)
    print("---------------------------------------------------",flush=True)

Time_0 = time.time()
totalTimeEdges = 0.; totalTimePutRho = 0.; 
totalTimeStream = 0.; totalTimeMacro = 0.; 
totalTimeEquil = 0.; totalTimeCollis = 0.;

for time_loop in np.arange(time_loops):
  for t in np.arange(nt+1):
    if t % Tinc == 0:
    #if 1:    
       if rank==0: print("run = ",run_number,", Time = ",t,flush=True)

#   Get the edges of the MPI domains

    t0 = time.time()
    get_edges()
    Dt = time.time()-t0;  totalTimeEdges+=Dt             
    if Timer and t % Tinc == 0: print('Time edges  = ',Dt,flush=True)

#   Density left and right boundary conditions

    t0 = time.time()
    if not point_source or timing: put_rho_boundaries(rho_left,rho_right)
    Dt = time.time()-t0;  totalTimePutRho+=Dt             
    if Timer and t % Tinc == 0: print('Time put_rho= ',Dt,flush=True)

#   Streaming step with bounce back boundary conditions from solid surfaces

    t0 = time.time()

    if stream_opt == normal:
        for a in range(na):
            for x in range(1,nx_i-1):
                x_xa = (x - c[a][0] + nx_i)%nx_i
                for z in range(nz):
                    z_za = (z - c[a][1] + nz)%nz
                    if solid[z_za][x_xa]:
                        f_stream[a][z][x] = f[ai[a]][z][x]    # Bounce-back BC
                    else:
                        f_stream[a][z][x] = f[a][z_za][x_xa]  # Streaming step
                        
    #correct but without BC
    elif stream_opt == gabrieleNoBC:
        for a in np.arange(na):
            f_new = f[a].reshape(nx_i*nz)[indexes[a]]
            f_stream[a] = f_new.reshape(nz,nx_i)

    #apparently correct with BC
    elif stream_opt == gabrieleWithBC:
        for a in np.arange(na):          # Right to left
            f_new = f[a].reshape(nx_i*nz)[indexes[a]]
            f_bounce = f[ai[a]]  #bounce back           
            f_stream1[a] = Solid[a]*f_bounce + (1-Solid[a])*f_new.reshape(nz,nx_i)
        for a in np.arange(na-1,-1,-1): # Left  to right
            f_new = f[a].reshape(nx_i*nz)[indexes[a]]
            f_bounce = f[ai[a]]  #bounce back           
            f_stream2[a] = Solid[a]*f_bounce + (1-Solid[a])*f_new.reshape(nz,nx_i)
        f_stream = f_stream1  # Right to left
        f_stream = f_stream2  # Left  to right
        f_stream = (f_stream1 + f_stream2)/2 # Average of (L->R + R->L)

    f = f_stream
    Dt = time.time()-t0;  totalTimeStream+=Dt             
    if Timer and t % Tinc == 0: print('Time stream = ',Dt,flush=True)

#   Macroscopic properties

    t0 = time.time()
    rho = np.sum(f,axis=0)
    Pi = np.einsum('azx,ad->dzx',f,c)
    u[0:D]=Pi[0:D]/rho
    Dt = time.time()-t0;  totalTimeMacro+=Dt             
    if Timer and t % Tinc == 0: print('Time macro  = ',Dt,flush=True)

#   Equilibrium distribution

    t0 = time.time()
    u2 = u[0]*u[0]+u[1]*u[1]#np.einsum('ijk,ijk->jk', u, u)#np.linalg.norm(u,axis=0)**2#
    for a in np.arange(na):
        cu = c[a][0]*u[0] + c[a][1]*u[1]#np.einsum('j,jkl->kl',c[a],u) #
        f_eq[a] = rho * w[a] * (c1 + c2*cu + c3*cu**2 + c4*u2)
    Dt = time.time()-t0;  totalTimeEquil+=Dt             
    if Timer and t % Tinc == 0: print('Time equil  = ',Dt,flush=True)

#   Collision term

    t0 = time.time()
    Delta_f = (f_eq - f)/tau_f
    f      += Delta_f
    Dt = time.time()-t0;  totalTimeCollis+=Dt                 
    if Timer and t % Tinc == 0: print('Time collis = ',Dt,flush=True)
        
#   Output
    print('before output: t % Inc', t % Inc, 'timing',timing,'output',output,'rank',rank,'nr',nr)
    if t % Inc == 0 and not timing and output == plots:
        t0 = time.time() 
        print('going to get rho')
        get_rho()
        print('getting rho')
        get_u  ()
        print('getting u')
        if Timer and t % Tinc == 0: print('Get variables = ',Dt,flush=True)
        t0 = time.time()
        print('start plotting')
        if rank == 0:
            print(t,run_number,RHO ,RHO_0,clip_rho,gpow_rho)
            if plot_rho: plot_field(1,"rho (t=%d)" %(t),str(run_number)  + "_rho_" + "%07d" %(t),RHO ,RHO_0,clip_rho,gpow_rho)
            if plot_u_x: plot_field(2,"u_x (t=%d)" %(t),str(run_number)  + "_u_x_" + "%07d" %(t),U[0],0    ,clip_u_i,gpow_u_i)
            if plot_u_z: plot_field(3,"u_z (t=%d)" %(t),str(run_number)  + "_u_z_" + "%07d" %(t),U[1],0    ,clip_u_i,gpow_u_i)
            if plot_u:   plot_field(4,"|u| (t=%d)" %(t),str(run_number)  + "_uu_" + "%07d" %(t),np.sqrt(U[0]**2+U[1]**2),-1,clip_u,gpow_u)
            if plot_U:   plot_field(5,"|u| (t=%d)" %(t),str(run_number)  + "_UU_" + "%07d" %(t),np.sqrt(U[0]**2+U[1]**2),-1,clip_U,gpow_U)
            if t == 0:
                plot_field(5,"Solid matrix",str(run_number)  + "_solid",SOLID,1,1,1)
        if Timer and t % Tinc == 0: print('Plot fields   = ',Dt,flush=True)
    elif t % Inc == 0 and not timing and output == files and rank > 0 and rank < nr:
        file = './'+str(run_number)  + "_rho_" + "%07d" %(t) + "_rank_%03d" %(rank)
        print('saving '+file)
        np.save(file,rho)#,fmt="%12.8f")
        for d in range(D):
            file = str(run_number)  + "_ux_" + "%07d" %(t) + "_rank_%03d" %(rank)
            print('saving '+file)
            np.save(file,u[0])#,fmt="%12.8f")
            file = str(run_number)  + "_uz_" + "%07d" %(t) + "_rank_%03d" %(rank)
            print('saving '+file)
            np.save(file,u[1])#,fmt="%12.8f")

    if t == nt:
        Time_1 = time.time()
        Dt =Time_1 - Time_0
        if rank == 0:
            print("===================================================",flush=True)
            print("run # ",run_number,flush=True)
            print("---------------------------------------------------",flush=True)
            print("Elapsed time = %8.4f" %(Dt),", # nodes = ",size,flush=True)
            totalTime = totalTimeEdges + totalTimePutRho + totalTimeStream + totalTimeMacro + totalTimeEquil + totalTimeCollis
            print("Edges:   %14.6f secs"%(totalTimeEdges),   " = %4.1f"%(100*totalTimeEdges /totalTime),"%",flush=True)
            print("Put rho: %14.6f secs"%(totalTimePutRho),  " = %4.1f"%(100*totalTimePutRho/totalTime),"%",flush=True)
            print("Stream:  %14.6f secs"%(totalTimeStream),  " = %4.1f"%(100*totalTimeStream/totalTime),"%",flush=True)
            print("Macro:   %14.6f secs"%(totalTimeMacro),   " = %4.1f"%(100*totalTimeMacro /totalTime),"%",flush=True)
            print("Equil:   %14.6f secs"%(totalTimeEquil),   " = %4.1f"%(100*totalTimeEquil /totalTime),"%",flush=True)
            print("Collis:  %14.6f secs"%(totalTimeCollis),  " = %4.1f"%(100*totalTimeCollis/totalTime),"%",flush=True)
            print("---------------------------------------------------",flush=True)
            print("TOTAL:   %14.6f secs"%(totalTime),        " = %4.1f"%(100*totalTime/totalTime),      "%",flush=True)
            print("===================================================",flush=True)

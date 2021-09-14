# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 22:26:54 2021

@author: parham
"""
"""
Parameters
----------
 pump_pos: *numpy array*:
        A system indicating the placement and the amplitude of the pumps,
        while also indicating the number of modes. In case one has a system 
        with n modes it is expected to have an input array of length 2*(2*n-1),
        as to consider the space between all modes as a pump position.The 
        negative one is used to insure that the first and last index are
        always modes.In this array the first half of the values are choosen 
        for the amplitude of the pumps in their respective possition 
        relative the previously  explained indexing leaving all the pumpless
        points equal to "0".The second half is then choosen to express the
        phase of each case make sure that they are in the same position as the
        the amplitudes with the same indexing.
        
        example:
            pump=[0,0,30e3*1e-9,30e3*1e-9,30e3*1e-9,0,0,0,0,0,np.pi,0,0,0]

n_edges : *int*
        Indicates the size of the square cluster based on the number of 
        edges on each side of the square. 
        for example:
            n_edges=2 for a square of 4 points.
"""
import numpy as np
import networkx as nx
from scipy.optimize import minimize
import scipy.linalg as LA    
import matplotlib.pyplot as plt 

##reorder cov matrix between (x1,p1,...,xn, pn) and (x1,...xn, p1,...,pn)
def permute_matrix(matrix, d):
    perm=np.zeros([2*d,2*d])
    for i in range(0,d):
        perm[2*i,i] = 1
        perm[2*i+1,d+i] = 1
    return perm.transpose() @ matrix @ perm

def covariance(pump,n):
    pumpinfo=np.array_split(pump,2)
    pump_amp=pumpinfo[0]
    pump_phase=pumpinfo[1]
    #number of modes in covariance matrix
    pump_length=len(pump_amp)
    n_modes=(pump_length+1)//2
     #mode index
     #generates a comb of modes centered on index 0 
     #the index for the pump positions considered remember the firt and last one 
     #in this index and mode index are the same "point".
     
    n_index = np.arange(pump_length)//2
     
     #the indexes for modes in the system
    mode_index=n_index[0::2]
    #--------3 wave mixing -----------------------
    #pump_frequencies (kHz)
    p1_freq = 2*  2*np.pi*4.3023e6
    p2_freq = 2*  2*np.pi*4.2977e6
    
    #mode frequencies, set up frequency comb
    center_comb_freq = (p2_freq + p1_freq)/4
    comb_det = np.abs(p2_freq - p1_freq)/4
    
    comb_freqs = center_comb_freq + (mode_index - n_modes//2 )*comb_det
    
    
    #-----------set pump positions and strengths -------------------
    #pump mode positions, enter a numpy array
    pump_pos=np.asarray(np.where(pump_amp!=0))
    pump_position=pump_pos[0]
    #effective pump amplitude  
    pump_strength = pump_amp[pump_position]*np.exp(1j*pump_phase[pump_position])
    
    #-------------- resonances and losses --------------------------
    
    # #JPA resonance (kHz)
    omega_0 = np.array([ 2*np.pi*4.3e6])
    
    #SAW resonances (kHz)
    #omega_0p = comb_freqs + 3*2*np.pi
    
    #dissipation rates (kHZ)
    gamma = 2*np.pi*20
    
    
    #bias point
    phi_DC = 0.6*np.pi
    
    #------generate scattering matrix--------------
    #using only up to first order
    #mode matrix
    M = np.zeros(( n_modes*2, n_modes*2 ), dtype = complex)
    
    #print("mode index", mode_index)
    #print("pump_position", pump_position)
    
    for s in mode_index: # mode index is "regular array"
    
        idler_positions = pump_position - s
        #print("idler pos", idler_positions, "n_idx", s)
    
        #constructing the mode coupling matrix M
        for pos, i1 in enumerate(idler_positions):
            #print("POS",pos, i1)
            if np.any(mode_index == i1):
                #print("n %d, pos %d, i1 %d"%(s, pos, i1))
                #coupling strength given below
                conver_rate = -pump_strength[pos]
                M[s,i1+n_modes] = conver_rate
                M[s+n_modes, i1] = -np.conjugate(conver_rate)
    
        diag_val = comb_freqs[n] - omega_0 + 1j*gamma/2
        diag_val_conj = -np.conjugate(diag_val)
    
        M[s, s] = diag_val
        M[s+n_modes, s+n_modes] = diag_val_conj
    
    #print(np.real(M[:n_modes,n_modes:]))
    #exit()
    #print(M)
    
    K = np.identity(2*n_modes)*np.sqrt(gamma)
    
    #calculating the scattering matrix
    M_inv = np.linalg.inv(M)
    S = 1j*K.dot(M_inv).dot(K) - np.identity(2*n_modes)
    
    # ----------------------- calculate the covariance matrix ----------------
    #convert scattering basis to the x, p basis
    #the vector ordering becomes (x1, x2, ...., p1,p2,....)
    X = np.zeros_like(S, dtype = complex)
    for ii in range(n_modes):
        X[ii, ii], X[ii, ii+n_modes] = np.ones(2)
        X[ii+n_modes, ii] = -1j
        X[ii+n_modes, ii+n_modes] = 1j
    
    #transform scattering matrix
    X_inv = np.linalg.inv(X)
    S_xp = X.dot(S).dot(X_inv)
    
    #check if it is symplectic
    # symp = np.block( [ [np.zeros((n_modes, n_modes)), np.identity(n_modes)], [-1*np.identity(n_modes), np.zeros((n_modes, n_modes))] ] )
    
    # symp2 = S_xp.dot(symp).dot(np.transpose(S_xp))
    
    # result = np.all(np.round( np.real(symp2) , 3) == symp )
    # print(' The transformation is symplectic: ' + str(result) )
    
    # calculate resulting covariance matrix
    #input noise:
    n_noise = 0
    V_input = (2*n_noise + 1)*np.identity(2*n_modes)
    #noise from loss channel, we can assume it is at the same temperature as the input channel
    #V_loss = V_input
    #output statistics
    V_outputxp=np.real(S_xp.dot(V_input).dot( np.transpose(S_xp) ))
    #rewrite in xxpp
    V_output=permute_matrix(V_outputxp, n_modes )
    
    G=nx.grid_2d_graph(n, n, periodic=False, create_using=None) 
    adj=nx.to_numpy_matrix(G)
    # $U=(I+iV)(V^2+I)^{-1/2}=AB^{-1/2}
    d=len(adj)
    I=np.eye(d)
    Binv=adj@adj+I
    B=np.linalg.inv(Binv)
    X1=LA.sqrtm(B)
    Y1=adj*X1
    S=np.block([[X1,-Y1],[Y1,X1]])
    #tau=1/gamma
    #R=np.mean(pump_amp*tau)
    R=0.5
    Q=np.block([[np.exp(-2*R)*I,np.zeros_like(I)],[np.zeros_like(I),np.exp(2*R)*I]])
    SQ=np.dot(S,Q)
    V_teori=np.dot(SQ,S.T)
    #calculate the differance and norm.
    V_diff=np.linalg.norm(V_teori-V_output)
    return [V_output,V_teori,V_diff]  

def build_square_adjacancey(n):
    G=nx.grid_2d_graph(n, n, periodic=False, create_using=None) 
    A=nx.to_numpy_matrix(G)
    return A

def calculateNullifiers(covar,n):
    d=n**2
    V=build_square_adjacancey(n)
    nullifiers = np.zeros(d)
    for i in range(0,d):
        #calculate nullifier i
        nullifier=covar[d+i,d+i]
        for k in range(0,d):
            nullifier+=-2*V[i,k]*covar[d+i,k]
            for j in range(0,d):
                nullifier+=V[i,j]*V[i,k]*covar[j,k]
        nullifiers[i]=nullifier
        
    return nullifiers

def optimizationfunction(pump,n):
    C=covariance(pump,n)
    covar=C[0]
    nullifiers=calculateNullifiers(covar,n)
    #nullifiers=calculateNullifiers(covar,n)
    limit=np.max(nullifiers)
    return limit

def optimizer(n,pump):
    #pump=[0,0,30e3*1e-6,30e3*1e-6,30e3*1e-6,0,0,0,0,0,np.pi,0,0,0]
    n_2=n**2
    pmp_tot=4*n_2-2
    boundarys=[]
    boundary=(0,20)
    for bnds in range(pmp_tot):
        if bnds<b1:
            boundarys.append(boundary)
        else:
            boundarys.append((0,2*np.pi))
    
    out=minimize(optimizationfunction,pump,n,method ='nelder-mead',bounds=boundarys)
    pump1=out.x
    return pump1
n=2
n_2=n**2
pmp_tot=4*n_2-2
b1=2*n_2-1
pump=np.zeros(pmp_tot)
pump1=optimizer(n,pump)
pumpamp=pump1[:b1]
pumpphi=pump1[b1:]
range1=np.arange(b1)
C1=covariance(pump1,n)
output=C1[0]   
plt.bar(range1,pumpamp)
plt.ylabel("The amplitude")
plt.xlabel("The pump index")
plt.xticks(range1)
plt.title('pump positions and amplitudes ')


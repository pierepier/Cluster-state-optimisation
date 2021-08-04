# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 11:16:34 2021

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
def covariance_differance(pump,n):
#pump=[0,0,30e3*1e-9,30e3*1e-9,30e3*1e-9,0,0,0,0,0,np.pi,0,0,0]  
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
    
    n_index = np.arange(pump_length) - pump_length//2
    
    #the indexes for modes in the system
    mode_index=n_index[0::2]//2
    #--------3 wave mixing -----------------------
    #pump_frequencies (GHz---->KHz)
    p1_freq = 2*  2*np.pi*4.3023*1e6
    p2_freq = 2*  2*np.pi*4.2977*1e6
    
    #mode frequencies, set up frequency comb
    
    center_comb_freq = (p2_freq + p1_freq)/4
    comb_det = np.abs(p2_freq - p1_freq)/4
    comb_freqs = center_comb_freq + (mode_index)*comb_det
    
    
    #--------4 wave mixing -----------------------
    #pump_frequencies (GHz)
    # p1_freq = 2*np.pi*4.3023
    # p2_freq = 2*np.pi*4.2977
    
    # #mode frequencies, set up frequency comb
    # center_comb_freq = (p2_freq + p1_freq)/2
    # comb_det = np.abs(p2_freq - p1_freq)/2
    # comb_freqs = center_comb_freq + n_index*comb_det
    
    
    #-----------set pump positions and strengths -------------------
    #pump mode positions, enter a numpy array
    
    pump_pos1=np.where(pump_amp!=0)
    ps =np.subtract(pump_pos1,pump_length//2)
    pump_position=ps[0]
    #center pump amp and phase (enter a complex number)
    #effective pump amplitude
    # phi1 = 30e3*np.exp(1j*np.pi)*1e-9
    # phi2 = 30e3*np.exp(0)*1e-9
    # phi3 = 30e3*np.exp(0)*1e-9
       
    pump_strength =pump_amp[pump_pos1]*np.exp(1j*pump_phase[pump_pos1])
    # pump_position = np.array([-1])
    # pump_strength = np.array([phi4])
    
    #-------------- resonances and losses --------------------------
    
    # #JPA resonance (GHz)
    # omega_0p = np.array([ 2*np.pi*4.3*10**9 /10**9 ])
    
    #SAW resonances (GHz------>KHz)
    omega_0p = comb_freqs + 3e3*1e6/1e9*2*np.pi
    
    #dissipation rates (GHZ------->KHz)
    gamma_int = 2*np.pi*20e3*1e6/1e9 * np.ones_like(omega_0p)
    #leave at 0, calculation does not take internal noise channels into account
    # gamma_ext = 2*np.pi*300e6 /1e9 * np.ones_like(omega_0p)
    gamma_ext = 2*np.pi*20e3*1e6/1e9 * np.ones_like(omega_0p)
    gamma_p = gamma_ext + gamma_int
    
    
    
    
    #------generate scattering matrix--------------
    #using only up to first order 
    #mode matrix
    M = np.zeros(( n_modes*2, n_modes*2 ), dtype = complex)
    
    for i, s in enumerate(mode_index):
        
        idler_positions = (2*pump_position) - s
        
        if len(omega_0p) == 1:
            omega_0 = omega_0p[0]
            gamma = gamma_p[0]
            K = np.identity(2*n_modes)*np.sqrt(gamma_ext)
            K_loss = np.identity(2*n_modes)*np.sqrt(gamma_int)
        else:
            omega_0 = omega_0p[i]
            gamma = gamma_p[i]
            K = np.block( [[ np.diag(np.sqrt(gamma_ext)), np.zeros((n_modes, n_modes)) ], 
                            [ np.zeros((n_modes, n_modes)), np.diag(np.sqrt(gamma_ext)) ]] )
            K_loss = np.block( [[ np.diag(np.sqrt(gamma_int)), np.zeros((n_modes, n_modes)) ], 
                                [ np.zeros((n_modes, n_modes)), np.diag(np.sqrt(gamma_int)) ]] )
    
        
        #constructing the mode coupling matrix M
            for pos, i1 in enumerate(idler_positions):
                if np.any(mode_index == i1):
                    j = np.where(mode_index == i1)[0][0]
                    #coupling strength given below
                    #coupling depends on system, for SAW see the off-diagonal terms below eq. S35 in SAW supplementary
                    #should be much smaller than the linewidth
                    # conver_rate = -omega_0/4*np.tan(phi_DC)*pump_strength[pos]*np.pi #for JPA
                    conver_rate = -pump_strength[pos]
                    M[i,j+n_modes] = conver_rate
                    M[i+n_modes, j] = -np.conjugate(conver_rate)
        
    
        diag_val = comb_freqs[i] - omega_0 + 1j*gamma/2
        diag_val_conj = -np.conjugate(diag_val)
       
        M[i, i] = diag_val
        M[i+n_modes, i+n_modes] = diag_val_conj
        
        #figure 1   
        # # fig, ax = plt.subplots(figsize=(8,8))
        # # ax.imshow(np.real(M[::2,::2]))
        
        # grid_arr = np.arange(0, 2*n_modes, 1)
        # plt.xticks( grid_arr)
        # plt.yticks( grid_arr)
        # # ax.set_xticks(grid_arr, minor=True)
        # # ax.set_yticks(grid_arr, minor=True)
        # ax.grid(True, which='major', axis='both', linestyle='-', color='w', linewidth=0.5)
        # # K = np.identity(2*n_modes)*np.sqrt(gamma_ext)
        # ax.set_xticklabels(np.concatenate((mode_index,mode_index)))
        # ax.set_yticklabels(np.concatenate((mode_index,mode_index)))
        # #calculating the scattering matrix
        
    M_inv = np.linalg.inv(M)
    S = 1j*K.dot(M_inv).dot(K) - np.identity(2*n_modes)
    #for the internal loss channel, we need a second term
    Sp = 1j*K.dot(M_inv).dot(K_loss)
        
        
    #plot gain across the frequency comb
    # gain_curve = np.abs(np.diag(S))[:n_modes]
    #figure 2
    # fig, ax = plt.subplots(1)
    # ax.plot( comb_freqs/(2*np.pi), 20*np.log10(gain_curve) )
        
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
    Sp_xp = X.dot(Sp).dot(X_inv)
        
        
        #check if it is symplectic
    symp = np.block( [ [np.zeros((n_modes, n_modes)), np.identity(n_modes)], [-1*np.identity(n_modes), np.zeros((n_modes, n_modes))] ] )
        
    symp2 = S_xp.dot(symp).dot(np.transpose(S_xp))
        
    result = np.all(np.round( np.real(symp2) , 3) == symp )
    print(' The transformation is symplectic: ' + str(result) )
        
        # calculate resulting covariance matrix
        #input noise:
    n_noise = 0
    V_input = (2*n_noise + 1)*np.identity(2*n_modes)
    #noise from loss channel, we can assume it is at the same temperature as the input channel
    V_loss = V_input
    #output statistics
    noise_from_loss_channel = np.real(Sp_xp.dot(V_loss).dot( np.transpose(Sp_xp) ))
    V_output = np.real(S_xp.dot(V_input).dot( np.transpose(S_xp) )) + noise_from_loss_channel
    #calculate the theoretical covariance matrix
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
    return V_diff

def covariances(pump,n):
#pump=[0,0,30e3*1e-9,30e3*1e-9,30e3*1e-9,0,0,0,0,0,np.pi,0,0,0]  
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
    
    n_index = np.arange(pump_length) - pump_length//2
    
    #the indexes for modes in the system
    mode_index=n_index[0::2]
    #--------3 wave mixing -----------------------
    #pump_frequencies (GHz----->KHz)
    p1_freq = 2*  2*np.pi*4.3023*1e6
    p2_freq = 2*  2*np.pi*4.2977*1e6
    
    #mode frequencies, set up frequency comb
    
    center_comb_freq = (p2_freq + p1_freq)/4
    comb_det = np.abs(p2_freq - p1_freq)/4
    comb_freqs = center_comb_freq + (mode_index//2)*comb_det
    
    
    #--------4 wave mixing -----------------------
    #pump_frequencies (GHz)
    # p1_freq = 2*np.pi*4.3023
    # p2_freq = 2*np.pi*4.2977
    
    # #mode frequencies, set up frequency comb
    # center_comb_freq = (p2_freq + p1_freq)/2
    # comb_det = np.abs(p2_freq - p1_freq)/2
    # comb_freqs = center_comb_freq + n_index*comb_det
    
    
    #-----------set pump positions and strengths -------------------
    #pump mode positions, enter a numpy array
    
    pump_pos1=np.where(pump_amp!=0)
    ps =np.subtract(pump_pos1,pump_length//2)
    pump_position=ps[0]
    #enter pump amp and phase (enter a complex number)
    #effective pump amplitude
    # phi1 = 30e3*np.exp(1j*np.pi)*1e-9
    # phi2 = 30e3*np.exp(0)*1e-9
    # phi3 = 30e3*np.exp(0)*1e-9
       
    pump_strength = pump_amp[pump_pos1]*np.exp(1j*pump_phase[pump_pos1])
    # pump_position = np.array([-1])
    # pump_strength = np.array([phi4])
    
    #-------------- resonances and losses --------------------------
    
    # #JPA resonance (GHz)
    # omega_0p = np.array([ 2*np.pi*4.3*10**9 /10**9 ])
    
    #SAW resonances (GHz------>KHz)
    omega_0p = comb_freqs + 3e3*1e6/1e9*2*np.pi
    
    #dissipation rates (GHZ------->KHz)
    gamma_int = 2*np.pi*20e3*1e6/1e9 * np.ones_like(omega_0p)
    #leave at 0, calculation does not take internal noise channels into account
    # gamma_ext = 2*np.pi*300e6 /1e9 * np.ones_like(omega_0p)
    gamma_ext = 2*np.pi*20e3*1e6/1e9 * np.ones_like(omega_0p)
    gamma_p = gamma_ext + gamma_int
    
    
    
    
    #------generate scattering matrix--------------
    #using only up to first order 
    #mode matrix
    M = np.zeros(( n_modes*2, n_modes*2 ), dtype = complex)
    
    for i, s in enumerate(mode_index):
        
        idler_positions = (2*pump_position) - s
        
        if len(omega_0p) == 1:
            omega_0 = omega_0p[0]
            gamma = gamma_p[0]
            K = np.identity(2*n_modes)*np.sqrt(gamma_ext)
            K_loss = np.identity(2*n_modes)*np.sqrt(gamma_int)
        else:
            omega_0 = omega_0p[i]
            gamma = gamma_p[i]
            K = np.block( [[ np.diag(np.sqrt(gamma_ext)), np.zeros((n_modes, n_modes)) ], 
                            [ np.zeros((n_modes, n_modes)), np.diag(np.sqrt(gamma_ext)) ]] )
            K_loss = np.block( [[ np.diag(np.sqrt(gamma_int)), np.zeros((n_modes, n_modes)) ], 
                                [ np.zeros((n_modes, n_modes)), np.diag(np.sqrt(gamma_int)) ]] )
    
        
        #constructing the mode coupling matrix M
            for pos, i1 in enumerate(idler_positions):
                if np.any(mode_index == i1):
                    j = np.where(mode_index == i1)[0][0]
                    #coupling strength given below
                    #coupling depends on system, for SAW see the off-diagonal terms below eq. S35 in SAW supplementary
                    #should be much smaller than the linewidth
                    # conver_rate = -omega_0/4*np.tan(phi_DC)*pump_strength[pos]*np.pi #for JPA
                    conver_rate = -pump_strength[pos]
                    M[i,j+n_modes] = conver_rate
                    M[i+n_modes, j] = -np.conjugate(conver_rate)
        
    
        diag_val = comb_freqs[i] - omega_0 + 1j*gamma/2
        diag_val_conj = -np.conjugate(diag_val)
       
        M[i, i] = diag_val
        M[i+n_modes, i+n_modes] = diag_val_conj
        
        #figure 1   
        # # fig, ax = plt.subplots(figsize=(8,8))
        # # ax.imshow(np.real(M[::2,::2]))
        
        # grid_arr = np.arange(0, 2*n_modes, 1)
        # plt.xticks( grid_arr)
        # plt.yticks( grid_arr)
        # # ax.set_xticks(grid_arr, minor=True)
        # # ax.set_yticks(grid_arr, minor=True)
        # ax.grid(True, which='major', axis='both', linestyle='-', color='w', linewidth=0.5)
        # # K = np.identity(2*n_modes)*np.sqrt(gamma_ext)
        # ax.set_xticklabels(np.concatenate((mode_index,mode_index)))
        # ax.set_yticklabels(np.concatenate((mode_index,mode_index)))
        # #calculating the scattering matrix
        
    M_inv = np.linalg.inv(M)
    S = 1j*K.dot(M_inv).dot(K) - np.identity(2*n_modes)
    #for the internal loss channel, we need a second term
    Sp = 1j*K.dot(M_inv).dot(K_loss)
        
        
    #plot gain across the frequency comb
    # gain_curve = np.abs(np.diag(S))[:n_modes]
    #figure 2
    # fig, ax = plt.subplots(1)
    # ax.plot( comb_freqs/(2*np.pi), 20*np.log10(gain_curve) )
        
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
    Sp_xp = X.dot(Sp).dot(X_inv)
        
        
        #check if it is symplectic
    symp = np.block( [ [np.zeros((n_modes, n_modes)), np.identity(n_modes)], [-1*np.identity(n_modes), np.zeros((n_modes, n_modes))] ] )
        
    symp2 = S_xp.dot(symp).dot(np.transpose(S_xp))
        
    result = np.all(np.round( np.real(symp2) , 3) == symp )
    print(' The transformation is symplectic: ' + str(result) )
        
        # calculate resulting covariance matrix
        #input noise:
    n_noise = 0
    V_input = (2*n_noise + 1)*np.identity(2*n_modes)
    #noise from loss channel, we can assume it is at the same temperature as the input channel
    V_loss = V_input
    #output statistics
    noise_from_loss_channel = np.real(Sp_xp.dot(V_loss).dot( np.transpose(Sp_xp) ))
    V_output = np.real(S_xp.dot(V_input).dot( np.transpose(S_xp) )) + noise_from_loss_channel
    #calculate the theoretical covariance matrix
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
    return[V_output,V_teori]

n=2
n_2=n**2
pmp_tot=4*n_2-2
pump=np.zeros(pmp_tot)
b1=2*n_2-1
boundarys=[]
boundary=(0,20)
for bnds in range(pmp_tot):
    if bnds<b1:
        boundarys.append(boundary)
    else:
        boundarys.append((0,2*np.pi))

out=minimize(covariance_differance,pump,n,method = 'Nelder-Mead',bounds=boundarys)
pump1=out.x
pumpamp=pump1[:b1]
pumpphi=pump1[b1:]
range1=np.arange(b1)-(b1//2)
plt.bar(range1,pumpamp)
plt.ylabel("The amplitude")
plt.xlabel("The pump index")
plt.xticks(range1)
plt.title('pump positions and amplitudes ')
V_general=covariances(pump1,n)
V_output=V_general[0]
V_teori=V_general[1]
V_diff=np.linalg.norm(V_teori-V_output)

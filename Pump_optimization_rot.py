# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 12:10:38 2021

@author: parham
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 10:39:19 2021

@author: parham
"""

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
        the amplitudes with the same indexing. we remove the two edge cases and
        rotation factors that are equal to the number of modes.
        
        example:
            pump=[0,20,20,20,0,0,0,np.pi,0,0,0,0,0,0]

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
import cma 
import ipdb


##reorder cov matrix between (x1,p1,...,xn, pn) and (x1,...xn, p1,...,pn)
def permute_matrix(matrix, d):
    perm=np.zeros([2*d,2*d])
    for i in range(0,d):
        perm[2*i,i] = 1
        perm[2*i+1,d+i] = 1
    return perm.transpose() @ matrix @ perm

def rotation_matrix(angles):
    nr_modes = len(angles)
    single_mode_rotations = []
    for i in np.arange(nr_modes):
        single_mode_rotations.append([[np.cos(angles[i]),-np.sin(angles[i])],[np.sin(angles[i]),np.cos(angles[i])]])
    return LA.block_diag(*single_mode_rotations)



def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))





def covariance(pump,n): 
    modenumber=n**2
    pump_info=np.array(pump)
    arbetrary=len(pump)
    rotation=pump_info[(arbetrary-modenumber):]
    pump_info=pump_info[:(arbetrary-modenumber)]
    pumpbase=np.array_split(pump_info,2)
    pump_amp=pumpbase[0]
    pump_phase=pumpbase[1]
    pump_amp=np.append(pump_amp,[0])
    pump_amp=np.append([0],pump_amp)
    pump_phase=np.append(pump_amp,[0])
    pump_phase=np.append([0],pump_amp)
    pump_length=len(pump_amp)
    n_modes=(pump_length+1)//2
     #mode index
     #generates a comb of modes centered on index 0 
     #the index for the pump positions considered remember the firt and last one 
     #in this index and mode index are the same "point".
     
    n_index = np.arange(pump_length)//2
     
     #the indexes for modes in the system
    mode_index=n_index[1::2]
    #--------3 wave mixing -----------------------
    #pump_frequencies (kHz)
    p1_freq = 2* 2*np.pi*4.3023e6
    p2_freq = 2*  2*np.pi*4.2977e6
    
    #mode frequencies, set up frequency comb
    center_comb_freq = (p2_freq + p1_freq)/4
    comb_det = np.abs(p2_freq - p1_freq)/4
    
    comb_freqs = center_comb_freq + (mode_index - n_modes )*comb_det
    
    
    #-----------set pump positions and strengths -------------------
    #pump mode positions, enter a numpy array
    pump_pos=np.asarray(np.where(pump_amp!=0))
    pump_position=pump_pos[0]
    #effective pump amplitude  
    pump_strength = np.absolute(pump_amp[pump_position])*np.exp(1j*pump_phase[pump_position])
    
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
            if np.any(mode_index == i1) and s != i1 :
                #print("n %d, pos %d, i1 %d"%(s, pos, i1))
                #coupling strength given below
                conver_rate = -pump_strength[pos]
                #conver_rate = -omega_0/4*np.tan(phi_DC)*pump_strength[s]*np.pi
                M[s,i1+n_modes] = conver_rate
                M[s+n_modes, i1] = -np.conjugate(conver_rate)
    
        diag_val = comb_freqs[s] - omega_0 + 1j*gamma/2
        diag_val_conj = -np.conjugate(diag_val)
    
        M[s, s] = diag_val
        M[s+n_modes, s+n_modes] = diag_val_conj
  
    
    # ax.set_title('weighted covariance matrix ON')
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
    rot=rotation_matrix(rotation)
    V_output=rot@V_output@rot.T
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
    R=1.2
    Q=np.block([[np.exp(-2*R)*I,np.zeros_like(I)],[np.zeros_like(I),np.exp(2*R)*I]])
    SQ=np.dot(S,Q)
    V_teori=NormalizeData(np.dot(SQ,S.T))
    #calculate the differance and norm.
    V_diff=np.linalg.norm(V_teori-V_output)
    
    return [V_output,V_teori,V_diff,rotation]  


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
  
    # pump2=[0,30,-30,30,0]
    # C2=covariance(pump2,n)
    C=covariance(pump,n)
    diff=C[2]
    #diff=np.linalg.norm(C2[0]-C[0])
    #covar=C[0]
    #nullifiers=calculateNullifiers(covar,n)
    #limit=np.max(nullifiers)
    #return limit
    return diff


def optimizer(n,pump): 
    #out=minimize(optimizationfunction,pump,n,method ='Nelder-Mead')
    #pump1=out.x
    # out = cma.fmin2(optimizationfunction, pump, 0.2,args=(n,))
    # pump1=out[0]
    es = cma.CMAEvolutionStrategy(pump, 0.2,{'bounds': [[0], [50]]})
    es.optimize(optimizationfunction,args=(n,))
    out=es.result_pretty()
    pump1=out[0]
    return pump1


def Mmatrixmaker(pumpsmall,n):
    pumpinfo=np.array_split(pumpsmall,2)
    pump_amp=pumpinfo[0]
    pump_amp=np.append(pump_amp,[0])
    pump_amp=np.append([0],pump_amp)

    pump_phase=pumpinfo[1]
    pump_phase=np.append(pump_phase,[0])
    pump_phase=np.append([0],pump_phase)
    pump_length=len(pump_amp)
    n_modes=(pump_length+1)//2
    #number of modes in covari
     #mode index
     #generates a comb of modes centered on index 0 
     #the index for the pump positions considered remember the firt and last one 
     #in this index and mode index are the same "point".
     
    n_index = np.arange(pump_length)//2
     
     #the indexes for modes in the system
    mode_index=n_index[0::2]
    #--------3 wave mixing -----------------------
    #pump_frequencies (kHz)
    p1_freq = 2*  2*np.pi*4.3023
    p2_freq = 2*  2*np.pi*4.2977
    
    #mode frequencies, set up frequency comb
    center_comb_freq = (p2_freq + p1_freq)/4
    comb_det = np.abs(p2_freq - p1_freq)/4
    
    comb_freqs = center_comb_freq + (mode_index - n_modes )*comb_det
    
    
    #-----------set pump positions and strengths -------------------
    #pump mode positions, enter a numpy array
    pump_pos=np.asarray(np.where(pump_amp!=0))
    pump_position=pump_pos[0]
    #effective pump amplitude  
    pump_strength = np.absolute(pump_amp[pump_position])*np.exp(1j*pump_phase[pump_position])
    
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
            if np.any(mode_index == i1) and s != i1 :
                #print("n %d, pos %d, i1 %d"%(s, pos, i1))
                #coupling strength given below
                conver_rate = -pump_strength[pos]
                #conver_rate = -omega_0/4*np.tan(phi_DC)*pump_strength[s]*np.pi
                M[s,i1+n_modes] = conver_rate
                M[s+n_modes, i1] = -np.conjugate(conver_rate)
    
        diag_val = comb_freqs[s] - omega_0 + 1j*gamma/2
        diag_val_conj = -np.conjugate(diag_val)
    
        M[s, s] = diag_val
        M[s+n_modes, s+n_modes] = diag_val_conj
  
    return M

n=2
n_2=n**2
pmp_tot=4*n_2+(n_2-6)
b1=2*n_2-1
pump=np.zeros(pmp_tot)
pump1=optimizer(n,pump)
#pump1=[0,20,20,20,0,0,0,np.pi,0,0,0,0,0,0]


C1=covariance(pump1,n)
Teori_case=C1[1]
output=C1[0]
framerot=C1[3]
print(framerot)

null=calculateNullifiers(C1[0],n)
nullref=calculateNullifiers(C1[1], n)
nulldif=np.divide(null,nullref)
pump2=pump1[:(pmp_tot-n_2)]
pumpinfo=np.array_split(pump2,2)
pump_amp=pumpinfo[0]
pump_amp=np.append(pump_amp,[0])
pump_amp=np.append([0],pump_amp)
pump_phase=pumpinfo[1]
pump_phase=np.append(pump_phase,[0])
pump_phase=np.append([0],pump_phase) 

M=Mmatrixmaker(pump2,n)
M1=np.abs(M)
range1=np.arange(b1)


plt.figure(0)
plt.bar(range1,pump_amp)
plt.ylabel("The amplitude")
plt.xlabel("The pump index")
plt.xticks(range1)
plt.title('pump positions and amplitudes ')

plt.figure(1)
plt.imshow(M1)
plt.colorbar()
plt.show()
plt.title("M_matrix")


plt.figure(3)
plt.imshow(output)
plt.colorbar()
plt.show()
plt.title("output")

plt.figure(2)
plt.imshow(Teori_case)
plt.colorbar()
plt.show()
plt.title("Target case")
    

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 12:56:29 2021

@author: parham
"""

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
            pump=[0,0,30,30,30,0,0,0,0,0,np.pi,0,0,0]

n_edges : *int*
        Indicates the size of the square cluster based on the number of 
        edges on each side of the square. 
        for example:
            n_edges=2 for a square of 4 points.
"""
import numpy as np
import matplotlib.pyplot as plt 

##reorder cov matrix between (x1,p1,...,xn, pn) and (x1,...xn, p1,...,pn)
def permute_matrix(matrix, d):
    perm=np.zeros([2*d,2*d])
    for i in range(0,d):
        perm[2*i,i] = 1
        perm[2*i+1,d+i] = 1
    return perm.transpose() @ matrix @ perm


#The menicuci version for a square 2x2 cluster
n=2
pumpamp=[0,30,-30,30,0]
pump=np.array(pumpamp)
pump_length=len(pump)
pump_phase=np.zeros(pump_length)
for phs in range (pump_length):
    if pump[phs]<0:
        pump_phase[phs]=np.pi
   
#number of modes in covariance matrix

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

pump_pos=np.asarray(np.where(pump!=0))
pump_position=pump_pos[0]
#effective pump amplitude  
pump_strength = pump[pump_position]*np.exp(1j*pump_phase[pump_position])

#-------------- resonances and losses --------------------------

# #JPA resonance (kHz)
omega_0 = np.array([ 2*np.pi*4.3e6])

#SAW resonances (kHz)
#omega_0p = comb_freqs + 3*2*np.pi

#dissipation rates (kHZ)
gamma = 2*np.pi*20


#bias point
#phi_DC = 0.6*np.pi

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
            M[s,i1+n_modes] = conver_rate
            M[s+n_modes, i1] = -np.conjugate(conver_rate)

    diag_val = comb_freqs[n] - omega_0 + 1j*gamma/2
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
    
  
plt.figure(1)
plt.imshow(V_output)
plt.colorbar()
plt.show()
plt.title("Menicucci covariance")





 
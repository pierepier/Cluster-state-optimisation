# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 13:51:07 2021

@author: parham
"""
import numpy as np
import networkx as nx
import scipy.linalg as LA    
import matplotlib.pyplot as plt 

#The ryuji version of a square 2x2 cluster

def build_square_adjacancey(n):
    G=nx.grid_2d_graph(n, n, periodic=False, create_using=None) 
    A=nx.to_numpy_matrix(G)
    return A

n=2
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
V_teori=10*np.dot(SQ,S.T)

plt.figure(1)
plt.imshow(V_teori)
plt.colorbar()
plt.show()
plt.title("Ryuji covariance")

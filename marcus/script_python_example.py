# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 14:39:02 2022

@author: au621298
"""
#import sys
#sys.modules[__name__].__dict__.clear()

import numpy as np
import scipy.io as sc

model = 'digitbench_frame.mat'
#model = 'digitbench_beam.mat'
mat = sc.loadmat(model)

K = mat["K"]
gDof = mat["gDof"] # MATLAB assumes array numbering starting from 1 !!!
nDof = np.size(K,1) # number of DoFs

# list of BCs
if model == 'digitbench_beam.mat':
    BCset = np.array([[2,1,0],[2,2,0],[2,3,0],[1,2,1]], dtype=int)
    BCval = np.array([[0.0],[0.0],[0.0],[100.0]], dtype=float)

if model == 'digitbench_frame.mat':
    P = 100 
    M = 1000
    BCset = np.array([[1,1,0],[1,2,0],[2,1,0],[2,2,0],[3,1,0],[3,2,0],[9,2,1],[9,3,1]], dtype=int)
    BCval = np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[P],[M]], dtype=float)

# default BCs (force controlled DoFs with zero force applied)   
fKnown = np.array(range(nDof),dtype=np.intp())
uKnown = np.array([],dtype=np.intp())

# initialization of force and displacement vectors (how to inizialize array)
f = np.zeros([nDof,1],dtype=float)
u = np.zeros([nDof,1],dtype=float)

for i in range(np.size(BCset,0)):

    # index of the DoF to which BC is applied
    index = gDof[BCset[i,1]-1,BCset[i,0]-1]-1
       
    if BCset[i,2] == 0:
        print('disp BC')
        u[index] = BCval[i,0]
        uKnown = np.append(uKnown,np.intp(index))
        fKnown = np.setdiff1d(fKnown,np.intp(index))
    
    if BCset[i,2] == 1:
        print('force BC')
        f[index] = BCval[i,0]
        
# partitioning of the stiffness matrix (how you select sub-matrices)
Kuu = K[np.ix_(uKnown,uKnown)]
Kfu = K[np.ix_(fKnown,uKnown)]
Kuf = K[np.ix_(uKnown,fKnown)]
Kff = K[np.ix_(fKnown,fKnown)]

# initialization of displacement and force vectors
uu = u[np.ix_(uKnown,[0])]
ff = f[np.ix_(fKnown,[0])]

# displacements of unconstrained DoFs (how to solve a linear system)
uf = np.linalg.solve(Kff,ff - np.linalg.multi_dot((Kfu,uu)))
 
# reactions of constrained DoFs (how to perform matrix-vector product)
fu = np.linalg.multi_dot((Kuu,uu)) + np.linalg.multi_dot((Kuf,uf))
    
# filling of displacement and force vectors
u[np.ix_(fKnown,[0])] = uf
f[np.ix_(uKnown,[0])] = fu
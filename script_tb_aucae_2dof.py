#%% Test setup
import matplotlib.pyplot as plt
from module_python_fem import *
from script_manipulator_sympy import *

## Initialization
# Parameters
E = 200e9   #Young's modulus [Pa]
nu = 0.3    #Poisson's ratio 
h = 150.0 #0.1     #Height of beam [mm]
b = 150.0 #0.1     #Width of beam [mm]
Lh = 1642.0   #Length of horizontal beams [mm]
Lv = 1786.0   #Length of vertical beams [mm]
t = 4.0   #Thickness of beam [mm]
I = 1/12 * (b*h**3 - (b-2*t)*(h-2*t)**3)     #Second moment of inertia [m4]
A = b*h - (b-2*t)*(h-2*t)     #Area [m2]
type = 'beam'   #beam or shell

params = {
  'youngs_modulus': E,
  'inertia': I,
  'area' : A,
  'poisson' : nu,
  'thickness' : t}

# test of a single element

#node_test = np.array([[1,-1,-1],[2,1,-1],[3,1,1],[4,-1,1]])

#elem_test = membrane2D(params=params,node=node_test)

#ke_rank = np.linalg.matrix_rank(elem_test.Ke) #should be 5
#print(ke_rank)

#U, S, VT = np.linalg.svd(elem_test.Ke)

# last three modes are rigid
#phi = VT.T
#phi = phi[:, -3:]

#sca = 1.0
#mode = 1
#plt.plot(node_test[:,1],node_test[:,2],'ob')
#plt.plot(node_test[:,1]+sca*phi[0::2,mode],node_test[:,2]+sca*phi[1::2,mode],'xr')


BCm = []                                            #Leading nodes
BCs = []                                            #Following nodes

if type == 'shell':
    coord = np.zeros((19,3),dtype=int)
    eTop = np.zeros((6,4),dtype=int)

    coord[0,:] = [0, 0.0, 0.0]
    coord[1,:] = [1, 1842.0, 0.0]
    coord[2,:] = [2, 0.0, h]
    coord[3,:] = [3, h, h]
    coord[4,:] = [4, 996.0, h]
    coord[5,:] = [5, 996+h, h]
    coord[6,:] = [6, 1842.0-h, h]
    coord[7,:] = [7, 1842.0, h]
    coord[8,:] = [8, h, 817.0+h]
    coord[9,:] = [9, 996.0, 817.0+h]
    coord[10,:] = [10, 996.0+h, 817.0+h]
    coord[11,:] = [11, h, 817.0+2*h]
    coord[12,:] = [12, 996.0+h, 817.0+2*h]
    coord[13,:] = [13, 0.0, 1786.0+h]
    coord[14,:] = [14, h, 1786.0+h]
    coord[15,:] = [15, 1842.0-h, 1786.0+h]
    coord[16,:] = [16, 1742.0, 1686+h]
    coord[17,:] = [17, 0.0, 1786.0+2*h]
    coord[18,:] = [18, 1842.0, 1786.0+2*h]

    eTop[0,:] = [0, 1, 7, 2]
    eTop[1,:] = [2, 3, 14, 13]
    eTop[2,:] = [4, 5, 10, 9]
    eTop[3,:] = [6, 7, 16, 15]
    eTop[4,:] = [8, 10, 12, 11]
    eTop[5,:] = [13, 16, 18, 17]

    # Parameters
    nE = eTop.shape[0]              #number of beams
    kr = 1                           #number of elements in which the shorter side will be divided
    lg = kr*100                       #number of elements in which the longer side will be divided
    NoN = (kr+1)*(lg+1)              #number of nodes per beam
    NoE = kr*lg                      #number of elements per beam
    NL = np.zeros((NoN*nE, 3),dtype="float")    #extended node list
    EL = np.zeros((NoE*nE, 4),dtype="int")      #extended element list 

    # Boundary conditions
    BCm = []   #Leading nodes
    BCs = []   #Following nodes

    for row in range(0,eTop.shape[0]):
        ## Nodes
        n = 0       #this will allow us to go through rows in node list

        lh = coord[eTop[row,1],1]-coord[eTop[row,0],1]
        lv = coord[eTop[row,2],2]-coord[eTop[row,0],2]
        if lh > lv:
            p = lg          #number of elements in which the horizontal line will be divided
            m = kr          #number of elements in which the vertical line will be divided
        elif lv > lh:
            p = kr          #number of elements in which the horizontal line will be divided
            m = lg          #number of elements in which the vertical line will be divided

        for i in range(0,m+1):
            for j in range(0,p+1):
                NL[row*NoN+n,0] = row*NoN + n
                NL[row*NoN+n,1] = coord[eTop[row,0],1] + j*lh/p 
                NL[row*NoN+n,2] = coord[eTop[row,0],2] + i*lv/m
                n += 1 

    
    # Boundary conditions
    BCd = np.array([[0,1],[0,2],[NoE,1],[NoE,2]])           #Node, dof with blocked displacements
    #force_dof = np.array([[NoE*2-p,1,force]])               #Node, dof, value of acting force
    force_dof = np.array([[7*NoE,0,8.62406523e-6],[10*NoE,1,-1842.55524]])
    disp_dof = np.zeros((0,3))


elif type == 'beam':
    # Node geometry
    NL = np.zeros((11,3),dtype="float")
    NL[0,:] = [0, 0.0, 0.0]
    NL[1,:] = [1, 917.0, 0.0]
    NL[2,:] = [2, Lv, 0.0]
    NL[3,:] = [3, Lv, Lh-330]
    NL[4,:] = [4, Lv, Lh]
    NL[5,:] = [5, 0.0, Lh]
    NL[6,:] = [6, 0.0, 996.0]
    NL[7,:] = [7, 660.0, 996.0]
    NL[8,:] = [8, 917.0, 996.0]
    NL[9,:] = [9, 917, 996.0-21.0]
    NL[10,:] = [10, 917.0, 675.0]
    #NL[11,:] = [11, Lv-500, Lh-330]

    # Elements
    EL = np.zeros((12,2),dtype="int")
    for i in range(len(NL)-1):
        EL[i,0] = NL[i,0]
        EL[i,1] = NL[i+1,0]
    EL[10,0] = NL[10,0]
    EL[10,1] = NL[1,0]
    EL[11,0] = NL[0,0]
    EL[11,1] = NL[6,0]
    #EL[12,0] = NL[3,0]
    #EL[12,1] = NL[11,0]

    # Boundary conditions
    BCd = np.array([[0,0],[0,1],[0,2],[2,0],[2,1],[2,2]])           #Node, dof with blocked displacements
    force_dof = np.array([[7,0,8.62406523e-6],[10,1,-1842.55524]])         #Node, dof, value of acting force
    disp_dof = np.zeros((0,3)) #np.array([[5,1,10]])


## Main code
myModel = model(EL,NL,BCd,params,BCm,BCs,force_dof,disp_dof,type)

x1 = -5
x2 = 2000
y1 = -5
y2 = 2000
direction = 0 #0=sigma_xx, 1=sigma_yy, 2=gamma_xy
myModel.compute_displacement()
myModel.plot(1e9,x1,x2,y1,y2)

#%%
myModel.compute_stress()

#resulting force into the frame at the actuator connection
if myModel.l.shape[0] == 1:
    fx = 0.0
else:
    fx = myModel.l[-2]
tau,theta1,theta2 = calculate_reaction_force(myModel.u[-2],myModel.u[-1],fx,myModel.l[-1])

#plot
myModel.plot(1e5,x1,x2,y1,y2)
myModel.plot_stresses(x1,x2,y1,y2,direction)

#%% Main code
myModel = model(EL,NL,BCd,params,BCm,BCs,force_dof,disp_dof,type)

x1 = -5
x2 = 2000
y1 = -5
y2 = 2000
myModel.compute_displacement()

#plot
myModel.plot(1e7,x1,x2,y1,y2)
#plt.plot(NL[7,1],NL[7,2], 'rx')
#plt.plot(NL[10,1],NL[10,2], 'bx')



# %% Validating the dense2sparse function 

test_matrix1 = np.zeros((6,6))
test_matrix1[0,0] = 2
test_matrix1[0,3] = 1
test_matrix1[1,2] = 1
test_matrix1[2,3] = 2
test_matrix1[3,0] = 1
test_matrix1[3,4] = 3
test_matrix1[4,1] = 2
test_matrix1[4,5] = 2
test_matrix1[5,2] = 1
test_matrix1[5,4] = 1

test_matrix2 = np.zeros((6,6))
test_matrix2[0,0] = 2
test_matrix2[0,4] = 1
test_matrix2[1,3] = 1
test_matrix2[2,5] = 2
test_matrix2[3,1] = 1
test_matrix2[3,3] = 3
test_matrix2[4,0] = 2
test_matrix2[4,4] = 2
test_matrix2[5,2] = 1
test_matrix2[5,5] = 1

print(test_matrix1)
print('=====')
print(test_matrix2)

test_sparse1 = dense2sparse(test_matrix1)
tets_sparse2 = dense2sparse(test_matrix2)

test_trans = test_sparse1.transpose()

test = test_sparse1 + tets_sparse2

print('=============================================')
print(test)
print('-----')
print(test_trans)

#%%


test_matrix1 = np.zeros((6,6))
test_matrix1[0,0] = 2
test_matrix1[0,3] = 1
test_matrix1[1,2] = 1
test_matrix1[2,3] = 2
test_matrix1[3,0] = 1
test_matrix1[3,4] = 3
test_matrix1[4,1] = 2
test_matrix1[4,5] = 2
test_matrix1[5,2] = 1
test_matrix1[5,4] = 1

total_matrix = np.zeros((6,6))
total_sp = np.zeros((6,6))

for i in range(len(test_matrix1)):
    test_matrix1[i,1] = i 
    test_matrix1[4,i] = 1

    test_sp = dense2sparse(test_matrix1)

    total_matrix += test_matrix1
    total_sp += test_sp

print(total_matrix)
print('------------')
print(total_sp)

#%% difference append and concatenate

import numpy as np

result_append = np.zeros((0,3))
for i in range(0,5):
    # Example arrays
    #arr1 = np.array([1, 2*i, 3])
    arr2 = np.array([4*i, 5, 6+i])

    # Using np.append
    result_append = np.append(result_append, arr2)
    print("np.append result:", result_append)

    # Using np.concatenate
    result_concatenate = np.concatenate((result_append, arr2))
    print("np.concatenate result:", result_concatenate)

#%%

Zu = sp.sparse.csr_matrix((np.ones((5)),(np.ones((5)),np.ones((5)))),shape=(10,10))

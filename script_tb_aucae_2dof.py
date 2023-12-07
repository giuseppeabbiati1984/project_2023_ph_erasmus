#%%
from module_python_fem import *

#%% Test setup

## Initialization
# Parameters
E = 200e9   #Young's modulus [Pa]
nu = 0.3    #Poisson's ratio 
h = 100.0 #0.1     #Height of beam [mm]
b = 100.0 #0.1     #Width of beam [mm]
Lh = 1642.0   #Length of horizontal beams [mm]
Lv = 1786.0   #Length of vertical beams [mm]
t = 4.0   #Thickness of beam [mm]
I = 1/12 * (b*h**3 - (b-2*t)*(h-2*t)**3)     #Second moment of inertia [m4]
A = b*h     #Area [m2]
Ndof = 3    #Number of degrees of freedom
p = 10       #number of elements in which the horizontal line will be divided
m = 10       #number of elements in which the vertical line will be divided
force = 100
type = 'shell'   #beam or shell

params = {
  'youngs_modulus': E,
  'inertia': I,
  'area' : A,
  'p' : p,
  'm' : m,
  'poisson' : nu,
  'thickness' : t}

#%% test of a single element

node_test = np.array([[1,-1,-1],[2,1,-1],[3,1,1],[4,-1,1]])

elem_test = membrane2D(params=params,node=node_test)

ke_rank = np.linalg.matrix_rank(elem_test.Ke) #should be 5
print(ke_rank)

U, S, VT = np.linalg.svd(elem_test.Ke)

# last three modes are rigid
phi = VT.T
phi = phi[:, -3:]

sca = 1.0
mode = 1
plt.plot(node_test[:,1],node_test[:,2],'ob')
plt.plot(node_test[:,1]+sca*phi[0::2,mode],node_test[:,2]+sca*phi[1::2,mode],'xr')


#%% Test setup cont'd

BCm = []                                            #Leading nodes
BCs = []                                            #Following nodes

if type == 'shell':
    p = 10       #number of elements in which the horizontal line will be divided
    m = 10       #number of elements in which the vertical line will be divided
    NoN = (p+1)*(m+1)                #number of nodes
    NoE = p*m                        #number of elements
    NL = np.zeros((13*NoN, 3),dtype="int")    #extended node list
    EL = np.zeros((13*NoE, 4),dtype="int")                  #extended element list 
    coord = np.zeros((19,3),dtype=int)
    eTop = np.zeros((6,4),dtype=int)

    coord[0,:] = [0, 0.0, 0.0]
    coord[1,:] = [1, 1742.0, 0.0]
    coord[2,:] = [2, 0.0, h]
    coord[3,:] = [3, h, h]
    coord[4,:] = [4, 996.0, h]
    coord[5,:] = [5, 996+h, h]
    coord[6,:] = [6, 1742.0-h, h]
    coord[7,:] = [7, 1742.0, h]
    coord[8,:] = [8, h, 817.0+h]
    coord[9,:] = [9, 996.0, 817.0+h]
    coord[10,:] = [10, 996.0+h, 817.0+h]
    coord[11,:] = [11, h, 817.0+2*h]
    coord[12,:] = [12, 996.0+h, 817.0+2*h]
    coord[13,:] = [13, 0.0, 1686.0+h]
    coord[14,:] = [14, h, 1686.0+h]
    coord[15,:] = [15, 1742.0-h, 1686.0+h]
    coord[16,:] = [16, 1742.0, 1686+h]
    coord[17,:] = [17, 0.0, 1686.0+2*h]
    coord[18,:] = [18, 1742.0, 1686.0+2*h]

    eTop[0,:] = [0, 1, 7, 2]
    eTop[1,:] = [2, 3, 14, 13]
    eTop[2,:] = [4, 5, 10, 9]
    eTop[3,:] = [6, 7, 16, 15]
    eTop[4,:] = [8, 10, 12, 11]
    eTop[5,:] = [13, 16, 18, 17]

    for row in range(0,eTop.shape[0]):
        ## Nodes
        n = 0       #this will allow us to go through rows in node list
        lh = coord[eTop[row,1],1]-coord[eTop[row,0],1]
        lv = coord[eTop[row,2],2]-coord[eTop[row,1],2]

        for i in range(0,m+1):
            for j in range(0,p+1):
                NL[row*NoN+n,0] = row*NoN + n
                NL[row*NoN+n,1] = coord[eTop[row,0],1] + j*lh/p 
                NL[row*NoN+n,2] = coord[eTop[row,0],2] + i*lv/m
                n += 1 

        ## Elements
        for i in range(0,m):
            for j in range(0,p):
                if j == 0:      #most left elements
                    EL[row*NoE+(i*p), 0] = row*NoN + i*(p+1)
                    EL[row*NoE+(i*p), 1] = EL[row*NoE, 0] + 1
                    EL[row*NoE+(i*p), 3] = row*NoN + (i+1)*(p+1)
                    EL[row*NoE+(i*p), 2] = EL[row*NoE, 3] + 1
                else:
                    EL[row*NoE+(i*p)+j, 0] = EL[row*NoE+(i*p)+(j-1), 1]
                    EL[row*NoE+(i*p)+j, 1] = EL[row*NoE+(i*p)+j, 0] + 1
                    EL[row*NoE+(i*p)+j, 3] = EL[row*NoE+(i*p)+(j-1), 2]
                    EL[row*NoE+(i*p)+j, 2] = EL[row*NoE+(i*p)+j, 3] + 1
    
    # Boundary conditions
    BCd = np.array([[0,1],[0,2],[p,1],[p,2]])           #Node, dof with blocked displacements
    force_dof = np.array([[NoE*2-p,1,force]])         #Node, dof, value of acting force
    disp_dof = np.array([[NoE*2-p,1,10]])

    ## Plot
    plt.plot([NL[EL[:,0],1],NL[EL[:,1],1]],[NL[EL[:,0],2],NL[EL[:,1],2]])
    plt.plot([NL[EL[:,1],1],NL[EL[:,2],1]],[NL[EL[:,1],2],NL[EL[:,2],2]])
    plt.plot([NL[EL[:,2],1],NL[EL[:,3],1]],[NL[EL[:,2],2],NL[EL[:,3],2]])
    plt.plot([NL[EL[:,0],1],NL[EL[:,3],1]],[NL[EL[:,0],2],NL[EL[:,3],2]])


elif type == 'beam':
    # Node geometry
    NL = np.zeros((12,3))
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
    NL[11,:] = [11, Lv-500, Lh-330]

    # Elements
    EL = np.zeros((13,2),dtype=int)
    for i in range(len(coord)-2):
        EL[i,0] = NL[i,0]
        EL[i,1] = NL[i+1,0]
    EL[10,0] = NL[10,0]
    EL[10,1] = NL[1,0]
    EL[11,0] = NL[0,0]
    EL[11,1] = NL[6,0]
    EL[12,0] = NL[3,0]
    EL[12,1] = NL[11,0]

    # Boundary conditions
    BCd = np.array([[0,1],[0,2],[2,1],[2,2]])           #Node, dof with blocked displacements
    force_dof = np.array([[11,1,force],[11,2,force]])         #Node, dof, value of acting force
    disp_dof = np.array([[5,1,10]])

    # Plot the initial geometry
    plt.plot([NL[EL[:,0],1],NL[EL[:,1],1]],[NL[EL[:,0],2],NL[EL[:,1],2]])

#print(NL[80:200,:])


## Main code
myModel = model(EL,NL,BCd,params,BCm,BCs,force_dof,disp_dof,type)
pos = NL
for i in range(NL.shape[0]):
    for j in range(myModel.df_dof.shape[0]):
        if pos[i,0] == myModel.df_dof[j,0]:
            if myModel.df_dof[j,1] == 1:
                pos[i,1] = pos[i,1] + myModel.df_dof[j,2]
            elif myModel.df_dof[j,1] ==2:
                pos[i,2] = pos[i,2] + myModel.df_dof[j,2]
            elif myModel.df_dof[j,1] == 3:
                pos[i,1] = pos[i,1] + myModel.df_dof[j,2]*(pos[i,1]-pos[i,0])
                pos[i,2] = pos[i,2] + myModel.df_dof[j,2]*(pos[i,2]-pos[i,1])

#plot new configuration
plt.plot([pos[EL[:,0],1],pos[EL[:,1],1]],[pos[EL[:,0],2],pos[EL[:,1],2]])
plt.plot([pos[EL[:,1],1],pos[EL[:,2],1]],[pos[EL[:,1],2],pos[EL[:,2],2]])
plt.plot([pos[EL[:,2],1],pos[EL[:,3],1]],[pos[EL[:,2],2],pos[EL[:,3],2]])
plt.plot([pos[EL[:,0],1],pos[EL[:,3],1]],[pos[EL[:,0],2],pos[EL[:,3],2]])



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

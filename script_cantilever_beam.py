#%%

from module_python_fem import *

#%% Cantilever beam

## Initialization
# Parameters
E = 200e9       #Young's modulus [Pa]
nu = 0.3        #Poisson ratio
h = 10          #Height of beam [mm]
b = 10          #Width of beam [mm]
t = 0.4         #Thickness of beam [mm]
I = 1/12 * (b*h**3 - (b-2*t)*(h-2*t)**3)     #Second moment of inertia [mm4]
A = b*h         #Area [mm2]
#Ndof = 3        #Number of degrees of freedom
force = 100
type = 'beam'  #beam or solid

params = {
  'youngs_modulus': E,
  'inertia': I,
  'area' : A,
  'poisson' : nu,
  'thickness' : t}

if type == 'solid':

    # Node geometry
    nodes = np.zeros((4,3))
    nodes[0,:] = [0, 0.0, 0.0]
    nodes[1,:] = [1, 100, 0]
    nodes[2,:] = [2, 0, 10]
    nodes[3,:] = [3, 100, 10]

    # Elements
    elements = np.zeros((1,4),dtype=int)
    elements[0,:] = [0, 1, 3, 2]

    # Parameters
    kr = 4                           #number of elements in which the shorter side will be divided
    lg = kr*10                       #number of elements in which the longer side will be divided
    NoN = (kr+1)*(lg+1)              #number of nodes per beam
    NoE = kr*lg                      #number of elements per beam
    NL = np.zeros((NoN, 3),dtype="float")    #extended node list
    EL = np.zeros((NoE, 4),dtype="int")    #extended element list 

    # Boundary conditions
    BCm = []   #Leading nodes
    BCs = []   #Following nodes
    BCd = np.zeros((2*(kr+1),2))
    for i in range(0,kr+1):
        BCd[2*i,0] = i*(lg+1)
        BCd[2*i+1,0] = i*(lg+1)
        BCd[2*i,1] = 0
        BCd[2*i+1,1] = 1
        
    BCf = np.array([[(kr+1)*(lg+1)-1,1,-force]])      #Prescribed forces [Node, dof, value of acting force]
    pre_disp = np.zeros((0,3)) #np.array([[0,0,0],[0,1,0]])                   #Prescribed displacement [Node, dof, value of displacement]

    for row in range(0,elements.shape[0]):
        ## Nodes
        n = 0       #this will allow us to go through rows in node list

        lh = nodes[elements[row,1],1]-nodes[elements[row,0],1]
        lv = nodes[elements[row,2],2]-nodes[elements[row,0],2]
        if lh > lv:
            p = lg          #number of elements in which the horizontal line will be divided
            m = kr          #number of elements in which the vertical line will be divided
        elif lv > lh:
            p = kr          #number of elements in which the horizontal line will be divided
            m = lg          #number of elements in which the vertical line will be divided

        for i in range(0,m+1):
            for j in range(0,p+1):
                NL[row*NoN+n,0] = row*NoN + n
                NL[row*NoN+n,1] = nodes[elements[row,0],1] + j*lh/p 
                NL[row*NoN+n,2] = nodes[elements[row,0],2] + i*lv/m
                n += 1 

        ## Elements
        for i in range(0,m):
            for j in range(0,p):
                if j == 0:      #most left elements
                    EL[row*NoE+(i*p), 0] = row*NoN + i*(p+1)
                    EL[row*NoE+(i*p), 1] = EL[row*NoE+(i*p), 0] + 1
                    EL[row*NoE+(i*p), 3] = row*NoN + (i+1)*(p+1)
                    EL[row*NoE+(i*p), 2] = EL[row*NoE+(i*p), 3] + 1
                else:
                    EL[row*NoE+(i*p)+j, 0] = EL[row*NoE+(i*p)+(j-1), 1]
                    EL[row*NoE+(i*p)+j, 1] = EL[row*NoE+(i*p)+j, 0] + 1
                    EL[row*NoE+(i*p)+j, 3] = EL[row*NoE+(i*p)+(j-1), 2]
                    EL[row*NoE+(i*p)+j, 2] = EL[row*NoE+(i*p)+j, 3] + 1

elif type == 'beam':
    NoE = 50
    L = 100.0

    # Node geometry
    NL = np.zeros((NoE+1,3),dtype="float") # these are positions
    Le = L/NoE
    
    NL[0,:] = [0, 0.0, 0.0]
    for n in range(1,NoE+1):
        NL[n,:] = [n, n*Le, 0.0]

    # Elements
    EL = np.zeros((NoE,2),dtype="int") # these are indices
    for n in range(0,NoE):
        EL[n,:] = [n,n+1]

    # Boundary conditions
    BCd = np.array([[0,0],[0,1],[0,2]])               #Node, dof with blocked displacements
    BCm = []  #Leading nodes
    BCs = []  #Following nodes
    BCf = np.array([[NoE,1,-force]])             #Prescribed forces [Node, dof, value of acting force]
    pre_disp = np.zeros((0,3))                        #Prescribed displacement [Node, dof, value of displacement]


## Main code
myModel = model(EL,NL,BCd,params,BCm,BCs,BCf,pre_disp,type)
myModel.compute_displacement()
myModel.plot(1e7)

analytical_displacement = force*100**3/(3*E*I)
analytical_rotation = force*100**2/(2*E*I)

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

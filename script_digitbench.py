#%%
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# Function for conversion of dense to sparse matrix
def dense2sparse(A):

    row_ind,col_ind = np.nonzero(A)
    data = A[row_ind,col_ind]
    B = sp.sparse.csr_matrix((data,(row_ind,col_ind)),shape=A.shape)

    return B


class beam2D_v2:

    def __init__(self, params, node):
        self.I = params['inertia']
        self.E = params['youngs_modulus']
        self.A = params['area']
        self.node_ID = node[:,0].astype(int)
        self.node_coord = node[:,1:3]
        self.dof = np.array([[self.node_ID[0],1],   #dof 1 = axial load
                             [self.node_ID[0],2],   #dof 2 = bending 
                             [self.node_ID[0],3],   #dof 3 = moment 
                             [self.node_ID[1],1],
                             [self.node_ID[1],2],
                             [self.node_ID[1],3]],dtype="int")
        self.L = np.linalg.norm(self.node_coord[1,:] - self.node_coord[0,:])
        self.s = (self.node_coord[[1],:] - self.node_coord[[0],:])/self.L
        t = np.array([[-self.s[0,1]],[self.s[0,0]]])
        self.t = np.transpose(t)

        # Rotation matrix
        one = [[1]]
        self.T = np.concatenate((self.s,self.t),axis=0)
        self.R = sp.linalg.block_diag(self.T,one,self.T,one)

        # Compute the stiffness matrix
        self.compute_K()

    def compute_K(self):
        # Element stiffness matrix
        self.Kl = self.E*self.I/self.L**3 * np.array([[self.E*self.A/self.L, 0,0,-self.E*self.A/self.L,0,0],
                                                    [0, self.E*self.I/self.L**2*12, self.E*self.I/self.L**2*6, 0, -self.E*self.I/self.L**3*12, self.E*self.I/self.L**2*6],
                                                    [0, self.E*self.I/self.L**2*6,  self.E*self.I/self.L*4,    0, -self.E*self.I/self.L**2*6,  self.E*self.I/self.L*2   ],
                                                    [-self.E*self.A/self.L, 0,0,self.E*self.A/self.L,0,0],
                                                    [0, -self.E*self.I/self.L**3*12, -self.E*self.I/self.L**2*6, 0, self.E*self.I/self.L**3*12, -self.E*self.I/self.L**2*6],
                                                    [0, self.E*self.I/self.L**2*6,   self.E*self.I/self.L*2,     0, -self.E*self.I/self.L**3*6, self.E*self.I/self.L*4    ]],dtype="float")

        # Global stiffness matrix per element
        self.Ke = dense2sparse(np.linalg.multi_dot([np.transpose(self.R),self.Kl,self.R]))


    def compute_Z(self,model_dofs):

        self.row_index = []
        self.col_index = []

        for i,edof in enumerate(self.dof):
            for j,mdof in enumerate(model_dofs):
                if np.array_equal(edof,mdof):
                    self.row_index.append(i)
                    self.col_index.append(j)

        # compact row, compat col or compact diag
        self.Ze = sp.sparse.csr_matrix((np.ones((len(self.row_index))),(self.row_index,self.col_index)),shape=(self.dof.shape[0],model_dofs.shape[0]))

#%%

## Initialization
# Parameters
E = 200e9   #Young's modulus [Pa]
h = 0.1     #Height of beam [m]
b = 0.1     #Width of beam [m]
Lh = 1642   #Length of horizontal beams [m]
Lv = 1786   #Length of vertical beams [m]
t = 0.004   #Thickness of beam [m]
I = 1/12 * (b*h**3 - (b-2*t)*(h-2*t)**3)     #Second moment of inertia [m4]
A = b*h     #Area [m2]
Ndof = 3    #Number of degrees of freedom

params = {
  'youngs_modulus': E,
  'inertia': I,
  'area' : A}

# Node geometry
coord = np.zeros((11,3))
coord[0,:] = [0, 0.0, 0.0]
coord[1,:] = [1, 917.0, 0.0]
coord[2,:] = [2, Lv, 0.0]
coord[3,:] = [3, Lv, Lh-330]
coord[4,:] = [4, Lv, Lh]
coord[5,:] = [5, 0.0, Lh]
coord[6,:] = [6, 0.0, 996.0]
coord[7,:] = [7, 660.0, 996.0]
coord[8,:] = [8, 917.0, 996.0]
coord[9,:] = [9, 917, 996.0-21.0]
coord[10,:] = [10, 917.0, 675.0]

# Elements
eTop = np.zeros((12,2),dtype=int)
for i in range(len(coord)-1):
    eTop[i,0] = coord[i,0]
    eTop[i,1] = coord[i+1,0]
eTop[10,0] = coord[10,0]
eTop[10,1] = coord[1,0]
eTop[11,0] = coord[0,0]
eTop[11,1] = coord[6,0]

# Plot the initial geometry
plt.plot([coord[eTop[:,0],1],coord[eTop[:,1],1]],[coord[eTop[:,0],2],coord[eTop[:,1],2]])

# Arrays
myElements = []
K = []
model_dofs = np.zeros((0,2),dtype=int)

#%% Prototype of the model class

# Get unique dof list
for i in range(0,eTop.shape[0]):

    myElements.append(beam2D_v2(params,coord[eTop[i,:],:])) 
    model_dofs = np.concatenate((model_dofs,myElements[i].dof),axis=0)      #List of all dofs

model_dofs = np.unique(model_dofs,axis=0)                                   #Remove double dofs

# Compute stiffness matrix
for i in range(0,eTop.shape[0]):

    myElements[i].compute_Z(model_dofs)

    Ze_sp = myElements[i].Ze
    Ke_sp = myElements[i].Ke

    K += Ze_sp.transpose() @ Ke_sp @ Ze_sp                  #Global stiffness matrix


#%%

# Loads
actuator = 100                       #force on frame due to the actuators
cantilever = 100                     #force on frame due to the cantilever beam
load = np.zeros(((len(coord)+1)*Ndof,1)).astype(object)
load[4*Ndof+1] = cantilever       #cantilever beam attached to node 4
load[7*Ndof+1] = actuator/2       #first actuator attached to node 7 and 8
load[8*Ndof+1] = actuator/2
load[9*Ndof+1] = actuator/2       #second actuator attahed to node 9 and 10
load[10*Ndof+1] = actuator/2

# Boundary conditions
BC = np.ones(((len(coord)+1)*Ndof,1)).astype(object)
BC[0,0] = 0     #x-translation node 0
BC[1,0] = 0     #y-translation node 0
BC[2*Ndof+1,0] = 0  #y-translation node 2

#        # Boundary conditions       #column 1: node_ID, column 2: dof, column 3: type (0=displacement, 1=force), column 4: value
#        BC = np.array([[self.dof[0,0],self.dof[0,1],0,0],
#                       [self.dof[1,0],self.dof[1,1],0,0],
#                       [self.dof[2,0],self.dof[2,1],0,0],
#                       [self.dof[3,0],self.dof[3,1],0,0],
#                       [self.dof[4,0],self.dof[4,1],0,0],
#                       [self.dof[5,0],self.dof[5,1],0,0]])





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


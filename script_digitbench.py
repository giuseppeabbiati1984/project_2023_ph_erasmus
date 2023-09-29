#%%
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# function for conversion of dense to sparse matrix
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
                             [self.node_ID[0],2],   #dof 2 = bending load
                             [self.node_ID[0],3],   #dof 3 = moment load
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

        # compute the stiffness matrix
        self.compute_K()

    def compute_K(self):
        # Element stiffness matrix
        self.Kl = self.E*self.I/self.L**3 * np.array([[self.E*self.A/self.L, 0,0,-self.E*self.A/self.L,0,0],
                                                    [0, self.E*self.I/self.L**2*12, self.E*self.I/self.L**2*6, 0, -self.E*self.I/self.L**3*12, self.E*self.I/self.L**2*6],
                                                    [0, self.E*self.I/self.L**2*6,  self.E*self.I/self.L*4,    0, -self.E*self.I/self.L**2*6,  self.E*self.I/self.L*2   ],
                                                    [-self.E*self.A/self.L, 0,0,self.E*self.A/self.L,0,0],
                                                    [0, -self.E*self.I/self.L**3*12, -self.E*self.I/self.L**2*6, 0, self.E*self.I/self.L**3*12, -self.E*self.I/self.L**2*6],
                                                    [0, self.E*self.I/self.L**2*6,   self.E*self.I/self.L*2,     0, -self.E*self.I/self.L**3*6, self.E*self.I/self.L*4    ]],dtype="float")

        # Global stiffness matrix
        self.Ke = dense2sparse(np.linalg.multi_dot([np.transpose(self.R),self.Kl,self.R]))       #per element

        # dense to sparse conversion
        # row_ind,col_ind = np.nonzero(self.Kg)
        # data = self.Kg[row_ind,col_ind]
        # self.Kg = sp.sparse.csr_matrix((data,(row_ind,col_ind)),shape=self.Kg.shape)


    def compute_Z(self,model_dofs):
        #put a 1 where self.dof == model_dofs

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
coord[1,:] = [1, 0.0, 996.0]
coord[2,:] = [2, 0.0, Lh]
coord[3,:] = [3, Lv, Lh-330]
coord[4,:] = [4, Lv, Lh]
coord[5,:] = [5, Lv, 0.0]
coord[6,:] = [6, 917.0, 0.0]
coord[7,:] = [7, 917.0, 675.0]
coord[8,:] = [8, 917.0, 996.0-21.0]
coord[9,:] = [9, 917.0, 996.0]
coord[10,:] = [10, 660.0, 996.0]

# Elements
eTop = np.zeros((12,2),dtype=int)
for i in range(len(coord)-1):
    eTop[i,0] = coord[i,0]
    eTop[i,1] = coord[i+1,0]
eTop[10,0] = coord[10,0]
eTop[10,1] = coord[1,0]
eTop[11,0] = coord[0,0]
eTop[11,1] = coord[6,0]

# Arrays
myElements = []

#%% prototype of the model class


model_dofs = np.zeros((0,2),dtype=int)

## Main code
# Get unique dof list
for i in range(0,eTop.shape[0]):

    myElements.append(beam2D_v2(params,coord[eTop[i,:],:])) 
    model_dofs = np.concatenate((model_dofs,myElements[i].dof),axis=0)      #All dofs

model_dofs = np.unique(model_dofs,axis=0)                                   #Remove double dofs

# Compute stiffness matrix
for i in range(0,eTop.shape[0]):

    myElements[i].compute_Z(model_dofs)

#%%

Ze_sp = myElements[0].Ze
Ke_sp = myElements[0].Ke

#print(Ke)
#print(Ke_sp.toarray())

K = Ze_sp.transpose() @ Ke_sp @ Ze_sp

#%%

for myElement in myElements:
    K = Ze_sp.transpose() @ Ke_sp @ Ze_sp

#%%

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


## Main code
Kg = np.zeros(((len(coord)+1)*Ndof+2,(len(coord)+1)*Ndof+2)).astype(object)
Ze = np.zeros((6,(len(coord)+1)*Ndof+2)).astype(object)
force = np.zeros(((len(coord)+1)*Ndof+2,1)).astype(object)

for i in range(len(eTop)-1):
    id0 = eTop[i,0]
    id1 = eTop[i,1]
    node = np.zeros((2,3)).astype(object)
    for j in range(len(coord)):
        if coord[j,0] == id0:
            node[0,:] = coord[j,:]
        if coord[j,0] == id1:
            node[1,:] = coord[j,:]

    element = beam2D_v2(params, node)
    element.__init__(params,node)
    Keg = element.stiffness_matrix()

    Ze[0:6,3*i:3*i+6] = np.ones((6,6))
    Kg += np.linalg.multi_dot([np.transpose(Ze),Keg,Ze])    #Global stiffness matrix
    force_e = load[Ndof*i:Ndof*i+6]
    force += np.dot(np.transpose(Ze),force_e)




#u = np.linalg.lstsq(Kg, force)



# %%

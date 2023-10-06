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
        # Local element stiffness matrix
        self.Kl = self.E*self.I/self.L**3 * np.array([[self.E*self.A/self.L, 0,0,-self.E*self.A/self.L,0,0],
                                                    [0, self.E*self.I/self.L**2*12, self.E*self.I/self.L**2*6, 0, -self.E*self.I/self.L**3*12, self.E*self.I/self.L**2*6],
                                                    [0, self.E*self.I/self.L**2*6,  self.E*self.I/self.L*4,    0, -self.E*self.I/self.L**2*6,  self.E*self.I/self.L*2   ],
                                                    [-self.E*self.A/self.L, 0,0,self.E*self.A/self.L,0,0],
                                                    [0, -self.E*self.I/self.L**3*12, -self.E*self.I/self.L**2*6, 0, self.E*self.I/self.L**3*12, -self.E*self.I/self.L**2*6],
                                                    [0, self.E*self.I/self.L**2*6,   self.E*self.I/self.L*2,     0, -self.E*self.I/self.L**3*6, self.E*self.I/self.L*4    ]],dtype="float")

        # Global element stiffness matrix
        self.Ke = dense2sparse(np.linalg.multi_dot([np.transpose(self.R),self.Kl,self.R]))


    def compute_Z(self,modeldofs):
        self.row_index = []
        self.col_index = []

        for i,edof in enumerate(self.dof):
            for j,mdof in enumerate(modeldofs):
                if np.array_equal(edof,mdof):
                    self.row_index.append(i)
                    self.col_index.append(j)

        # compact row, compat col or compact diag
        self.Ze = sp.sparse.csr_matrix((np.ones((len(self.row_index))),(self.row_index,self.col_index)),shape=(self.dof.shape[0],modeldofs.shape[0]))


class model:

    def __init__(self,eTop,BCd,params,coord,BCm,BCs,force_dof):
        self.BCd = BCd
        self.BCm = BCm
        self.BCs = BCs
        self.BCf = force_dof[:,0:2]
        self.f = force_dof[:,2]
        self.model_dofs = np.zeros((0,2),dtype=int)
        self.get_unique_dof(eTop,params,coord)
        self.stiffness_matrix(eTop)
        self.compute_force()

    # Get unique DOF list
    def get_unique_dof(self,eTop,params,coord):
        self.myElements = []

        for i in range(0,eTop.shape[0]):

            self.myElements.append(beam2D_v2(params,coord[eTop[i,:],:])) 
            np.append(self.model_dofs, self.myElements[i].dof, axis=0)

            for idr, element_row in enumerate(self.myElements[i].dof):
                for j, row in enumerate(self.BCs):
                    if np.array_equal(element_row,row):
                        self.myElements[i].dof[idr] = self.BCm[j]                                #Replace BCs with BCm

            self.model_dofs = np.concatenate((self.model_dofs,self.myElements[i].dof),axis=0)      #List of all dofs

        self.model_dofs = np.unique(self.model_dofs,axis=0)                                        #Remove double dofs

        for i, row in enumerate(self.BCd):
            for j, model_row in enumerate(self.model_dofs):
                if np.array_equal(row,model_row):
                    self.model_dofs = np.delete(self.model_dofs, j, axis=0)                   #Remove BC dofs

    # Compute stiffness matrix
    def stiffness_matrix(self,eTop):
        self.K = []

        for i in range(0,eTop.shape[0]):
            self.myElements[i].compute_Z(self.model_dofs)
            Ze_sp = self.myElements[i].Ze
            Ke_sp = self.myElements[i].Ke

        self.K += Ze_sp.transpose() @ Ke_sp @ Ze_sp                  #Global stiffness matrix

    def compute_force(self):
        self.row_index = []
        self.col_index = []

        for i,fdof in enumerate(self.BCf):
            for j,mdof in enumerate(self.model_dofs):
                if np.array_equal(fdof,mdof):
                    self.row_index.append(i)
                    self.col_index.append(j)

        self.Zf = sp.sparse.csr_matrix((np.ones((len(self.row_index))),(self.row_index,self.col_index)),shape=(self.BCf.shape[0],self.model_dofs.shape[0]))
        
        self.Kd = self.Zf.transpose() @ self.f
        print(self.Zf)
        print('------------------------')
        print(self.K)
        print('------------------------')
        print(self.Kd)

        #self.d = sp.sparse.linalg.spsolve(self.K, self.Kd)
        self.d = np.linalg.lstsq(self.K, self.Kd)




#%% Cantilever beam

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
coord = np.zeros((12,3))
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
coord[11,:] = [11, Lv-500, Lh-330]

# Elements
eTop = np.zeros((13,2),dtype=int)
for i in range(len(coord)-2):
    eTop[i,0] = coord[i,0]
    eTop[i,1] = coord[i+1,0]
eTop[10,0] = coord[10,0]
eTop[10,1] = coord[1,0]
eTop[11,0] = coord[0,0]
eTop[11,1] = coord[6,0]
eTop[12,0] = coord[3,0]
eTop[12,1] = coord[11,0]

# Plot the initial geometry
#plt.plot([coord[eTop[:,0],1],coord[eTop[:,1],1]],[coord[eTop[:,0],2],coord[eTop[:,1],2]])

# Boundary conditions
BCd = np.array([[0,1],[0,2],[2,1],[2,2]])           #Node, dof with blocked displacements
force = 100
force_dof = np.array([[11,1,force],[11,2,force]])         #Node, dof, value of acting force
BCm = []                                            #Leading nodes
BCs = []                                            #Following nodes


## Main code

myModel = model(eTop,BCd,params,coord,BCm,BCs,force_dof)

#%% Simple frame

## Initialization
# Parameters
E = 200e9   #Young's modulus [Pa]
h = 0.1     #Height of beam [m]
b = 0.1     #Width of beam [m]
t = 0.004   #Thickness of beam [m]
I = 1/12 * (b*h**3 - (b-2*t)*(h-2*t)**3)     #Second moment of inertia [m4]
A = b*h     #Area [m2]
Ndof = 3    #Number of degrees of freedom

params = {
  'youngs_modulus': E,
  'inertia': I,
  'area' : A}

# Node geometry
nodes = np.zeros((6,3))
nodes[0,:] = [0, 0.0, 0.0]
nodes[1,:] = [1, 0.0, 10.0]
nodes[2,:] = [2, 10.0, 0.0]
nodes[3,:] = [3, 10.0, 10.0]
nodes[4,:] = [4, 0.0, 10.0]
nodes[5,:] = [5, 10.0, 10.0]

# Elements
elements = np.zeros((3,2),dtype=int)
elements[0,0] = nodes[0,0]
elements[0,1] = nodes[1,0]
elements[1,0] = nodes[2,0]
elements[1,1] = nodes[3,0]
elements[2,0] = nodes[4,0]
elements[2,1] = nodes[5,0]

# Plot the initial geometry
#plt.plot([nodes[elements[:,0],1],nodes[elements[:,1],1]],[nodes[elements[:,0],2],nodes[elements[:,1],2]])

# Boundary conditions
BCd = np.array([[0,1],[0,2],[2,1],[2,2]])           #Node, dof with blocked displacements
force = 100
force_dof = np.array([[4,1,force],[4,2,force]])         #Node, dof, value of acting force
BCm = np.array([[1,1],[1,2],[1,3],[3,1],[3,2],[3,3]])   #Leading nodes
BCs = np.array([[4,1],[4,2],[4,3],[5,1],[5,2],[5,3]])   #Following nodes

## Main code

myModel = model(elements,BCd,params,nodes,BCm,BCs,force_dof)



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


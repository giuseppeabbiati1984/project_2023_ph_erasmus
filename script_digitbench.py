#%%
import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt

# Function for conversion of dense to sparse matrix
def dense2sparse(A):

    row_ind,col_ind = np.nonzero(A)
    data = A[row_ind,col_ind]
    B = sp.sparse.csr_matrix((data,(row_ind,col_ind)),shape=A.shape)

    return B

def delta(i,j):
    if i == j:
        delta = 1
    else:
        delta = 0

    return delta


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
        self.Kl = np.array([[self.E*self.A/self.L, 0,0,-self.E*self.A/self.L,0,0],
                            [0, 12*self.E*self.I/self.L**3, -6*self.E*self.I/self.L**2, 0, -12*self.E*self.I/self.L**3, -6*self.E*self.I/self.L**2],
                            [0, -6*self.E*self.I/self.L**2,  4*self.E*self.I/self.L,    0,  6*self.E*self.I/self.L**2,   2*self.E*self.I/self.L   ],
                            [-self.E*self.A/self.L, 0,0,self.E*self.A/self.L,0,0],
                            [0, -12*self.E*self.I/self.L**3, 6*self.E*self.I/self.L**2, 0,  12*self.E*self.I/self.L**3,  6*self.E*self.I/self.L**2],
                            [0, -6*self.E*self.I/self.L**2,  2*self.E*self.I/self.L,    0,  6*self.E*self.I/self.L**2,   4*self.E*self.I/self.L    ]],dtype="float")

        # Global element stiffness matrix
        self.Ke = dense2sparse(np.linalg.multi_dot([np.transpose(self.R),self.Kl,self.R]))
        self.Ke = self.Ke.todense()


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

    def __init__(self,eTop,BCd,params,coord,BCm,BCs,force_dof,disp_dof,type):
        self.BCd = BCd
        self.BCm = BCm
        self.BCs = BCs
        self.BCf = force_dof[:,0:2]
        self.f = force_dof[:,2]
        self.BCu = disp_dof[:,0:2]
        self.u = disp_dof[:,2]
        self.model_dofs = np.zeros((0,2),dtype=int)
        self.get_unique_dof(eTop,params,coord,type)
        self.stiffness_matrix(eTop)
        self.compute_displacement()
        

    # Get unique DOF list
    def get_unique_dof(self,eTop,params,coord,type):
        self.myElements = []

        for i in range(0,eTop.shape[0]):
            if type == 'beam':
                self.myElements.append(beam2D_v2(params,coord[eTop[i,:],:]))
            elif type == 'shell':
                self.myElements.append(shell2D(params, coord[eTop[i,:],:]))
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
        self.K = np.zeros((len(self.model_dofs),len(self.model_dofs)))

        for i in range(0,eTop.shape[0]):
            self.myElements[i].compute_Z(self.model_dofs)
            Ze_sp = self.myElements[i].Ze
            Ke = self.myElements[i].Ke

            Ze_dense = Ze_sp.todense()

            self.K += Ze_dense.transpose() @ Ke @ Ze_dense                  #Global stiffness matrix

    def compute_Zf(self):
        self.row_index = []
        self.col_index = []

        for i,fdof in enumerate(self.BCf):
            for j,mdof in enumerate(self.model_dofs):
                if np.array_equal(fdof,mdof):
                    self.row_index.append(i)
                    self.col_index.append(j)           

        self.Zf = sp.sparse.csr_matrix((np.ones((len(self.row_index))),(self.row_index,self.col_index)),shape=(self.BCf.shape[0],self.model_dofs.shape[0]))
        
    def compute_Zu(self):
        self.row_ind = []
        self.col_ind = []

        for i,udof in enumerate(self.BCu):
            for j,mdof in enumerate(self.model_dofs):
                if np.array_equal(udof,mdof):
                    self.row_ind.append(i)
                    self.col_ind.append(j)

        self.Zu = sp.sparse.csr_matrix((np.ones((len(self.row_ind))),(self.row_ind,self.col_ind)),shape=(self.BCu.shape[0],self.model_dofs.shape[0]))

    def compute_displacement(self):
        self.compute_Zf()
        self.compute_Zu()
        self.Zu = self.Zu.todense()
        self.u = self.u.reshape(-1,1)

        self.Kd = self.Zf.transpose() @ self.f
        self.Kd = self.Kd.reshape(-1, 1)        # Reshape to a 2D array
        self.Kinv = np.linalg.inv(self.K)

        #displacement because of prescribed displacement
        self.lamb = np.linalg.lstsq(self.Zu @ self.Kinv @ self.Zu.transpose(), self.u - self.Zu @ self.Kinv @ self.Kd)
        self.lamb = self.lamb[0]
        self.du = np.linalg.lstsq(self.K, self.Kd + self.Zu.transpose() @ self.lamb)
        self.du = self.du[0]
        self.du_dof = np.concatenate([self.model_dofs, self.du], axis=1)

        #displacement because of prescribed force
        self.df = np.linalg.lstsq(self.K, self.Kd)
        self.df = self.df[0]
        self.df_dof = np.concatenate([self.model_dofs, self.df], axis=1)


class membrane2D:

    def __init__(self, params, node):
        #self.I = params['inertia']
        self.E = params['youngs_modulus']
        self.nu = params['poisson']
        #self.A = params['area']
        #self.p = params['p']
        #self.m = params['m']
        self.NPE = 4                        #nodes per element
        self.PD = 2                         #problem dimension
        self.h = params['thickness']
        self.node_ID = node[:,0].astype(int)
        self.node_coord = node[:,1:3]
        self.dof = np.array([[self.node_ID[0],1],   #dof 1 = axial load
                             [self.node_ID[0],2],   #dof 2 = bending 
                             [self.node_ID[1],1],
                             [self.node_ID[1],2],
                             [self.node_ID[2],1],   #dof 1 = axial load
                             [self.node_ID[2],2],   #dof 2 = bending 
                             [self.node_ID[3],1],
                             [self.node_ID[3],2]],dtype="int")
        self.a = (self.node_coord[0,0]-self.node_coord[1,0])   #increment in horizontal direction (length of element)
        self.b = (self.node_coord[3,1]-self.node_coord[0,1])   #increment in vertical direction (length of element)

        self.D = # plane stress  formulation

        #self.assemble_stiffness()
        self.compute_K()
    

    def compute_K(self):

        self.ke = np.zeros((8,8))                     #element stiffness
        
        #self.x = np.zeros([self.NPE,self.PD],dtype=int)
        #self.x[0:self.NPE,0:self.PD] = self.NL[self.nl[0:self.NPE]-1,0:self.PD]     #coordinates of the four corner nodes
        #self.coor = self.x.transpose()
        #self.GPE = 4
        #for i in range(0,self.NPE):
        #for j in range(0,self.NPE):
        #k = np.zeros([self.PD,self.PD]) #stiffness between each node

        # bending (ri,rj are the isoparametric coordinates (xi and eta in your previous version))

        N = lambda(ri,rj) : np.array([[],[]]) #  [[phi1(ri,rj),0,phi2,0,phi3,0,phi4,0],[0,phi1,0,phi2,0,phi3,0,phi4]]
        B = lambda(ri,rj) : np.array([[],[]]) #  [[dphi1dx,0,dphi2dx,0,dphi3dx,0,dphi4dx,0],[0,dphi1dy,0,dphi2y,0,dphi3y,0,dphi4y]]

        # quadrature rule
        r,w = self.GaussPoints(2)

        # numerical ingration
        for ri,wi in zip(r,w):
            for rj,wj in zip(r,w):

                J = N(ri,rj) @ self.node_coord([:,0:2])

                grad = np.linalg.inv(self.J).transpose() @ self.N



                self.ke += Bb @ D @ B * wi * wj



                for gp in range(1,self.GPE+1):
                    self.J = np.zeros([self.PD,self.PD]) #jacobian
                    grad = np.zeros([self.PD,self.NPE])

                    self.GaussPoint(gp)
                    self.N = np.array([[-1/4*(1-self.eta),    1/4*(1-self.eta), -1/4*(1+self.eta),   1/4*(1+self.eta)],
                                       [-1/4*(1-self.xi),     -1/4*(1+self.xi),  1/4*(1-self.xi),    1/4*(1+self.xi) ]])
                                #3 en 4 omgedraaid

                    #self.J = self.coor @ self.N.transpose()
                    self.J = self.N @ self.x
                    grad = np.linalg.inv(self.J).transpose() @ self.N

                    for a in range(0,self.PD):
                        for c in range(0,self.PD):
                            for b in range(0,self.PD):
                                for d in range(0,self.PD):
                                    C = (self.E/(2*(1+self.nu))) * (delta(a,d)*delta(b,c) + delta(a,c)*delta(b,d)) + (self.E*self.nu)/(1-nu**2)*delta(a,b)*delta(c,d)
                                    k[a,c] = k[a,c] + grad[b, i] * C * grad[d,j] * np.linalg.det(self.J) * self.alpha

                self.ke[i*self.PD:(i+1)*self.PD, j*self.PD:(j+1)*self.PD] = k   # element stiffness matrix

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

    def GaussPoints(self,order):
        # quadrature rules in 1D (2D rules are obtained by combining 1Ds as in a grid)
        if order == 1:
            r = np.array[0.0]
            w = np.array[2.0]
        elif order == 2:
            r = np.array([-1/math.sqrt(3),+1/math.sqrt(3)])
            w = np.array([1.0,1.0])

        return r,w
    
    '''
    def GaussPoint(self,gp):
        #if self.NPE == 4:    #quadralatic element
            if self.GPE == 1:
                if gp == 1:
                    self.xi = 0
                    self.eta = 0
                    self.alpha = 4

            if self.GPE == 4:
                if gp == 1:
                    self.xi = -1/math.sqrt(3)
                    self.eta = -1/math.sqrt(3)
                    self.alpha = 1

                if gp == 2:
                    self.xi = 1/math.sqrt(3)
                    self.eta = -1/math.sqrt(3)
                    self.alpha = 1

                if gp == 3:
                    self.xi = 1/math.sqrt(3)
                    self.eta = 1/math.sqrt(3)
                    self.alpha = 1

                if gp == 4:
                    self.xi = -1/math.sqrt(3)
                    self.eta = 1/math.sqrt(3)
                    self.alpha = 1


    #def grad_N_nat(self):   #get derivatives of shape functions
    #    self.N = np.zeros([self.PD,self.NPE])

    #    if self.NPE == 4:
    #        self.N = np.array([[-1/4*(1-self.eta),    1/4*(1-self.eta),   1/4*(1+self.eta), -1/4*(1+self.eta)],
    #                                [-1/4*(1-self.xi),     -1/4*(1+self.xi),   1/4*(1+self.xi),  1/4*(1-self.xi)  ]])
            #3 en 4 misschien omdraaien?
    

    def assemble_stiffness(self):
        self.uniform_mesh()
        self.Ke = np.zeros([self.NoN*self.PD, self.NoN*self.PD])     #global stiffness matrix

        for i in range(0,self.NoE):
            self.nl = self.EL[i,0:self.NPE]     #mini node list, list of the four nodes of the element
            self.element_stiffness()            #element stiffness

            #self.Ke = self.ke
            self.Ke[i*2:i*2+8,i*2:i*2+8] = self.Ke[i*2:i*2+8,i*2:i*2+8] + self.ke
            #print(self.Ke)
            #print(self.Ke.shape)


    def uniform_mesh(self):
        n = 0       #this will allow us to go through rows in node list

        ## Nodes
        self.NL = np.zeros([self.NoN, self.PD],dtype="int")    #node list
        for i in range(0,self.m+1):
            for j in range(0,self.p+1):
                self.NL[n,0] = self.q[0,0] + j*self.a
                self.NL[n,1] = self.q[0,1] + i*self.b
                n += 1

        ## Elements
        self.EL = np.zeros([self.NoE, self.NPE],dtype="int")   #element list  
        for i in range(0,self.m):
            for j in range(0,self.p):
                if j == 0:      #most left elements
                    self.EL[i*self.p+j, 0] = i*(self.p+1) + (j+1)
                    self.EL[i*self.p+j, 1] = self.EL[i*self.p+j, 0] + 1
                    self.EL[i*self.p+j, 3] = self.EL[i*self.p+j, 0] + (self.p+1)
                    self.EL[i*self.p+j, 2] = self.EL[i*self.p+j, 3] + 1
                else:
                    self.EL[i*self.p+j, 0] = self.EL[i*self.p+j-1, 1]
                    self.EL[i*self.p+j, 3] = self.EL[i*self.p+j-1, 2]
                    self.EL[i*self.p+j, 1] = self.EL[i*self.p+j, 0] + 1

                    self.EL[i*self.p+j, 2] = self.EL[i*self.p+j, 3] + 1

        ## DOF list
        self.dof = np.zeros((3*len(self.NL),2))
        for i in range(0,len(self.NL)):
            self.dof[i*3,0] = i
            self.dof[i*3+1,0] = i
            self.dof[i*3+2,0] = i
            self.dof[i*3,1] = 1
            self.dof[i*3+1,1] = 2
            self.dof[i*3+2,1] = 3
    

'''
    

#%% Cantilever beam

## Initialization
# Parameters
E = 200e9   #Young's modulus [Pa]
nu = 0.3    #Poisson's ratio 
h = 0.1     #Height of beam [m]
b = 0.1     #Width of beam [m]
Lh = 1642   #Length of horizontal beams [m]
Lv = 1786   #Length of vertical beams [m]
t = 0.004   #Thickness of beam [m]
I = 1/12 * (b*h**3 - (b-2*t)*(h-2*t)**3)     #Second moment of inertia [m4]
A = b*h     #Area [m2]
Ndof = 3    #Number of degrees of freedom
p = 10       #number of elements in which the horizontal line will be divided
m = 10       #number of elements in which the vertical line will be divided
type = 'shell'   #beam or shell

params = {
  'youngs_modulus': E,
  'inertia': I,
  'area' : A,
  'p' : p,
  'm' : m,
  'poisson' : nu}

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
disp_dof = np.array([[5,1,10]])
BCm = []                                            #Leading nodes
BCs = []                                            #Following nodes



## Main code

myModel = model(eTop,BCd,params,coord,BCm,BCs,force_dof,disp_dof,type)

#%% Simple frame

## Initialization
# Parameters
E = 200e9   #Young's modulus [Pa]
nu = 0.3
h = 0.1     #Height of beam [m]
b = 0.1     #Width of beam [m]
t = 0.004   #Thickness of beam [m]
I = 1/12 * (b*h**3 - (b-2*t)*(h-2*t)**3)     #Second moment of inertia [m4]
A = b*h     #Area [m2]
Ndof = 3    #Number of degrees of freedom
p = 10       #number of elements in which the horizontal line will be divided
m = 10       #number of elements in which the vertical line will be divided

params = {
  'youngs_modulus': E,
  'inertia': I,
  'area' : A,
  'p' : p,
  'm' : m,
  'poisson' : nu}

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
BCd = np.array([[0,1],[0,2],[2,1],[2,2]])               #Node, dof with blocked displacements
force = 100
pre_forces = np.array([[1,1,force]])                        #Prescribed forces [Node, dof, value of acting force]
pre_disp = np.array([[1,1,10]])                              #Prescribed displacement [Node, dof, value of displacement]
BCm = np.array([[1,1],[1,2],[1,3],[3,1],[3,2],[3,3]])   #Leading nodes
BCs = np.array([[4,1],[4,2],[4,3],[5,1],[5,2],[5,3]])   #Following nodes
type = 'shell'

## Main code

myModel = model(elements,BCd,params,nodes,BCm,BCs,pre_forces,pre_disp,type)
print(myModel.df)

#displacement = myModel.d
#print(displacement)


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

#%% Membrane based model

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

n = 0       #this will allow us to go through rows in node list

## Nodes
NL = np.zeros([self.NoN, self.PD],dtype="int")    #node list
        for i in range(0,self.m+1):
            for j in range(0,self.p+1):
                self.NL[n,0] = self.q[0,0] + j*self.a
                self.NL[n,1] = self.q[0,1] + i*self.b
                n += 1

        ## Elements
        self.EL = np.zeros([self.NoE, self.NPE],dtype="int")   #element list  
        for i in range(0,self.m):
            for j in range(0,self.p):
                if j == 0:      #most left elements
                    self.EL[i*self.p+j, 0] = i*(self.p+1) + (j+1)
                    self.EL[i*self.p+j, 1] = self.EL[i*self.p+j, 0] + 1
                    self.EL[i*self.p+j, 3] = self.EL[i*self.p+j, 0] + (self.p+1)
                    self.EL[i*self.p+j, 2] = self.EL[i*self.p+j, 3] + 1
                else:
                    self.EL[i*self.p+j, 0] = self.EL[i*self.p+j-1, 1]
                    self.EL[i*self.p+j, 3] = self.EL[i*self.p+j-1, 2]
                    self.EL[i*self.p+j, 1] = self.EL[i*self.p+j, 0] + 1

                    self.EL[i*self.p+j, 2] = self.EL[i*self.p+j, 3] + 1


 self.L = np.linalg.norm(self.node_coord[1,:] - self.node_coord[0,:])
        d1 = self.node_coord[1,0] - self.node_coord[0,0]
        d2 = self.node_coord[1,1] - self.node_coord[0,1]
        if d1 == 0:
            d1 = 10.0
        if d2 == 0:
            d2 = 10.0
        self.q = np.array([[0,0],[d1,0],[0,d2],[d1,d2]])  #four corners
        self.NoN = (self.p+1)*(self.m+1)                #number of nodes
        self.NoE = self.p*self.m                        #number of elements
        self.a = (self.q[1,0]-self.q[0,0])/self.p   #increment in horizontal direction (length of element)
        self.b = (self.q[2,1]-self.q[0,1])/self.m   #icrement in vertical direction (length of element)

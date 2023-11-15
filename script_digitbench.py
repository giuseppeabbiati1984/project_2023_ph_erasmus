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

class model:

    def __init__(self,EL,NL,BCd,params,BCm,BCs,force_dof,disp_dof,type):
        self.BCd = BCd
        self.BCm = BCm
        self.BCs = BCs
        self.BCf = force_dof[:,0:2]
        self.f = force_dof[:,2]
        self.BCu = disp_dof[:,0:2]
        self.u = disp_dof[:,2]
        self.EL = EL
        self.model_dofs = np.zeros((0,2),dtype=int)

        self.get_unique_dof(params,NL,type)
        self.stiffness_matrix()
        self.compute_displacement()
        
    # Get unique DOF list
    def get_unique_dof(self,params,NL,type):
        self.myElements = []

        for i in range(0,EL.shape[0]):
            if type == 'beam':
                self.myElements.append(beam2D(params,NL[self.EL[i,:],:]))
            elif type == 'shell':
                self.myElements.append(membrane2D(params, NL[self.EL[i,:],:]))
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
    def stiffness_matrix(self):
        self.K = np.zeros((len(self.model_dofs),len(self.model_dofs)))

        for i in range(0,self.EL.shape[0]):
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

        print(self.df_dof)



class beam2D:

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
        self.Ke = np.linalg.multi_dot([np.transpose(self.R),self.Kl,self.R])


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


class membrane2D:

    def __init__(self, params, node):
        self.E = params['youngs_modulus']
        self.nu = params['poisson']        
        self.t = params['thickness']
        self.NPE = 4                        #nodes per element
        self.PD = 2                         #problem dimension
        self.node_ID = node[:,0].astype(int)
        self.node_coord = node[:,1:3]
        self.dof = np.array([[self.node_ID[0],1],   #dof 1 = axial load
                             [self.node_ID[0],2],   #dof 2 = bending 
                             [self.node_ID[1],1],
                             [self.node_ID[1],2],
                             [self.node_ID[2],1],   
                             [self.node_ID[2],2],   
                             [self.node_ID[3],1],
                             [self.node_ID[3],2]],dtype="int")
        self.a = (self.node_coord[0,0]-self.node_coord[1,0])   #increment in horizontal direction (length of element)
        self.b = (self.node_coord[3,1]-self.node_coord[0,1])   #increment in vertical direction (length of element)
    
        self.compute_K() 


    def compute_K(self):
        self.Ke = np.zeros((8,8))                     #element stiffness
        Ke_b = np.zeros((8,8)) 
        #Ke_sh = np.zeros((8,8)) 
        N = lambda s, t : 1/4* np.array([[-(1-t),  (1-t), (1+t), -(1+t)],
                                         [-(1-s), -(1+s), (1+s),  (1-s)]])
        D = np.array([[self.E/(1-self.nu**2),          self.nu*self.E/(1-self.nu**2),  0],         #plane stress
                      [self.nu*self.E/(1-self.nu**2),  self.E/(1-self.nu**2),          0],
                      [0,                              0,                              self.E/(2*(1+self.nu))]])   

        # quadrature rule (bending)
        r,w = self.GaussPoints(2)

        # numerical ingration
        for si,wi in zip(r,w):
            for tj,wj in zip(r,w):

                # Jacobian matrix [dx/ds,dx/dt;dy/ds,dy/dt]
                J = N(si,tj) @ self.node_coord

                print(np.linalg.det(J))

                Bs = np.zeros((4,8))
                Bs[0,[0,2,4,6]] = N(si,tj)[0,:] #dphi_ds_val
                Bs[1,[0,2,4,6]] = N(si,tj)[1,:] #dphi_dt_val
                Bs[2,[1,3,5,7]] = N(si,tj)[0,:]
                Bs[3,[1,3,5,7]] = N(si,tj)[1,:]

                B = np.array([[1,0,0,0],[0,0,0,1]]) @ sp.linalg.block_diag(np.linalg.inv(J),np.linalg.inv(J)) @ Bs

                Ke_b += self.t * B.transpose() @ D[0:2,0:2] @ B * np.linalg.det(J) * wi * wj

        # quadrature rule (shear)
        r,w = self.GaussPoints(1)

        # Jacobian matrix [dx/ds,dx/dt;dy/ds,dy/dt]
        Jsh = N(r,r) @ self.node_coord

        print(np.linalg.det(Jsh))

        Bssh = np.zeros((4,8))
        Bssh[0,[0,2,4,6]] = N(r,r)[0,:] #dphi_ds_val
        Bssh[1,[0,2,4,6]] = N(r,r)[1,:] #dphi_dt_val
        Bssh[2,[1,3,5,7]] = N(r,r)[0,:]
        Bssh[3,[1,3,5,7]] = N(r,r)[1,:]

        Bsh = np.array([[0,1,1,0]]) @ sp.linalg.block_diag(np.linalg.inv(Jsh),np.linalg.inv(Jsh)) @ Bssh

        Ke_sh = self.t * Bsh.transpose() * D[2,2]  * Bsh * np.linalg.det(Jsh) * w * w

        self.Ke = Ke_b + Ke_sh

    def GaussPoints(self,order):
        # quadrature rules in 1D (2D rules are obtained by combining 1Ds as in a grid)
        if order == 1:
            r = 0.0 #np.array([0.0])
            w = 2.0 #np.array([2.0])
        elif order == 2:
            r = np.array([-1/math.sqrt(3),+1/math.sqrt(3)])
            w = np.array([1.0,1.0])

        return r,w
    
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


#%% Cantilever beam

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

node_test = np.array([[1,-1,-1,0],[2,1,-1,0],[3,1,1,0],[4,-1,1,0]])

elem_test = membrane2D(params=params,node=np.array([[1,-1,-1,0],[2,1,-1,0],[3,1,1,0],[4,-1,1,0]]))

ke_rank = np.linalg.matrix_rank(elem_test.Ke)

U, S, VT = np.linalg.svd(elem_test.Ke)

# last three modes are rigid
phi = VT.T
phi = phi[:, -4:-1]

sca = 1.0
mode = 0
plt.plot(node_test[:,1],node_test[:,2],'ob')
plt.plot(node_test[:,1]+sca*phi[0::2,mode],node_test[:,2]+sca*phi[1::2,mode],'xr')


#%%

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


#%% Simple frame

## Initialization
# Parameters
E = 200e9   #Young's modulus [Pa]
nu = 0.3
h = 0.1     #Height of beam [mm]
b = 0.1     #Width of beam [mm]
t = 0.004   #Thickness of beam [mm]
I = 1/12 * (b*h**3 - (b-2*t)*(h-2*t)**3)     #Second moment of inertia [mm4]
A = b*h     #Area [mm2]
Ndof = 3    #Number of degrees of freedom
p = 5.0       #number of elements in which the horizontal line will be divided
m = 5.0       #number of elements in which the vertical line will be divided
force = 100
type = 'shell'

params = {
  'youngs_modulus': E,
  'inertia': I,
  'area' : A,
  'p' : p,
  'm' : m,
  'poisson' : nu,
  'thickness' : t}


if type == 'shell':
    h = 10     #Height of beam [mm]
    b = 10     #Width of beam [mm]
    t = 0.4   #Thickness of beam [mm]
    I = 1/12 * (b*h**3 - (b-2*t)*(h-2*t)**3)     #Second moment of inertia [mm4]
    p = 5       #number of elements in which the horizontal line will be divided
    m = 5       #number of elements in which the vertical line will be divided
    NoN = (p+1)*(m+1)                #number of nodes
    NoE = p*m                        #number of elements
    NL = np.zeros((3*NoN, 3),dtype="int")    #extended node list
    EL = np.zeros((3*NoE, 4),dtype="int")                  #extended element list 

    # Node geometry
    nodes = np.zeros((12,3))
    nodes[0,:] = [0, 0.0, 0.0]
    nodes[1,:] = [1, h, 0.0]
    nodes[2,:] = [2, 0.0, 100.0]
    nodes[3,:] = [3, h, 100.0]
    nodes[4,:] = [4, 100.0-h, 0.0]
    nodes[5,:] = [5, 100.0, 0.0]
    nodes[6,:] = [6, 100.0-h, 100.0]
    nodes[7,:] = [7, 100.0, 100.0]
    nodes[8,:] = [8, 0.0, 100.0]
    nodes[9,:] = [9, 100.0, 100.0]
    nodes[10,:] = [10, 0.0, 100.0+h]
    nodes[11,:] = [11, 100.0, 100.0+h]

    # Elements
    elements = np.zeros((3,4),dtype=int)
    elements[0,:] = [0, 1, 3, 2]
    elements[1,:] = [4, 5, 7, 6]
    elements[2,:] = [8, 9, 11, 10]

    # Plot
    #plt.plot([nodes[elements[:,0],1],nodes[elements[:,1],1]],[nodes[elements[:,0],2],nodes[elements[:,1],2]])
    #plt.plot([nodes[elements[:,1],1],nodes[elements[:,2],1]],[nodes[elements[:,1],2],nodes[elements[:,2],2]])
    #plt.plot([nodes[elements[:,2],1],nodes[elements[:,3],1]],[nodes[elements[:,2],2],nodes[elements[:,3],2]])
    #plt.plot([nodes[elements[:,0],1],nodes[elements[:,3],1]],[nodes[elements[:,0],2],nodes[elements[:,3],2]])

    BCm = np.array([[2,1],[2,2],[2,3],[7,1],[7,2],[7,3]])   #Leading nodes
    BCs = np.array([[8,1],[8,2],[8,3],[9,1],[9,2],[9,3]])   #Following nodes
    BCd = np.array([[0,1],[0,2],[1,1],[1,2],[4,1],[4,2],[5,1],[5,2]])               #Node, dof with blocked displacements
    pre_forces = np.array([[3,1,force]])                        #Prescribed forces [Node, dof, value of acting force]
    pre_disp = np.array([[3,1,0]])                              #Prescribed displacement [Node, dof, value of displacement]


    for row in range(0,elements.shape[0]):
        ## Nodes
        n = 0       #this will allow us to go through rows in node list

        lh = nodes[elements[row,1],1]-nodes[elements[row,0],1]
        lv = nodes[elements[row,2],2]-nodes[elements[row,0],2]

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

    ## Plot
    #plt.plot([NL[EL[:,0],1],NL[EL[:,1],1]],[NL[EL[:,0],2],NL[EL[:,1],2]])
    #plt.plot([NL[EL[:,1],1],NL[EL[:,2],1]],[NL[EL[:,1],2],NL[EL[:,2],2]])
    #plt.plot([NL[EL[:,2],1],NL[EL[:,3],1]],[NL[EL[:,2],2],NL[EL[:,3],2]])
    #plt.plot([NL[EL[:,0],1],NL[EL[:,3],1]],[NL[EL[:,0],2],NL[EL[:,3],2]])


elif type == 'beam':

    # Node geometry
    NL = np.zeros((6,3))
    NL[0,:] = [0, 0.0, 0.0]
    NL[1,:] = [1, 0.0, 1.0]
    NL[2,:] = [2, 1.0, 0.0]
    NL[3,:] = [3, 1.0, 1.0]
    NL[4,:] = [4, 0.0, 1.0]
    NL[5,:] = [5, 1.0, 1.0]

    # Elements
    EL = np.zeros((3,2),dtype=int)
    EL[0,0] = NL[0,0]
    EL[0,1] = NL[1,0]
    EL[1,0] = NL[2,0]
    EL[1,1] = NL[3,0]
    EL[2,0] = NL[4,0]
    EL[2,1] = NL[5,0]

    # Plot the initial geometry
    plt.plot([nodes[elements[:,0],1],nodes[elements[:,1],1]],[nodes[elements[:,0],2],nodes[elements[:,1],2]])

    # Boundary conditions
    BCd = np.array([[0,1],[0,2],[2,1],[2,2]])               #Node, dof with blocked displacements
    BCm = np.array([[1,1],[1,2],[1,3],[3,1],[3,2],[3,3]])   #Leading nodes
    BCs = np.array([[4,1],[4,2],[4,3],[5,1],[5,2],[5,3]])   #Following nodes
    pre_forces = np.array([[1,1,force]])                        #Prescribed forces [Node, dof, value of acting force]
    pre_disp = np.array([[1,1,0]])                              #Prescribed displacement [Node, dof, value of displacement]


## Main code
myModel = model(EL,NL,BCd,params,BCm,BCs,pre_forces,pre_disp,type)
pos = NL
for i in range(NL.shape[0]):
    for j in range(myModel.df_dof.shape[0]):
        if pos[i,0] == myModel.df_dof[j,0]:
            if myModel.df_dof[j,1] == 1:
                pos[i,1] = pos[i,1] + myModel.df_dof[j,2]
            elif myModel.df_dof[j,1] ==2:
                pos[i,2] = pos[i,2] + myModel.df_dof[j,2]
            elif myModel.df_dof[j,1] == 3:
                pos[i,1] = pos[i,1] + myModel.df_dof[j,2]*(pos[i,0]-lh/2)
                pos[i,2] = pos[i,2] + myModel.df_dof[j,2]*(pos[i,0]-lv/2)

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

#%% Membrane based model

import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt


## Initialization
# Parameters
E = 200e9   #Young's modulus [Pa]
nu = 0.3    #Poisson's ratio 
h = 10 #0.1     #Height of beam [m]
b = 10 #0.1     #Width of beam [m]
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
  'poisson' : nu,
  'thickness' : t}

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

p = 10       #number of elements in which the horizontal line will be divided
m = 10       #number of elements in which the vertical line will be divided
NoN = (p+1)*(m+1)                #number of nodes
NoE = p*m                        #number of elements
NL = np.zeros((13*NoN, 3),dtype="int")    #extended node list
EL = np.zeros((13*NoE, 4),dtype="int")                  #extended element list 


for row in range(0,eTop.shape[0]):
    ## Nodes
    n = 0       #this will allow us to go through rows in node list

    lh = coord[eTop[row,1],1]-coord[eTop[row,0],1]
    lv = coord[eTop[row,1],2]-coord[eTop[row,0],2]
    if lh == 0:
        lh = h
    if lv == 0:
        lv = b

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
                EL[row*NoE+(i*p), 0] = row*NoN + i*p
                EL[row*NoE+(i*p), 1] = EL[row*NoE, 0] + 1
                EL[row*NoE+(i*p), 3] = row*NoN + (i+1)*p
                EL[row*NoE+(i*p), 2] = EL[row*NoE, 3] + 1
            else:
                EL[row*NoE+(i*p)+j, 0] = EL[row*NoE+(i*p)+(j-1), 1]
                EL[row*NoE+(i*p)+j, 1] = EL[row*NoE+(i*p)+j, 0] + 1
                EL[row*NoE+(i*p)+j, 3] = EL[row*NoE+(i*p)+(j-1), 2]
                EL[row*NoE+(i*p)+j, 2] = EL[row*NoE+(i*p)+j, 3] + 1





class membrane2D:

    def __init__(self, params, node):
        self.E = params['youngs_modulus']
        self.nu = params['poisson']        
        self.t = params['thickness']
        self.NPE = 4                        #nodes per element
        self.PD = 2                         #problem dimension
        self.node_ID = node[:,0].astype(int)
        self.node_coord = node[:,1:3]
        self.dof = np.array([[self.node_ID[0],1],   #dof 1 = axial load
                             [self.node_ID[0],2],   #dof 2 = bending 
                             [self.node_ID[1],1],
                             [self.node_ID[1],2],
                             [self.node_ID[2],1],   
                             [self.node_ID[2],2],   
                             [self.node_ID[3],1],
                             [self.node_ID[3],2]],dtype="int")
        self.a = (self.node_coord[0,0]-self.node_coord[1,0])   #increment in horizontal direction (length of element)
        self.b = (self.node_coord[3,1]-self.node_coord[0,1])   #increment in vertical direction (length of element)

        self.D = np.array([[self.E/(1-self.nu**2),          self.nu*self.E/(1-self.nu**2),  0],         #plane stress
                           [self.nu*self.E/(1-self.nu**2),  self.E/(1-self.nu**2),          0],
                           [0,                              0,                              self.E/(2*(1+self.nu))]])
        self.compute_K() 


    def compute_K(self):
        self.Ke = np.zeros((8,8))                     #element stiffness
        N = lambda s, t : 1/4* np.array([[-(1-t),  (1-t), (1+t), -(1+t)],
                                          [-(1-s), -(1+s), (1+s),  (1-s)]])

        # quadrature rule
        r,w = self.GaussPoints(2)

        # numerical ingration
        for si,wi in zip(r,w):
            for tj,wj in zip(r,w):

                # Jacobian matrix [dx/ds,dx/dt;dy/ds,dy/dt]
                J = N(si,tj) @ self.node_coord

                Bs = np.zeros((4,8))
                Bs[0,[0,2,4,6]] = N(si,tj)[0,:] #dphi_ds_val
                Bs[1,[0,2,4,6]] = N(si,tj)[1,:] #dphi_dt_val
                Bs[2,[1,3,5,7]] = N(si,tj)[0,:]
                Bs[3,[1,3,5,7]] = N(si,tj)[1,:]

                B = np.array([[1,0,0,0],[0,0,0,1],[0,1,1,0]]) @ sp.linalg.block_diag(np.linalg.inv(J),np.linalg.inv(J)) @ Bs

                self.Ke += self.t * B.transpose() @ self.D @ B * np.linalg.det(J) * wi * wj

    def GaussPoints(self,order):
        # quadrature rules in 1D (2D rules are obtained by combining 1Ds as in a grid)
        if order == 1:
            r = np.array([0.0])
            w = np.array([2.0])
        elif order == 2:
            r = np.array([-1/math.sqrt(3),+1/math.sqrt(3)])
            w = np.array([1.0,1.0])

        return r,w



## Main code
for i in range(0,EL.shape[0]):
    myModel = membrane2D(params,NL[EL[i,:],:])

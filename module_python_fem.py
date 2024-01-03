import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import dill
from scipy.sparse.linalg import spsolve

# Function for conversion of dense to sparse matrix
def dense2sparse(A):

    row_ind,col_ind = np.nonzero(A)
    data = A[row_ind,col_ind]
    B = sp.sparse.csr_matrix((data,(row_ind,col_ind)),shape=A.shape)

    return B

class model:

    def __init__(self,etop,ncoord,BCd,params,BCm,BCs,force_dof,disp_dof,type):
        self.BCd = BCd
        self.BCm = BCm
        self.BCs = BCs
        self.BCf = force_dof[:,0:2]
        self.fc = force_dof[:,[2]]  #controlled
        self.BCu = disp_dof[:,0:2]
        self.uc = disp_dof[:,[2]]   #controlled
        self.EL = etop
        self.model_dofs = np.zeros((0,2),dtype=int)
        
        self.get_unique_dof(params,ncoord,type)
        self.stiffness_matrix()
        
    # Get unique DOF list
    def get_unique_dof(self,params,ncoord,type):
        self.myElements = []

        for i in range(0,self.EL.shape[0]):
            if type == 'beam':
                self.myElements.append(beam2D(params,ncoord[self.EL[i,:],:]))
            elif type == 'solid':
                self.myElements.append(solid2D(params,ncoord[self.EL[i,:],:]))

            #Replace following with leading nodes
            for idr, element_row in enumerate(self.myElements[i].dof):
                for j, row in enumerate(self.BCs):
                    if np.array_equal(element_row,row):
                        self.myElements[i].dof[idr] = self.BCm[j]                                #Replace BCs with BCm

            self.model_dofs = np.concatenate((self.model_dofs,self.myElements[i].dof),axis=0)      #List of all dofs

        self.model_dofs = np.unique(self.model_dofs,axis=0)                                        #Remove double dofs

        #Remove BC dofs
        for i, row in enumerate(self.BCd):
            for j, model_row in enumerate(self.model_dofs):
                if np.array_equal(row,model_row):
                    self.model_dofs = np.delete(self.model_dofs, j, axis=0)

    # Compute stiffness matrix
    def stiffness_matrix(self):
        self.K = np.zeros((len(self.model_dofs),len(self.model_dofs)))

        for i in range(0,self.EL.shape[0]):
            self.myElements[i].compute_Z(self.model_dofs)
            Ze_sp = self.myElements[i].Ze
            Ke_sp = self.myElements[i].Ke

            self.K += Ze_sp.transpose() @ Ke_sp @ Ze_sp   #Global stiffness matrix

    def compute_Zf(self):
        # compute collocation matrix for controlled forces
        if len(self.fc) == 0:
            self.Zf = np.zeros((0,self.model_dofs.shape[0]))
        else:
            self.row_index = []
            self.col_index = []

            for i,fdof in enumerate(self.BCf):
                for j,mdof in enumerate(self.model_dofs):
                    if np.array_equal(fdof,mdof):
                        self.row_index.append(i)
                        self.col_index.append(j)           

            self.Zf = sp.sparse.csr_matrix((np.ones((len(self.row_index))),(self.row_index,self.col_index)),shape=(self.BCf.shape[0],self.model_dofs.shape[0]))     

    def compute_Zu(self):
        #Compute collocation matrix for controlled displacements
        if len(self.uc) == 0:
            self.Zu = np.zeros((0,self.model_dofs.shape[0]))
        else:
            self.row_ind = []
            self.col_ind = []

            for i,udof in enumerate(self.BCu):
                for j,mdof in enumerate(self.model_dofs):
                    if np.array_equal(udof,mdof):
                        self.row_ind.append(i)
                        self.col_ind.append(j)

            self.Zu = sp.sparse.csr_matrix((np.ones((len(self.row_ind))),(self.row_ind,self.col_ind)),shape=(self.BCu.shape[0],self.model_dofs.shape[0]))

    def compute_displacement(self):
        self.compute_Zf()       #collocation matrix for controlled forces
        self.compute_Zu()       #collocation matrix for controlled displacement

        #Only force controlled dofs
        if self.Zu.shape[0] == 0 and self.Zf.shape[0] != 0 :
            self.u = spsolve(self.K, self.Zf.transpose() @ self.fc)
            self.l = np.zeros((0,1))

        #Only displacement controlled dofs
        elif self.Zu.shape[0] != 0 and self.Zf.shape[0] == 0 :
            Kiuu = self.Zu @ spsolve(self.K,self.Zu.transpose())
            self.l = spsolve(Kiuu,self.uc)
            self.u = spsolve(self.K,self.Zu.transpose() @ self.l)

        #Both force and displacement controlled dofs
        elif self.Zu.shape[0] != 0 and self.Zf.shape[0] != 0 :
            Kiuf = self.Zu @ spsolve(self.K,self.Zf.transpose())
            Kiuu = self.Zu @ spsolve(self.K,self.Zu.transpose())
            self.l = spsolve(Kiuu,self.uc - Kiuf @ self.fc)
            self.u = np.linalg.lstsq(self.K,self.Zu.transpose() @ self.l + self.Zf.transpose() @ self.fc)[0][:,0]

    def compute_stress(self):
        self.sigma = np.zeros((self.EL.shape[0]*4,3))
        for i in range(0,self.EL.shape[0]):
            self.myElements[i].calculate_stress(self.myElements[i].Ze @ self.u)
            self.sigma[4*i:4*i+4] = self.myElements[i].stress
            

    def plot(self,uscale,x1,x2,y1,y2):
        self.fig, self.ax = plt.subplots()

        for myElement in self.myElements:
            myElement.plot(self.ax, myElement.Ze @ self.u, uscale)

        self.ax.set_xlim(x1, x2)
        self.ax.set_ylim(y1, y2)
        plt.show()


    def plot_stresses(self, x1, x2, y1, y2, direction):
        self.fig1, self.ax1 = plt.subplots()

        max_value = np.max(self.sigma[:,direction])
        min_value = np.min(self.sigma[:,direction])

        for myElement in self.myElements:
            myElement.plot_stress(self.ax1, min_value, max_value, direction, myElement.Ze @ self.u)

        im = plt.imshow([self.sigma], cmap='viridis', aspect='auto')
        plt.colorbar(im, label='Normalized Stress Values')

        self.ax1.set_xlim(x1, x2)
        self.ax1.set_ylim(y1, y2)
        plt.show()


class beam2D:

    def __init__(self, params, node):
        self.I = params['inertia']
        self.E = params['youngs_modulus']
        self.A = params['area']
        self.node_ID = node[:,0].astype(np.int32)
        self.node_coord = node[:,1:3]
        self.dof = np.array([[self.node_ID[0],0],   #dof 0 = axial load
                             [self.node_ID[0],1],   #dof 1 = bending 
                             [self.node_ID[0],2],   #dof 2 = moment 
                             [self.node_ID[1],0],
                             [self.node_ID[1],1],
                             [self.node_ID[1],2]],dtype="int")
        self.L = np.linalg.norm(self.node_coord[1,:] - self.node_coord[0,:])
        self.s = (self.node_coord[[1],:] - self.node_coord[[0],:])/self.L
        self.t = np.array([[-self.s[0,1], self.s[0,0]]])
        disp = np.zeros((1,12))

        # Rotation matrix
        one = [[1]]
        self.T = np.concatenate((self.s,self.t),axis=0)
        self.R = sp.linalg.block_diag(self.T,one,self.T,one)

        # Compute the stiffness matrix
        self.compute_K(disp)

    def compute_K(self):
        # Local element stiffness matrix
        self.Kl = np.array([[self.E*self.A/self.L, 0,0,-self.E*self.A/self.L,0,0],
                            [0, 12*self.E*self.I/self.L**3, 6*self.E*self.I/self.L**2, 0, -12*self.E*self.I/self.L**3, 6*self.E*self.I/self.L**2],
                            [0, 6*self.E*self.I/self.L**2,  4*self.E*self.I/self.L,    0, -6*self.E*self.I/self.L**2,   2*self.E*self.I/self.L   ],
                            [-self.E*self.A/self.L, 0,0,self.E*self.A/self.L,0,0],
                            [0, -12*self.E*self.I/self.L**3, -6*self.E*self.I/self.L**2, 0,  12*self.E*self.I/self.L**3,  -6*self.E*self.I/self.L**2],
                            [0, 6*self.E*self.I/self.L**2,  2*self.E*self.I/self.L,      0,  -6*self.E*self.I/self.L**2,   4*self.E*self.I/self.L    ]],dtype="float")

        # Global element stiffness matrix
        self.Ke = dense2sparse(self.R.transpose() @ self.Kl @ self.R)

    def compute_Z(self,modeldofs):
        self.row_index = []
        self.col_index = []

        for i,edof in enumerate(self.dof):
            for j,mdof in enumerate(modeldofs):
                if np.array_equal(edof,mdof):
                    self.row_index.append(i)
                    self.col_index.append(j)

        self.Ze = sp.sparse.csr_matrix((np.ones((len(self.row_index))),(self.row_index,self.col_index)),shape=(self.dof.shape[0],modeldofs.shape[0]))

    def plot(self,ax,ue,uscale):

        # Add the polygon patch to the axes
        ax.add_patch(patches.Polygon(self.node_coord[:,0:2], color='blue', alpha=0.5))

        # Update position
        pos = self.node_coord
        pos[0,0] = pos[0,0] + ue[0] * uscale
        pos[0,1] = pos[0,1] + ue[1] * uscale
        pos[1,0] = pos[1,0] + ue[3] * uscale
        pos[1,1] = pos[1,1] + ue[4] * uscale

        ax.add_patch(patches.Polygon(pos[:,0:2], color='red', alpha=0.5))

class solid2D:

    def __init__(self, params, node):
        self.E = params['youngs_modulus']
        self.nu = params['poisson']        
        self.t = params['thickness']
        self.node_ID = node[:,0].astype(int)
        self.node_coord = node[:,1:3]
        self.dof = np.array([[self.node_ID[0],0],   #dof 0 = axial load
                             [self.node_ID[0],1],   #dof 1 = bending 
                             [self.node_ID[1],0],
                             [self.node_ID[1],1],
                             [self.node_ID[2],0],   
                             [self.node_ID[2],1],   
                             [self.node_ID[3],0],
                             [self.node_ID[3],1]],dtype="int")
        disp = np.zeros((1,8))
        self.compute_K(disp) 


    def compute_K(self,disp):
        self.Ke = np.zeros((8,8))                     #element stiffness
        Ke_b = np.zeros((8,8))

        #Derivative of shape functions 
        N = lambda s, t : 1/4* np.array([[-(1-t),  (1-t), (1+t), -(1+t)],
                                         [-(1-s), -(1+s), (1+s),  (1-s)]])
        #Plane stress
        D = np.array([[self.E/(1-self.nu**2),          self.nu*self.E/(1-self.nu**2),  0],         #plane stress
                      [self.nu*self.E/(1-self.nu**2),  self.E/(1-self.nu**2),          0],
                      [0,                              0,                              self.E/(2*(1+self.nu))]])   

        #Quadrature rule (shear)
        r,w = self.GaussPoints(1)

        #Jacobian matrix [dx/ds,dx/dt;dy/ds,dy/dt]
        Jsh = N(r,r) @ self.node_coord

        Bssh = np.zeros((4,8))
        Bssh[0,[0,2,4,6]] = N(r,r)[0,:] #dphi_ds_val
        Bssh[1,[0,2,4,6]] = N(r,r)[1,:] #dphi_dt_val
        Bssh[2,[1,3,5,7]] = N(r,r)[0,:]
        Bssh[3,[1,3,5,7]] = N(r,r)[1,:]

        Bsh = np.array([[0,1,1,0]]) @ sp.linalg.block_diag(np.linalg.inv(Jsh),np.linalg.inv(Jsh)) @ Bssh

        Ke_sh = self.t * Bsh.transpose() * D[2,2]  * Bsh * np.linalg.det(Jsh) * w * w 
        
        #Quadrature rule (bending)
        r,w = self.GaussPoints(2)

        #Numerical ingration
        self.stress = np.zeros((4,3))
        n = 0
        for si,wi in zip(r,w):
            for tj,wj in zip(r,w):

                # Jacobian matrix [dx/ds,dx/dt;dy/ds,dy/dt]
                J = N(si,tj) @ self.node_coord

                Bs = np.zeros((4,8))
                Bs[0,[0,2,4,6]] = N(si,tj)[0,:] #dphi_ds_val
                Bs[1,[0,2,4,6]] = N(si,tj)[1,:] #dphi_dt_val
                Bs[2,[1,3,5,7]] = N(si,tj)[0,:]
                Bs[3,[1,3,5,7]] = N(si,tj)[1,:]

                Bb = np.array([[1,0,0,0],[0,0,0,1]]) @ sp.linalg.block_diag(np.linalg.inv(J),np.linalg.inv(J)) @ Bs

                Ke_b += self.t * Bb.transpose() @ D[0:2,0:2] @ Bb * np.linalg.det(J) * wi * wj 

                #set displacement to zero, and run again later with known displacement. calculate the stress here
                B = np.concatenate((Bb,Bsh),axis=0)
                strain = B @ disp.transpose()
                stress = D @ strain
                self.stress[n,:] = stress.transpose()
                n += 1

        self.Ke = Ke_b + Ke_sh

    def GaussPoints(self,order):
        # quadrature rules in 1D (2D rules are obtained by combining 1Ds as in a grid)
        if order == 1:
            r = 0.0
            w = 2.0 
        elif order == 2:
            r = np.array([-1/np.sqrt(3),1/np.sqrt(3)])
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

        self.Ze = sp.sparse.csr_matrix((np.ones((len(self.row_index))),(self.row_index,self.col_index)),shape=(self.dof.shape[0],modeldofs.shape[0]))

    def calculate_stress(self,u):
        self.compute_K(u)
    
    def plot(self,ax,ue,uscale):

        # Add the polygon patch to the axes
        ax.add_patch(patches.Polygon(self.node_coord[:,0:2],color='blue', alpha=0.5)) 

        # Update position
        self.pos = self.node_coord
        self.pos[0,0] = self.pos[0,0] + ue[0] * uscale
        self.pos[0,1] = self.pos[0,1] + ue[1] * uscale
        self.pos[1,0] = self.pos[1,0] + ue[2] * uscale
        self.pos[1,1] = self.pos[1,1] + ue[3] * uscale
        self.pos[2,0] = self.pos[2,0] + ue[4] * uscale
        self.pos[2,1] = self.pos[2,1] + ue[5] * uscale
        self.pos[3,0] = self.pos[3,0] + ue[6] * uscale
        self.pos[3,1] = self.pos[3,1] + ue[7] * uscale

        ax.add_patch(patches.Polygon(self.pos[:,0:2],color='red', alpha=0.5))

    def plot_stress(self,ax,stress_min,stress_max,direction,u):
        self.calculate_stress(u)
        avg_stress = (self.stress[0,direction] + self.stress[1,direction] + self.stress[2,direction] + self.stress[3,direction])/4.0
        stress_norm = np.interp(avg_stress,np.array([stress_min,stress_max]),np.array([0.0,1.0]))

        ax.add_patch(patches.Polygon(self.pos[:,0:2], closed=True, edgecolor='black', facecolor=plt.cm.viridis(stress_norm)))



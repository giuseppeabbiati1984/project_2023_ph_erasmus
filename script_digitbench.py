#%%

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

class Node:
    __ID = 0

    def __init__(self, coord):
        self.__ID = Node.__ID
        Node.__ID += 1
        self.x = coord[self.__ID,0]
        self.y = coord[self.__ID,1]
        self.__coord = coord[self.__ID,:]

    def get_id(self):
        return self.__ID

    def get_coordinate(self):
        return self.__coord

## Bernoulli beam
class beam2D:
#    __ID = 0

    def __init__(self, E,I,bar,node):
        self.__I = I
        self.__E = E
        self.node = node.astype(float)
        self.bar = eTop.astype(int)
        self.dof = 2
        self.point_load = np.zeros_like(node)
        self.support = np.ones_like(node).astype(int)
        self.force = np.zeros([len(bar), 2*self.dof])           #contains shear force in first en third column and bending moment in second and fourth
        self.displacement = np.zeros([len(bar), 2*self.dof])    #first and third column are translation and second and fourth are rotation

#        self.__ID = beam2D.__ID
#        beam2D.__ID += 1

    def analysis(self):
        nNodes = len(self.node)
        nElem = len(self.bar)
        ndof = self.dof * nNodes
        d = np.zeros_like(self.bar)
        length = np.zeros([nElem,1])

        # Structural stiffness
        k = np.zeros([nElem, 2*self.dof, 2*self.dof])
        ss = np.zeros([ndof, ndof])
        for i in range(nElem):
            # Length
            d[i,:] = self.node[self.bar[i,1],:] - self.node[self.bar[i,0],:]
            length[i] = np.sqrt(d[i,0]**2+d[i,1]**2)

            #DOF
            aux = self.dof * self.bar[i,:]
            index = np.r_[aux[0]:aux[0]+self.dof, aux[1]:aux[1] + self.dof]

            #Element stiffness matrix
            k[i,:,:] = self.__E*self.__I/length[i]**3 * np.array([[12*length[i], 6*length[i], -12, 6*length[i]],
                                                              [6*length[i], 4*length[i]**2, -6*length[i], 2*length[i]**2],
                                                              [-12, -6*length[i], 12, -6*length[i]],
                                                              [6*length[i], 2*length[i]**2, -6, 4*length[i]**2]],dtype="object")

            #Global stiffness matrix
            ss[np.ix_(index, index)] += k[i]

        # Solution
        free_dof = self.support.flatten().nonzero()[0]
        kff = ss[np.ix_(free_dof, free_dof)]
        p = self.point_load.flatten()                   #load
        pf = p[free_dof]
        uf = np.linalg.solve(kff,pf)
        u = self.support.astype(float).flatten()
        u[free_dof] = uf
        u = u.reshape(nNodes, self.dof)
        u_elem = np.concatenate((u[self.bar[:,0]], u[self.bar[:,1]]), axis=1)
        for i in range(nElem):
            self.force[i] = np.dot(k[i],u_elem[i])
            self.displacement[i] = u_elem[i]

    def plot(self, scale=None):
        nElem = len(self.bar)
        fig, axs = plt.subplots(3)
        # Deformed shape
        for i in range(nElem):
            xi, xf = self.node[self.bar[i,0],0], self.node[self.bar[i,1],0] #x-coordinate of start point and end point of 1 element
            yi, yf = self.node[self.bar[i,0],1], self.node[self.bar[i,1],1] #y-coordinate of start point and end point of 1 element
            axs[0].plot([xi,xf],[yi,yf],'b',linewidth=1)
        for i in range(nElem):
            dxi, dxf = self.node[self.bar[i,0],0], self.node[self.bar[i,1],0]
            dyi = self.node[self.bar[i,0],1] + self.displacement[i,0]*scale
            dyf = self.node[self.bar[i,1],1] + self.displacement[i,2]*scale
            axs[0].plot([dxi,dxf],[dyi,dyf],'r',linewidth=2)
            axs[0].text(dxi,dyi, str(round(dyi/scale,4)),rotation=90)
        # Bending moment
        axs[1].invert_yaxis()
        for i in range(nElem):
            mxi, mxf = self.node[self.bar[i,0],0], self.node[self.bar[i,1],0] #x-coordinate of start point and end point of 1 element
            myi, myf = self.node[self.bar[i,0],1], self.node[self.bar[i,1],1] #y-coordinate of start point and end point of 1 element
            axs[1].plot([mxi,mxf],[myi,myf],'b',linewidth=1)
        for i in range(nElem):
            dmxi, dmxf = self.node[self.bar[i,0],0], self.node[self.bar[i,1],0]
            dmyi = -self.force[i,1]
            dmyf = self.force[i,3]
            axs[1].plot([dmxi, dmxi, dmxf ,dmxf],[0, dmyi,dmyf,0],'g',linewidth=1)
            axs[1].fill([dmxi, dmxi, dmxf ,dmxf],[0, dmyi,dmyf,0],'c',alpha=0.3)
            axs[1].text(dmxi,dmyi, str(round(dmyi,4)),rotation=90)
        # Shear force
        for i in range(nElem):
            fxi, fxf = self.node[self.bar[i,0],0], self.node[self.bar[i,1],0] #x-coordinate of start point and end point of 1 element
            fyi, fyf = self.node[self.bar[i,0],1], self.node[self.bar[i,1],1] #y-coordinate of start point and end point of 1 element
            axs[2].plot([fxi,fxf],[fyi,fyf],'b',linewidth=1)
        for i in range(nElem):
            dfxi, dfxf = self.node[self.bar[i,0],0], self.node[self.bar[i,1],0]
            dfyi = -self.force[i,0]
            dfyf = self.force[i,2]
            axs[2].plot([dfxi, dfxi, dfxf ,dfxf],[0, dfyi,dfyf,0],'r',linewidth=1)
            axs[2].fill([dfxi, dfxi, dfxf ,dfxf],[0, dfyi,dfyf,0],'orange',alpha=0.3)
            axs[2].text(dfxi,dfyi, str(round(dfyi,4)),rotation=90)


## Initialization
# Parameters
E = 200e9   #Young's modulus [Pa]
h = 0.1     #Height of beam [m]
b = 0.1     #Width of beam [m]
Lh = 1642   #Length of horizontal beams [m]
Lv = 1786   #Length of vertical beams [m]
t = 0.004   #Thickness of beam [m]
I = 1/12 * (b*h**3 - (b-2*t)*(h-2*t)**3)     #Second moment of inertia [m4]

# Node geometry
coord = np.zeros((11,2))
coord[1,:] = [996.0,0.0]
coord[2,:] = [Lh,0.0]
coord[3,:] = [Lh,Lv]
coord[4,:] = [Lh-330,Lv]
coord[5,:] = [0.0,Lv]
coord[6,:] = [0.0,917.0]
coord[7,:] = [675.0,917.0]
coord[8,:] = [996.0-21.0,917.0]
coord[9,:] = [996.0,917.0]
coord[10,:] = [996.0,660.0]

# Elements
eTop = np.zeros((11+1,2))
for i in range (11-1):
    eTop[i,:] = [i, i+1]
eTop[10,:] = [10, 1]
eTop[11,:] = [6, 0]


## Main code
beam_1 = beam2D(E, I, eTop, coord)

point_load_1 = beam_1.point_load
point_load_1[1,0] = -100
point_load_1[6,0] = -100
point_load_1[4,0] = -100

support_1 = beam_1.support
support_1[0,:] = 0          #fixed
support_1[5,:] = 0          #fixed

beam_1.analysis()
beam_1.plot(5)



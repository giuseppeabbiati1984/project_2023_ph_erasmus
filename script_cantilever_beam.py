#%% Cantilever beam
from module_python_fem import *
from script_manipulator_sympy import *

## Initialization
# Parameters
E = 200e9       #Young's modulus [Pa]
nu = 0.3        #Poisson ratio
h = 10          #Height of beam [mm]
b = 10          #Width of beam [mm]
t = 0.4         #Thickness of beam [mm]
I = 1/12 * (b*h**3 - (b-2*t)*(h-2*t)**3)     #Second moment of inertia [mm4]
A = b*h         #Area [mm2]
force = 100
type = 'solid'  #beam or solid

params = {
'youngs_modulus': E,
'inertia': I,
'area' : A,
'poisson' : nu,
'thickness' : t}

if type == 'solid':

    # Node geometry
    nodes = np.zeros((4,3),dtype="float")
    nodes[0,:] = [0, 0.0, 0.0]
    nodes[1,:] = [1, 100, 0]
    nodes[2,:] = [2, 0, 10]
    nodes[3,:] = [3, 100, 10]

    # Elements
    elements = np.zeros((1,4),dtype="int")
    elements[0,:] = [0, 1, 3, 2]

    # Parameters
    kr = 5                           #number of elements in which the shorter side will be divided
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
        
    BCf =  np.zeros((0,3)) #np.array([[(kr+1)*(lg+1)-1,1,-force]])      #Prescribed forces [Node, dof, value of acting force]
    pre_disp = np.array([[(kr+1)*(lg+1)-1,1,-10**(-5)]])  #np.zeros((0,3))                    #Prescribed displacement [Node, dof, value of displacement]

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
    NoE = 1
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
    BCf = np.zeros((0,3)) #np.array([[NoE,0,-force]])             #Prescribed forces [Node, dof, value of acting force]
    #displacement = eval_rhs(ti_val,xi_val,mb_val,kb_val,zb_val,lb_val,mt_val,kt_val,zt_val,Ig_val,dg_val,fw_val,g_val,myModel.l)
    pre_disp = np.array([[NoE,0,-10**(-6)],[NoE,1,-10**(-6)]])                         #Prescribed displacement [Node, dof, value of displacement]



## Main code
myModel = model(EL,NL,BCd,params,BCm,BCs,BCf,pre_disp,type)

x1 = -1
x2 = 120
y1 = -120
y2 = 20
direction = 0 #0=sigma_xx, 1=sigma_yy, 2=gamma_xy
myModel.compute_displacement()
myModel.compute_stress()

#resulting force into the frame at the actuator connection
#tau,theta1,theta2 = calculate_reaction_force(pre_disp[0,2],pre_disp[1,2],myModel.l[-2],myModel.l[-1])

#plot
myModel.plot(1e7,x1,x2,y1,y2)
myModel.plot_stresses(x1,x2,y1,y2,direction)



#%% Analytical check
#analytical_displacement = force*100**3/(3*E*I)
#analytical_rotation = force*100**2/(2*E*I)

#analytical_displacement_solid = force*100.0**3/(3.0*E*(1/12)*0.4*10.0**3)


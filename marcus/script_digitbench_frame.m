%% input

% here I define my model

close all
clear
clc

b1 = 0.6 ;
b2 = 1.0 ;
b3 = 0.2 ;
h1 = 0.6 ;
h2 = 0.2 ;
h3 = 0.4 ;

P = 100 ; % fy applied to node 9 [N]
M = 1000 ; % mz applied to node 9 [Nm]

% commont to all elements
ePar_ref.b = 0.100 ;
ePar_ref.h = 0.100 ;
ePar_ref.s = 0.005 ;
ePar_ref.E = 210e9 ;
ePar_ref.Nu = 0.3 ;

% degrees of freedom number vs meaning
% dofs = [1,2,3] -> ux,uy,rz

% 9                 10
% o------------------o   /
% |                  |   |
% |                  |   |h3
% |      7           |8  |
% |     o            o   /
% |     |            |   |
% |     |            |   |h2
% |     |  b3        |   |
% o-----o-----o      |   /
% |4    |5     6     |   |
% |     |            |   |h1
% |     |            |   |
% o-----o------------o   /
% 1  b1 2      b2    3

nCoord = [ 0    ,0  ; % n1
           b1   ,0  ; % n2
           b1+b2,0  ; % n3
           0    ,h1 ; % n4
           b1   ,h1 ; % n5
           b1+b3,h1 ; % n6
           b1   ,h1+h2 ; % n7           
           b1+b2,h1+h2 ; % n8
           0    ,h1+h2+h3 ; % n9
           b1+b2,h1+h2+h3 ] ; % n10

eTop = [1,4 ; %e1 - vertical
        4,9 ; %e2 - vertical
        2,5 ; %e3 - vertical
        5,7 ; %e4 - vertical
        3,8 ; %e5 - vertical
        8,10 ; %e6 - vertical
        1,2 ; %e7 - horizontal
        2,3 ; %e8 - horizontal
        4,5 ; %e9 - horizontal
        5,6 ; %e10 - horizontal
        9,10] ; %e11 - horizontal

% node, dofs, type (1=force, 0=displacement), value
BC = [1,1,0,0 ; % node 1 direction 1 (ux) displacement-BC set to 0
      1,2,0,0 ;
      2,1,0,0 ;
      2,2,0,0 ;
      3,1,0,0 ;
      3,2,0,0 ;
      9,2,1,P ; % node 9 direction 2 (fy) force-BC set to P
      9,3,1,M ] ; % node 9 direction 3 (mz) force-BC set to M

DofsPerNode = 3 ; % DoFs per node

%% pre-processor

% here I assemble stiffness matrix and load vectors

Ne = size(eTop,1) ; % number of elements
Nn = size(nCoord,1) ; % number of nodes
Ndof = Nn * DofsPerNode ; % number of DoFs

% DoFs mapping matrix (dof x node)
gDof = reshape(1:1:Ndof,DofsPerNode,Nn) ;

% initialization of model matrices
K = zeros(Ndof,Ndof) ;

for i = 1:1:Ne
    
    ePar{i}.id = i ; % identification number
    ePar{i} = ePar_ref ; % common parameters
    
    ePar{i}.xe = nCoord(eTop(i,:),:) ; % nodal coordinates of element i-th
    ePar{i}.eDof = gDof(:,eTop(i,:)) ; % DoFs indices of element i-th
    
    % element evaluation (K,fc,ft)
    ePar{i} = fun_rod2d(ePar{i}) ;
    
    % model stiffness assembly
    K(ePar{i}.eDof,ePar{i}.eDof) = ...
        K(ePar{i}.eDof,ePar{i}.eDof) + ePar{i}.Kg ;
        
end

% default BCs (force controlled DoFs with zero force applied)
Q = [ones(Ndof,1),...
    zeros(Ndof,1)] ;

% updated BCs (i take the info in BC and flush it into Q)
for i = 1:1:size(BC,1)
    Q(gDof(BC(i,2),BC(i,1)),:) = ...
        BC(i,[3,4]) ;
end

% indices of known displacements and forces
uKnown = Q(:,1) == 0 ;
fKnown = Q(:,1) == 1 ;

u = zeros(Ndof,1) ; 
u(uKnown,:) = Q(uKnown,2) ; % partial filling of u

f = zeros(Ndof,1) ; 
f(fKnown,:) = Q(fKnown,2) ; % partial filling of f

% figure
% hold all
% grid on
% spy(K)
% return

%% solver

% here we solve the static analysis problem (if possible!)

% stiffness matrix partitioning
Kff = K(fKnown,fKnown) ;
Kuf = K(uKnown,fKnown) ;
Kfu = K(fKnown,uKnown) ;
Kuu = K(uKnown,uKnown) ;

uu = u(uKnown,1) ;
ff = f(fKnown,1) ;

% if det(Kff) ~= 0
if rank(Kff) == size(Kff,1)
    
    % displacements of unconstrained DoFs
    uf = Kff \ (ff - Kfu * uu) ; % from the 2nd row-block
    
    % reactions of constrained DoFs
    fu = Kuu * uu + Kuf * uf ; % from the 1st row-block
    
    u(fKnown,1) = uf ;
    f(uKnown,1) = fu ;
    
else
    
    error('the stiffness matrix is singular')
    
end

for i = 1:1:numel(ePar)
    ePar{i} = fun_rod2d(ePar{i},u(ePar{i}.eDof,1)) ;
end

%% post-processor

opts.scaleU = 0.1 ;
opts.scaleF = 0.1 ;
opts.type = 'none' ;

fun_rod2d_plot(nCoord,ePar,gDof,u,f,fKnown,uKnown,opts)

save digitbench_frame gDof K

return

%% code prototype for the FMU

close all
clear
clc

load digitbench_frame K gDof

% K: unconstrained stiffness matrix
% gDof: mapping matrix (e.g., gDoF(3,5) provide the index of DoF 3 of node 5)

Ndof = size(K,1) ;

% here we set BCs:
BC = [1,1,0,0 ; % node 1 direction 1 (ux) displacement-BC set to 0
      1,2,0,0 ;
      2,1,0,0 ;
      2,2,0,0 ;
      3,1,0,0 ;
      3,2,0,0 ;
      9,2,1,100 ; % node 9 direction 2 (fy) force-BC set to P
      9,3,1,100*0.5 ] ; % node 9 direction 3 (mz) force-BC set to M

% default BCs (force controlled DoFs with zero force applied)
Q = [ones(Ndof,1),zeros(Ndof,1)] ;

% updated BCs (i take the info in BC and flush it into Q)
for i = 1:1:size(BC,1)
    Q(gDof(BC(i,2),BC(i,1)),:) = BC(i,[3,4]) ;
end

% indices of known displacements and forces
uKnown = Q(:,1) == 0 ;
fKnown = Q(:,1) == 1 ;

% application of BCs
u = zeros(Ndof,1) ; u(uKnown,:) = Q(uKnown,2) ; % set displacements where displacement are known
f = zeros(Ndof,1) ; f(fKnown,:) = Q(fKnown,2) ; % set forces where forces are known

% stiffness matrix partitioning
Kff = K(fKnown,fKnown) ;
Kuf = K(uKnown,fKnown) ;
Kfu = K(fKnown,uKnown) ;
Kuu = K(uKnown,uKnown) ;

uu = u(uKnown,1) ; 
ff = f(fKnown,1) ; 

% if det(Kff) ~= 0
if rank(Kff) == size(Kff,1)
    
    % displacements of unconstrained DoFs
    uf = Kff \ (ff - Kfu * uu) ; % from the 2nd row-block
    
    % reactions of constrained DoFs
    fu = Kuu * uu + Kuf * uf ; % from the 1st row-block
    
    u(fKnown,1) = uf ;
    f(uKnown,1) = fu ;
    
else
    
    error('the stiffness matrix is singular')
    
end

r2z = u(gDof(3,9))  % DoF 3 on node 9, displacement
m2z = f(gDof(3,9))  % DoF 3 on node 9, force

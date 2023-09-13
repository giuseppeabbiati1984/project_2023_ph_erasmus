%% input

% here I define my model

close all
clear
clc

b1 = 0.5 ;

P = -100 ; % fy applied to node 9 [N]

% commont to all elements
ePar_ref.b = 0.100 ;
ePar_ref.h = 0.100 ;
ePar_ref.s = 0.005 ;
ePar_ref.E = 210e9 ;
ePar_ref.Nu = 0.3 ;

% degrees of freedom number vs meaning
% dofs = [1,2,3] -> ux,uy,rz

% |P
% |
% v               //
% o-------------o //
% 1             2 //

nCoord = [ 0  , 0 ; % n1
           b1 , 0 ] ; % n2

eTop = [ 1 ,2 ] ; %e11 - horizontal

% node, dofs, type (1=force, 0=displacement), value
BC = [2,1,0,0 ; % node 2 direction 1 (ux) displacement-BC set to 0
      2,2,0,0 ;
      2,3,0,0 ;
      1,2,1,P ] ; % node 1 direction 2 (fy) force-BC set to P

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
    K(ePar{i}.eDof,ePar{i}.eDof) = K(ePar{i}.eDof,ePar{i}.eDof) + ePar{i}.Kg ;
        
end

% default BCs (force controlled DoFs with zero force applied)
Q = [ones(Ndof,1),...
    zeros(Ndof,1)] ;

% updated BCs (i take the info in BC and flush it into Q)
for i = 1:1:size(BC,1)
    Q(gDof(BC(i,2),BC(i,1)),:) = BC(i,[3,4]) ;
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

save digitbench_beam gDof K

%% post-processor

opts.scaleU = 0.1 ;
opts.scaleF = 0.1 ;
opts.type = 'none' ;

fun_rod2d_plot(nCoord,ePar,gDof,u,f,fKnown,uKnown,opts)

return

%% code prototype for the FMU

close all
clear
clc

load digitbench_beam K gDof

% K: unconstrained stiffness matrix
% gDof: mapping matrix (e.g., gDoF(3,5) provide the index of DoF 3 of node 5)

Ndof = size(K,1) ;

% node, dofs, type (1=force, 0=displacement), value
BC = [2,1,0,0 ; % node 2 direction 1 (ux) displacement-BC set to 0
      2,2,0,0 ;
      2,3,0,0 ;
      1,2,1,100 ] ; % node 1 direction 2 (fy) force-BC set to P

% default BCs (force controlled DoFs with zero force applied)
Q = [ones(Ndof,1),zeros(Ndof,1)] ;

% updated BCs (i take the info in BC and flush it into Q)
for i = 1:1:size(BC,1)
    Q(gDof(BC(i,2),BC(i,1)),:) = BC(i,[3,4]) ;
end

% indices of known displacements and forces
uKnown = Q(:,1) == 0 ;
fKnown = Q(:,1) == 1 ;

u = zeros(Ndof,1) ; 
u(uKnown,:) = Q(uKnown,2) ; % partial filling of u

f = zeros(Ndof,1) ; 
f(fKnown,:) = Q(fKnown,2) ; % partial filling of f

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

% here we read BCs:

u1x = u(gDof(1,1))  % horizontal displaxcement on node 1
u1y = u(gDof(2,1))  % vertical displaement on node 2
m2z = f(gDof(3,2))  % bending moment reaction on node 2
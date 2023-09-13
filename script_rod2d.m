%% input

% here I define my model

close all
clear
clc

b = 0.1 ;
h = 0.2 ;
E = 30e9 ;
P = 100 ; % externally applied load [N]
L = 0.2 ; % beam length [m]
f = 0 ; % uniformly distributed load [N/m]
n = 10 ; % number of elements

% commont to all elements
ePar_ref.b = b ;
ePar_ref.h = h ;
ePar_ref.E = E ;
ePar_ref.f = f ;

DofsPerNode = 3 ; % DoFs per node

nCoord = [0.0,0.0];
for i = 1:1:n
    nCoord=[nCoord;[L/n*i,0.0]] ;
end

eTop = [] ;
for i = 1:1:n
    eTop = [eTop;[i,i+1]] ; % element 1 is between nodes 1 and 2
end

% node, dofs, type (1=force, 0=displacement), value
BC = [1,1,0,0;
    1,2,0,0;
    1,3,0,0;
    n+1,2,1,P] ; % node 2 direction 2 has force-type BC of value P

%% pre-processor

% here I assemble stiffness matrix and load vectors

Ne = size(eTop,1) ; % number of elements
Nn = size(nCoord,1) ; % number of nodes
Ndof = Nn * DofsPerNode ; % number of DoFs

% DoFs mapping matrix (dof x node)
gDof = reshape(1:1:Ndof,DofsPerNode,Nn) ;

% initialization of model matrices
K = zeros(Ndof,Ndof) ;
fc = zeros(Ndof,1) ;
ft = zeros(Ndof,1) ;

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
    
    % consistent load vector assembly
    fc(ePar{i}.eDof,1) = fc(ePar{i}.eDof,1) + ePar{i}.fcg ;
    
    % thermal load vector assembly
    ft(ePar{i}.eDof,1) = ft(ePar{i}.eDof,1) + ePar{i}.ftg ;
    
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

u = zeros(Ndof,1) ; u(uKnown,:) = Q(uKnown,2) ; % partial filling of u
fb = zeros(Ndof,1) ; fb(fKnown,:) = Q(fKnown,2) ; % partial filling of f

%% solver

% here we solve the static analysis problem (if possible!)

% stiffness matrix partitioning
Kff = K(fKnown,fKnown) ;
Kuf = K(uKnown,fKnown) ;
Kfu = K(fKnown,uKnown) ;
Kuu = K(uKnown,uKnown) ;

uu = u(uKnown,1) ;

fbf = fb(fKnown,1) ;

fcf = fc(fKnown,1) ;
ftf = ft(fKnown,1) ;

fcu = fc(uKnown,1) ;
ftu = ft(uKnown,1) ;

% if det(Kff) ~= 0
if rank(Kff) == size(Kff,1)
    
    % displacements of unconstrained DoFs
    %     uf = Kff \ (fbf - Kfu * uu) ; % from the 2nd row-block
    uf = Kff \ (fbf + fcf + ftf - Kfu * uu) ; % from the 2nd row-block
    
    % reactions of constrained DoFs
    %     fbu = Kuu * uu + Kuf * uf ; % from the 1st row-block
    fbu = Kuu * uu + Kuf * uf - fcu - ftu ; % from the 1st row-block
    
    u(fKnown,1) = uf ;
    fb(uKnown,1) = fbu ;
    
else
    
    error('the stiffness matrix is singular')
    
end

% checks
v2 = P*L^3 / (3*ePar{1}.I*ePar{1}.E), u(gDof(2,n+1),1)
t2 = P*L^2 / (2*ePar{1}.I*ePar{1}.E), u(gDof(3,n+1),1)

for i = 1:1:numel(ePar)
    ePar{i} = fun_rod2d(ePar{i},u(ePar{i}.eDof,1)) ;
end

%% post-processor

opts.scaleU = 0.1 ;
opts.scaleF = 0.4 ;
opts.type = 'M' ;

% fun_rod2d_plot(nCoord,ePar,gDof,u,fb,fKnown,uKnown) % default
fun_rod2d_plot(nCoord,ePar,gDof,u,fb,fKnown,uKnown,opts)

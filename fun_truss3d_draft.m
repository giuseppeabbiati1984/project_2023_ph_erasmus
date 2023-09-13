close all
clear
clc

% u1,f1                      u2,f2
% ---> o-------------------o --->

% u1,u2: displacement DoFs
% f1,f2: corresponding forces

L = 0.5 ; % truss length [m]
E = 210e9 ; % young modulus [N/m2]
A = pi*(1e-2)^2/4 ; % cross-section area [m2]
Ke = E*A/L * [1,-1;-1,1] ; % stiffness matrix [N/m]

null(Ke)
rank(Ke)
det(Ke)

% -->u1,f1  -->u2,f2  -->u3,f3  -->u4,f4  -->u5,f5
% o---------o---------o---------o---------o

Nn = 5;
Ne = Nn - 1 ;
dofPerNode = 1 ;
Ndofs = Nn * dofPerNode ;

K = zeros(Ndofs,Ndofs) ;
for i = 1:1:Ne
    index = [i,i+1] ; % element i-th DoFs indices
    K(index,index) = K(index,index) + Ke ; % matrix assembly
end

% u1==0
K = K(2:end,2:end) ;

f=[0;0;0;1]; % externally applied load
u=K\f; % displacement vector (linear static analysis)

%% 

close all
clear 
clc

xe = [0,0,0 ;  % [x1,y1,z1] node 1
      1,2,3] ; % [x2,y2,z2] node 2
  
Dx = xe(2,1) - xe(1,1) ; % x2 - x1
Dy = xe(2,2) - xe(1,2) ; % y2 - y1
Dz = xe(2,3) - xe(1,3) ; % z2 - z1

% orthogonal vectors
xp = [Dx,Dy,Dz] ;
yp = [-Dy,Dx,0] ;
zp = cross(xp,yp) ;

% normalization
xp = xp/norm(xp) ;
yp = yp/norm(yp) ;
zp = zp/norm(zp) ;

% L = sqrt(Dx^2 + Dy^2) ;
% xp = [ Dx/L , Dy/L] ;
% yp = [-Dy/L , Dx/L] ;

% rotation matrix
T = [xp;yp;zp] ;
G = blkdiag(T,T) ;

E = 210e9 ;
A = pi*(1e-2)^2/4 ;
L = norm([Dx,Dy,Dz]) ;

% local stiffness
Kl = E*A/L * [1,0,0,-1,0,0 ;   % up1x
              0,0,0, 0,0,0 ;   % up1y
              0,0,0, 0,0,0 ;   % up1z
             -1,0,0, 1,0,0 ;   % up2x
              0,0,0, 0,0,0 ;   % up2y
              0,0,0, 0,0,0 ] ; % up2z
          
% global stiffness
Kg = G' * Kl * G ;

% u2 = [0.1;0.2] ;
% u2p = T * u2 ; % coordinate transformation

ePar.E = E ;
ePar.A = A ;
ePar.xe = xe ;
ePar = fun_truss3d(ePar) ;

%% VERIFICATION

isequal(G,ePar.G)
isequal(T,ePar.T)
isequal(Kl,ePar.Kl)
isequal(Kg,ePar.Kg)












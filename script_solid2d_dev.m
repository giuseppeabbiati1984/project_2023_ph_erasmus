%% Melosh element

% Here the stiffness matrix of the Melosh element (2D 4-node solid element)
% is computed. Analytical integration is used. The element can be only
% rectangular and suffer for shear locking.

close all
clear
clc

syms E Nu % Young modulus, Poisson ratio

% shear modulus
G = E / (2*(1+Nu)) ;

% elastic compliance (eps = C * sig)
C = [  1/E , -Nu/E , -Nu/E , 0   , 0   , 0 ;
     -Nu/E ,   1/E , -Nu/E , 0   , 0   , 0 ;
     -Nu/E , -Nu/E ,   1/E , 0   , 0   , 0 ;
         0 ,     0 ,     0 , 1/G , 0   , 0 ;
         0 ,     0 ,     0 , 0   , 1/G , 0 ;
         0 ,     0 ,     0 , 0   , 0   , 1/G ] ;
     
% elastic stiffness ( sig = D * eps)
D = C^-1 ;

% compliance & stiffness for plane stress
Cs = C([1,2,6],[1,2,6]) ; % compliance
Ds = Cs^-1 ; % stiffness

% stiffness & compliance for plane strain
De = D([1,2,6],[1,2,6]) ; % stiffness
Ce = De^-1 ; % compliance

% generation of numerical implementation functions
matlabFunction(Ds,'file','fun_solid2d_Ds','vars',{E,Nu}) ;
matlabFunction(De,'file','fun_solid2d_De','vars',{E,Nu}) ;

% b,h,c : base, height and thickness
% x y : spatial coordinates
syms b h c x y 

assume(h,'real') ; assumeAlso(h,'positive') ;
assumeAlso(b,'real') ; assumeAlso(b,'positive') ;
assumeAlso(x,'real') ; assumeAlso(x,'positive') ;
assumeAlso(y,'real') ; assumeAlso(y,'positive') ;

% bi-linear shape functions
Phi(1) = (b-x)*(h-y)/(4*b*h) ;
Phi(2) = (b+x)*(h-y)/(4*b*h) ;
Phi(3) = (b+x)*(h+y)/(4*b*h) ;
Phi(4) = (b-x)*(h+y)/(4*b*h) ;

% testing of shape function 4
simplify(subs(Phi(4),{x,y},{-b,-h}))
simplify(subs(Phi(4),{x,y},{ b,-h}))
simplify(subs(Phi(4),{x,y},{ b, h}))
simplify(subs(Phi(4),{x,y},{-b, h}))

% shape functions partial derivatives
dPhi_dx = diff(Phi,x) ;
dphi_dy = diff(Phi,y) ; 

% strain interpolation matrix
B = sym(zeros(3,8)) ;
B(1,1:2:end) = dPhi_dx ;
B(2,2:2:end) = dphi_dy ;
B(3,1:2:end) = dphi_dy ; B(3,2:2:end) = dPhi_dx ; 

% element stiffness matrix for plane stress condition
Ks = int(int(c * B.' * Ds * B,x,-b,b),y,-h,h) ;

% element stiffness matrix for plane strain condition
Ke = int(int(c * B.' * De * B,x,-b,b),y,-h,h) ;

matlabFunction(Ke,'file','fun_solid2d_Ke','vars',{b,h,c,E,Nu})
matlabFunction(Ks,'file','fun_solid2d_Ks','vars',{b,h,c,E,Nu})

% distributed element load [N/m2]
syms px py
assumeAlso(px,'real') ; assumeAlso(px,'positive') ;
assumeAlso(py,'real') ; assumeAlso(py,'positive') ;

% displacement interpolation matrix
N = sym(zeros(2,8)) ;
N(1,1:2:end) = Phi ;
N(2,2:2:end) = Phi ;

% consistent nodal load vector
fc = int(int(N.' * [px;py],x,-b,b),y,-h,h) ;

%% iso-parametric element (function generation)

% here we generate all functions necessary to implement a 4-node
% iso-parametric solid element

close all ; clear ; clc

% iso-parametric coordinates
syms s t

% shape functions
Phi(1) = (1-s)*(1-t)/4 ;
Phi(2) = (1+s)*(1-t)/4 ;
Phi(3) = (1+s)*(1+t)/4 ;
Phi(4) = (1-s)*(1+t)/4 ;

% partial derivatives
dPhids = diff(Phi,s) ;
dPhidt = diff(Phi,t) ;

% element nodal coordinates
xe = sym('xe',[4,2]) ; % [x1,y1;x2,y2;x3,y3;x4,y4]
  
% jacobian matrix [dx/ds,dy/ds;dx/dt,dy/dt]
Jac = [dPhids * xe ; dPhidt * xe] ;

% element size
syms b h 

% test with rectangular geometry 
det(simplify(subs(Jac,xe,[-b,-h;
                           b,-h;
                           b, h;
                          -b, h])))

% numberical implementation of the Jacobian function
matlabFunction(Jac,'file','fun_solid2d_Jac','vars',{s,t,xe}) ;
matlabFunction(Phi,'file','fun_solid2d_Phi','vars',{s,t}) ;
matlabFunction(dPhids,'file','fun_solid2d_dPhids','vars',{s,t}) ;
matlabFunction(dPhidt,'file','fun_solid2d_dPhidt','vars',{s,t}) ;

%% iso-parametric element (verification against Melosh)

close all ; clear ; clc

E = 30e9 ; Nu = 0.2 ;

% stiffness matrix associated with the constitutive model of the material
% sigma = De * epsilon
De = fun_solid2d_De(E,Nu) ; % plane strain
Ds = fun_solid2d_Ds(E,Nu) ; % plane stress

% element geometry
b = 2 ; % base
h = 1 ; % heigth
c = 0.1 ; % thickness

% nodal coordinates
xe = [-b,-h;
       b,-h;
       b, h;
      -b, h] ;

% full-order integration
[sf,wf] = gl_quadrature(2) ;

% element stiffness matrix
Ke = zeros(8,8) ; % plane strain
Ks = zeros(8,8) ; % plane stress

% full-order integration of the component of the stiffness matrix
% associated with normal stress
for i = 1:1:numel(sf)
    for j = 1:1:numel(sf)
        Bb = zeros(4,8) ;
        % sf(i) = s , sf(j) = t in the notes
        Bb(1,1:2:end) = fun_solid2d_dPhids(sf(i),sf(j)) ; 
        Bb(2,1:2:end) = fun_solid2d_dPhidt(sf(i),sf(j)) ; 
        Bb(3,2:2:end) = fun_solid2d_dPhids(sf(i),sf(j)) ; 
        Bb(4,2:2:end) = fun_solid2d_dPhidt(sf(i),sf(j)) ; 
        
        % Jacobian of the iso-parametric mapping
        Jac = fun_solid2d_Jac(sf(i),sf(j),xe) ;
        
        % strain interpolation matrix
        B = [1,0,0,0;
             0,0,0,1] * (blkdiag(Jac,Jac) \ Bb) ;
         
        % stiffness matrix update
        Ke = Ke + c * B.' * De(1:2,1:2) * B * det(Jac) * wf(i) * wf(j) ;
        Ks = Ks + c * B.' * Ds(1:2,1:2) * B * det(Jac) * wf(i) * wf(j) ;
         
    end
end

% reduced-order integration
[sr,wr] = gl_quadrature(1) ; % if this is set to 2, you obtain the same results of the Melosh element

% reduced-order integration of the component of the stiffness matrix
% associated with shear stress
for i = 1:1:numel(sr)
    for j = 1:1:numel(sr)
        Bb = zeros(4,8) ;
        % sf(i) = s , sf(j) = t in the notes
        Bb(1,1:2:end) = fun_solid2d_dPhids(sr(i),sr(j)) ; 
        Bb(2,1:2:end) = fun_solid2d_dPhidt(sr(i),sr(j)) ; 
        Bb(3,2:2:end) = fun_solid2d_dPhids(sr(i),sr(j)) ; 
        Bb(4,2:2:end) = fun_solid2d_dPhidt(sr(i),sr(j)) ; 
        
        % Jacobian of the iso-parametric transformation
        Jac = fun_solid2d_Jac(sr(i),sr(j),xe) ;
        
        % strain interpolation matrix
        B = [0,1,1,0] * (blkdiag(Jac,Jac) \ Bb) ;
         
        % update of the stiffness matrix
        Ke = Ke + c * B.' * De(3,3) * B * det(Jac) * wr(i) * wr(j) ;
        Ks = Ks + c * B.' * Ds(3,3) * B * det(Jac) * wr(i) * wr(j) ;
         
    end
end

% matrices computed via analytical integration (Melosh element)
Ke_check=fun_solid2d_Ke(b,h,c,E,Nu) ; % plane strain
Ks_check=fun_solid2d_Ks(b,h,c,E,Nu) ; % plane strss

% Melosh vs. iso-parametric element (to get the same results, full-order 
% integration shall be used also for shear in the iso-paramtric element)
figure
subplot(2,1,1)
hold all
plot(diag(Ke),'-o')
plot(diag(Ke_check),'-x')
legend('Iso-param.','Melosh')

subplot(2,1,2)
hold all
plot(diag(Ks),'-o')
plot(diag(Ks_check),'-x')

% pure bending displacement
d = [-1,0,1,0,-1,0,1,0].' ;

% elastic energy associate with pure bending
d.' * Ks * d % gauss-legendre (selective)
d.' * Ks_check * d % analytical (melosh)

% to plot rigid body modes:
null(Ks)

%% consistent nodal load vector

qx = 100 ; % uniformly distributed load in x [N/m2]
qy = 100 ; % uniformly distributed load in y [N/m2]

fc = zeros(8,1) ;

for i = 1:1:numel(sf)
    for j = 1:1:numel(sf)
        
        % displacement interpolation matrix
        N = zeros(2,8) ;
        N(1,1:2:end) = fun_solid2d_Phi(sf(i),sf(j)) ; % u(s,t)
        N(2,2:2:end) = fun_solid2d_Phi(sf(i),sf(j)) ; % v(s,t)
        
        % Jacobian of the iso-parametric transformation
        Jac = fun_solid2d_Jac(sf(i),sf(j),xe) ;
        
        % Gauss-Legendre summation
        fc = fc + N.' * [qx;qy] * det(Jac) * wf(i) * wf(j) ;
        
    end
end

%% testing

ePar.E = E ;
ePar.Nu = Nu ;
ePar.xe = xe ;
ePar.c = c ;
ePar.type = 'peps' ;
ePar.qx = qx ;
ePar.qy = qy ;
ePar = fun_solid2d(ePar) ;

% stiffness matrices
switch ePar.type
    case 'psig'
        disp('plane stress check')
        isequal(ePar.Kg,Ks)
        isequal(ePar.D,Ds)
    case 'peps'
        disp('plane strain check')
        isequal(ePar.Kg,Ke)
        isequal(ePar.D,De)
end
% consistent nodal load vector
isequal(ePar.fcg,fc) 



























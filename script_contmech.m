%% symbolic expression

close all
clear
clc

% 4           3
% o===========o   /
% |           |   |
% |     ^ y   |   b
% |     |     |   |
% |     o-->  |   /
% |        x  |   |
% |           |   b
% |           |   |
% o===========o   /
% 1           2
% /--a--/--a--/

syms x y a b

assume(x,'real') ; assumeAlso(y,'real')
assumeAlso(a,'real') ; assumeAlso(a,'positive') ; 
assumeAlso(b,'real') ; assumeAlso(b,'positive') ;

% displacement/temperature interpolation functions
phi(1) = (a-x) * (b-y) / (4*a*b) ;
phi(2) = (a+x) * (b-y) / (4*a*b) ;
phi(3) = (a+x) * (b+y) / (4*a*b) ;
phi(4) = (a-x) * (b+y) / (4*a*b) ;

%% numerical testing

% anonymous implementations of all shape functions
phi1_h = matlabFunction(phi(1),'vars',{a,b,x,y}) ;
phi2_h = matlabFunction(phi(2),'vars',{a,b,x,y}) ;
phi3_h = matlabFunction(phi(3),'vars',{a,b,x,y}) ;
phi4_h = matlabFunction(phi(4),'vars',{a,b,x,y}) ;

% grid generation for numerical evaluation of shape functions
a_val = 2.0 ;
b_val = 3.0 ;
% 1d grids
x_val = linspace(-a_val,a_val,20) ;
y_val = linspace(-b_val,b_val,20) ;
% 2d grid
[X_val,Y_val] = meshgrid(x_val,y_val) ;

% shape function plots
figure
subplot(2,2,1)
hold all
surf(X_val,Y_val,phi1_h(a_val,b_val,X_val,Y_val))

subplot(2,2,2)
hold all
surf(X_val,Y_val,phi2_h(a_val,b_val,X_val,Y_val))

subplot(2,2,3)
hold all
surf(X_val,Y_val,phi3_h(a_val,b_val,X_val,Y_val))

subplot(2,2,4)
hold all
surf(X_val,Y_val,phi4_h(a_val,b_val,X_val,Y_val))

% check if shape functions are ok with rigid-body motions:
u_const = phi1_h(a_val,b_val,X_val,Y_val) + ...
    phi2_h(a_val,b_val,X_val,Y_val) + ...
    phi3_h(a_val,b_val,X_val,Y_val) + ...
    phi4_h(a_val,b_val,X_val,Y_val) ;

figure
subplot(1,1,1)
title('u_{const}')
hold all
surf(X_val,Y_val,u_const)

% interpolation of a generic nodal displacement
u1 = 1.0 ;
u2 = 1.0 ;
u3 = 2.0 ;
u4 = 3.0 ;

v1 = 0.0 ;
v2 = 2.0 ;
v3 = 3.0 ;
v4 = 5.0 ;

% check if shape functions are ok with rigid-body motions:
u_int = u1*phi1_h(a_val,b_val,X_val,Y_val) + ...
    u2*phi2_h(a_val,b_val,X_val,Y_val) + ...
    u3*phi3_h(a_val,b_val,X_val,Y_val) + ...
    u4*phi4_h(a_val,b_val,X_val,Y_val) ;

figure
subplot(1,1,1)
title('u_{const}')
hold all
surf(X_val,Y_val,u_int)

% try to epress normal strain in x direction
% epsx_int = 

%%

dphi_dx = diff(phi,x) ;
dphi_dy = diff(phi,y) ;

% anonymous implementations of all shape functions
dphi1_dx_h = matlabFunction(dphi_dx(1),'vars',{a,b,x,y}) ;
dphi2_dx_h = matlabFunction(dphi_dx(2),'vars',{a,b,x,y}) ;
dphi3_dx_h = matlabFunction(dphi_dx(3),'vars',{a,b,x,y}) ;
dphi4_dx_h = matlabFunction(dphi_dx(4),'vars',{a,b,x,y}) ;

% anonymous implementations of all shape functions
dphi1_dy_h = matlabFunction(dphi_dy(1),'vars',{a,b,x,y}) ;
dphi2_dy_h = matlabFunction(dphi_dy(2),'vars',{a,b,x,y}) ;
dphi3_dy_h = matlabFunction(dphi_dy(3),'vars',{a,b,x,y}) ;
dphi4_dy_h = matlabFunction(dphi_dy(4),'vars',{a,b,x,y}) ;

%%

% interpolation of a generic nodal displacement
u1 = 0.0 ; u2 = 0.0 ; u3 = 0.0 ; u4 = 0.0 ;
v1 = 0.0 ; v2 = 1.0 ; v3 = 1.0 ; v4 = 0.0 ;

u = u1*phi1_h(a_val,b_val,X_val,Y_val) + ...
    u2*phi2_h(a_val,b_val,X_val,Y_val) + ...
    u3*phi3_h(a_val,b_val,X_val,Y_val) + ...
    u4*phi4_h(a_val,b_val,X_val,Y_val) ;

v = v1*phi1_h(a_val,b_val,X_val,Y_val) + ...
    v2*phi2_h(a_val,b_val,X_val,Y_val) + ...
    v3*phi3_h(a_val,b_val,X_val,Y_val) + ...
    v4*phi4_h(a_val,b_val,X_val,Y_val) ;

% epsilon xx
exx = u1*dphi1_dx_h(a_val,b_val,X_val,Y_val) + ...
      u2*dphi2_dx_h(a_val,b_val,X_val,Y_val) + ...
      u3*dphi3_dx_h(a_val,b_val,X_val,Y_val) + ...
      u4*dphi4_dx_h(a_val,b_val,X_val,Y_val) ;

% epsilon yy
eyy = v1*dphi1_dy_h(a_val,b_val,X_val,Y_val) + ...
      v2*dphi2_dy_h(a_val,b_val,X_val,Y_val) + ...
      v3*dphi3_dy_h(a_val,b_val,X_val,Y_val) + ...
      v4*dphi4_dy_h(a_val,b_val,X_val,Y_val) ;

% gamma xy
gxy = u1*dphi1_dy_h(a_val,b_val,X_val,Y_val) + ...
      u2*dphi2_dy_h(a_val,b_val,X_val,Y_val) + ...
      u3*dphi3_dy_h(a_val,b_val,X_val,Y_val) + ...
      u4*dphi4_dy_h(a_val,b_val,X_val,Y_val) + ...
      v1*dphi1_dx_h(a_val,b_val,X_val,Y_val) + ...
      v2*dphi2_dx_h(a_val,b_val,X_val,Y_val) + ...
      v3*dphi3_dx_h(a_val,b_val,X_val,Y_val) + ...
      v4*dphi4_dx_h(a_val,b_val,X_val,Y_val) ;

close all

figure
subplot(1,2,1)
% title('u')
hold all
surf(X_val,Y_val,u)
colorbar
xlabel('x')
ylabel('y')

subplot(1,2,2)
% title('\epsilon_{xx}')
surf(X_val,Y_val,gxy)
colorbar
xlabel('x')
ylabel('y')

%%

close all
% clear
clc

syms E nu alpha T
 
assume(E,'real') ;
assumeAlso(E,'positive')
assumeAlso(nu,'real') ;
assumeAlso(alpha,'real') ;
assumeAlso(alpha,'positive') ;
assumeAlso(T,'real') ;

G = E/(2*(1+nu)) ;

% elastic compliance matrix
C = sym(zeros(6)) ;
C(1,1) = 1/E ; C(2,2) = 1/E ; C(3,3) = 1/E ;
C(1,2) = -nu/E ; C(1,3) = -nu/E ; C(2,3) = -nu/E ;
C(2,1) = -nu/E ; C(3,1) = -nu/E ; C(3,2) = -nu/E ;
C(4,4) = 1/G ; C(5,5) = 1/G ; C(6,6) = 1/G ;

% elastic stiffness matrix
D = inv(C) ;

% vector of thermal strain
Te = sym([1;1;1;0;0;0]) * T * alpha ;

%%  plane strain case

% components of the strain vector that are different than zero
syms exx eyy gxy
assumeAlso(exx,'real') ; assumeAlso(eyy,'real') ; assumeAlso(gxy,'real') ;

% correponding stress from constitutive model (linear w.r.t. strain and temperature)
sig_pe = C\([exx;eyy;0;0;0;gxy] - Te) ; sig_pe = sig_pe([1,2,6],1) ;

% elastic stiffness
D_pe = simplify(jacobian(sig_pe,[exx;eyy;gxy])) ;

% thermal-stress matrix
Q_pe = simplify(jacobian(sig_pe,T)) ;

%% plane stress case

% components of the stress vector that are different than zero
syms sxx syy txy

% total strain from constitutive model (linear in sigma and temperature)
eps_ps = C*[sxx;syy;0;0;0;txy] + Te ; eps_ps = eps_ps([1,2,6],1) ;

% eps_ps = Jps_sig * sig_ps + Jps_tmp * T
Jps_sig = jacobian(eps_ps,[sxx;syy;txy]) ; 
Jps_tmp = jacobian(eps_ps,T) ; 

% sig_ps = D_ps * eps_ps + Q_ps * T

% elastic stiffness
D_ps = Jps_sig^-1 ;

% thermal stress matrix
Q_ps = -Jps_sig \ Jps_tmp ; 


%% Melosh element implementation

% we consider the plane stress

B = sym(zeros(3,8)) ;
B(1,1:2:end) = dphi_dx ;
B(2,2:2:end) = dphi_dy ;
B(3,1:2:end) = dphi_dy ;
B(3,2:2:end) = dphi_dx ;

% temperature interpolation matrix (Nt(x,y)*t is the temperature on a generic
% point of coordinates x,y and t is the vector of nodal temperatures)
Nt = phi ;

% thickness
syms h
assumeAlso(h,'real')
assumeAlso(h,'positive')

% in python you can use sympy (D_ps*B*d = eps)
Ke = h*simplify(int(int(B.'*D_ps*B,x,-a,a),y,-b,b)) ;

% matrix that convert temperature to force *** not necessary for Python
% implementation ***
Qe = h*simplify(int(int(B.'*Q_ps*Nt,x,-a,a),y,-b,b)) ;

% internal force (t is the nodal temperature)
% fi = Ke*d + Qe*t

% in python you can generate a "lambified function" from a sympy expression
Ke_h = matlabFunction(Ke,'vars',{a,b,h,E,nu}) ;

a_val = 0.1 ;
b_val = 0.2 ;
h_val = 0.01 ;
E_val = 210e9 ;
nu_val = 0.3 ;

Ke_val = Ke_h(a_val,b_val,h_val,E_val,nu_val) ;

% it must be 5 = 8 (dofs) - 3 (rigid-body-modes)
rank(Ke_val)

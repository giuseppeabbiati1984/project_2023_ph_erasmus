close all
clear
clc

syms L x

assume(L,'real')
assumeAlso(L,'positive')

assume(x,'real')

a = sym('a',[1,4]);

y = sym(0) ;

for i = 1:1:numel(a)
   y = y + a(i)*x^(i-1) ;    
end

yd = diff(y,x) ;

%% phi1

% linear system
equ1 = [subs(y,x,0)==1; % phi1(0)=1
        subs(y,x,L)==0; % phi1(L)=0
        subs(yd,x,0)==0; % dphi1(0)=0
        subs(yd,x,L)==0]; % dphi1(L)=0
    
% linear system
equ2 = [subs(y,x,0)==0; % phi1(0)=0
        subs(y,x,L)==0; % phi1(L)=0
        subs(yd,x,0)==1; % dphi1(0)=1
        subs(yd,x,L)==0]; % dphi1(L)=0

% linear system
equ3 = [subs(y,x,0)==0; % phi1(0)=0
        subs(y,x,L)==1; % phi1(L)=1
        subs(yd,x,0)==0; % dphi1(0)=0
        subs(yd,x,L)==0]; % dphi1(L)=0
    
% linear system
equ4 = [subs(y,x,0)==0; % phi1(0)=1
        subs(y,x,L)==0; % phi1(L)=0
        subs(yd,x,0)==0; % dphi1(0)=0
        subs(yd,x,L)==1]; % dphi1(L)=0

% symbolic solution of the linear system
sol1 = solve(equ1,a) ;
sol2 = solve(equ2,a) ;
sol3 = solve(equ3,a) ;
sol4 = solve(equ4,a) ;

% SYMBOLIC function
phi1 = subs(y,a,[sol1.a1,sol1.a2,sol1.a3,sol1.a4]) ;
phi2 = subs(y,a,[sol2.a1,sol2.a2,sol2.a3,sol2.a4]) ;
phi3 = subs(y,a,[sol3.a1,sol3.a2,sol3.a3,sol3.a4]) ;
phi4 = subs(y,a,[sol4.a1,sol4.a2,sol4.a3,sol4.a4]) ;

% anonymous NUMERICAL function
fun_phi1 = matlabFunction(phi1,'vars',{L,x}) ;
fun_phi2 = matlabFunction(phi2,'vars',{L,x}) ;
fun_phi3 = matlabFunction(phi3,'vars',{L,x}) ;
fun_phi4 = matlabFunction(phi4,'vars',{L,x}) ;

L_val = 2 ;
x_val = linspace(0,L_val,100) ;

figure
hold all
plot(x_val,fun_phi1(L_val,x_val))
plot(x_val,fun_phi2(L_val,x_val))
plot(x_val,fun_phi3(L_val,x_val))
plot(x_val,fun_phi4(L_val,x_val))
legend('show')

%% Bernoulli stiffness (Kb)

N = [phi1,phi2,phi3,phi4] ; % Displacement interpolation matrix
B = diff(N,x,2) ; % Deformation interpolation matrix

syms I E f

assume(f,'real') 

assume(I,'positive')
assumeAlso(I,'real')

assume(E,'positive')
assumeAlso(E,'real')

% Bernoulli stiffness matrix following Galerkin (FEM)
Kb = int(B.'*E*I*B,x,0,L) ;
fc = int(N.'*f,x,0,L) ;

vars = {E,I,L,f} ;

matlabFunction(Kb,'file','fun_beam2d_Kb','vars',vars)
matlabFunction(fc,'file','fun_beam2d_fc','vars',vars)

%% Winkler stiffness

syms k0 b

assume(k0,'real')
assumeAlso(k0,'positive')

assume(b,'real')
assumeAlso(b,'positive')

Kw = k0*b*int(N.'*N,x,0,L) ;

vars = {L,b,k0} ;
matlabFunction(Kw,'file','fun_beam2d_Kw','vars',vars)

%% Timoshenko beam

close all
clear
clc

syms E G A I ky L x

assume(E,'real') ; assumeAlso(E,'positive') % Young modulus
assume(G,'real') ; assumeAlso(G,'positive') % shear modulus
assume(A,'real') ; assumeAlso(A,'positive') % cross-section area
assume(I,'real') ; assumeAlso(I,'positive') % cross-section inertia
assume(ky,'real') ; assumeAlso(ky,'positive') % shaer factor
assume(L,'real') ; assumeAlso(L,'positive') % length
assume(x,'real') ;

P = [1 , x;
     0 , 1];

D = diag([1/(E*I),ky/(G*A)]) ;

H = int(P.'*D*P,x,0,L) ;

R = [ 0 ,  1 ;
     -1 ,  0 ;
      0 , -1 ;
      1 ,  L ] ;
  
Kt = simplify((H^-1*R.').' * H * (H^-1*R.')) ;

rank(Kt)
null(Kt)
d=[0;1;0;-1] ; simplify(Kt*d)

% limit(K,0,inf) 
subs(Kt,ky,0) % Bernoulli is Timoshenko with ky==0

vars = {E,G,A,I,ky,L} ;

matlabFunction(Kt,'file','fun_beam2d_Kt','vars',vars)































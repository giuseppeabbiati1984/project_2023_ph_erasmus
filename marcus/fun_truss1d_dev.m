close all
clear
clc

% this is the development script for fun_truss3d.m

% x : local reference axis
% L : truss length [m]

syms x L E A p alpha theta

assume(x,'real')
assume(p,'real')
assume(theta,'real')
assume(alpha,'real')

assume(L,'real')
assume(L,'positive')

assume(E,'real')
assume(E,'positive')

assume(A,'real')
assume(A,'positive')

% truss element with linear shape functions
phi1 = 1 - x/L ; dphi1 = diff(phi1,x) ;
phi2 = x/L ; dphi2 = diff(phi2,x) ;

% stiffness matrix
% K = E*A*[int(dphi1*dphi1,x,0,L),int(dphi1*dphi2,x,0,L);
%          int(dphi2*dphi1,x,0,L),int(dphi2*dphi2,x,0,L)] ;
     
K = E*A*int([dphi1*dphi1,dphi1*dphi2;
             dphi2*dphi1,dphi2*dphi2],x,0,L) ;
         
fc = p * int([phi1;
              phi2],x,0,L) ;
          
ft = E*A*alpha*theta * int([dphi1;
                            dphi2],x,0,L) ;

% input parameters for numerical implementations
vars = {E,alpha,L,A,p,theta} ;

% generation of numerical implementation files
matlabFunction(K,'file','fun_truss1d_K','vars',vars) ;
matlabFunction(fc,'file','fun_truss1d_fc','vars',vars) ;
matlabFunction(ft,'file','fun_truss1d_ft','vars',vars) ;






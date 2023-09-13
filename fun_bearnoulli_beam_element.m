function [Kb,Mb] = fun_bearnoulli_beam_element(E,Rho,L,I,A)

% bearnoulli beam element parameters:
% E : Young modulus [Pa]
% Rho : density [kg/m3]
% L : length [m]
% A : cross-section area [m2]
% I : cross-section inertia [m4]

% bernoulli beam element degress-of-freedom:
%
%   ^              ^
%   |u1            |u3
% G o--------------o G
% u2                 u4

% stiffness matrix
Kb = [ (12*E*I)/L^3,  (6*E*I)/L^2, -(12*E*I)/L^3,  (6*E*I)/L^2 ;
        (6*E*I)/L^2,    (4*E*I)/L,  -(6*E*I)/L^2,    (2*E*I)/L ;
      -(12*E*I)/L^3, -(6*E*I)/L^2,  (12*E*I)/L^3, -(6*E*I)/L^2 ;
        (6*E*I)/L^2,    (2*E*I)/L,  -(6*E*I)/L^2,    (4*E*I)/L ] ;

% mass matrix
Mb = [  (13*A*L*Rho)/35, (11*A*L^2*Rho)/210,      (9*A*L*Rho)/70, -(13*A*L^2*Rho)/420 ;
     (11*A*L^2*Rho)/210,    (A*L^3*Rho)/105,  (13*A*L^2*Rho)/420,    -(A*L^3*Rho)/140 ;
         (9*A*L*Rho)/70, (13*A*L^2*Rho)/420,     (13*A*L*Rho)/35, -(11*A*L^2*Rho)/210 ;
    -(13*A*L^2*Rho)/420,   -(A*L^3*Rho)/140, -(11*A*L^2*Rho)/210,     (A*L^3*Rho)/105 ] ;

end
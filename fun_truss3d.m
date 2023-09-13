function ePar = fun_truss3d(ePar)

% ePar.xe : nodal coordinates [m] -> L,G,T
% ePar.E : young modulus [N/m2]
% ePar.A : cross-section area [m2]
% ePar.p : uniformly distributed load [N/m]
% ePar.alpha : coefficient of thermal expansion [1/K]
% ePar.theta : relative temperature [K]

% xe = [0,0,0 ;  % [x1,y1,z1] node 1
%       1,2,3] ; % [x2,y2,z2] node 2

Dx = ePar.xe(2,1) - ePar.xe(1,1) ; % x2 - x1
Dy = ePar.xe(2,2) - ePar.xe(1,2) ; % y2 - y1
Dz = ePar.xe(2,3) - ePar.xe(1,3) ; % z2 - z1

% orthogonal vectors
xp = [Dx,Dy,Dz] ;
if Dx == 0 && Dy == 0
    yp = [0,1,0] ; % workaround when xp is vertical
else
    yp = [-Dy,Dx,0] ;
end
zp = cross(xp,yp) ;

% normalization
xp = xp/norm(xp) ;
yp = yp/norm(yp) ;
zp = zp/norm(zp) ;

% L = sqrt(Dx^2 + Dy^2) ;
% xp = [ Dx/L , Dy/L] ;
% yp = [-Dy/L , Dx/L] ;

% rotation matrix
ePar.T = [xp;yp;zp] ;
ePar.G = blkdiag(ePar.T,ePar.T) ;

% E = 210e9 ;
% A = pi*(1e-2)^2/4 ;
ePar.L = norm([Dx,Dy,Dz]) ;

% vars = {E,alpha,L,A,p,theta} ;
% fun_truss3d_K(E,alpha,L,A,p,theta)
% fun_truss3d_fc(E,alpha,L,A,p,theta)
% fun_truss3d_ft(E,alpha,L,A,p,theta)
ePar.Kl = zeros(6,6) ;
ePar.Kl([1,4],[1,4]) = fun_truss3d_K(ePar.E,ePar.alpha,...
                                     ePar.L,ePar.A,...
                                     ePar.p,ePar.theta) ;
                                 
ePar.fcl = zeros(6,1) ;                                
ePar.fcl([1,4],1) = fun_truss3d_fc(ePar.E,ePar.alpha,...
                                   ePar.L,ePar.A,...
                                   ePar.p,ePar.theta) ;
  
ePar.ftl = zeros(6,1) ;
ePar.ftl([1,4],1) = fun_truss3d_ft(ePar.E,ePar.alpha,...
                                   ePar.L,ePar.A,...
                                   ePar.p,ePar.theta) ;

% % local stiffness
% ePar.Kl = ePar.E * ePar.A / ePar.L * [1,0,0,-1,0,0 ;   % up1x
%                                       0,0,0, 0,0,0 ;   % up1y
%                                       0,0,0, 0,0,0 ;   % up1z
%                                      -1,0,0, 1,0,0 ;   % up2x
%                                       0,0,0, 0,0,0 ;   % up2y
%                                       0,0,0, 0,0,0 ] ; % up2z

% element output (global reference system)
ePar.Kg = ePar.G' * ePar.Kl * ePar.G ; % stiffness matrix
ePar.fcg = ePar.G' * ePar.fcl ; % consistent load vector
ePar.ftg = ePar.G' * ePar.ftl ; % thermal load vector

end
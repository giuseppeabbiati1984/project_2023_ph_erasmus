function ePar = fun_rod2d(ePar,varargin)

if isempty(varargin)
    ePar.dg = zeros(6,1) ;
else
    ePar.dg = varargin{1} ;
end

% cross-section parameters (b,h width and depth [m])
if ~isfield(ePar,'b'), ePar.b = 0.2 ; end 
if ~isfield(ePar,'h'), ePar.h = 0.4 ; end 
if ~isfield(ePar,'ky'), ePar.ky = 1.2 ; end % shear factor for rectangual cross-section
% ky = 0 produces turns the Timoshenko beam into a Bernoulli beam

ePar.I = ePar.b * ePar.h^3 / 12 ;
ePar.A = ePar.b * ePar.h ;

% material parameters (concrete by default)
if ~isfield(ePar,'E'), ePar.E = 30e9 ; end % Young modulus
if ~isfield(ePar,'G1'), ePar.G1 = 30e9/(2*(1+0.2)) ; end % shear modulus
if ~isfield(ePar,'alpha'), ePar.alpha = 1e-5 ; end % thermal expantion coeff.
if ~isfield(ePar,'k0'), ePar.k0 = 0 ; end % by default the beam has not soil underneath

% element loads
if ~isfield(ePar,'f') , ePar.f = 0 ; end 
if ~isfield(ePar,'p') , ePar.p = 0 ; end 
if ~isfield(ePar,'theta') , ePar.theta = 0 ; end 

% ePar.xe = [x1,y1;
%            x2,y2];

xp = ePar.xe(2,:) - ePar.xe(1,:) ; % local x axis
yp = [-xp(2),xp(1)] ; % local y axis
L = norm(xp) ; % element length
xp = xp/L ; % local x axis normalization
yp = yp/L ; % local y axis normalization

ePar.L = L ;

% coordinate transformation matrix for single node
T = [xp,0;
     yp,0;
     0,0,1] ;

% coordinate transformation matrix for the element (two nodes)
G = blkdiag(T,T) ;

indb = [2,3,5,6] ; % indices for bernoulli DoFs
inda = [1,4] ; % indices for axial DoFs

% element matrices and vectors in the local reference system
Kl = zeros(6,6) ;
fcl = zeros(6,1) ;
ftl = zeros(6,1) ;

% bending 
Klb=fun_beam2d_Kt(ePar.E,ePar.G1,ePar.A,ePar.I,ePar.ky,ePar.L) ;
Kl(indb,indb)=Klb ;
fcl(indb,1)=fclb ;

% axial
% vars = {ePar.E,ePar.alpha,ePar.L,ePar.A,ePar.p,ePar.theta} ;
Kla=fun_truss3d_K(ePar.E,ePar.alpha,ePar.L,ePar.A,ePar.p,ePar.theta) ;
fcla=fun_truss3d_fc(ePar.E,ePar.alpha,ePar.L,ePar.A,ePar.p,ePar.theta) ;
ftla=fun_truss3d_ft(ePar.E,ePar.alpha,ePar.L,ePar.A,ePar.p,ePar.theta) ;

Kl(inda,inda)=Kla ;
fcl(inda,1)=fcla ;
ftl(inda,1)=ftla ;

Kg = G' * Kl * G ;
fcg = G' * fcl ;
ftg = G' * ftl ;

% element output
ePar.Kg = Kg ;
ePar.fcg = fcg ;
ePar.ftg = ftg ;

ePar.T = T ;
ePar.G = G ;

ePar.Kla = Kla ; % stiffness
ePar.fcla = fcla ; % mechanical load
ePar.ftla = ftla ; % thermal load
ePar.inda = inda ; % DoFs indices

ePar.Klb = Klb ;
ePar.fclb = fclb ;
ePar.indb = indb ;

ePar.dl = ePar.G * ePar.dg ; % element displacement in local ref.

ePar.dla = ePar.dl(ePar.inda,1) ; % local disp at axial dofs
ePar.rla = ePar.Kla * ePar.dla ; % local forc at axial dofs
% ePar.rla = ePar.rl(ePar.inda) *** NEVER DO THAT !!! ***

ePar.dlb = ePar.dl(ePar.indb,1) ; % local disp at bernoulli dofs
ePar.rlb = ePar.Klb * ePar.dlb ; % local forc at bernoulli dofs

% internal forces
ePar.N = [-ePar.rla(1),ePar.rla(2)] ;
ePar.V = [ePar.rlb(1),-ePar.rlb(3)] ;
ePar.M = [-ePar.rlb(2),ePar.rlb(4)] ;

end
function fc = fun_truss1d_fc(E,alpha,L,A,p,theta)
%FUN_TRUSS1D_FC
%    FC = FUN_TRUSS1D_FC(E,ALPHA,L,A,P,THETA)

%    This function was generated by the Symbolic Math Toolbox version 8.7.
%    14-Jul-2022 09:51:21

t2 = (L.*p)./2.0;
fc = [t2;t2];

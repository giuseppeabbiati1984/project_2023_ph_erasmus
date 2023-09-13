function fc = fun_beam2d_fc(E,I,L,f)
%FUN_BEAM2D_FC
%    FC = FUN_BEAM2D_FC(E,I,L,F)

%    This function was generated by the Symbolic Math Toolbox version 8.7.
%    31-Mar-2022 13:04:14

t2 = L.^2;
t3 = (L.*f)./2.0;
t4 = (f.*t2)./1.2e+1;
fc = [t3;t4;t3;-t4];

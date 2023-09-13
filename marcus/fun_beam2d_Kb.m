function Kb = fun_beam2d_Kb(E,I,L,f)
%FUN_BEAM2D_KB
%    KB = FUN_BEAM2D_KB(E,I,L,F)

%    This function was generated by the Symbolic Math Toolbox version 8.7.
%    14-Jul-2022 09:52:34

t2 = 1.0./L;
t3 = t2.^2;
t4 = t2.^3;
t5 = E.*I.*t2.*2.0;
t6 = E.*I.*t2.*4.0;
t7 = E.*I.*t3.*6.0;
t9 = E.*I.*t4.*1.2e+1;
t8 = -t7;
t10 = -t9;
Kb = reshape([t9,t7,t10,t7,t7,t6,t8,t5,t10,t8,t9,t8,t7,t5,t8,t6],[4,4]);

function [UT,VT,AT,ENE_KIN,ENE_ELA,ENE_EXT,ENE_DAM,ENE_ALG] = fun_Newmark_method(MT,CT,KT,FT,UTi,VTi,Dt)

% la seguente implementazione funziona con matrici di massa lumped e quindi
% non full rank.

% si aggiunge il calcolo di tutte le componenti dell'energia come suggerito
% da Krenk 2006.
    
    GAMMA = 1/2;
    BETA  = 1/4;
    STEP  = length(FT(1,:));
        
    UT = zeros(length(KT),STEP);
    VT = zeros(length(KT),STEP);
    AT = zeros(length(KT),STEP);
    
    UT(:,1) = UTi;
    VT(:,1) = VTi;
    AT(:,1) = MT\(FT(:,1) - KT*UT(:,1) - CT*VT(:,1));
    
    ENE_KIN = zeros(1,STEP);
    ENE_ELA = zeros(1,STEP);
    ENE_EXT = zeros(1,STEP);
    ENE_DAM = zeros(1,STEP);
    ENE_ALG = zeros(1,STEP);
    
    Di = (MT + CT*GAMMA*Dt + KT*BETA*Dt^2)^-1;

    for i = 1:1:STEP-1
        
        % Predictor
        UT(:,i+1) = UT(:,i) + VT(:,i)*Dt + AT(:,i)*(1/2-BETA)*Dt^2;
        VT(:,i+1) = VT(:,i) + AT(:,i)*(1-GAMMA)*Dt;

        AT(:,i+1) = Di * (FT(:,i+1) - CT*VT(:,i+1) - KT*UT(:,i+1));
        
        % Corrector
        UT(:,i+1) = UT(:,i+1) + AT(:,i+1)*BETA*Dt^2;
        VT(:,i+1) = VT(:,i+1) + AT(:,i+1)*GAMMA*Dt;
        
        % Energies: dE_kin + dE_ela + dE_alg + dE_ext + dE_dam = 0;
        
        u1 = UT(:,i+1);
        u0 = UT(:,i);
        du = u1-u0;
        
        v1 = VT(:,i+1);
        v0 = VT(:,i);
        dv = v1-v0;
        
        a1 = AT(:,i+1);
        a0 = AT(:,i);
        da = a1-a0;
        
        f1 = FT(:,i+1);
        f0 = FT(:,i);
        df = f1-f0;

        % Kinetic:
        ENE_KIN(i+1) = 1/2 * (v1+v0)' * MT * dv;

        % Elastic:
        ENE_ELA(i+1) = 1/2 * (u1+u0)' * KT * du;

        % Load:
        ENE_EXT(i+1) = -du' * ((f1+f0)/2 + (GAMMA-1/2) * df);

        % Damping:
        ENE_DAM(i+1) = du' * CT * ((v1+v0)/2 + (GAMMA-1/2) * dv);

        % Algorithmic:
        ENE_ALG(i+1) = 1/2 * Dt^2 * (BETA-GAMMA/2) * (a1+a0)' * MT * da + (GAMMA-1/2) * (du' * KT * du + (BETA-GAMMA/2) * Dt^2 * da' * MT * da);

    end

end
function fun_truss3d_plot(...
    nCoord,ePar,gDof,u,fb,scaleU,scaleF,fKnown,uKnown)

% here I plot the results
% scaleU = 0.10 ;
% scaleF = 0.20 ;

uMax = max(abs(u)) ;
fMax = max(abs(fb)) ;

fbf = zeros(size(fb)) ;
fbf(fKnown) = fb(fKnown) ;

fbu = zeros(size(fb)) ;
fbu(uKnown) = fb(uKnown) ;

fbfMat = fbf(gDof') ;
fbuMat = fbu(gDof') ;

uMat = u(gDof') ; % nodal displacements arranged as nCoord ;

for i = 1:1:size(nCoord,1)
    for j = i+1:1:size(nCoord,1)
        modelSize(i,j) = norm(nCoord(i,:) - nCoord(j,:)) ;
    end
end

% characteristic length of the model
modelSize = max(modelSize(:)) ;
modelCentroid = mean(nCoord,1) ;

% nodal coordinates in deformed configuration (with scaling)
nCoord_def = nCoord + uMat/uMax * modelSize * scaleU ;

% nodal coordiantes in deformed configuration (with scaling)
for i = 1:1:numel(ePar)
    ePar{i}.dxe = u(ePar{i}.eDof)'/uMax * modelSize * scaleU ;
end

figure
hold all
box on
xlim(modelCentroid(1) + [-modelSize,+modelSize])
ylim(modelCentroid(2) + [-modelSize,+modelSize])
zlim(modelCentroid(3) + [-modelSize,+modelSize])

% plot(x,y,z) create a line linking all points
plot3(nCoord(:,1),nCoord(:,2),nCoord(:,3),'ko')
plot3(nCoord_def(:,1),nCoord_def(:,2),nCoord_def(:,3),'rx')

for i = 1:1:size(nCoord,1)
    % text(x,y,z,string) create a string at x,y,z
    text(nCoord(i,1),nCoord(i,2),nCoord(i,3),num2str(i),...
        'fontsize',16)
    text(nCoord_def(i,1),nCoord_def(i,2),nCoord_def(i,3),...
        num2str(i),'fontsize',16)
end

% alternative way of plotting elements:
% plot3(nCoord(eTop',1),nCoord(eTop',2),nCoord(eTop',3),'--k')
% plot3(nCoord_def(eTop',1),nCoord_def(eTop',2),nCoord_def(eTop',3),'b')

% element plot
for i = 1:1:numel(ePar)
    
    % undeformed configuration
    plot3(ePar{i}.xe(:,1),ePar{i}.xe(:,2),ePar{i}.xe(:,3),'--k')
    
    % deformed configuration
    if ePar{i}.N > 0
        
        % tensile force
        plot3(ePar{i}.xe(:,1) + ePar{i}.dxe(:,1),...
            ePar{i}.xe(:,2) + ePar{i}.dxe(:,2),...
            ePar{i}.xe(:,3) + ePar{i}.dxe(:,3),'-r') ;
    else
        % compressive force
        plot3(ePar{i}.xe(:,1) + ePar{i}.dxe(:,1),...
            ePar{i}.xe(:,2) + ePar{i}.dxe(:,2),...
            ePar{i}.xe(:,3) + ePar{i}.dxe(:,3),'-b') ;
    end
end

% nodal loads
quiver3(nCoord(:,1),nCoord(:,2),nCoord(:,3),...
    fbfMat(:,1) / fMax * modelSize * scaleF ,...
    fbfMat(:,2) / fMax * modelSize * scaleF ,...
    fbfMat(:,3) / fMax * modelSize * scaleF , 0, 'm') ;

% nodal reactions
quiver3(nCoord(:,1),nCoord(:,2),nCoord(:,3),...
    fbuMat(:,1) / fMax * modelSize * scaleF ,...
    fbuMat(:,2) / fMax * modelSize * scaleF ,...
    fbuMat(:,3) / fMax * modelSize * scaleF , 0, 'g') ;

end
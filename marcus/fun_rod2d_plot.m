function fun_rod2d_plot(nCoord,ePar,gDof,u,fb,fKnown,uKnown,varargin)

% plot options
if isempty(varargin)
    opts = [] ;
else
    opts = varargin{1} ;
end
% default setting
if ~isfield(opts,'scaleU') , opts.scaleU = 0.1 ; end
if ~isfield(opts,'scaleF') , opts.scaleF = 0.2 ; end
if ~isfield(opts,'scaleP') , opts.scaleP = 0.1 ; end % scaling factors for soil pressure
if ~isfield(opts,'type') , opts.type = 'none' ; end % N, V, M, P

% applied forces
fbf = zeros(size(fb)) ;
fbf(fKnown) = fb(fKnown) ;

% reaction forces
fbu = zeros(size(fb)) ;
fbu(uKnown) = fb(uKnown) ;

% nodal forces arranged as nCoord (moments are discarded!!!) ;
fbMat = fb(gDof(1:2,:)') ;
fbfMat = fbf(gDof(1:2,:)') ;
fbuMat = fbu(gDof(1:2,:)') ;

% nodal displacements arranged as nCoord (rotations are discarded!!!);
uMat = u(gDof(1:2,:)') ;

uMax = max(abs(uMat(:))) ;
fMax = max(abs(fbMat(:))) ;

for i = 1:1:size(nCoord,1)
    for j = i+1:1:size(nCoord,1)
        modelSize(i,j) = norm(nCoord(i,:) - nCoord(j,:)) ;
    end
end

% characteristic length of the model
modelSize = max(modelSize(:)) ;
modelCentroid = mean(nCoord,1) ;

% nodal coordinates in deformed configuration (with scaling)
nCoord_def = nCoord + uMat/uMax * modelSize * opts.scaleU ;

% nodal coordiantes in deformed configuration (with scaling)
for i = 1:1:numel(ePar)
    ePar{i}.dxe = u(ePar{i}.eDof)'/uMax * modelSize * opts.scaleU ;
end

figure
hold all
box on
xlim(modelCentroid(1) + [-modelSize,+modelSize])
ylim(modelCentroid(2) + [-modelSize,+modelSize])

% plot(x,y,z) create a line linking all points
plot(nCoord(:,1),nCoord(:,2),'ko')
plot(nCoord_def(:,1),nCoord_def(:,2),'rx')

for i = 1:1:size(nCoord,1)
    % text(x,y,z,string) create a string at x,y,z
    text(nCoord(i,1),nCoord(i,2),num2str(i),'fontsize',16)
    text(nCoord_def(i,1),nCoord_def(i,2),num2str(i),'fontsize',16)
end

% alternative way of plotting elements:
% plot3(nCoord(eTop',1),nCoord(eTop',2),nCoord(eTop',3),'--k')
% plot3(nCoord_def(eTop',1),nCoord_def(eTop',2),nCoord_def(eTop',3),'b')

% element plot
for i = 1:1:numel(ePar)
    
    % undeformed configuration
    plot(ePar{i}.xe(:,1),ePar{i}.xe(:,2),'--k')
    
    % deformed configuration
    plot(ePar{i}.xe(:,1) + ePar{i}.dxe(:,1),...
         ePar{i}.xe(:,2) + ePar{i}.dxe(:,2),'-r') ;
    
end

% nodal loads
quiver(nCoord(:,1),nCoord(:,2),...
    fbfMat(:,1) / fMax * modelSize * opts.scaleF ,...
    fbfMat(:,2) / fMax * modelSize * opts.scaleF , 0, 'm') ;

% nodal reactions
quiver(nCoord(:,1),nCoord(:,2),...
    fbuMat(:,1) / fMax * modelSize * opts.scaleF ,...
    fbuMat(:,2) / fMax * modelSize * opts.scaleF , 0, 'g') ;

% element results
if ~strcmp(opts.type,'none')
    
    for i = 1:1:numel(ePar)
        
        X(1,:) = ePar{i}.xe(1,:) ;
        X(4,:) = ePar{i}.xe(2,:) ;
        
        % local axis orthogonal to beam axis (unit length)
        yp = ePar{i}.T(2,1:2) ;
        
        switch opts.type
            
            case 'N'
                
                X(2,:) = ePar{i}.xe(1,:) + ...
                    yp * ePar{i}.N(1) / fMax * modelSize * opts.scaleF ;
                
                X(3,:) = ePar{i}.xe(2,:) + ...
                    yp * ePar{i}.N(2) / fMax * modelSize * opts.scaleF ;
                
            case 'V'
                
                X(2,:) = ePar{i}.xe(1,:) + ...
                    yp * ePar{i}.V(1) / fMax * modelSize * opts.scaleF ;
                
                X(3,:) = ePar{i}.xe(2,:) + ...
                    yp * ePar{i}.V(2) / fMax * modelSize * opts.scaleF ;
                
            case 'M'
                
                X(2,:) = ePar{i}.xe(1,:) - ...
                    yp * ePar{i}.M(1) / fMax * opts.scaleF ;
                
                X(3,:) = ePar{i}.xe(2,:) - ...
                    yp * ePar{i}.M(2) / fMax * opts.scaleF ;
                
            case 'P'
                
                % proportional to v1
                X(2,:) = ePar{i}.xe(1,:) - ...
                    yp * ePar{i}.k0 * ePar{i}.dlb(1) * ...
                    uMax^2 / fMax * modelSize * opts.scaleP ; 
                
                % proportional to v2
                X(3,:) = ePar{i}.xe(2,:) - ...
                    yp * ePar{i}.k0 * ePar{i}.dlb(3) * ...
                    uMax^2 / fMax * modelSize * opts.scaleP ; 
                
        end
        
        % internal force diagram
        plot(X(:,1),X(:,2),'m') ;
        
    end
    
end

end
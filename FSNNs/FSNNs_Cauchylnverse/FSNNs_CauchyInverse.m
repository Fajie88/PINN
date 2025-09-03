
%  ========================================================================
%          FSNNs for solving inverse Cauchy problems 
%          ---- 2D & 3D
%          ---- Laplace, biharmonic & PDEs with fundamental solutions
%  ========================================================================
%  COPYRIGHT NOTICE
%  © 2025 Fajie Wang, Xin Li. All Rights Reserved.
%   
%  Redistribution and use in source or binary forms, with or without
%  modification, are strictly prohibited unless expressly permitted
%  in writing by the copyright holder.
%  
%  DISCLAIMER:
%  This code is provided "AS IS," without warranty of any kind, express or implied, including 
%  but not limited to the warranties of merchantability, fitness for a particular purpose, and 
%  non-infringement. In no event shall the copyright holder be liable for any claim, damages, 
%  or other liability, whether in an action of contract, tort, or otherwise, arising from, out 
%  of or in connection with the code or the use or other dealings in the code.
%  --------------------------------------------------------------------------
%  References：
%  [1] https://ww2.mathworks.cn/help/deeplearning/ug/solve-partial-differential-equations-with-lbfgs-method-and-deep-learning.html
%  [2] F. Wang, X. Li, H. Liu, L. Qiu, X. Yue. An adaptive method of fundamental solutions using physics-informed neural networks. 
%      Eng Anal Bound Elem. 2025, 178:106295. https://doi.org/10.1016/j.enganabound.2025.106295
%  --------------------------------------------------------------------------
%  Qingdao University 
%  College of Mechanical and Electrical Engineering, Qingdao University
%  E-mail: wfj1218@126.com
%  ========================================================================
clear; clc; format long;
clearvars -global
t0 = clock; 
global u_exact activation_function numLayers du_dn_exact 
% -----------------------------------------------------
activation_function = 1; % activation_function
numLayers = 2; % number of layer
numNeurons = 3; % number of neurons
%------------------------------------------------------
options = optimoptions('fmincon', ...
        'HessianApproximation', 'lbfgs', ...                                                                                                                                                                                                                                                                                                                                                                                                                                              
        'MaxIterations',200, ...
        'MaxFunctionEvaluations',5000, ...
        'OptimalityTolerance',1e-15, ...
        'SpecifyObjectiveGradient',true, ...
        'Display', 'iter', ...
        'UseParallel',true, ...
        'Algorithm', 'sqp', ...
        'FunctionTolerance', 1e-20, ...
        'StepTolerance', 1e-20);
%   'Algorithm', 'active-set'
%                'sqp'
%                'interior-point'
% -----------------------------------------------------
nb = 300; % number of boundary points

N1 = 250; % number of accessible boundary points

[bp, inp, bp_D, bp_N, N_flux, S] = Geometry_Data(nb, N1);  
%------------------------------
nbn = 13; % number of source points

theta = linspace(0, 2*pi-2*pi/nbn, nbn); % source angle

aerf = 0.01; % noise
a = aerf*100; % for display
u_exact = @(x,y) exp(x).*sin(y); %analytical solution
du_dx = @(x,y) exp(x).*sin(y);
du_dy = @(x,y) exp(x).*cos(y);
du_dn_exact = @(x,y,n1,n2) n1.*du_dx(x,y) + n2.*du_dy(x,y);

%-------------------Boundary conditions-----------------------
[BC] = Boundary_Conditions(bp_D,N_flux,aerf);
% -----------------------------------------------------
dlBC = dlarray(BC,'CB');
dltheta = dlarray(theta(1,:),'CB');
dlbp_x_accessible = dlarray(bp_D(:,1),'CB');
dlbp_y_accessible = dlarray(bp_D(:,2),'CB');
dlbp_x_inaccessible = dlarray(bp_N(:,1),'CB');
dlbp_y_inaccessible = dlarray(bp_N(:,2),'CB');
dlflux = dlarray(N_flux,'CB');
%--------------------------------------------
[parameters] = Parameters_Initialize(numLayers, numNeurons, nbn);
% ====================================================================
%           Train Network Using fmincon-Optimization 
%
%     The fmincon function requires the learnable parameters to be 
%     specified as a vector. So, convert the parameters to a vector 
%     using the paramsStructToVector function.
% ====================================================================
[parametersV,parameterNames,parameterSizes] = parameterStructToVector(parameters);
parametersV = extractdata(parametersV);
% Create a function handle with one input that defines the objective function.
objFun = @(parameters) objectiveFunction(parameters,dltheta,dlbp_x_accessible,dlbp_y_accessible,dlBC,parameterNames,parameterSizes,dlflux);
% Update the learnable parameters using the fmincon function. 
parametersV = fmincon(objFun,parametersV,[],[],[],[],[],[],[],options);
% ============================================================
%                          Out-Put
% ============================================================
% convert the vector of parameters to a structure
parameters = parameterVectorToStruct(parametersV,parameterNames,parameterSizes);

[s, E_globalS] = numerical_solution(parameters, dltheta, theta, S);

figure(1)
plot(bp_D(:,1),bp_D(:,2),'k.')
hold on
plot(bp_N(:,1),bp_N(:,2),'b.')
hold on 
plot(s(:,1), s(:,2),'r*')
xlabel('{\itx}(m)', 'FontSize', 12)
ylabel('{\ity}(m)', 'FontSize', 12)
 legend( 'Accessible boundary', 'Inaccessible boundary', 'Sourse points');
 set (gca, 'Fontname', 'Times New Roman','FontSize',15);
axis equal
hold off 

disp(['Elapsed total_time is: ',num2str(etime(clock,t0)),' seconds.'])
disp(['Error_global on the whole domain: ',num2str(E_globalS)])


function sig = activ_function(x)
global activation_function
% ------------------------------------------------------------
if activation_function == 1
    sig = logsig(x);                 % nonlinear function used in neural networks
elseif activation_function == 2
    sig = tanh(x); 
elseif activation_function == 3
    sig = x.*logsig(x);  
elseif activation_function == 4
    sig =  log(1 + exp(x));
elseif activation_function == 5
    sig = sin(x); 
elseif activation_function == 6 
    sig = atan(x); 
elseif activation_function == 7
    sig = x.*tanh(log(1 + exp(x))); 
end
end

function [bp,inp,bp_D,bp_N,N_flux,S] = Geometry_Data(nb,N1)
theta=linspace(0,2*pi-2*pi/nb,nb);
r=1;
xb=r.*cos(theta);
yb=r.*sin(theta);
pn1=cos(theta);
pn2=sin(theta);

pn1=pn1./sqrt(pn1.^2+pn2.^2);
pn2=pn2./sqrt(pn1.^2+pn2.^2);
pn1=pn1';pn2=pn2';
x0=-6 ;  x1=-x0;  y0=x0;  y1=-x0;
nx=200;  ny=nx;
ti=linspace(x0,x1,nx);
tj=linspace(y0,y1,ny);
NN=size(ti,2)*size(tj,2);
[xt,yt] = meshgrid(ti,tj);
B=[xt(:)';yt(:)'];
xx=B(1,:);
yy=B(2,:);
    for i=1:NN
xx(i)=xx(i);
yy(i)=yy(i);
    end  
in=inpolygon(xx,yy,xb,yb);
k=0;
for j=1:NN
    if in(j)==1
        k=k+1;
        xi(k)=xx(j);
        yi(k)=yy(j);
    else
    end
end
ni=k;
bp=[xb' yb']; inp=[xi' yi'];
N_flux=[pn1 pn2];
S = [bp;inp];

% =========================================================================
Accessible_BC_double = 1 : N1;
unknow = N1 + 1 : nb;
bp_D = bp(Accessible_BC_double,:);
bp_N = bp(unknow,:);
end


function [BC,N_flux_x, N_flux_y] = Boundary_Conditions(bp_D,N_N,aerf)
global u_exact du_dn_exact Accessible_BC
% ---------------------------------------------
Accessible_BC = length(bp_D); 
% =========================================================================
%                               Boundary conditions
% =========================================================================
BC = zeros(2*Accessible_BC, 1);
% ---------------------------------------------------
BC_Dirichlet = u_exact(bp_D(1:Accessible_BC, 1), bp_D(1:Accessible_BC, 2)).*(1+(-1+2*rand(size(bp_D(1:Accessible_BC, 2))))*aerf);
BC_Neumann = du_dn_exact(bp_D(1:Accessible_BC, 1), bp_D(1:Accessible_BC, 2), N_N(1:Accessible_BC, 1), N_N(1:Accessible_BC, 2)).*(1+(-1+2*rand(size(bp_D(1:Accessible_BC, 2))))*aerf);
% % ---------------------------------------------------
BC(1:Accessible_BC, :) = BC_Dirichlet; 
BC(Accessible_BC+1:2*Accessible_BC, :) = BC_Neumann; 
N_flux_x = N_N(:, 1);
N_flux_y = N_N(:, 2);
end

function [parameters] = Parameters_Initialize(numLayers, numNeurons, nb)
% =========================================================================
%  The first layer has two input channels corresponding to the inputs theta.
%  The last layer has one output r(theta).
% -------------------- Initialize the parameters for the first layer
sz = [numNeurons 1];   %  one input channels
parameters.fc1_Weights = initializeHe(sz,1,'double');
parameters.fc1_Bias = initializeZeros([numNeurons 1],'double');

for layerNumber = 2 : numLayers - 1   % hidden layers
    name = "fc" + layerNumber;

    sz = [numNeurons numNeurons];
    numIn = numNeurons;
    parameters.(name + "_Weights") = initializeHe(sz,numIn,'double');
    parameters.(name + "_Bias") = initializeZeros([numNeurons 1],'double');
end

sz = [1 numNeurons];    %  Initialize the parameters for the output layer
numIn = numNeurons;
parameters.("fc" + numLayers + "_Weights") = initializeHe(sz,numIn,'double');
parameters.("fc" + numLayers + "_Bias") = initializeZeros([1 1],'double');
parameters.("alafa") = initializeOnes([1 nb],'double');
end

% --------------------------------------------

function parameter = initializeZeros(sz,className)
arguments
    sz
    className = 'single'
end
parameter = zeros(sz,className);
parameter = dlarray(parameter);
end

%-------------------------------------

function parameter = initializeOnes(sz,className)
arguments
    sz
    className = 'single'
end
parameter = 0.1.*ones(sz,className);
parameter = dlarray(parameter);
end

%-------------------------------------

function parameter = initializeHe(sz,numIn,className)
arguments
    sz
    numIn
    className = 'single'
end
parameter = sqrt(2/numIn) *randn(sz,className);
parameter = dlarray(parameter);
end

%-----------------------------------------

function [loss,gradientsV] = objectiveFunction(parametersV,dltheta,dlbp_x_accessible,dlbp_y_accessible,dlBC,parameterNames,parameterSizes,dlflux)
parametersV = dlarray(parametersV);
parameters = parameterVectorToStruct(parametersV,parameterNames,parameterSizes);
% Evaluate model gradients and loss.
accmodelGradients = dlaccelerate(@modelGradients);
[gradients,loss] = dlfeval(accmodelGradients,parameters,dltheta,dlbp_x_accessible,dlbp_y_accessible,dlBC,dlflux);
% Return loss and gradients for fmincon.
gradientsV = parameterStructToVector(gradients);
gradientsV = extractdata(gradientsV);
loss = extractdata(loss);    
end

function [gradients,loss] = modelGradients(parameters,dltheta,dlbp_x_accessible,dlbp_y_accessible,dlBC,dlflux)
global Accessible_BC
% ----------------------------------------------------------------------
% Make predictions with the initial conditions.
U = model(parameters,dltheta);

r = (sin(U)+1)./2.*(3 + 1) - (sin(U)-1)./2.*(0.2 + 1); 
s_x = r.*cos(dltheta);
s_y = r.*sin(dltheta);
r = sqrt((dlbp_x_accessible - s_x).^2+(dlbp_y_accessible - s_y).^2); 
G = -1./2./pi.*log(r);
alfa = parameters.alafa;
U_all = sum(G.*alfa,2);

Gx = -1./2./pi./r.^(2).*(dlbp_x_accessible - s_x);
Gy = -1./2./pi./r.^(2).*(dlbp_y_accessible - s_y);
% =========================================================================
%           Calculate lossU. Enforce Dirichlet boundary conditions
% =========================================================================
U_Dirichlet  = dlarray(U_all(1:Accessible_BC),'CB');
lossU = mse(U_Dirichlet, dlBC(1:Accessible_BC));
 
% =========================================================================
%           Calculate lossN. Enforce Neumann boundary conditions
% =========================================================================

dlflux_x = dlflux(1:Accessible_BC, 1); 
dlflux_y = dlflux(1:Accessible_BC, 2); 
G_N = Gx.*dlflux_x + Gy.*dlflux_y ; 
U_Neumann = sum(G_N.*alfa,2);
U_Neumann = dlarray(U_Neumann, 'CB');
lossN = mse(U_Neumann, dlBC(1+Accessible_BC:2*Accessible_BC));

% Combine losses.
loss = lossU + lossN;

% Combine losses.
% Calculate gradients with respect to the learnable parameters.
gradients = dlgradient(loss, parameters,'EnableHigherDerivatives',false, 'RetainData',true);
end

function dlU = model(parameters,dltheta)
global numLayers
% First fully connect operation.
weights = parameters.fc1_Weights;
bias = parameters.fc1_Bias;
dlU = fullyconnect(dltheta,weights,bias);
%fully connect operations for hidden layers.
for i = 2 : numLayers
    name = "fc" + i;
    dlU = activ_function(dlU);
    weights = parameters.(name + "_Weights");
    bias = parameters.(name + "_Bias");
    dlU = fullyconnect(dlU, weights, bias);
end
end

function [parametersV,parameterNames,parameterSizes] = parameterStructToVector(parameters)
% parameterStructToVector converts a struct of learnable parameters to a
% vector and also returns the parameter names and sizes.

% Parameter names.
parameterNames = fieldnames(parameters);

% Determine parameter sizes.
numFields = numel(parameterNames);
parameterSizes = cell(1,numFields);
for i = 1:numFields
    parameter = parameters.(parameterNames{i});
    parameterSizes{i} = size(parameter);
end
% Calculate number of elements per parameter.
numParameterElements = cellfun(@prod,parameterSizes);
numParamsTotal = sum(numParameterElements);
% Construct vector
parametersV = zeros(numParamsTotal,1,'like',parameters.(parameterNames{1}));
count = 0;
for i = 1:numFields
    parameter = parameters.(parameterNames{i});
    numElements = numParameterElements(i);
    parametersV(count+1:count+numElements) = parameter(:);
    count = count + numElements;
end
end

function parameters = parameterVectorToStruct(parametersV,parameterNames,parameterSizes)
% parameterVectorToStruct converts a vector of parameters with specified
% names and sizes to a struct.
parameters = struct;
numFields = numel(parameterNames);
count = 0;
for i = 1:numFields
    numElements = prod(parameterSizes{i});
    parameter = parametersV(count+1:count+numElements);
    parameter = reshape(parameter,parameterSizes{i});
    parameters.(parameterNames{i}) = parameter;
    count = count + numElements;
end
end

function [s, E_globalS] = numerical_solution(parameters, dltheta, theta, S)
global u_exact
U = model(parameters,dltheta);
U = extractdata(U);
r = (sin(U)+1)./2.*(3 + 1) - (sin(U)-1)./2.*(0.2 + 1); 
s_x = r.*cos(theta);
s_y = r.*sin(theta);
s = [s_x' s_y'];
alfa = parameters.alafa;
alfa = dlarray(alfa,'CB');
r_S = sqrt((S(:,1) - s_x).^2+(S(:,2) - s_y).^2); 
G_S = -1./2./pi.*log(r_S);
U_S = sum(G_S.*alfa,2);
U_S = extractdata(U_S);
E_S = u_exact(S(:,1),S(:,2));
E_globalS = (sum(abs(U_S - E_S).^2)./(sum(E_S.^2))).^0.5;
end









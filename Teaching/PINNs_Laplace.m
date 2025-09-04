%  COPYRIGHT NOTICE
%  © 2025 Fajie Wang, All Rights Reserved.
%   
%  Redistribution and use in source or binary forms, with or without
%  modification, are strictly prohibited unless expressly permitted
%  in writing by the copyright holder.
%  
 clc
 clear all
 format long
 clearvars
 t0 = clock;
 % Time Start
 %----------------------------------------------------
global u_exact f_source
 global numLayers
 global BC Dirichlet_BC Neumann_BC Robin_BC
 global activation_function omiga
%----------------------------------------------------
numLayers = 3; % 神经网络层数
numNeurons = 10; % 神经元个数
%----------------------------------------------------
options = optimoptions('fmincon', ...
 'HessianApproximation', 'lbfgs', ...
 'MaxIterations',1000, ... %训练次数
'MaxFunctionEvaluations',30000, ...
 'OptimalityTolerance',1e-15, ...
'SpecifyObjectiveGradient',true, ...
 'Display', 'iter', ...
 'UseParallel',true, ...
 'Algorithm', 'sqp', ...
 'FunctionTolerance', 1e-20, ...
 'StepTolerance', 1e-20);
 % 'Algorithm', 'active-set'
 %'sqp'
 %'interior-point'
 %----------------------------------------------------
activation_function = 1;
 % = 1: 激活函数(Sigmoid) logsig(x)
 % = 2: 激活函数(Tanh) tanh(x)
 % = 3: 激活函数(Swish) x*logsig(x) good 1
 % = 4: 激活函数(Softplus) log(1 + exp(x))
 % = 5: 激活函数(Sinusoid) sin(x)
 % = 6: 激活函数(ArcTan) arctan(x)
 % = 7: 激活函数(Mish) x*tanh(log(1 + exp(x)))
 %----------------------------------------------------
xscal = [0 1]; % Computational domain
 yscal = [0 1];
 nx = 40;
 ny = 40;
 %----------------------------------------------------
cases = 1;
 omiga = 2;
 Exacts(cases);
 %----------------------------------------------------
[bp1, bp2, bp3, bp4, inp, N_flux, bp, S] = geodata_rectangle(xscal, yscal, nx,ny);
 N_total = length(S);
 %画出布点
figure(1)
 plot(S(:,1),S(:,2),'b.'); axis equal;
 axis([xscal(1),xscal(2),yscal(1),yscal(2)]); % 设置当前坐标轴x轴和 y轴的限制范围
% =======================================================================
 % Boundary conditions
 % =======================================================================
 [BC, Dirichlet_BC, Neumann_BC, Robin_BC] = Boundary_Conditions(bp, bp1, bp2,bp3, bp4, N_flux);
 FSource = ones(N_total,1).*f_source(S(:, 1), S(:, 2));
 % =====================================================================
 % Convert the training data to dlarray objects
% =====================================================================
 dlX = dlarray(S(:, 1)','CB'); % 1*N
 dlY = dlarray(S(:, 2)','CB');
 dlBC = dlarray(BC','CB');
 dlFSource = dlarray(FSource','CB');
 dlflux = dlarray(N_flux','CB');
 % ====================================================================
 % Initialize the learnable parameters
 % ====================================================================
 [parameters] = Parameters_Initialize(numLayers, numNeurons);
 % ====================================================================
 %Train Network Using fmincon-Optimization
%  The fmincon function requires the learnable parameters to be
%  specified as a vector. So, convert the parameters to a vector
%  using the paramsStructToVector function.
 % ====================================================================
 [parametersV,parameterNames,parameterSizes] =parameterStructToVector(parameters);
 parametersV = extractdata(parametersV);
 % Create a function handle with one input that defines the objective function.
 objFun = @(parameters)objectiveFunction(parameters,dlX,dlY,dlBC,dlFSource,dlflux,parameterNames,parameterSizes);
 % Update the learnable parameters using the fmincon function.
 parametersV = fmincon(objFun,parametersV,[],[],[],[],[],[],[],options);
 % ============================================================
 %Out-Put
 % ============================================================
 % convert the vector of parameters to a structure
 parameters = parameterVectorToStruct(parametersV,parameterNames,parameterSizes);
 %----------------------------------------------------------
 u_num = model(parameters,dlX,dlY);
 u_num = extractdata(u_num);
 %----------------------------------------------------------
 err_norm = norm(u_num'- u_exact(S(:, 1), S(:, 2))) / norm(u_exact(S(:, 1), S(:,2))); %relative error
err_absolute = max(abs(u_num'- u_exact(S(:, 1), S(:, 2)))); %max error
b1 = linspace(min(S(:, 1)), max(S(:, 1)), 150);
 b2 = linspace(min(S(:, 2)), max(S(:, 2)), 150);
 [x, y] = meshgrid(b1, b2);
 abs_er = griddata(S(:, 1), S(:, 2), abs(u_num'- u_exact(S(:, 1), S(:, 2))),x, y, 'cubic');
 figure(2); mesh(x,y,abs_er)
 re_er = griddata(S(:, 1), S(:, 2), abs(u_num'- u_exact(S(:, 1), S(:,2)))./abs(u_exact(S(:,1),S(:,2))), x, y, 'cubic');
 figure(3); mesh(x,y,re_er)
 % =========================================================================
 disp(['Elapsed total_time is: ',num2str(etime(clock,t0)),' seconds.'])
 disp(['Error_Norm: ',num2str(err_norm)])
 disp(['Error_Absolute: ',num2str(err_absolute)])
 % =========================================================================
 function sig = activ_function(x)
 global activation_function
 %------------------------------------------------------------
 if activation_function == 1
 sig = logsig(x);
 % nonlinear function used in neural networks
 elseif activation_function == 2
 sig = tanh(x);
 elseif activation_function == 3
 sig = x.*logsig(x);
 elseif activation_function == 4
 sig = log(1 + exp(x));
 elseif activation_function == 5
 sig = sin(x);
 elseif activation_function == 6
 sig = atan(x);
 elseif activation_function == 7
 sig = x.*tanh(log(1 + exp(x)));
 end
 end
 % =========================================================================
 function Exacts(cases)
 global u_exact f_source du_dn_exact du_R_exact omiga
 if cases == 1 %线性问题
u_exact = @(x, y) exp(x).*cos(y); %精确解
f_source = @(x, y) 0; %右端项
du_dx = @(x, y) exp(x).*cos(y); %对x求一阶导
du_dy = @(x, y)-exp(x).*sin(y); %对y求一阶导
du_dn_exact = @(x, y, n1, n2) n1.*du_dx(x, y) + n2.*du_dy(x, y);
du_R_exact = @(x, y, n1, n2) n1.*du_dx(x, y) + n2.*du_dy(x, y) + omiga*u_exact(x,y);
 %cases == 2，cases == 3对应不同的控制方程
elseif cases == 2 %Cauchy非线性问题
u_exact = @(x, y)-0.1*exp(sqrt(20)*y).*sin(sqrt(10)*x);
 f_source = @(x, y)-0.01*exp(2*sqrt(20)*y).*(sin(sqrt(10)*x)).^2;
 du_dx = @(x, y)-0.1*sqrt(10)*exp(sqrt(20)*y).*cos(sqrt(10)*x);
 du_dy = @(x, y)-0.1*sqrt(20)*exp(sqrt(20)*y).*sin(sqrt(10)*x);
 du_dn_exact = @(x, y, n1, n2) n1.*du_dx(x, y) + n2.*du_dy(x, y);
 du_R_exact = @(x, y, n1, n2) n1.*du_dx(x, y) + n2.*du_dy(x, y) + omiga*u_exact(x,y);
 elseif cases == 3 %Helmholtz非线性
u_exact = @(x, y) x.^2.*y.^2;
 f_source = @(x, y)-x.^4.*y.^4+2*(x.^2+y.^2)+50*x.^2.*y.^2;
 du_dx = @(x, y) 2*x.*y.^2;
 du_dy = @(x, y) 2*x.^2.*y;
 du_dn_exact = @(x, y, n1, n2) n1.*du_dx(x, y) + n2.*du_dy(x, y);
 du_R_exact = @(x, y, n1, n2) n1.*du_dx(x, y) + n2.*du_dy(x, y) + omiga*u_exact(x,y);
 end
 end
 % =========================================================================
 function [bp1,bp2,bp3,bp4,inp,N,bp,S] = geodata_rectangle(x, y, nx, ny)
 bp1 = zeros(ny-2, 2);
 % left boundary
 bp2 = zeros(nx-2, 2);
 bp3 = zeros(ny-2, 2);
 bp4 = zeros(nx-2, 2);
 % lower boundary
 % right boundary
 % upper boundary
 x_Temp = linspace(x(1),x(2),nx);
 y_Temp = linspace(y(1),y(2),ny);
 bp1(:,1) = x(1);
 bp1(:,2) = y_Temp(2:ny-1); % bp1 x=0
 bp2(:,1) = x_Temp(2:nx-1);
 bp2(:,2) = y(1); % bp2 y=0
 bp3(:,1) = x(2);
 bp3(:,2) = y_Temp(2:ny-1); % bp3 x=1
bp4(:,1) = x_Temp(2:nx-1);
 bp4(:,2) = y(2); % bp4 y=1
 n1 = zeros(ny-2,2);
 n2 = zeros(nx-2,2);
 n3 = zeros(ny-2,2);
 n4 = zeros(nx-2,2);
 n1(:,1) =-1;
 n2(:,2) =-1;
 n3(:,1) = 1;
 n4(:,2) = 1;
 N = [n1;n2;n3;n4];
 % ============= interior points =============
 [Allx,Ally] = meshgrid(x_Temp,y_Temp);
 XX = reshape(Allx(2:end-1,2:end-1),(nx-2)*(ny-2),1);
 YY = reshape(Ally(2:end-1,2:end-1),(nx-2)*(ny-2),1);
 inp = [XX YY];
 % ====================== 内部随机布点=============
 % dx = (x(2)-x(1))/(nx-1);
 % dy = (y(2)-y(1))/(ny-1);
 % rr = 2*rand(length(inp),1)-1;
 % inp(:,1) = inp(:,1)+rr*dx/3;
 % inp(:,2) = inp(:,2)+rr*dy/3;
 %====================================
 bp = [bp1;bp2;bp3;bp4];
 % whole boundary
 S = [bp;inp]; % the total points = boundary points + interior points
 end
 % =========================================================================
 function [BC, Dirichlet_BC, Neumann_BC, Robin_BC] = Boundary_Conditions(bp, bp1,bp2, bp3, bp4, N_N)
 global u_exact du_dn_exact du_R_exact
 %--------------------------------------------
N1 = length(bp1);
 N2 = length(bp2);
 N3 = length(bp3);
 N4 = length(bp4);
 %--------------------------------------------- 编号
N_1 = 1 : N1;
 N_2 = N1 + 1 : N1 + N2;
 N_3 = N1 + N2 + 1 : N1 + N2 + N3;
N_4 = N1 + N2 + N3 + 1 : N1 + N2 + N3 + N4;
 % =========================================================================
 Dirichlet_BC = [N_1, N_2, N_3, N_4];
 Neumann_BC = [];
 Robin_BC = [];
 % =========================================================================
 %边界条件
% =========================================================================
 N_bp = length(bp);
 BC = zeros(N_bp, 1);
 %--------------------------------------------------
BC_Dirichlet = u_exact(bp(:, 1), bp(:, 2));
 BC_Neumann = du_dn_exact(bp(:, 1), bp(:, 2), N_N(:, 1), N_N(:, 2));
 BC_Robin = du_R_exact(bp(:, 1), bp(:, 2), N_N(:, 1), N_N(:, 2));
 %--------------------------------------------------
BC(Dirichlet_BC, :) = BC_Dirichlet(Dirichlet_BC, :);
 BC(Neumann_BC, :) = BC_Neumann(Neumann_BC, :);
 BC(Robin_BC, :) = BC_Robin(Robin_BC, :);
 end
 % =========================================================================
 function [parameters] = Parameters_Initialize(numLayers, numNeurons)
 % The first layer has two input channels corresponding to the inputs x and y.
 % The last layer has one output u(x, y).
 %------------------------------------------------------------------------
 parameters = struct;
 %-------------------- Initialize the parameters for the first layer
 sz = [numNeurons 2]; % two input channels
 parameters.fc1_Weights = initializeHe(sz,2,'double');
 parameters.fc1_Bias = initializeZeros([numNeurons 1],'double');
 for layerNumber = 2 : numLayers- 1 % the remaining intermediate layers
 name = "fc" + layerNumber;
 sz = [numNeurons numNeurons];
 numIn = numNeurons;
 parameters.(name + "_Weights") = initializeHe(sz,numIn,'double');
 parameters.(name + "_Bias") = initializeZeros([numNeurons 1],'double');
 end
 sz = [1 numNeurons];
 numIn = numNeurons;
 % Initialize the parameters for the final layer
 parameters.("fc" + numLayers + "_Weights") = initializeHe(sz,numIn,'double');
 parameters.("fc" + numLayers + "_Bias") = initializeZeros([1 1],'double');
 end
% =========================================================================
 function parameter = initializeZeros(sz,className)
 arguments
 sz
 className = 'single'
 end
 parameter = zeros(sz,className);
 parameter = dlarray(parameter);
 end
 % =========================================================================
 function parameter = initializeHe(sz,numIn,className)
 arguments
 sz
 numIn
 className = 'single'
 end
 parameter = sqrt(2/numIn) * randn(sz,className);
 parameter = dlarray(parameter);
 end
 % =========================================================================
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
 % =========================================================================
 function [loss,gradientsV] =objectiveFunction(parametersV,dlX,dlY,dlBC,dlFSource,dlflux,parameterNames,parameterSizes)
 % Convert parameters to structure of dlarray objects.
 parametersV = dlarray(parametersV);
 parameters = parameterVectorToStruct(parametersV,parameterNames,parameterSizes);
 % Evaluate model gradients and loss.
 accmodelGradients = dlaccelerate(@modelGradients); % 加速
[gradients,loss] = dlfeval(accmodelGradients,parameters,dlX,dlY,dlBC,dlFSource,dlflux);
 % Return loss and gradients for fmincon.
 gradientsV = parameterStructToVector(gradients);
 gradientsV = extractdata(gradientsV);
 loss = extractdata(loss);
 end
 % =========================================================================
 function [gradients,loss] = modelGradients(parameters,dlX,dlY,dlBC,dlFSource,dlflux)
 global Dirichlet_BC Neumann_BC Robin_BC omiga
 %----------------------------------------------------------------------
 % Make predictions with the initial conditions.
 U = model(parameters,dlX,dlY);
 % Calculate derivatives with respect to X and Y.
gradientsU = dlgradient(sum(U,'all'),{dlX,dlY},'EnableHigherDerivatives',true,'RetainData',true);
 Ux = gradientsU{1};
 Uy = gradientsU{2};
 % Calculate second-order derivatives with respect to X and Y.
 Uxx = dlgradient(sum(Ux,'all'),dlX,'EnableHigherDerivatives',true,'RetainData',true);
 Uyy = dlgradient(sum(Uy,'all'),dlY,'EnableHigherDerivatives',true,'RetainData',true);
 Uxy = dlgradient(sum(Ux,'all'),dlY,'EnableHigherDerivatives',true,'RetainData',true);
 % =========================================================================
 %Calculate lossF. Enforce the governing equation
 % =========================================================================
 N_inp = length(dlBC) + 1 : length(dlFSource);
 f = Uxx(N_inp) + Uyy(N_inp); %case1
 %f = Uxx(N_inp) + Uyy(N_inp)-10*U(N_inp)-U(N_inp).^2; %case2
 %f = Uxx(N_inp) + Uyy(N_inp) +50*U(N_inp)-U(N_inp).^2; %case3
 f = dlarray(f, 'CB');
 lossF = mse(f, dlFSource(N_inp));
 % =========================================================================
 %Calculate lossU. Enforce Dirichlet boundary conditions
 % =========================================================================
 if isempty(Dirichlet_BC)
 lossU = 0;
 else
 U_Dirichlet = dlarray(U(Dirichlet_BC), 'CB');
 lossU = mse(U_Dirichlet, dlBC(Dirichlet_BC));
 end
 % =========================================================================
 %Calculate lossN. Enforce Neumann boundary conditions
 % =========================================================================
 if isempty(Neumann_BC)
 lossN = 0;
 else
 dlflux_x = dlflux(1, :);
 dlflux_y = dlflux(2, :);
 U_Neumann = Ux(Neumann_BC).*dlflux_x(Neumann_BC) + Uy(Neumann_BC).*dlflux_y(Neumann_BC);
 U_Neumann = dlarray(U_Neumann, 'CB');
 lossN = mse(U_Neumann, dlBC(Neumann_BC));
 end
% =========================================================================
 % Calculate lossR. Enforce Robin boundary conditions
 % =========================================================================
 if isempty(Robin_BC)
 lossR = 0;
 else
 dlflux_x = dlflux(1, :);
 dlflux_y = dlflux(2, :);
 U_Robin = Ux(Robin_BC).*dlflux_x(Robin_BC) + Uy(Robin_BC).*dlflux_y(Robin_BC) + omiga*U(Robin_BC);
 U_Robin = dlarray(U_Robin, 'CB');
 lossR = mse(U_Robin, dlBC(Robin_BC));
 end
 % Combine losses.
 loss = lossF + lossU + lossN + lossR;
 % Calculate gradients with respect to the learnable parameters.
 gradients = dlgradient(loss, parameters,'EnableHigherDerivatives',false, 'RetainData',true);
 end
 % =========================================================================
 function dlU = model(parameters,dlX,dlY)
 global numLayers
 dlXY = [dlX;dlY];
 % numLayers = numel(fieldnames(parameters))/2;
 % First fully connect operation.
 weights = parameters.fc1_Weights;
 bias = parameters.fc1_Bias;
 dlU = fullyconnect(dlXY,weights,bias);
 % tanh and fully connect operations for remaining layers.
 for i = 2 : numLayers
 name = "fc" + i;
 dlU = activ_function(dlU);
 weights = parameters.(name + "_Weights");
 bias = parameters.(name + "_Bias");
 dlU = fullyconnect(dlU, weights, bias);
 end
 end
% =========================================================================
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
 % =========================================================================

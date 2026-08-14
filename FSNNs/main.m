clc;
clear all;
format long
clearvars -global
% t0 = clock; 
tStart = tic;   
global u_exact activation_function numLayers du_dn_exact  du_R_exact
global omiga Dirichlet_BC unknow Robin_BC k2
% -----------------------------------------------------
activation_function = 1;
numLayers = 4; 
numNeurons = 3; 
% -----------------------------------------------------
addpath('Geometry_Data', 'Initialize_the_parameters', 'Training','2D'); 
%------------------------------------------------------
options = optimoptions('fmincon', ...
        'HessianApproximation', 'lbfgs', ...                                                                                                                                                                                                                                                                                                                                                                                                                                              
        'MaxIterations',300, ... 
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
iboundary = 2;
idomain = 1; 
%---------------------------
[bp1, bp2, bp3, bp4,inp, bp, S, N_flux] = BoundaryGeometry(iboundary , idomain); 
cases = 1;     
omiga = 2;
N1 = 150;
N2 = 150;
N_1 = 1 : N1; 
N_2 = N1 + 1 : N1 + N2; 
Exacts(cases);
Dirichlet_BC_double = [N_1];
unknow = [N_2];
bp_D = bp(Dirichlet_BC_double,:);
bp_N = bp(unknow,:);
nb = 20;
%----------------------------
k2 = 2;
beta = sqrt(k2^2 - 0.25);  % sqrt(3.75)
u_exact_1 = @(x,y) cos(0.5*x + beta*y);
u = u_exact_1(S(:,1),S(:,2));
ubp_N = u_exact_1(bp_N(:,1),bp_N(:,2));
ubp_N =real(ubp_N );
aerf = 0.03;

u_exact = @(x,y) cos(0.5*x + beta*y);
%-------------------boundaty condation-----------------------
[BC] = Boundary_Conditions(bp_D,N_flux,aerf);
jiaodu = linspace(0,2*pi-2*pi/nb,nb);
% -----------------------------------------------------
tic  %  time start
dlBC = dlarray(BC,'CB');
dljiaodu = dlarray(jiaodu(1,:),'CB');
dlbp_x_zhenshi = dlarray(bp_D(:,1),'CB');
dlbp_y_zhenshi = dlarray(bp_D(:,2),'CB');
dlinp_x = dlarray(inp(:,1),'CB');
dlinp_y = dlarray(inp(:,2),'CB');
dlflux = dlarray(N_flux,'CB');

%--------------------------------------------
[parameters] = Parameters_Initialize(numLayers, numNeurons, nb);

% ====================================================================
%           Train Network Using fmincon-Optimization 
%     The fmincon function requires the learnable parameters to be 
%     specified as a vector. So, convert the parameters to a vector 
%     using the paramsStructToVector function.
% ====================================================================
[parametersV,parameterNames,parameterSizes] = parameterStructToVector(parameters);
parametersV = extractdata(parametersV);
% Create a function handle with one input that defines the objective function.
objFun = @(parameters) objectiveFunction(parameters,dljiaodu,dlbp_x_zhenshi,dlbp_y_zhenshi,dlBC,dlinp_x,dlinp_y,parameterNames,parameterSizes,dlflux);
% Update the learnable parameters using the fmincon function. 
tStart = tic;   %Time start
parametersV = fmincon(objFun,parametersV,[],[],[],[],[],[],[],options);
train_time = toc;   %training time
% ============================================================
%                          Out-Put
% ============================================================
% convert the vector of parameters to a structure
parameters = parameterVectorToStruct(parametersV,parameterNames,parameterSizes);
U = model(parameters,dljiaodu);
U = extractdata(U);
% Calculate derivatives with respect to X and Y.
x = jiaodu;
changdu = (sin(U)+1)./2.*(4+exp(sin(x)).*(sin(2*x).^2)+exp(cos(x)).*(cos(2*x).^2)) - (sin(U)-1)./2.*(0.5+exp(sin(x)).*(sin(2*x).^2)+exp(cos(x)).*(cos(2*x).^2)); %
% changdu = U;
s_x = changdu.*cos(jiaodu);
s_y = changdu.*sin(jiaodu);

coeff_1 = parameters.alafa_1;
coeff_2 = parameters.alafa_2;

coeff = coeff_1' + 1i*coeff_2';
r_inp = sqrt( (S(:,1) - s_x).^2+(S(:,2) - s_y).^2 );
G_inp = besselh(0, 1, k2 .* r_inp);

zuobian = (1i/4).*G_inp*coeff;      %Numericl rusult
% ---------------------------------------------------------

err_norm = abs(zuobian - u) ./ abs(u); %RE
err_absolute = abs(zuobian - u);  %AE

% E_global = (sum(err_absolute.^2)./(sum(u.^2))).^0.5;
r_bp_N = sqrt( (bp_N(:,1) - s_x).^2+(bp_N(:,2) - s_y).^2 );
G_bp_N = besselh(0, 1, k2 .* r_bp_N);

zuobianbp_N = (1i/4).*G_bp_N*coeff;
zuobianbp_N=real(zuobianbp_N);
% ---------------------------------------------------------

err_normbp_N = abs(zuobianbp_N - ubp_N) ./ abs(ubp_N); 
err_absolutebp_N = abs(zuobianbp_N - ubp_N);  

% ===== All points（RRMSE）========================================
RRMSE = sqrt( mean((real(zuobian) - real(u)).^2) / mean(real(u)) );
% ===== ==================================Training Time=======================
total_time = toc(tStart); 
fprintf('====================================\n');
fprintf('Training time (fmincon): %.4f seconds\n', train_time);
fprintf('Total runtime: %.4f seconds\n', total_time);
fprintf('====================================\n');






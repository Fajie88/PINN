function [gradients,loss] = modelGradients(parameters,dljiaodu,dlbp_x_zhenshi,dlbp_y_zhenshi,dlBC,dlinp_x,dlinp_y,dlflux)
global Dirichlet_BC Neumann_BC Robin_BC omiga  u_exact k2
% ----------------------------------------------------------------------
% Make predictions with the initial conditions.
U_D = model(parameters,dljiaodu);
% Calculate derivatives with respect to X and Y.
x = dljiaodu;
changdu = (sin(U_D)+1)./2.*(4+exp(sin(x)).*(sin(2*x).^2)+exp(cos(x)).*(cos(2*x).^2)) - (sin(U_D)-1)./2.*(0.5+exp(sin(x)).*(sin(2*x).^2)+exp(cos(x)).*(cos(2*x).^2)); %%changdu = U;
s_x = changdu.*cos(dljiaodu);
s_y = changdu.*sin(dljiaodu);
r = sqrt((dlbp_x_zhenshi - s_x).^2+(dlbp_y_zhenshi - s_y).^2); %得到距离r
r_1 = extractdata(r);
H0 = besselh(0, k2 .* r_1);
G = (1i/4).*H0;%基本解
G_1 = real(G);%实部
G_2 = imag(G);%虚部
G1 = [G_1,-G_2;G_2,G_1];
H1 = besselh(1,1, k2 .* r_1);%第二类边界条件
dlflux_x = dlflux(1:Dirichlet_BC, 1); 
dlflux_y = dlflux(1:Dirichlet_BC, 2); 
s_x1 = extractdata(s_x);
s_y1 = extractdata(s_y);
G_D_x = - (1i * k2 / 4) .* H1 .* ((dlbp_x_zhenshi - s_x1) ./ r_1);
G_D_y = - (1i * k2 / 4) .* H1 .* ((dlbp_y_zhenshi - s_y1) ./ r_1);
G_N = dlflux_x.*G_D_x + dlflux_y.*G_D_y;


alfa_1 = parameters.alafa_1;
alfa_2 = parameters.alafa_2;

alfa = [alfa_1,alfa_2];
U_D = sum(G1.*alfa,2);
U_D_real = U_D(1:length(dlBC)/2);

U_D_imag =U_D(length(dlBC)/2+1:length(dlBC));



% G_N = N_flux(:,1).*G_D_x + N_flux(:,2).*G_D_y;

G_N_real = real(G_N);
G_N_imag = imag(G_N);
G2 = [G_N_real,-G_N_imag;G_N_imag,G_N_real];
U_N = sum(G2.*alfa,2);
U_N_real = U_N(1:length(dlBC)/2);
U_N_imag =U_N(length(dlBC)/2+1:length(dlBC));

dlBC_real = real(dlBC);
dlBC_imag = imag(dlBC);

 U_Dirichlet_real  = dlarray(U_D_real,'CB');
 lossU_real = mse(U_Dirichlet_real, dlBC_real(1:Dirichlet_BC));
  U_Dirichlet_imag  = dlarray(U_D_imag,'CB');
 lossU_imag = mse(U_Dirichlet_imag, dlBC_imag(1:Dirichlet_BC));
U_Neuman_imag  = dlarray(U_N_imag,'CB');
U_Neuman_real  = dlarray(U_N_real,'CB');
 lossN_real = mse(U_Neuman_real, dlBC_real(1+Dirichlet_BC:2*Dirichlet_BC));
 lossN_imag = mse(U_Neuman_imag, dlBC_imag(1+Dirichlet_BC:2*Dirichlet_BC));


loss = lossU_real + lossU_imag + lossN_real+ lossN_imag;

% Combine losses.
% Calculate gradients with respect to the learnable parameters.
gradients = dlgradient(loss, parameters,'EnableHigherDerivatives',false, 'RetainData',true);
end
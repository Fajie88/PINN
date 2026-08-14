function [BC] = Boundary_Conditions(bp_D,N_N,aerf)
%function [BC, Dirichlet_BC, Neumann_BC, Robin_BC] = Boundary_Conditions(bp, bp1, bp2, bp3, bp4, N_N)
global u_exact du_dn_exact du_R_exact omiga Dirichlet_BC
% ---------------------------------------------
Dirichlet_BC = length(bp_D); 
%N2 = length(bp2);
%N3 = length(bp3);
% =========================================================================
%                               ±ß½çÌõ¼þ
% =========================================================================
BC = zeros(2*Dirichlet_BC, 1);
% ---------------------------------------------------
BC_Dirichlet = u_exact(bp_D(1:Dirichlet_BC, 1), bp_D(1:Dirichlet_BC, 2)).*(1+(-1+2*rand(size(bp_D(1:Dirichlet_BC, 2))))*aerf);
BC_Neumann = du_dn_exact(bp_D(1:Dirichlet_BC, 1), bp_D(1:Dirichlet_BC, 2), N_N(1:Dirichlet_BC, 1), N_N(1:Dirichlet_BC, 2)).*(1+(-1+2*rand(size(bp_D(1:Dirichlet_BC, 2))))*aerf);

% % ---------------------------------------------------
BC(1:Dirichlet_BC, :) = BC_Dirichlet; 
BC(Dirichlet_BC+1:length(BC), :) = BC_Neumann; 
N_flux_x = N_N(1:Dirichlet_BC, 1);
N_flux_y = N_N(1:Dirichlet_BC, 2);
% BC(Robin_BC, :) = BC_Robin(Robin_BC, :);
end
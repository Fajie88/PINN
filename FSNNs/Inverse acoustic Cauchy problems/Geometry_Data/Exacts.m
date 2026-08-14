function Exacts(cases)
global u_exact  f_source  du_dn_exact du_R_exact
global du_dx du_dy du_dz du_dxx du_dyy du_dzz
global omiga  alpha
if cases == 1   
k2 = 2;
beta = sqrt(k2^2 - 0.25);  % sqrt(3.75)
u_exact = @(x,y) cos(0.5*x + beta*y);
du_dx = @(x,y) -0.5 .* sin(0.5*x + beta*y);
du_dy = @(x,y) -beta .* sin(0.5*x + beta*y);
du_dn_exact = @(x,y,n1,n2) n1.*du_dx(x,y) + n2.*du_dy(x,y);
du_R_exact = @(x,y,n1,n2) n1.*du_dx(x, y) + n2.*du_dy(x, y) + omiga.*u_exact(x,y);
else cases == 2
    u_exact = @(x,y,z) 3*x.^2-3*y.^2+2*x.*y+x+6*y;
    f_source = @(x,y,z) 0;

    du_dx = @(x, y,z) 6*x+2*y+1;     
    du_dy = @(x, y,z) -6*y+2*x+6;
    du_dz = @(x, y,z) 0*z;

    du_dn_exact = @(x,y,z,n1,n2,n3) n1.*du1_dx(x, y,z) + n2.*du1_dy(x, y,z) + n3.*du1_dz(x, y,z);
    
    du_R_exact = @(x,y,z,n1,n2,n3) n1.*du1_dx(x, y,z) + n2.*du1_dy(x, y,z) + n3.*du1_dz(x, y,z)+ omiga*u1_exact(x,y,z);
    end
end
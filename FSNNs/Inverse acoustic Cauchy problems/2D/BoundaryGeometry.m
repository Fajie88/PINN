function [bp1, bp2, bp3, bp4,inp, bp, S, normal_vector]=BoundaryGeometry(iboundary, idomain)
%----------Boundary Geometry------------------
%       --iboundary=1: Rectangle
%       --iboundary=2: SimplyConnDomain
%             idomain=1: Circle
%             idomain=2: Gear wheel shape
%             idomain=3: Amoeba-like domain 变形虫
%             idomain=4: Peanut1  花生
%             idomain=5: Peanut2
%             idomain=6: Triangle  三角形物体
%             idomain=7: Heart   心
%             idomain=8: Gear-shaped domain   齿形域
%             idomain=9: Epitrochoid boundary shape  长短辐圆外旋轮
%             idomain=10: Petal   花瓣
%             idomain=11: Irregular  不规则的
%       --iboundary=3: MultiConnDomain
%             idomain=1: Peanut with two petals   二瓣花生
%       --iboundary=4: MultiConnDomain
%             idomain=1: Ring with nc holes (nc=10\15\20) 孔环
%
addpath('F:\Matlab_Subroutine\Boundary_Geometry_data');
%===================== geometry=====
irand=2;       % Style of collocation nodes
delta=0.0;     % Perturbation for Irregular I
x0=-6 ;  x1=-x0;  y0=x0;  y1=-x0;
nx=200;  ny=nx;
if iboundary==1
    x_scale=[0 2]; y_scale=[0 2];
   [xb,yb,xi,yi,nb,ni,pn1,pn2,lii] = Rectangle(x_scale,y_scale,irand,delta);   
%    xscal = [0 1]; yscal = [0 1]; nx = 52;    ny = nx;
%    [bp1,bp2,bp3,bp4,inp,N,bp,S] = geodata_rectangle(xscal, yscal, nx, ny); 
elseif iboundary==2
   nb=300;
   [xb,yb,xi,yi,ni,pn1,pn2,lii] = IrregularBoundary(nb,idomain,irand,delta,x0,x1,y0,y1,nx,ny);
elseif iboundary==3
   nb1=500; nb2=100; nb3=300; nb4=300;
   [xb,yb,xi,yi,nb,ni,pn1,pn2] = MultiConnBoundary(nb1,nb2,nb3,nb4,irand,delta,x0,x1,y0,y1,nx,ny);
elseif iboundary==4
   nc=15;
   nb1=100; nb2=50; nb3=10;
   [xb,yb,xi,yi,nb,ni,pn1,pn2] = RingMultiHoles(nc,nb1,nb2,nb3,irand,delta,x0,x1,y0,y1,nx,ny);
elseif iboundary==5
   nb1=400; nb2=400;  
   [xb,yb,xi,yi,nb,ni,pn1,pn2,lii]=MultiConnBoundary_1(nb1,nb2,irand,delta,x0,x1,y0,y1,nx,ny);
end
bp=[xb' yb']; inp=[xi' yi'];
normal_vector=[pn1 pn2];
S = [bp;inp];
bp1 = S(1:length(bp)/4,:);
bp2 = S(length(bp)/4+1:length(bp)/2,:);
bp3 = S(length(bp)/2+1:3*length(bp)/4,:);
bp4 = S(3*length(bp)/4+1:length(bp),:);

% figure(1)
% plot(xi,yi,'b.')
% hold on;
% plot(xb,yb,'r.')
end
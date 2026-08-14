function rbfcoeff=Regularization(svdmA,svdb,bmA)
% This program is use Regularization Tools to solve linear equations
%
% Ref: Per Christian Hansen.  Regularization tools: A Matlab package for
% analysis and solution of discrete ill-posed problems, Numerical
% Algorithms 6 (1994) 1-35.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   [svdmA]*[rbfcoeff]=[svdb], that is, A*x=b 
%   "bmA" denote the type of regularization techniques
%    bmA(1)=(0=gauss; 1=csvd; 2=one-order cgsvd; 3=two-order cgsvd)
%    bmA(2)=(0=gcv; 1=l-curve)
%    bmA(3)=(0=tikhonov; 1=tsvd; 2=dsvd)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    bmA=[0 0 0]; gauss
%    csvd + gcv + Tikh: bmA=[1,0,0]
%               + tsvd: bmA=[1,0,1]
%               + dsvd: bmA=[1,0,2]
%         + L_curve + Tikh: bmA=[1,1,0]
%                   + tsvd: bmA=[1,1,1]
%                   + dsvd: bmA=[1,1,2]
%      *****************************************************
%  (one-order) cgsvd + gcv + Tikh: bmA=[2,0,0]
%                          + tgsvd: bmA=[2,0,1]
%                          + dsvd: bmA=[2,0,2]
%                    + L_curve + Tikh: bmA=[2,1,0]
%                              + tgsvd: bmA=[2,1,1]
%                              + dsvd: bmA=[2,1,2]
%      *****************************************************
%  (two-order) cgsvd + gcv + Tikh: bmA=[3,0,0]
%                          + tgsvd: bmA=[3,0,1]
%                          + dsvd: bmA=[3,0,2]
%                    + L_curve + Tikh: bmA=[3,1,0]
%                              + tgsvd: bmA=[3,1,1]
%                              + dsvd: bmA=[3,1,2]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if bmA(1)==0   % Gauss
    if bmA(2)==0
        if bmA(3)==0
            rbfcoeff=svdmA\svdb;
        end
    end
elseif bmA(1)==1   % csvd for singular value decomposition (SVD)
    [U,s,V]=csvd(svdmA);
    if bmA(2)==0   % gcv
        if bmA(3)==0  % different reg-methods
            lambda_l=gcv(U,s,svdb,'Tikh');
            rbfcoeff=tikhonov(U,s,V,svdb,lambda_l);
        elseif bmA(3)==1  
            [reg_min,alpha,reg_param]=gcv(U,s,svdb,'tsvd');
            rbfcoeff= tsvd(U,s,V,svdb,reg_min);
        elseif bmA(3)==2  
            [reg_min,alpha,reg_param]=gcv(U,s,svdb,'dsvd');
            rbfcoeff= dsvd(U,s,V,svdb,reg_min);
        end
    elseif bmA(2)==1  % L_curve
        if bmA(3)==0  % different reg-methods
            lambda_l=l_curve(U,s,svdb,'Tikh');
            rbfcoeff=tikhonov(U,s,V,svdb,lambda_l);
        elseif bmA(3)==1
            [reg_min,alpha,reg_param]=l_curve(U,s,svdb,'tsvd');
            rbfcoeff= tsvd(U,s,V,svdb,reg_min);
        elseif bmA(3)==2
            [reg_min,alpha,reg_param]=l_curve(U,s,svdb,'dsvd');
            rbfcoeff= dsvd(U,s,V,svdb,reg_min);
        end
    end
elseif bmA(1)==2  % one-order cgsvd for generalized SVD
    L_gsvd=diag(-ones(1,size(svdmA,2)-1),0)+diag(ones(1,size(svdmA,2)-2),1);  % obtain the one-order matrix L
    L_gsvd(1:size(svdmA,2)-2,size(svdmA,2))=0;
    L_gsvd(size(svdmA,2)-1,size(svdmA,2))=1;
    [U,sm,X,V]=cgsvd(svdmA,L_gsvd);
    if bmA(2)==0  % gcv
        if bmA(3)==0  % different reg-methods
            lambda_l=gcv(U,sm,svdb,'Tikh');
            rbfcoeff=tikhonov(U,sm,X,svdb,lambda_l);
        elseif bmA(3)==1
            [reg_min,alpha,reg_param]=gcv(U,sm,svdb,'tsvd');
            rbfcoeff= tgsvd(U,sm,X,svdb,reg_min);
        elseif bmA(3)==2
            [reg_min,alpha,reg_param]=gcv(U,sm,svdb,'dsvd');
            rbfcoeff= dsvd(U,sm,X,svdb,reg_min);
        end
    elseif bmA(2)==1  % L_curve
        if bmA(3)==0  % different reg-methods
            lambda_l=l_curve(U,sm,svdb,'Tikh');
            rbfcoeff=tikhonov(U,sm,X,svdb,lambda_l);
        elseif bmA(3)==1
            [reg_min,alpha,reg_param]=l_curve(U,sm,svdb,'tsvd');
            rbfcoeff= tgsvd(U,sm,X,svdb,reg_min);
        elseif bmA(3)==2
            [reg_min,alpha,reg_param]=l_curve(U,sm,svdb,'dsvd');
            rbfcoeff= dsvd(U,sm,X,svdb,reg_min);
        end
    end
elseif bmA(1)==3  % two-order cgsvd for generalized SVD
    L_gsvd=zeros(size(svdmA,2)-2,size(svdmA,2));  % obtain the two-order matrix L
    L_gsvd(1:size(svdmA,2)-2,2:size(svdmA,2)-1)=diag(-2.*ones(1,size(svdmA,2)-2),0)+diag(ones(1,size(svdmA,2)-3),1)+diag(ones(1,size(svdmA,2)-3),-1);
    L_gsvd(1,1)=1;
    L_gsvd(size(svdmA,2)-2,size(svdmA,2))=1;
    [U,sm,X,V]=cgsvd(svdmA,L_gsvd);
    if bmA(2)==0  % gcv
        if bmA(3)==0   % different reg-methods
            lambda_l=gcv(U,sm,svdb,'Tikh');
            rbfcoeff=tikhonov(U,sm,X,svdb,lambda_l);
        elseif bmA(3)==1
            [reg_min,alpha,reg_param]=gcv(U,sm,svdb,'tsvd');
            rbfcoeff= tgsvd(U,sm,X,svdb,reg_min);
        elseif bmA(3)==2
            [reg_min,alpha,reg_param]=gcv(U,sm,svdb,'dsvd');
            rbfcoeff= dsvd(U,sm,X,svdb,reg_min);
        end
    elseif bmA(2)==1  % L_curve
        if bmA(3)==0  % different reg-methods
            lambda_l=l_curve(U,sm,svdb,'Tikh');
            rbfcoeff=tikhonov(U,sm,X,svdb,lambda_l);
        elseif bmA(3)==1
            [reg_min,alpha,reg_param]=l_curve(U,sm,svdb,'tsvd');
            rbfcoeff= tgsvd(U,sm,X,svdb,reg_min);
        elseif bmA(3)==2
            [reg_min,alpha,reg_param]=l_curve(U,sm,svdb,'dsvd');
            rbfcoeff= dsvd(U,sm,X,svdb,reg_min);
        end
    end
end
end


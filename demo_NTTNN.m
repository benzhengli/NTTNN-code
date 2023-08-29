% clc;
clear;
close all;
%% load data
load('Balloons.mat');
X  = Omsi_1;
Y_tensorT = X;
%% sampling rate
sample_ratio = 0.05;
fprintf('\n');
fprintf('================Results=p=%f======================\n',sample_ratio);
[n1,n2,n3]  = size(Y_tensorT);
Ndim        = ndims(Y_tensorT);
Nway        = size(Y_tensorT);
rand('state',0);
Omega       = find(rand(prod(Nway),1)<sample_ratio);
Ind         = zeros(size(Y_tensorT));
Ind(Omega)  = 1;
Y_tensor0   = zeros(Nway);
Y_tensor0(Omega) = Y_tensorT(Omega);

%% Interpolation
A = Y_tensor0;
B = padarray(A,[20,20,20],'symmetric','both');
C = padarray(Ind,[20,20,20],'symmetric','both');
%a0 = interpolate2(B,C);
a1 = interpolate(shiftdim(B,1),shiftdim(C,1));
a1(a1<0) = 0;
a1(a1>1) = 1;
a1 = a1(21:end-20,21:end-20,21:end-20);
a1 = shiftdim(a1,2);
a1(Omega) = Y_tensorT(Omega);

a2 = interpolate(shiftdim(B,2),shiftdim(C,2));
a2(a2<0) = 0;
a2(a2>1) = 1;
a2 = a2(21:end-20,21:end-20,21:end-20);
a2 = shiftdim(a2,1);
a2(Omega) = Y_tensorT(Omega);
a3(Omega) = Y_tensorT(Omega);
a = 0.5*a1+0.5*a2;
X0 = a;
%% Parameters
alpha = 100;
beta  = 10;
d     = 8;
rho   = 0.001;
%% Initialize
X_m = Unfold(X0,size(X0),3);
Ind_m = Unfold(Ind,size(Ind),1);
Ind   = sum(Ind_m,1);
[~,D_omega] = sort(Ind(:),'descend');
[DI,~,~] = svds(X_m,d);
opts = [];
opts.D0  = DI;
opts.d   = d;
opts.rho = rho;
opts.tol = 10^-4;
opts.max_iter = 500;
opts.inner = 10;
opts.alpha = alpha;
opts.beta = beta;

[Re_tensor,OUT,iter] = NTTNN_PAM(Y_tensor0,Omega,opts,Y_tensorT,X0);
[psnr, ssim ] = quality_my(Re_tensor * 255, Y_tensorT * 255)

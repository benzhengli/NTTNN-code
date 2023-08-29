function [X,OUT,iter] = NTTNN_PAM(M,omega,opts,M_true,X0)
%Solve the nonlinear transform-based tensor nuclear norm
%for tensor completion by PAM algorithm
%
% min_{X,Y,Z,T} \sum_{r=1}^d||tanh(Z)||_*,
% s.t. P_Omega(X) = P_Omega(O), 
%      X = Z ¡Á_3 D
%      DD^T = I
% ---------------------------------------------
% Written by Ben-Zheng Li (lbz1604179601@126.com)
%
% References: 
%Ben-Zheng Li, Xi-Le Zhao, Teng-Yu Ji, Xiong-Jun Zhang, and Ting-Zhu Huang, 
%Nonlinear Transform Induced Tensor Nuclear Norm for Tensor Completion,
%Journal of Scientific Computing, 2022.

if isfield(opts, 'tol');        tol         =  opts.tol;         end
if isfield(opts, 'max_iter');   max_iter    =  opts.max_iter;	 end
if isfield(opts, 'alpha');      alpha       =  opts.alpha;       end
if isfield(opts, 'beta');       beta        =  opts.beta;        end
if isfield(opts, 'inner');      inner       =  opts.inner;       end
if isfield(opts, 'rho');         rho          =  opts.rho;          end
if isfield(opts, 'D0');         D0          =  opts.D0;          end
if isfield(opts, 'd');           d          =  opts.d;          end

X   = X0;
Dim = size(X);
D   = D0;
Z   = Fold(D' * Unfold(X,size(X),3),[Dim(1),Dim(2),d],3);
Dim = size(X);
dim = size(Z);
Y = tanh(Z);
OUT  = [];
PSNR = [];
SSIM = [];
for iter = 1 : max_iter
    Xk = X;
    Yk = Y;
    Zk = Z;
    Dk = D;
%% update X
    Z_mat = Unfold(Z,size(Z),3);
    Xk_mat = Unfold(Xk,size(X),3);
    if iter > 10
    X_mat     =  (alpha * D * Z_mat + rho * Xk_mat)/(alpha + rho);
    X     = Fold(X_mat,size(X),3);
    X(omega) = M(omega);
    end
%% update Y
      Y     = my_SVD((beta * tanh(Z) + rho * Yk) ./ (beta + rho) , 1/(beta + rho));
%% update Z
    temp1  = Unfold(Y,size(Y),3);
    DX     = Fold(D' * Unfold(X,size(X),3),[Dim(1),Dim(2),d],3);
    temp2  = (alpha * DX + rho * Zk) / (alpha + rho);
    temp2  = Unfold(temp2,size(temp2),3);
    Z_mat  = Newton(temp1,temp2,Z_mat,inner,alpha + rho,beta);
    Z      = Fold(Z_mat,size(Z),3);
%% update T
    Z_mat   =  reshape(Z,dim(1)*dim(2),dim(3))';
    X_mat   =  reshape(X,Dim(1)*Dim(2),Dim(3))';
    temp =  alpha * X_mat * Z_mat' + rho * Dk;
    [UI,~,VI] = svd( temp , 'econ');
    D   = UI * VI';         
%% check convergence
    chgX    = norm(Xk(:)-X(:))/norm(Xk(:));OUT.chgX(iter) = chgX;
    
    [psnr, ssim] = quality_my(M_true * 255,X * 255);
    PSNR = [PSNR,psnr];
    SSIM = [SSIM,ssim];    
    if  mod(iter, 10) == 0
        fprintf('NTTNN:  iter = %d   PSNR=%f SSIM=%f resX=%f \n', iter, psnr,ssim,chgX);
    end
    if iter > 10
        if chgX < tol
            break;
        end
    end
end
    OUT.X = X; OUT.Y = Y; OUT.Z = Z;
end
%% nested functions
function Z  = Newton(g,a,Z,inner,alpha, beta)
        i=0;
        relchg=1;
        tol=10^(-4);  
        while  i < inner  &&  relchg > tol 
                Zp=Z;
                Numer = beta .* (1 - tanh(Z).^2) .* (tanh(Z) - g) + alpha .* (Z - a);
                Denom = -2 .* beta .* tanh(Z) .* ( 1 - tanh(Z).^2) .* (tanh(Z) - g) + beta .* ( 1 - tanh(Z).^2).^2 + alpha;
                Z  = Z - Numer./Denom;
                relchg = norm(Z - Zp,'fro')/norm(Z,'fro');
                i=i+1;
        end
end
  function Y = my_SVD(Y,rho)
    [n1, n2, n3] = size(Y);
    n12 = min(n1, n2);
    Uf = zeros(n1, n12, n3);
    Vf = zeros(n2, n12, n3);
    Sf = zeros(n12,n12, n3);
    trank = 0;
    for i = 1 : n3
        [Uf(:,:,i), Sf(:,:,i), Vf(:,:,i)] = svd(Y(:,:,i), 'econ');
        s = diag(Sf(:, :, i));
        s = max(s - rho, 0);
        Sf(:, :, i) = diag(s);
        temp = length(find(s>0));
        trank = max(temp, trank);
        Y(:,:,i) = Uf(:,:,i)*Sf(:,:,i)*Vf(:,:,i)';
    end
    end
   function output = tanh(x)
      output =(exp(x) - exp(-x))./(exp(x) + exp(-x));
   end

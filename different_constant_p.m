clear all;
clc;
% %  close all;
%% real data set

L=0*2*10^(-10);
N=10000;
a=1;

% load('W.mat')
% load('H.mat')
% load('z0.mat')
%%
K=10;
%%
SW1=1;
SW2=2;
ImF21=zeros(N+1,K);
ImF22=zeros(N+1,K);
ImF23=zeros(N+1,K);
ImF24=zeros(N+1,K);
ExF_NAG=zeros(N+1,K);
ExF_GD=zeros(N+1,K);
%%
ImF212=zeros(N+1,1);
ImF222=zeros(N+1,1);
ImF232=zeros(N+1,1);
ImF242=zeros(N+1,1);
ExF_NAG2=zeros(N+1,1);
ExF_GD2=zeros(N+1,1);
%% june 12
h1=0.3;
h2=0.3;
h3=0.3;
h4=0.3;
h_NAG=0.1;
h_GD=0.1;

p1=2;
p2=4;
p3=6;
p4=8;
%%
for k=1:K
    
X=rand(10,10);
y=rand(10,1);
for i=1:10
    if y(i)>=0.5
        y(i)=1;
    else
        y(i)= -1;
    end
end
% % % %%
W=X;
H=y; 
d=size(W,2);
% %% initial conditions
 z11=rand(d,1); z22=zeros(d,1); z33=1;
z0=[z11;z22;z33];

%% 
% h1=0.02;
% h2=0.1;
% h3=0.2;
% h_NAG=0.02;
% h_GD=0.02;
%% june 6
% % % h1=0.3;
% % % h2=0.3;
% % % h3=0.3;
% % % h4=0.3;
% % % h_NAG=0.2;
% % % h_GD=0.2;


ImX21=gauss_bfgs_crj_diff(a, h1, N, z0, d, W, H, L, p1);
ImX22=gauss_bfgs_crj_diff(a, h2, N, z0, d, W, H, L, p2);
ImX23=gauss_bfgs_crj_diff(a, h3, N, z0, d, W, H, L, p3);
ImX24=gauss_bfgs_crj_diff(a, h4, N, z0, d, W, H, L, p4);

X_NAG=nesterov(h_NAG, N, z0, d, W, H, L, SW2);
X_GD=nesterov(h_GD, N, z0, d, W, H, L, SW1);
for j=1:N+1
%% loss 
ImF21(j, k) = logit_loss_fu(ImX21(:, j), W, H, L);
ImF22(j, k) = logit_loss_fu(ImX22(:, j), W, H, L);
ImF23(j, k) = logit_loss_fu(ImX23(:, j), W, H, L);
ImF24(j, k) = logit_loss_fu(ImX24(:, j), W, H, L);
ExF_NAG(j, k) = logit_loss_fu(X_NAG(:, j), W, H, L);
ExF_GD(j, k) = logit_loss_fu(X_GD(:, j), W, H, L);
end

ImF212 = ImF212 + ImF21(:, k) ;
ImF222 = ImF222 + ImF22(:, k);
ImF232 = ImF232 + ImF23(:, k) ;
ImF242 = ImF242 + ImF24(:, k) ;
ExF_NAG2 = ExF_NAG2 + ExF_NAG(:, k) ;
ExF_GD2 = ExF_GD2 + ExF_GD(:, k) ;
end
figure
% % % semilogy(1:N+1, ImF21,'r-','LineWidth', 1.5);hold on
% % % semilogy(1:N+1, ImF22,'b-','LineWidth', 1.5);hold on
% % % semilogy(1:N+1, ImF23,'r--','LineWidth', 1.5);hold on
% % % semilogy(1:N+1, ImF24,'b--','LineWidth', 1.5);hold on
% % % 
% % % semilogy(1:N+1, ExF_NAG,'k:','LineWidth', 1.5);hold on;
% % % semilogy(1:N+1, ExF_GD,'k-.','LineWidth', 1.5);

loglog(1:N+1, ImF212/K,'r-','LineWidth', 1.5);hold on
loglog(1:N+1, ImF222/K,'b-','LineWidth', 1.5);hold on
loglog(1:N+1, ImF232/K,'r--','LineWidth', 1.5);hold on
loglog(1:N+1, ImF242/K,'b--','LineWidth', 1.5);hold on

loglog(1:N+1, ExF_NAG2/K,'k:','LineWidth', 2.5);hold on;
loglog(1:N+1, ExF_GD2/K,'k-.','LineWidth', 1.5);
xlabel('Iterations', 'FontSize',16);
ylabel('Objective','FontSize',16);
%title('Minimizing regularized quadratic function on  set','FontSize',16);
legend({'Im p=2','Im p=4','Im p=6','Im p=8','NAG','GD'},'FontSize',16); %'Ex-2s2o-Kutta'
set(gca,'FontSize',16);
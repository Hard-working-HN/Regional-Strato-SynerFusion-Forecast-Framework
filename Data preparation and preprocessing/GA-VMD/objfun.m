function SeMin=objfun(para,x)

alpha=round(para(1));
K=round(para(2));

tau = 0;            
DC = 0;            
init = 1;          
tol = 1e-5;

[u, u_hat, omega] = VMD(x, alpha, tau, K, DC, init, tol);

[m,n]=size(u);
mm=2;
for ii=1:m
feature(ii)=SampEn(u(ii,:), mm, 0.2*std(u(ii,:)));%u�������أ�0.2*std(imf1(1,:))��ʾ���������r��ֵ��
end
SeMin=min(feature);
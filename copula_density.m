close all
%['ACM2278', 'CMP2946', 'PLJ1771','CDE1846','MBG3183']
user = 'ACM2278';
load(['C:\Users\xu wenjie\AnacondaProjects\CMU_anomaly\',user,'_feature'])
%for i = 1:size(x,2)
x_addnoise = zeros(size(x));
u = zeros(size(x));
for i = 1:size(x,2)
    x_addnoise(:,i) = x(:,i)+ max(sqrt(x(:,i).^2/20),sqrt(1/20)).*randn(size(x(:,1)));
    u(:,i) = ksdensity(x_addnoise(:,i), x_addnoise(:,i),'function','cdf');
    %figure
    %hist(u(:,i))
end
Rho = copulafit('Gaussian',u);
fx = zeros(size(x));
v = zeros(size(x));
for i = 1:size(x,2)
    fx(:,i) = log(ksdensity(x_addnoise(:,i),x(:,i),'function','pdf'));
    v(:,i) = ksdensity(x_addnoise(:,i),x(:,i),'function','cdf');
end
log_jointpdf = zeros(1,size(x,1));
for j = 1:size(x,1)
    log_jointpdf(j) = sum(fx(j,:))+log(copulapdf('Gaussian',v(j,:),Rho));
end
log_indpdf = sum(fx,2);
figure
hold on
plot(log_jointpdf,'go')
plot(find(label==1),log_jointpdf(label==1),'rx')
title('joint pdf')

figure
plot(log_indpdf,'go')
hold on
plot(find(label==1),log_indpdf(label==1),'rx')
title('ind pdf')



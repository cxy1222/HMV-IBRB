function [S]=ER_Rule(r,w,p)%r:可靠度 w:权重 p:置信分布矩阵
%The analystical ER rule without local ignorance
%input:r: reliability 1xM;w;weighxtM 1;P:belief degree MxN 
%output: P_fin: final belief degree 1xN
%没有局部忽略的分析ER规则
%输入：r：可靠度 1xM；w：权重 TM 1；P：置信度 MxN
%输出：P_fin：最终置信度1xN
% data=load('rocket_data.txt');
% T = length(data);
[m,n]=size(p);%size()函数用来获取矩阵的行数和列数。m行n列。m=2 n=4
P_fin=zeros(1,n);%生成一个1*n的零矩阵
mu1=0;

% sumw=w(1)+w(2);
% w(1)=w(1)/sumw;
% w(2)=1-w(1);
%     
for i=1:n%n列    i=1,2,3,4
    temp=1;
    for j=1:m%m行   m=1,2
        temp1=0;
        for h=1:n%n列   n=1,2,3,4
            temp1=p(j,h)+temp1;%算的是p矩阵每行的和
        end
        temp=((1-r(j))+(w(j)*p(j,i))+(w(j)*(1-temp1)))*temp;
    end
    mu1=temp+mu1;
end
mu2=1;
for i1=1:m%m行   m=1,2
    temp2=0;
    for j1=1:n%n列   n=1,2,3,4
        temp2=p(i1,j1)+temp2;%算的是p矩阵每行的和
    end
    mu2=((1-r(i1))+w(i1)*(1-temp2))*mu2;
end
mu2=(n-1)*mu2;
mu=mu1-mu2;
mu=1/mu;

for i2=1:n% n=4
    sec1=1;
    sec2=1;
    sec3=1;
    for j2=1:m%m=2
        temp3=0;
        for h2=1:n
            temp3=p(j2,h2)+temp3;
        end
        sec1=((1-r(j2))+w(j2)*p(j2,i2)+w(j2)*(1-temp3))*sec1;
        sec2=((1-r(j2))+w(j2)*(1-temp3))*sec2;
        sec3=(1-r(j2))*sec3;
    end
    numerator=mu*(sec1-sec2);
    denominator=1-(mu*sec3);
    P_fin(1,i2)=numerator/denominator;
end
    MPtheta=mu*sec3;
    r0min=(1-MPtheta*2)/(1-MPtheta);
    r0max=(1-MPtheta*(1+max(w)))/(1-MPtheta);
    r0=[r0min,r0max];
   S=[r0,P_fin];
%     Y=S(3)*1+S(4)*2+S(5)*3;
 
end




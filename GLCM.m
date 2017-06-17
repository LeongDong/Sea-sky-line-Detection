origin = imread('F:\1-sea-sky-pic\197.jpg');
% gray = rgb2gray(origin);
gray = double(origin(:,:,3));
[M,N] = size(gray);
gap = 20;  %划分为20个灰度共生矩阵
step = 3;   %步长为3
diag_sum = 0;
v_length = floor(M/gap);    %每个区域的高
count = zeros(1,gap);
E_count = zeros(1,gap);%纹理量化值
C_count = zeros(1,gap);
Sum_E_count = 0;%纹理值的和，用于归一化处理
Max_E_count = 0;%纹理的最大值，用于归一化处理
x = zeros(1,gap);%直线拟合横坐标
y = zeros(1,gap);%直线拟合纵坐标
Sum_x_y = 0;
Sum_x_2 = 0;
k = 0; % 直线拟合的斜率
l = 0; % 直线拟合的截距

y_sig = zeros(1,gap);%sigmoid函数中的y坐标

for i = 1 : M
    for j = 1 : N
        gray(i,j) = floor(gray(i,j)/16);   %将灰度共生矩阵的大小限制为16*16
    end
end

for k = 0 : gap-1
    v_start = k * v_length + 1;
    v_end = (k+1) * v_length;
    p = zeros(16,16,4);
    for m = 1 : 16
        for n = 1 : 16
            for i = v_start : v_end
                for j = 1: N/20
                    if j <= N/20 - step&gray(i,j)==m-1&gray(i,j + step)==n-1   %0度角
                        p(m,n,1) = p(m,n,1) + 1;
                        p(n,m,1) = p(m,n,1);
                    end
                    if i>= v_start + step&j < N/20 - step&gray(i,j)==m-1&gray(i - step,j + step)==n-1  %45度角
                        p(m,n,2) =  p(m,n,2) + 1;
                        p(n,m,2) = p(m,n,2);
                    end
                    if i>= v_start + step&gray(i,j)==m-1&gray(i - step,j)==n-1  %90度角
                        p(m,n,3) = p(m,n,3)+1;
                        p(n,m,3) = p(m,n,3);
                    end
                    if i>= v_start + step&j > step&gray(i,j)==m-1&gray(i - step,j - step)==n-1   %135度角，步长step为1，这里需要调整步长，以1为增量
                        p(m,n,4) = p(m,n,4)+1;
                        p(n,m,4) = p(m,n,4);
                    end
                end

            end
        end
   end
    for a=1 : 4
        p(:,:,a) = p(:,:,a)/sum(sum(p(:,:,a)));
    end
    H = zeros(1,4);
    I = H;
    Ux = H;
    Uy = H;
    deltaX = H;
    deltaY = H;
    C = H;
     for b=1 : 4
         E(b) = sum(sum(p(:,:,b).^2));
         for i = 1 : 16
             for j = 1 : 16
                 I(b) = (i - j) * (i - j) * p(i,j,b) + I(b);
                 Ux(b) = i*p(i,j,b)+Ux(b); %相关性中μx
                 Uy(b) = j*p(i,j,b)+Uy(b); %相关性中μy

             end
         end
     end
   for n = 1:4
    for i = 1:16
        for j = 1:16
            deltaX(n) = (i-Ux(n))^2*p(i,j,n)+deltaX(n); %相关性中σx
            deltaY(n) = (j-Uy(n))^2*p(i,j,n)+deltaY(n); %相关性中σy
            C(n) = i*j*p(i,j,n)+C(n);            
        end
    end
    C(n) = (C(n)-Ux(n)*Uy(n))/deltaX(n)/deltaY(n); %相关性  
   end
 % E_count(k+1) = E(1);
   % E_count(k+1) = sum(C);
    E_count(k+1) = sum(I);  %这里将四个角度的值全部加起来
%    sprintf('0,45,90,135方向上的对比度依次为： %f, %f, %f, %f',I(1),I(2),I(3),I(4));
end


Sum_E_count = sum(E_count);
Max_E_count = max(E_count);

for i = 1:gap
    E_count(i) = E_count(i) / Max_E_count + 0.001;
end

x(1) = floor(v_length / 2);
for i = 2:gap
    x(i) = x(i - 1) + v_length;
end

for i = 1:gap
    %if(E_count(i) == 0)
     %   E_count(i) = E_count(i) + 0.001;
    %end
    y(i) = log(1/E_count(i) - 1);
end
avg_x = sum(x) / gap;
avg_y = sum(y) / gap;

for i = 1:gap
    Sum_x_y = Sum_x_y + x(i)*y(i);
    Sum_x_2 = Sum_x_2 + x(i)*x(i);
end

k = (Sum_x_y - gap * avg_x * avg_y) / (Sum_x_2 - gap * avg_x * avg_x);
l = avg_y - k * avg_x;

c = -1 / k;
b = -l / k;

for i = 1:gap
    y_sig(i) = 1 / (1+exp(-(x(i) - b)/c));
end

plot(x,y_sig);
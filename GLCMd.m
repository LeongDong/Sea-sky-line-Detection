origin = imread('F:\1-sea-sky-pic\153_error.jpg');
% gray = rgb2gray(origin);
gray = double(origin(:,:,1));
[M,N] = size(gray);
gap = 15;  %划分为gap个灰度共生矩阵
step = 3;   %步长为3
diag_sum = 0;
v_length = floor(M / gap);    %每个区域的高
v_width = floor(N / gap);    %每个区域的宽
count = zeros(1,gap);
E_count = zeros(1,gap);%纹理量化值
C_count = zeros(1,gap);
Sum_E_count = 0;%纹理值的和，用于归一化处理
Max_E_count = 0;%纹理的最大值，用于归一化处理
x = zeros(1,gap);%直线拟合横坐标
y = zeros(1,gap);%直线拟合纵坐标
avg_gray = zeros(1,gap); %每个区域灰度平均值，倒数为权值
pro_y = zeros(1,gap); %概率拟合点y(probility)
pro_x = zeros(1,gap);   %概率拟合点x(x-axis)
Sum_x_y = 0;
Sum_x_2 = 0;
k = 0; % 直线拟合的斜率
l = 0; % 直线拟合的截距
Pro_count = 1;  %count the number of fitting dots
Min = 9999;
Pro_avg = 0;
Min_E_count = 999;%纹理的最小值，用于纹理值替换

y_sig = zeros(1,M);%sigmoid函数中的y坐标
x_sig = zeros(1,M);  %sigmoid函数中的x坐标

v_size = v_length * v_width; %区域大小

for i = 1 : M
    for j = 1 : N
        gray(i,j) = floor(gray(i,j)/16);   %将灰度共生矩阵的大小限制为16*16
    end
end

for k = 0 : gap-1
    v_start = k * v_length + 1;
    v_end = (k+1) * v_length;
    p = zeros(16,16,4);
    for i = v_start : v_end
        for j = 1: N / gap
            if j <= N / gap - step   %0度角
                p(gray(i,j) + 1,gray(i,j + step) + 1,1) = p(gray(i,j) + 1,gray(i,j + step) + 1,1) + 1;
                p(gray(i,j + step) + 1,gray(i,j) + 1,1) = p(gray(i,j) + 1,gray(i,j + step) + 1,1);
            end
            if i>= v_start + step&j < N/gap - step  %45度角
                p(gray(i,j) + 1,gray(i - step,j + step) + 1,2) =  p(gray(i,j) + 1,gray(i - step,j + step) + 1,2) + 1;
                p(gray(i - step,j + step) + 1,gray(i,j) + 1,2) = p(gray(i,j) + 1,gray(i - step,j + step) + 1,2);
            end
            if i>= v_start + step  %90度角
                p(gray(i,j) + 1,gray(i - step,j) + 1,3) =  p(gray(i,j) + 1,gray(i - step,j) + 1,3) + 1;
                p(gray(i - step,j) + 1,gray(i,j) + 1,3) =  p(gray(i,j) + 1,gray(i - step,j) + 1,3);
            end
            if i>= v_start + step&j > step  %135度角，步长step为1，这里需要调整步长，以1为增量
                p(gray(i,j) + 1,gray(i - step,j - step) + 1,4) =  p(gray(i,j) + 1,gray(i - step,j - step) + 1,4) + 1;
                p(gray(i - step,j - step) + 1,gray(i,j) + 1,4) =  p(gray(i,j) + 1,gray(i - step,j - step) + 1,4);
            end

            avg_gray(k + 1) = avg_gray(k + 1) + gray(i,j);

        end
    end
    
    avg_gray(k + 1) = avg_gray(k + 1) / v_size;
    avg_gray(k + 1) = avg_gray(k + 1) * avg_gray(k + 1);% * avg_gray(k + 1);
   
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

x(1) = floor(v_length / 2);

for i = 2:gap
    x(i) = x(i - 1) + v_length;
end

for i = 1:gap
    E_count(i) = E_count(i) / avg_gray(i);
end

Sum_E_count = sum(E_count);
Max_E_count = max(E_count);
for i = 1:gap
    if(Min_E_count > E_count(i) && E_count(i) ~= 0) %对于区域纹理值为0 的区域，用非零的最小值进行初始化
        Min_E_count = E_count(i);
    end
end

for i = 1:gap  %对数据进行归一化处理，同时找出纹理最小特征值
    E_count(i) = E_count(i) / Max_E_count ;
    if(E_count(i) == 0)
        E_count(i) = Min_E_count / Max_E_count ;  %使用最大值进行归一化，会导致最大值的归一化结果为1,1是错误的所以要适当减少
    end
    if (E_count(i) >= 1)
        E_count(i) = 0.99998;
    end
    if (Min >= E_count(i))
        Min = E_count(i);
    end
end

for i = 1:gap   %找出与最小值同数量级的其他纹理值，作为天空候选拟合点
    if(max(E_count) / E_count(i) > 2.5 && E_count(i) / Min < 4)
        pro_x(Pro_count) = x(i);
        pro_y(Pro_count) = 0.00001
        Pro_count = Pro_count + 1;
    end
end

Pro_avg = mean(E_count);    %找出纹理值大于平均值的其他纹理值，作为海面候选拟合点
for i = 1:gap
    if(E_count(i) > Pro_avg && x(i) > pro_x(1))
        pro_x(Pro_count) = x(i);
        pro_y(Pro_count) = 0.99998;
        Pro_count = Pro_count + 1;
    end
end
Pro_count = Pro_count - 1;
for i = 1:Pro_count
    %if(E_count(i) == 0)
     %   E_count(i) = E_count(i) + 0.001;
    %end
    pro_y(i) = log(1 / pro_y(i) - 1);
end
avg_x = sum(pro_x) / Pro_count;
avg_y = sum(pro_y) / Pro_count;

for i = 1:Pro_count
    Sum_x_y = Sum_x_y + pro_x(i) * pro_y(i);
    Sum_x_2 = Sum_x_2 + pro_x(i) * pro_x(i);
end

k = (Sum_x_y - Pro_count * avg_x * avg_y) / (Sum_x_2 - Pro_count * avg_x * avg_x);
l = avg_y - k * avg_x;

c = -1 / k;
b = -l / k;

for i = 1:M
    x_sig(i) = i;
    y_sig(i) = 1 / (1+exp(-(i - b)/c));
end

figure(1);
plot(x_sig,y_sig);



Cut_y_start = b-c*log(23 + sqrt(528)) - 10;
Cut_y_end = b-c*log(23 - sqrt(528)) + 10;

if (Cut_y_start <= 0)
    Cut_y_start = 1;
end 
if (Cut_y_end >= M)
    Cut_y_end = M;
end

Show_image = origin(Cut_y_start : Cut_y_end,1 : N,:);
figure(2);
imshow(Show_image);
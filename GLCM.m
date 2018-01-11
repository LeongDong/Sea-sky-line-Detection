tic;
ti = clock;
origin = imread('F:\1-sea-sky-pic\202.jpg');
% gray = rgb2gray(origin);
gray = double(origin(:,:,3));
[M,N] = size(gray);
gap = 15;  %划分为gap个灰度共生矩阵
step = 3;   %步长为3
diag_sum = 0;
v_length = floor(M / gap);    %每个区域的高
v_width = floor(N / gap);    %每个区域的宽
count = zeros(1,gap);
E_count = zeros(1,gap);%纹理量化值
C_count = zeros(1,gap);
E_count_color = zeros(1,26);
E_count_comp = zeros(1,3);
E_count_new = zeros(1,gap);
E_count_gradmax = -1;   % the maximum value of gradient
E_count_gradmaxloc = 1;   %the location of maximum gradient
E_location = 1;
flag_end = 0;
tmp = 0;
x = zeros(1,gap);%直线拟合横坐标
y = zeros(1,gap);%直线拟合纵坐标
avg_gray = zeros(1,gap); %每个区域灰度平均值，倒数为权值
flag_E_count = zeros(1,gap);
pro_y = zeros(1,gap); %概率拟合点y(probility)
pro_x = zeros(1,gap);   %概率拟合点x(x-axis)
Sum_x = 0;
Sum_y = 0;
Sum_x_y = 0;
Sum_x_2 = 0;
Pro_count = 1;  %count the number of fitting dots
Min = 9999;
Pro_avg = 0;
Min_E_count = 999;%纹理的最小值，用于纹理值替换
threshold  = 1;   %归一化纹理值大于0.5的最小值
Max_color = -1;   %maxium value for segmentation
locamin_color = 1;
thres_color = 1;   % threshold for segmentation
loca_color = 1;   %location for segmentation
Max_grad = -1;
loca_grad = 1;

loca_avg_gray = 0;  %定位区域最大平均灰度值 
loca_E_count = 0;  %定位区域最大纹理值
E_count_sky = 0;    %天空拟合点的数量
E_count_sea = 0;    %海面拟合点的数量
start_point = 1;    %开始拟合的位置


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
        for j = 1: N
            if j <= N - step   %0度角
                p(gray(i,j) + 1,gray(i,j + step) + 1,1) = p(gray(i,j) + 1,gray(i,j + step) + 1,1) + 1;
                p(gray(i,j + step) + 1,gray(i,j) + 1,1) = p(gray(i,j) + 1,gray(i,j + step) + 1,1);
            end
            if i>= v_start + step && j < N - step  %45度角
                p(gray(i,j) + 1,gray(i - step,j + step) + 1,2) =  p(gray(i,j) + 1,gray(i - step,j + step) + 1,2) + 1;
                p(gray(i - step,j + step) + 1,gray(i,j) + 1,2) = p(gray(i,j) + 1,gray(i - step,j + step) + 1,2);
            end
            if i>= v_start + step  %90度角
                p(gray(i,j) + 1,gray(i - step,j) + 1,3) =  p(gray(i,j) + 1,gray(i - step,j) + 1,3) + 1;
                p(gray(i - step,j) + 1,gray(i,j) + 1,3) =  p(gray(i,j) + 1,gray(i - step,j) + 1,3);
            end
            if i>= v_start + step && j > step  %135度角，步长step为1，这里需要调整步长，以1为增量
                p(gray(i,j) + 1,gray(i - step,j - step) + 1,4) =  p(gray(i,j) + 1,gray(i - step,j - step) + 1,4) + 1;
                p(gray(i - step,j - step) + 1,gray(i,j) + 1,4) =  p(gray(i,j) + 1,gray(i - step,j - step) + 1,4);
            end
            avg_gray(k + 1) = avg_gray(k + 1) + gray(i,j);
        end
    end 
   avg_gray(k + 1) = avg_gray(k + 1) / v_size;
   avg_gray(k + 1) = avg_gray(k + 1) * avg_gray(k + 1);
   
    for a = 1:4
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

for i = 1 : gap
    if(E_count(i) == max(E_count))
        loca_E_count = i;
        break;
    end
end

for i = 1 : gap
    if(avg_gray(i) == max(avg_gray))
        loca_avg_gray = i;
        break;
    end
end
for i =1:gap - 1
    if(Max_grad < avg_gray(i)  / avg_gray(i + 1))
        Max_grad = avg_gray(i) / avg_gray(i + 1);
        loca_grad = i + 1;
    end
end

% for i = 1:gap
%     t = ceil((sqrt(avg_gray(i)) + 1) / 10);
%     E_count_color(t) = E_count_color(t) + 1;
% end
% E_count_color = E_count_color / gap;
% colormax = ceil((sqrt(max(avg_gray)) + 1) / 10);
% for i = 1:colormax
%     if(E_count_color(i) ~=0)
%         locamin_color = i;
%         break;
%     end
% end
% b = 1:colormax;
% for i = locamin_color:colormax
%     w1 = sum(E_count_color(locamin_color:i));
%     w2 = 1 - w1;
%     mean1 = sum(E_count_color(locamin_color:i).*b(locamin_color:i)) / sum(E_count_color(locamin_color:i));
%     mean2 = sum(E_count_color(i + 1:colormax).*b(i + 1:colormax)) / sum(E_count_color(i + 1:colormax));
%     delta = w1 *w2 *(mean1 - mean2)^2;
%     if(delta > Max_color)
%         Max_color = delta;
%         thres_color = i;
%     end
% end
% thres_color = thres_color * thres_color * 100;
% for i = 1:gap
%     if(avg_gray(i) <= thres_color)
%         loca_color = i;
%         break;
%     end
% end

for i = 1 : gap
    if(avg_gray(i) == 0)
         avg_gray(i) = 1;
    end
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
        t = i;
    end
end
if(t > 1)
    E_count(t - 1) = E_count(t);
end

if(E_count(1) > E_count(2))
    E_count_new(1) = E_count(2);
    E_count(1) = E_count(2);
else
    E_count_new(1) = E_count(1);
end

if(E_count(gap - 1) > E_count(gap))
    E_count_new(gap) = E_count(gap - 1);
    E_count(gap) = E_count(gap - 1);
else
    E_count_new(gap) = E_count(gap);
end

for i = 2:gap
    if(E_count_gradmax < E_count(i) / E_count(i - 1))
        E_count_gradmax = E_count(i) / E_count(i - 1);
        E_count_gradmaxloc = i;
    end
end

for i = 2:(gap - 1)
    E_count_comp(1) = E_count_new(i - 1);
    E_count_comp(2) = E_count(i);
    E_count_comp(3) = E_count(i + 1);
    for j = 1:3
        for k = 1:3 - j
            if(E_count_comp(k) > E_count_comp(k + 1))
                tmp = E_count_comp(k);
                E_count_comp(k) = E_count_comp(k + 1);
                E_count_comp(k + 1) = tmp;
            end
        end
    end
    E_count_new(i) = E_count_comp(2);
end
while(1)
    flag_end = 0;
    for i = 1:gap - 1
        if(E_count_new(i) > E_count_new(i + 1))
            E_count_new(i) = E_count_new(i + 1);
            flag_end = 1;
        end
    end
    if(flag_end == 0)
        break;
    end
end
E_count_new_avg = mean(E_count_new);
E_count_new_min = min(E_count_new);
E_count_new_max = max(E_count_new);

for i = 1:gap
    if((E_count_new(i) / E_count_new_min < 10 && E_count_new_avg / E_count_new(i) >= 2) || E_count_new(i) == E_count_new_min)
        E_count_new(i) = 0.0001;
        E_location = i;
    else
        E_count_new(i) = E_count_new(i) / E_count_new_avg;
    end
    if(E_count_new(i) >= 1)
        E_count_new(i) = 0.9998;
    end
end
for i = 1:gap
      if(E_count_new(i) > 0.5)
          threshold = i;
          break;
      end
end
if(E_location + 1 == E_count_gradmaxloc)
    for i = threshold + 1:gap
       if(E_count_new(i) * i / threshold * 1.5> E_count_new(i))
          E_count_new(i) = E_count_new(i) * i / threshold * 1.5;
       end
       if(E_count_new(i) >= 1)
          E_count_new(i) = 0.9998;
       end
    end
elseif(E_location + 1 > E_count_gradmaxloc)
        while(1)
            if(E_count(threshold) / E_count(E_location) > 5)
                break;
            else
                E_count_new(E_location) = E_count(E_location) / E_count(threshold) * E_count_new(threshold);
                E_location = E_location - 1;
                %{
                if(E_location == E_count_gradmaxloc)
                    break;
                end
                %}
            end
        end
end

for i = 1:gap
    pro_x(i) = x(i);
    pro_y(i) = E_count_new(i);
end

for i = 1:gap
    if(pro_y(i) >= 1)
        pro_y(i) = 0.9998;
    end
    pro_y(i) = log(1 / pro_y(i) - 1);
end
avg_x = sum(pro_x) / gap;
avg_y = sum(pro_y) / gap;

for i = 1:gap
    Sum_x_y = Sum_x_y + pro_x(i) * pro_y(i);
    Sum_x_2 = Sum_x_2 + pro_x(i) * pro_x(i);
end
k = (Sum_x_y - gap * avg_x * avg_y) / (Sum_x_2 - gap * avg_x * avg_x);
l = avg_y - k * avg_x;

c = -1 / k;
b = -l / k;

for i = 1:M
    x_sig(i) = i;
    y_sig(i) = 1 / (1+exp(-(i - b)/c));
end

% figure(1);
% plot(x_sig,y_sig);
Cut_y_start = floor(b - c * 2);
Cut_y_end = floor(b + c * 3);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (Cut_y_start <= 0)
    Cut_y_start = 1;
end 
if (Cut_y_end >= M)
    Cut_y_end = M;
end

Show_image = origin(Cut_y_start : Cut_y_end,1 : N,:);
% figure(2);
% imshow(Show_image);	%extracted sea sky region
img = Show_image;
ai = double(img(:,:,3));
SE = strel ( 'square',5);
ai = imerode( ai, SE);
% aii = medfilt2(ai,[2 2],'symmetric');%
ai = imdilate( ai, SE);
edg = edge(ai, 'canny',[0.1 0.15]);
[H1,theta1,rho1] = hough(edg,'ThetaResolution',1,'RhoResolution',1);
[r1,r2] = sort(H1(:) / sum(edg(:)),'descend');
% [rhloc,thetaloc] = find(H1 == max(max(H1)));
figure,
imshow(origin);
hold on;
chomat = zeros(5,5);
chomatdown = zeros(5,5);
chopoll = zeros(1,10);
cogray = double(origin(:,:,3));
for i = 1:20
    [rhloc,thetaloc] = ind2sub(size(H1),r2(i));
    x_plot = 1:1:size(gray,2);
    y_plot = (rho1(rhloc) - x_plot * cosd(theta1(thetaloc))) / sind(theta1(thetaloc)) + Cut_y_start;
    plot(x_plot,y_plot,'Color',[1 0 0],'LineWidth',2);
    chomin = 65536;
    for chox = 1:size(gray,2)
        if(sind(theta1(thetaloc)) == 0)
            break;
        end
        choy = floor((rho1(rhloc) - chox * cosd(theta1(thetaloc))) / sind(theta1(thetaloc)) + Cut_y_start);
        chomod = mod(chox,5);
        if(choy - 5 <= 0 || choy + 5 >= size(origin,1))
            break;
        end
        if(chomod == 0)
            chomat(5,1) = cogray(choy - 1,chox);
            chomat(5,2) = cogray(choy - 2,chox);
            chomat(5,3) = cogray(choy - 3,chox);
            chomat(5,4) = cogray(choy - 4,chox);
            chomat(5,5) = cogray(choy - 5,chox);
            if(chomin > std2(chomat))
                chomin = std2(chomat);
            end
        else
            chomat(chomod,1) = cogray(choy - 1,chox);
            chomat(chomod,2) = cogray(choy - 2,chox);
            chomat(chomod,3) = cogray(choy - 3,chox);
            chomat(chomod,4) = cogray(choy - 4,chox);
            chomat(chomod,5) = cogray(choy - 5,chox);
        end
    end
    chopoll(i) =(1 / (chomin + 1)^2) * r1(i); 
end
[~,I] = max(chopoll);
[rhloc,thetaloc] = ind2sub(size(H1),r2(I));
y_plot = (rho1(rhloc) - x_plot * cosd(theta1(thetaloc))) / sind(theta1(thetaloc)) + Cut_y_start;
plot(x_plot,y_plot,'Color',[0 0 0],'LineWidth',2);

hold off;
disp(['whole runtime',num2str(etime(clock,ti))]);
origin = imread('F:\1-sea-sky-pic\2.png');
% gray = rgb2gray(origin);
gray = double(origin(:,:,3));
[M,N] = size(gray);
gap = 15;  %����Ϊgap���Ҷȹ�������
step = 3;   %����Ϊ3
diag_sum = 0;
v_length = floor(M / gap);    %ÿ������ĸ�
v_width = floor(N / gap);    %ÿ������Ŀ�
count = zeros(1,gap);
E_count = zeros(1,gap);%��������ֵ
C_count = zeros(1,gap);
E_count_color = zeros(1,26);
E_count_comp = zeros(1,3);
E_count_new = zeros(1,gap);
E_count_gradmax = -1;   % the maximum value of gradient
E_count_gradmaxloc = 1;   %the location of maximum gradient
E_location = 1;
E_count_new_avg = 0;
E_count_new_min = 0;
E_count_new_max = 0;
flag_end = 0;
tmp = 0;
Sum_E_count = 0;%����ֵ�ĺͣ����ڹ�һ������
Max_E_count = 0;%��������ֵ�����ڹ�һ������
x = zeros(1,gap);%ֱ����Ϻ�����
y = zeros(1,gap);%ֱ�����������
avg_gray = zeros(1,gap); %ÿ������Ҷ�ƽ��ֵ������ΪȨֵ
flag_E_count = zeros(1,gap);
pro_y = zeros(1,gap); %������ϵ�y(probility)
pro_x = zeros(1,gap);   %������ϵ�x(x-axis)
Sum_x = 0;
Sum_y = 0;
Sum_x_y = 0;
Sum_x_2 = 0;
k = 0; % ֱ����ϵ�б��
l = 0; % ֱ����ϵĽؾ�
Pro_count = 1;  %count the number of fitting dots
Min = 9999;
Pro_avg = 0;
Min_E_count = 999;%�������Сֵ����������ֵ�滻
threshold  = 1;   %��һ������ֵ����0.5����Сֵ
Max_color = -1;   %maxium value for segmentation
locamin_color = 1;
thres_color = 1;   % threshold for segmentation
loca_color = 1;   %location for segmentation
Max_grad = -1;
loca_grad = 1;

loca_avg_gray = 0;  %��λ�������ƽ���Ҷ�ֵ 
loca_E_count = 0;  %��λ�����������ֵ
E_count_sky = 0;    %�����ϵ������
E_count_sea = 0;    %������ϵ������
start_point = 1;    %��ʼ��ϵ�λ��


y_sig = zeros(1,M);%sigmoid�����е�y����
x_sig = zeros(1,M);  %sigmoid�����е�x����

v_size = v_length * v_width; %�����С

for i = 1 : M
    for j = 1 : N
        gray(i,j) = floor(gray(i,j)/16);   %���Ҷȹ�������Ĵ�С����Ϊ16*16
    end
end

for k = 0 : gap-1
    v_start = k * v_length + 1;
    v_end = (k+1) * v_length;
    p = zeros(16,16,4);
    for i = v_start : v_end
        for j = 1: N
            if j <= N - step   %0�Ƚ�
                p(gray(i,j) + 1,gray(i,j + step) + 1,1) = p(gray(i,j) + 1,gray(i,j + step) + 1,1) + 1;
                p(gray(i,j + step) + 1,gray(i,j) + 1,1) = p(gray(i,j) + 1,gray(i,j + step) + 1,1);
            end
            if i>= v_start + step&j < N - step  %45�Ƚ�
                p(gray(i,j) + 1,gray(i - step,j + step) + 1,2) =  p(gray(i,j) + 1,gray(i - step,j + step) + 1,2) + 1;
                p(gray(i - step,j + step) + 1,gray(i,j) + 1,2) = p(gray(i,j) + 1,gray(i - step,j + step) + 1,2);
            end
            if i>= v_start + step  %90�Ƚ�
                p(gray(i,j) + 1,gray(i - step,j) + 1,3) =  p(gray(i,j) + 1,gray(i - step,j) + 1,3) + 1;
                p(gray(i - step,j) + 1,gray(i,j) + 1,3) =  p(gray(i,j) + 1,gray(i - step,j) + 1,3);
            end
            if i>= v_start + step&j > step  %135�Ƚǣ�����stepΪ1��������Ҫ������������1Ϊ����
                p(gray(i,j) + 1,gray(i - step,j - step) + 1,4) =  p(gray(i,j) + 1,gray(i - step,j - step) + 1,4) + 1;
                p(gray(i - step,j - step) + 1,gray(i,j) + 1,4) =  p(gray(i,j) + 1,gray(i - step,j - step) + 1,4);
            end
            avg_gray(k + 1) = avg_gray(k + 1) + gray(i,j);
        end
    end 
   avg_gray(k + 1) = avg_gray(k + 1) / v_size;
   avg_gray(k + 1) = avg_gray(k + 1) * avg_gray(k + 1); % * avg_gray(k + 1);
   
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
                 Ux(b) = i*p(i,j,b)+Ux(b); %������Ц�x
                 Uy(b) = j*p(i,j,b)+Uy(b); %������Ц�y

             end
         end
     end
   for n = 1:4
    for i = 1:16
        for j = 1:16
            deltaX(n) = (i-Ux(n))^2*p(i,j,n)+deltaX(n); %������Ц�x
            deltaY(n) = (j-Uy(n))^2*p(i,j,n)+deltaY(n); %������Ц�y
            C(n) = i*j*p(i,j,n)+C(n);            
        end
    end
    C(n) = (C(n)-Ux(n)*Uy(n))/deltaX(n)/deltaY(n); %�����  
   end
 % E_count(k+1) = E(1);
   % E_count(k+1) = sum(C);
    E_count(k+1) = sum(I);  %���ｫ�ĸ��Ƕȵ�ֵȫ��������
    
%    sprintf('0,45,90,135�����ϵĶԱȶ�����Ϊ�� %f, %f, %f, %f',I(1),I(2),I(3),I(4));
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

for i = 1:gap
    t = ceil((sqrt(avg_gray(i)) + 1) / 10);
    E_count_color(t) = E_count_color(t) + 1;
end
E_count_color = E_count_color / gap;
colormax = ceil((sqrt(max(avg_gray)) + 1) / 10);
for i = 1:colormax
    if(E_count_color(i) ~=0)
        locamin_color = i;
        break;
    end
end
b = 1:colormax;
for i = locamin_color:colormax
    w1 = sum(E_count_color(locamin_color:i));
    w2 = 1 - w1;
    mean1 = sum(E_count_color(locamin_color:i).*b(locamin_color:i)) / sum(E_count_color(locamin_color:i));
    mean2 = sum(E_count_color(i + 1:colormax).*b(i + 1:colormax)) / sum(E_count_color(i + 1:colormax));
    delta = w1 *w2 *(mean1 - mean2)^2;
    if(delta > Max_color)
        Max_color = delta;
        thres_color = i;
    end
end
thres_color = thres_color * thres_color * 100;
for i = 1:gap
    if(avg_gray(i) <= thres_color)
        loca_color = i;
        break;
    end
end

for i = 1 : gap
    if(avg_gray(i) == 0)
         avg_gray(i) = 1;
    end
    E_count(i) = E_count(i) / avg_gray(i);
end

Sum_E_count = sum(E_count);
Max_E_count = max(E_count);
for i = 1:gap
    if(Min_E_count > E_count(i) && E_count(i) ~= 0) %������������ֵΪ0 �������÷������Сֵ���г�ʼ��
        Min_E_count = E_count(i);
    end
end

for i = 1:gap  %�����ݽ��й�һ������ͬʱ�ҳ�������С����ֵ
    E_count(i) = E_count(i) / Max_E_count ;
    if(E_count(i) == 0)
        E_count(i) = Min_E_count / Max_E_count ;  %ʹ�����ֵ���й�һ�����ᵼ�����ֵ�Ĺ�һ�����Ϊ1,1�Ǵ��������Ҫ�ʵ�����
    end
    if (E_count(i) >= 1)
        E_count(i) = 0.99998;
    end
    if (Min >= E_count(i))
        Min = E_count(i);
    end
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
%}
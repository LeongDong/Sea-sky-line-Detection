origin = imread('F:\1-sea-sky-pic\153_error.jpg');
% gray = rgb2gray(origin);
gray = double(origin(:,:,1));
[M,N] = size(gray);
gap = 15;  %����Ϊgap���Ҷȹ�������
step = 3;   %����Ϊ3
diag_sum = 0;
v_length = floor(M / gap);    %ÿ������ĸ�
v_width = floor(N / gap);    %ÿ������Ŀ�
count = zeros(1,gap);
E_count = zeros(1,gap);%��������ֵ
C_count = zeros(1,gap);
Sum_E_count = 0;%����ֵ�ĺͣ����ڹ�һ������
Max_E_count = 0;%��������ֵ�����ڹ�һ������
x = zeros(1,gap);%ֱ����Ϻ�����
y = zeros(1,gap);%ֱ�����������
avg_gray = zeros(1,gap); %ÿ������Ҷ�ƽ��ֵ������ΪȨֵ
pro_y = zeros(1,gap); %������ϵ�y(probility)
pro_x = zeros(1,gap);   %������ϵ�x(x-axis)
Sum_x_y = 0;
Sum_x_2 = 0;
k = 0; % ֱ����ϵ�б��
l = 0; % ֱ����ϵĽؾ�
Pro_count = 1;  %count the number of fitting dots
Min = 9999;
Pro_avg = 0;
Min_E_count = 999;%�������Сֵ����������ֵ�滻

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
        for j = 1: N / gap
            if j <= N / gap - step   %0�Ƚ�
                p(gray(i,j) + 1,gray(i,j + step) + 1,1) = p(gray(i,j) + 1,gray(i,j + step) + 1,1) + 1;
                p(gray(i,j + step) + 1,gray(i,j) + 1,1) = p(gray(i,j) + 1,gray(i,j + step) + 1,1);
            end
            if i>= v_start + step&j < N/gap - step  %45�Ƚ�
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

for i = 1:gap
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

for i = 1:gap   %�ҳ�����Сֵͬ����������������ֵ����Ϊ��պ�ѡ��ϵ�
    if(max(E_count) / E_count(i) > 2.5 && E_count(i) / Min < 4)
        pro_x(Pro_count) = x(i);
        pro_y(Pro_count) = 0.00001
        Pro_count = Pro_count + 1;
    end
end

Pro_avg = mean(E_count);    %�ҳ�����ֵ����ƽ��ֵ����������ֵ����Ϊ�����ѡ��ϵ�
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
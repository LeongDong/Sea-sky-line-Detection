origin = imread('F:\1-sea-sky-pic\197.jpg');
% gray = rgb2gray(origin);
gray = double(origin(:,:,3));
[M,N] = size(gray);
gap = 20;  %����Ϊ20���Ҷȹ�������
step = 3;   %����Ϊ3
diag_sum = 0;
v_length = floor(M/gap);    %ÿ������ĸ�
count = zeros(1,gap);
E_count = zeros(1,gap);%��������ֵ
C_count = zeros(1,gap);
Sum_E_count = 0;%����ֵ�ĺͣ����ڹ�һ������
Max_E_count = 0;%��������ֵ�����ڹ�һ������
x = zeros(1,gap);%ֱ����Ϻ�����
y = zeros(1,gap);%ֱ�����������
Sum_x_y = 0;
Sum_x_2 = 0;
k = 0; % ֱ����ϵ�б��
l = 0; % ֱ����ϵĽؾ�

y_sig = zeros(1,gap);%sigmoid�����е�y����

for i = 1 : M
    for j = 1 : N
        gray(i,j) = floor(gray(i,j)/16);   %���Ҷȹ�������Ĵ�С����Ϊ16*16
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
                    if j <= N/20 - step&gray(i,j)==m-1&gray(i,j + step)==n-1   %0�Ƚ�
                        p(m,n,1) = p(m,n,1) + 1;
                        p(n,m,1) = p(m,n,1);
                    end
                    if i>= v_start + step&j < N/20 - step&gray(i,j)==m-1&gray(i - step,j + step)==n-1  %45�Ƚ�
                        p(m,n,2) =  p(m,n,2) + 1;
                        p(n,m,2) = p(m,n,2);
                    end
                    if i>= v_start + step&gray(i,j)==m-1&gray(i - step,j)==n-1  %90�Ƚ�
                        p(m,n,3) = p(m,n,3)+1;
                        p(n,m,3) = p(m,n,3);
                    end
                    if i>= v_start + step&j > step&gray(i,j)==m-1&gray(i - step,j - step)==n-1   %135�Ƚǣ�����stepΪ1��������Ҫ������������1Ϊ����
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
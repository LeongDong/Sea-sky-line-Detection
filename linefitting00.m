x=imread('F:\1-sea-sky-pic\3.jpg');
%a=double(rgb2gray(x));
a=double(x(:,:,3));
SE = strel ( 'square',3);
a = imerode( a, SE);
a = imdilate( a, SE);

[m,n]=size(a);
N=m*n;
L=256;
interval=20;

zone_length=floor(n/interval);
totalNum=zone_length*m;

theta_f = 0;
intercept_f = 0;
max_val = 0;
max_num = 0;
slope = 0;
intercept = 0;
theta = 0;
num = zeros(1,interval * interval / 2);
line_theta = zeros(1,interval * interval / 2);
line_b = zeros(1,interval * interval / 2);

for k=0:(interval-1)
    zone_start=k*zone_length+1;
    zone_final=(k+1)*zone_length;
    count=zeros(1,256);
    for i=zone_start:zone_final
        for j=1:m
            t=a(j,i);
            t=t+1;%t的灰度值应该加1，范围为1~256
            count(t)=count(t)+1;
        end
    end
    c=a(:,zone_start:zone_final);
    count=count/totalNum;
    maxmax=max(max(c));
    b=1:maxmax;
    MaxNum=0;
    for l=1:maxmax
        w1=sum(count(1:l));
        w2=1-w1;
        mean1=sum(count(1:l).*b(1:l))/sum(count(1:l));
        mean2=sum(count(l+1:maxmax).*b(l+1:maxmax))/sum(count(l+1:maxmax));
        delta=w1*w2*(mean1-mean2)^2;
        if (delta>MaxNum)
            MaxNum=delta;
            threshold(k+1)=l;
        end
    end
end

figure,
imshow(x);
hold on;

for i=1:interval
    t=ceil((zone_length+1)/2+(i-1)*zone_length);    %从上往下找阈值点
    x1(i)=t;
    for j=1:m
        if (threshold(i)>a(j,x1(i)))
            y(i)=j;
            break;
        end
    end
   % plot(x1(i),y(i),'o','MarkerFaceColor','r');
end

th = 0;
flag = -1;
for i = 1 : interval - 1
    for j = (i + 1) : interval
        slope = (y(j) - y(i)) / (x1(j) - x1(i));
        intercept = (x1(j) * y(i) - y(j) * x1(i)) / (x1(j) - x1(i));
        theta = atand(slope);
        for k = 1 : th
            if(abs(line_theta(k) - theta) < 2 && abs(line_b(k) - intercept) < 2)
                line_theta(k) = (line_theta(k) * num(k) + theta) / (num(k) + 1);
                line_b(k) = (line_b(k) * num(k) + intercept) / (num(k) + 1);
                num(k) = num(k) + 1;
                flag = 1;
            end
        end
        if(flag == -1)
            th = th + 1;
            line_theta(th) = theta;
            line_b(th) = intercept;
        end
    end
end

[max_val,max_num] = max(num);
theta_f = line_theta(max_num);
intercept_f = line_b(max_num);

x_plot = 1:1:n;
y_plot = tand(theta_f) * x_plot + intercept_f;
plot(x_plot,y_plot,'Color',[0 0 0]);

hold off;
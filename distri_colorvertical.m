ori_img = imread('F:\1-sea-sky-pic\24.jpg');
img = ori_img(:,:,3);
img_red = ori_img(:,:,1);
img_green = ori_img(:,:,2);

img_size = size(img);

img_rows = img_size(1);
img_cols = img_size(2);

area_th = floor(img_cols/2);

illu_y = 0: (img_rows - 1);
illu_x_blue = img(:,area_th);
illu_x_red = img_red(:,area_th);
illu_x_green = img_green(:,area_th);

hold on;
%set(gca,'xaxislocation','top','yaxislocation','left','ydir','reverse');

plot(illu_x_blue,illu_y,'blue');
plot(illu_x_red,illu_y,'red');
plot(illu_x_green,illu_y,'green');
%plot(illu_y,illu_x_blue,'blue');
%set(gca,'xdir',reverse);
xlabel('rows of image');
ylabel('gray level of pixel');
hold off;
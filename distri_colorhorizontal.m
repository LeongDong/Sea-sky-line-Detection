ori_img = imread('F:\1-sea-sky-pic\24.jpg');
img = ori_img(:,:,3);
img_red = ori_img(:,:,1);
img_green = ori_img(:,:,2);

img_size = size(img);

img_rows = img_size(1);
img_cols = img_size(2);

area_th_sky = floor(23); %acquire the sequence of row
area_th_sea = floor(img_rows - 20);

illu_x = 0: (img_cols - 1);

illu_y_blue_sky = img(area_th_sky,:);%acquire a row of pixels in the sky
illu_y_red_sky = img_red(area_th_sky,:);
illu_y_green_sky = img_green(area_th_sky,:);

illu_y_blue_sea = img(area_th_sea,:);%acquire a row of pixels in the sea
illu_y_red_sea = img_red(area_th_sea,:);
illu_y_green_sea = img_green(area_th_sea,:);

figure(1);
hold on;

plot(illu_x,illu_y_blue_sky,'blue');
plot(illu_x,illu_y_red_sky,'red');
plot(illu_x,illu_y_green_sky,'green');

%set(gca,'xdir',reverse);
xlabel('cols of image');
ylabel('gray level of pixel');
title('sky');
hold off;

figure(2);
hold on;

plot(illu_x,illu_y_blue_sea,'blue');
plot(illu_x,illu_y_red_sea,'red');
plot(illu_x,illu_y_green_sea,'green');

xlabel('cols of image');
ylabel('gray level of pixel');

title('sea');

hold off;
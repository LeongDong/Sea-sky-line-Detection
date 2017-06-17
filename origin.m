img_origin = imread('F:\1-sea-sky-pic\15.jpg');

img = double(img_origin(:,:,3));
img_size = size(img);
rows = img_size(1);
cols = img_size(2);

avg_grad = 0;
grad = 0;
flag = 1;
sequence = zeros(1,rows);
x = 0 : (rows - 1);

for i = 1 : (rows - 1)
    for j = 1 : 20 %(cols - 1)
        grad = img(i,j+1) - img(i,j)
        grad = abs(grad)  / (img(i,j) + 1);
        %avg_grad = (avg_grad*(j-1) + grad) / j;
        avg_grad = avg_grad + grad; 
    end
    avg_grad = avg_grad / cols;
    sequence(i) = avg_grad;
end

sequence(1,rows) = sequence(1,rows-1);

hold on;
plot(x,sequence,'black');
xlabel('rows of image');
ylabel('gradient');
hold off;
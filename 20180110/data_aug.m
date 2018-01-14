f_list = dir('./*.jpg');
index = 0;
for n=1:length(f_list)
    index = index + 1
    img = imread(['./', f_list(n).name]);
	
	%%%%%%%
    b = 0.15*rand() + 1.3;
    img_bright_liang = imlincomb(b, img);
    img_bright_an = imlincomb(0.85,img);
    
    imwrite(img_bright_liang, ['./l', f_list(n).name]);
    imwrite(img_bright_an, ['./a', f_list(n).name]);
    
    %%%%%%%
    h = size(img, 1) - 7;
    w = size(img, 2) - 5;
    
    x = round(3*rand()) + 1;
    y = round(5*rand()) + 1;
    img_crop = imcrop(img,[x, y, w, h]);
    imwrite(img_crop, ['./c', f_list(n).name]);
    
    %%%%%%%
    img_ver = flipud(img);
    b = 0.65*rand() + 0.8;
    img_bright = imlincomb(b, img_ver);
    imwrite(img_bright, ['./vb', f_list(n).name]);
    
	%%%%%%%
    img_hor = fliplr(img_ver);
    b = 1.0*rand() + 0.5;
    w = fspecial('gaussian', 5, b);
    img_blur = imfilter(img_hor, w);
    imwrite(img_blur, ['./hb', f_list(n).name]);
end

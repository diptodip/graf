filename = '20160802 N2-0025.tif';

output = '20160802 N2-0025-trimmed.tif';


% I = imread(filename, 94);
% 
% %level = multithresh(I);
% 
% level = 2500;
% 
% seg_I = imquantize(I, level);
% 
% decrementer = ones(512, 512);
% 
% seg_I = seg_I - decrementer;
% 
% blur = imgaussfilt(seg_I, 10);
% 
% rows = size(I, 1);
% cols = size(I, 2);
% 
% I = reshape(blur, rows * cols, 1);
% 
% ncolors = 2;
% 
% [cluster_idx, cluster_center] = kmeans(I,ncolors,'distance','sqEuclidean','Replicates',3);
%                                   
% pixel_labels = reshape(cluster_idx, rows, cols);
% 
% imshow(pixel_labels, []), title('2 means clustered');

decrementer = ones(512, 512);

I = imread(filename, 2);

bg_brightness = mean(mean(I));

decrease = [151, 1];
start = 0;
for i = 2:152
    I = imread(filename, i);
    I = imgaussfilt(I, 5);
    seg_I = imquantize(I, 1.1 * bg_brightness);
    seg_I = seg_I - decrementer;
    total = sum(sum(seg_I));
    decrease(i-1, 1) = total;
    if total > 1
        if start == 0
            start = i - 1;
        end
    end
end

start = start - 2;

finish = 0;

for j = 1:146
    dx = gradient(decrease(j:j+5, 1));
    if dx < 0
        if finish == 0 || j < 151 - 15
            if j < 151 - 15
                finish = j+15;
            else
                finish = j+5;
            end
        end
    end
end

disp(start);
disp(finish);

for k = start:finish
    I = imread(filename, k);
    imwrite(I(:, :), output, 'WriteMode', 'append',  'Compression','none');
end

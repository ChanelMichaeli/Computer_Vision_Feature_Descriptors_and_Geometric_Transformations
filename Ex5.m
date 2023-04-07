clc; 
clear all;
disp('Noam Atias 311394357');
disp('Chanel Michaeli 208491787');

%% Q1 - Mona Surfaliza
%% Q1.1.3
mona_org_img = imread('mona_org.jpg');
mona_org_norm = img_norm(mona_org_img);
figure('Name','Q1.1.3');
imshow(mona_org_norm); 
title('Normalized Mona Image');

%% Q1.1.4
tic;
pointsDefault = detectSURFFeatures(mona_org_norm);
time1=toc ;

figure('Name','Q1.1.4');
imshow(mona_org_norm); hold on;
plot(pointsDefault); hold off;
title('Moan Image with Extracted Default Features ');

%% Q1.1.5
tic;
pointsROI = detectSURFFeatures(mona_org_norm,'ROI',[59 5 128 120]);
time2=toc;

figure('Name','Q1.1.5');
imshow(mona_org_norm); hold on;
plot(pointsROI); hold off;
title('Extracted Features with ROI - Mona Image');

%% Q1.1.6
tic;
points4Octaves = detectSURFFeatures(mona_org_norm,'NumOctaves',4);
time3=toc;
tic;
points1Octaves = detectSURFFeatures(mona_org_norm,'NumOctaves',1);
time4=toc;

figure('Name','Q1.1.6');
subplot(1,2,1);
imshow(mona_org_norm); hold on;
plot(points4Octaves); hold off;
sgtitle('Extracted features with:')
title('NumOctaves=4');
subplot(1,2,2);
imshow(mona_org_norm); hold on;
plot(points1Octaves); hold off;
title('NumOctaves=1');


%% Q1.1.7
tic;
points8ScaleLevels = detectSURFFeatures(mona_org_norm,'NumScaleLevels',8);
time5=toc;
tic;
points3ScaleLevels = detectSURFFeatures(mona_org_norm,'NumScaleLevels',3);
time6=toc;

figure('Name','Q1.1.7');
subplot(1,2,1);
imshow(mona_org_norm); hold on;
plot(points3ScaleLevels); hold off;
sgtitle('Extracted features with:')
title('NumScaleLevels=3');
subplot(1,2,2);
imshow(mona_org_norm); hold on;
plot(points8ScaleLevels); hold off;
title('NumScaleLevels=8');

%% Q1.2.1
straight_mona_img = imread('straight_mona.PNG');
crooked_mona_img = imread('crooked_mona.jpg');
straight_mona_norm = img_norm(straight_mona_img);
crooked_mona_gray = double(crooked_mona_img);
crooked_mona_norm = (crooked_mona_gray - min(crooked_mona_gray(:)))/(max(crooked_mona_gray(:)) - min(crooked_mona_gray(:)));

figure('Name','Q1.2.1');
subplot(1,2,1); 
imshow(straight_mona_norm); 
title("Normalized Straight Mona Image");
subplot(1,2,2); 
imshow(crooked_mona_norm);  
title("Normalized Crooked Mona Image");

%% Q1.2.2
features_straight_mona = detectSURFFeatures(straight_mona_norm);
features_crooked_mona  = detectSURFFeatures(crooked_mona_norm);

figure('Name','Q1.2.2');
subplot(1,2,1); 
imshow(straight_mona_norm);
sgtitle('Ten strongest features of:')
title('Straight Mona image');
hold on; 
plot(features_straight_mona.selectStrongest(10)); 
hold off;
subplot(1,2,2); 
imshow(crooked_mona_norm);
title('Crooked Mona image');
hold on; 
plot(features_crooked_mona.selectStrongest(10));  
hold off;

%% Q1.2.3
[features_str,points_str] = extractFeatures(straight_mona_norm,features_straight_mona);
[features_Crook,points_Crook] = extractFeatures(crooked_mona_norm,features_crooked_mona); 

index_Pairs = matchFeatures(features_str, features_Crook);
matched_points_str = points_str(index_Pairs(:,1));
matched_points_crooked = points_Crook(index_Pairs(:,2));

figure('Name','Q1.2.3 '); 
showMatchedFeatures(straight_mona_norm,crooked_mona_norm,matched_points_str,matched_points_crooked);
title('Matching points')
legend('Original','Crooked'); 

Geometric_Transform = estimateGeometricTransform(matched_points_crooked, matched_points_str,'similarity');
output_view_str = imref2d(size(straight_mona_norm));
mona_fixed = imwarp(crooked_mona_norm, Geometric_Transform, 'OutputView', output_view_str);
figure('Name','Q1.2.3 Fixed image');
imshow(mona_fixed);
title('Fix crooked image to straight image');

%% Q1.3.1
mona = imgaussfilt(imgaussfilt(mona_org_norm,0.5),0.5);
mona_features = detectSURFFeatures(mona,'ROI',[59 5 128 120],'NumOctaves',4,'NumScaleLevels',8);
[featuresOrg,pointsOrg] = extractFeatures(mona,mona_features);

images = dir('mona');
for i=1:length(images) 
   if(~images(i).bytes)
      continue; 
   end
   name_img = fullfile('mona',images(i).name);
   new_mona = imread(name_img);
   new_mona = img_norm(new_mona);
   new_mona = imgaussfilt(imgaussfilt(new_mona,0.5),0.5);

   new_features = detectSURFFeatures(new_mona,'NumOctaves',4,'NumScaleLevels',8);
   [new_mona_features,new_mona_points] = extractFeatures(new_mona,new_features);

   indexPairs = matchFeatures(featuresOrg,new_mona_features,'MatchThreshold',1.6);

    if(size(indexPairs,1) >= 3) 
        matched_Points_org = pointsOrg(indexPairs(:,1));
        matched_Points_new = new_mona_points(indexPairs(:,2));

        figure; 
        showMatchedFeatures(mona,new_mona,matched_Points_org,matched_Points_new,'montage');
        title(strcat('The image ',images(i).name,'--> Real'))
        disp(strcat('The image  ',images(i).name,' is Mona'));
        continue;
     end
end

%% Q2
%% Q2.3
% ID2QR('311394357');
% title('Our QR')
xfixed=[1200; 1200; 1; 1];
yfixed=[1; 1200; 1200; 1];

% (x,y) Load 
load("easy_xpoints.mat", 'x1');
load("easy_ypoints.mat", 'y1');
load("inter_xpoints.mat", 'x2');
load("inter_ypoints.mat", 'y2');
load("hard_xpoints.mat", 'x3');
load("hard_ypoints.mat", 'y3');

QR_easy = imread("QReasy.jpg");
QR_inter = imread("QRinter.jpg");
QR_hard = imread("QRhard.jpg");

QR_easy = img_norm(QR_easy);
QR_inter = img_norm(QR_inter);
QR_hard = img_norm(QR_hard);

figure('Name','Q2.1 easyQR'); 
imshow(QR_easy);
hold on; 
plot(x1,y1,'r+',LineWidth=1.5); 
title("Easy QR");
% [x1,y1] = ginput(4);

figure('Name','Q2.1 interQR'); 
imshow(QR_inter);
hold on; 
plot(x2,y2,'r+',LineWidth=1.5); 
title("Intermediate QR");
% [x2,y2] = ginput(4);

figure('Name','Q2.1 hardQR'); 
imshow(QR_hard);
hold on; 
plot(x3,y3,'r+',LineWidth=1.5); 
title("Hard QR");
% [x3,y3] = ginput(4);

% save("easy_xpoints.mat", 'x1');
% save("easy_ypoints.mat", 'y1');
% save("inter_xpoints.mat", 'x2');
% save("inter_ypoints.mat", 'y2');
% save("hard_xpoints.mat", 'x3');
% save("hard_ypoints.mat", 'y3');

%% 2.4

straight_easy_rigid = straight_image(QR_easy,[x1 y1],[xfixed yfixed],'nonreflectivesimilarity','Straight easyQR with transformation ');
straight_inter_rigid = straight_image(QR_inter,[x2 y2],[xfixed yfixed],'nonreflectivesimilarity','Straight interQR with transformation ');
straight_hard_rigid = straight_image(QR_hard,[x3 y3],[xfixed yfixed],'nonreflectivesimilarity','Straight hardQR with transformation ');

straight_easy_affine = straight_image(QR_easy,[x1 y1],[xfixed yfixed],'affine','Straight easyQR with transformation ');
straight_inter_affine = straight_image(QR_inter,[x2 y2],[xfixed yfixed],'affine','Straight interQR with transformation ');
straight_hard_affine = straight_image(QR_hard,[x3 y3],[xfixed yfixed],'affine','Straight hardQR with transformation ');

straight_easy_projective = straight_image(QR_easy,[x1 y1],[xfixed yfixed],'projective','Straight easyQR with transformation ');
straight_inter_projective = straight_image(QR_inter,[x2 y2],[xfixed yfixed],'projective','Straight interQR with transformation ');
straight_hard_projective = straight_image(QR_hard,[x3 y3],[xfixed yfixed],'projective','Straight hardQR with transformation ');

%% 2.5
% Explain in PDF

%% 2.6
rec_easyQR_rigid = img2QR(straight_easy_rigid);
rec_interQR_rigid = img2QR(straight_inter_rigid);
rec_hardQR_rigid = img2QR(straight_hard_rigid);
rec_easyQR_affine = img2QR(straight_easy_affine);
rec_interQR_affine = img2QR(straight_inter_affine);
rec_hardQR_afiine = img2QR(straight_hard_affine);
rec_easyQR_projective = img2QR(straight_easy_projective);
rec_interQR_projective = img2QR(straight_inter_projective);
rec_hardQR_projective = img2QR(straight_hard_projective);

%% 2.7
rec_ID_from_easyQR_rigid = QR2ID(rec_easyQR_rigid);
rec_ID_from_interQR_rigid = QR2ID(rec_interQR_rigid);
rec_ID_from_hardQR_rigid = QR2ID(rec_hardQR_rigid);
rec_ID_from_easyQR_affine = QR2ID(rec_easyQR_affine);
rec_ID_from_interQR_affine = QR2ID(rec_interQR_affine);
rec_ID_from_hardQR_afiine = QR2ID(rec_hardQR_afiine);
rec_ID_from_easyQR_projective = QR2ID(rec_easyQR_projective);
rec_ID_from_interQR_projective = QR2ID(rec_interQR_projective);
rec_ID_from_hardQR_projective = QR2ID(rec_hardQR_projective);


%% Q3
QR_easy = imread("QReasy.jpg");
easy_corners_auto = find_corners(QR_easy);
% inter_corners_auto = find_corners(QR_inter);

 %% Q3 - repet Q2 
 QR_easy = imread("QReasy.jpg");
 QR_easy = img_norm(QR_easy);
 x1_Q3= easy_corners_auto(:,1);
 y1_Q3= easy_corners_auto(:,2);

xfixed_Q3=[1200; 1; 1200; 1];
yfixed_Q3=[1; 1; 1200; 1200];

% 2.4
straight_easy_rigid_Q3 = straight_image(QR_easy,[x1_Q3 y1_Q3],[xfixed_Q3 yfixed_Q3],'nonreflectivesimilarity','Straight easyQR with transformation ');
straight_easy_affine_Q3 = straight_image(QR_easy,[x1_Q3 y1_Q3],[xfixed_Q3 yfixed_Q3],'affine','Straight easyQR with transformation ');
straight_easy_projective_Q3 = straight_image(QR_easy,[x1_Q3 y1_Q3],[xfixed_Q3 yfixed_Q3],'projective','Straight easyQR with transformation ');

% 2.6
rec_easyQR_rigid_Q3 = img2QR(straight_easy_rigid_Q3);
rec_easyQR_affine_Q3 = img2QR(straight_easy_affine_Q3);
rec_easyQR_projective_Q3 = img2QR(straight_easy_projective_Q3);

% 2.7
rec_ID_from_easyQR_rigid_Q3 = QR2ID(rec_easyQR_rigid_Q3);
rec_ID_from_easyQR_affine_Q3 = QR2ID(rec_easyQR_affine_Q3);
rec_ID_from_easyQR_projective_Q3 = QR2ID(rec_easyQR_projective_Q3);



%% Functions

function corners = find_corners(gray_img)
    gray_img = img_norm(gray_img);
    figure();
    imshow(gray_img)
    title("Original image normalized gray scale")
    gray_img((gray_img>0.2) & (gray_img<0.9)) = 1;  
    figure();
    imshow(gray_img) 
    sgtitle('Original image')
    title("Values in range of [0.2 0.9] turn white")

    bin_gray_img = imbinarize(gray_img);                
    figure()
    imshow(bin_gray_img)  
    sgtitle('Original image')
    title("Binary image")

    BW = edge(bin_gray_img,'canny');                    
    figure()
    imshow(BW);
    title("Canny Edge Detector on the Binary image")
    
    [H,theta,rho] = hough(BW);                       

    P = houghpeaks(H,8,'threshold',ceil(0.3*max(H(:))));

    lines = houghlines(BW,theta,rho,P,'MinLength',100);   

    figure, imshow(gray_img), hold on
    for k = 1:length(lines)
       xy = [lines(k).point1; lines(k).point2];
       plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');    
       plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
       plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');

    end
    title("All the detected lines")

    
    neg_tetha_min_rho = [0 1000]; %tetha,rho
    neg_tetha_max_rho = [0 -1000];
    pos_tetha_min_rho = [0 1000];
    pos_tetha_max_rho = [0 -1000];
    
    neg_tetha_min_rho_points = zeros(2,2); %tetha,rho,point1,point2
    neg_tetha_max_rho_points = zeros(2,2); 
    pos_tetha_min_rho_points = zeros(2,2); 
    pos_tetha_max_rho_points = zeros(2,2); 
    
    for k = 1:length(lines)
        curr_rho = lines(k).rho;
        curr_tetha = lines(k).theta;
        if curr_tetha<0 
            if curr_rho>neg_tetha_max_rho(1,2)   
                neg_tetha_max_rho(1,1) = curr_tetha;
                neg_tetha_max_rho(1,2) = curr_rho;
                neg_tetha_max_rho_points(1,:) = lines(k).point1;
                neg_tetha_max_rho_points(2,:) = lines(k).point2;
            end
            if curr_rho<neg_tetha_min_rho(1,2)
                neg_tetha_min_rho(1,1) = curr_tetha;
                neg_tetha_min_rho(1,2) = curr_rho;                
                neg_tetha_min_rho_points(1,:) = lines(k).point1;
                neg_tetha_min_rho_points(2,:) = lines(k).point2;
            end
        elseif curr_tetha>0 
            if curr_rho>pos_tetha_max_rho(1,2)
                pos_tetha_max_rho(1,1) = curr_tetha;
                pos_tetha_max_rho(1,2) = curr_rho;
                pos_tetha_max_rho_points(1,:) = lines(k).point1;
                pos_tetha_max_rho_points(2,:) = lines(k).point2;
            end
            if curr_rho<pos_tetha_min_rho(1,2)
                pos_tetha_min_rho(1,1) = curr_tetha;
                pos_tetha_min_rho(1,2) = curr_rho;                  
                pos_tetha_min_rho_points(1,:) = lines(k).point1;
                pos_tetha_min_rho_points(2,:) = lines(k).point2;
            end
        end
    end
    
    figure()
    imshow(gray_img)
    hold on
    plot(pos_tetha_min_rho_points(:,1),pos_tetha_min_rho_points(:,2),'LineWidth',2,'Color','green');
    plot(pos_tetha_max_rho_points(:,1),pos_tetha_max_rho_points(:,2),'LineWidth',2,'Color','green');
    plot(neg_tetha_min_rho_points(:,1),neg_tetha_min_rho_points(:,2),'LineWidth',2,'Color','green');
    plot(neg_tetha_max_rho_points(:,1),neg_tetha_max_rho_points(:,2),'LineWidth',2,'Color','green');
    title("The final 4 lines")
    
    neg_tetha_min_rho_ma = zeros(1,2);    
    neg_tetha_max_rho_ma = zeros(1,2);  
    pos_tetha_min_rho_ma = zeros(1,2);  
    pos_tetha_max_rho_ma = zeros(1,2); 

    
    neg_tetha_min_rho_ma(1,1) = (neg_tetha_min_rho_points(2,2) - neg_tetha_min_rho_points(1,2))/(neg_tetha_min_rho_points(2,1) - neg_tetha_min_rho_points(1,1));
    neg_tetha_min_rho_ma(1,2) = neg_tetha_min_rho_points(2,2)-neg_tetha_min_rho_ma(1,1)*neg_tetha_min_rho_points(2,1);
    neg_tetha_max_rho_ma(1,1) = (neg_tetha_max_rho_points(2,2) - neg_tetha_max_rho_points(1,2))/(neg_tetha_max_rho_points(2,1) - neg_tetha_max_rho_points(1,1));
    neg_tetha_max_rho_ma(1,2) = neg_tetha_max_rho_points(2,2)-neg_tetha_max_rho_ma(1,1)*neg_tetha_max_rho_points(2,1);    
    pos_tetha_min_rho_ma(1,1) = (pos_tetha_min_rho_points(2,2) - pos_tetha_min_rho_points(1,2))/(pos_tetha_min_rho_points(2,1) - pos_tetha_min_rho_points(1,1));
    pos_tetha_min_rho_ma(1,2) = pos_tetha_min_rho_points(2,2)-pos_tetha_min_rho_ma(1,1)*pos_tetha_min_rho_points(2,1);    
    pos_tetha_max_rho_ma(1,1) = (pos_tetha_max_rho_points(2,2) - pos_tetha_max_rho_points(1,2))/(pos_tetha_max_rho_points(2,1) - pos_tetha_max_rho_points(1,1));
    pos_tetha_max_rho_ma(1,2) = pos_tetha_max_rho_points(2,2)-pos_tetha_max_rho_ma(1,1)*pos_tetha_max_rho_points(2,1);    
    
    
    m_list = [neg_tetha_min_rho_ma(1,1) neg_tetha_max_rho_ma(1,1) pos_tetha_min_rho_ma(1,1) pos_tetha_max_rho_ma(1,1)];
    a_list = [neg_tetha_min_rho_ma(1,2) neg_tetha_max_rho_ma(1,2) pos_tetha_min_rho_ma(1,2) pos_tetha_max_rho_ma(1,2)];

    [m_list,idx]=sort(m_list,'descend');
    a_list=a_list(idx);

    corners = zeros(4,2);

    for i=1:2
        m = m_list(i);
        a = a_list(i);
        m1 = m_list(3);
        a1 = a_list(3);   
        m2 = m_list(4);
        a2 = a_list(4);  
        x_point = (a1-a)/(m-m1);
        y_point = m*x_point + a;
        corners(i,1) = x_point;
        corners(i,2) = y_point;
        x_point2 = (a2-a)/(m-m2);
        y_point2 = m*x_point2 + a;
        corners(i+2,1) = x_point2;
        corners(i+2,2) = y_point2;
    end

    corners = round(corners);
    
    figure()
    imshow(gray_img)
    hold on;
    for i=1:4
        plot(corners(i,1),corners(i,2),'x','LineWidth',2,'Color','red');
        hold on;
    end
    title("4 corners")
    hold off;
end



function ID = QR2ID(QR)
ID = zeros(9,1);
matrix2vec = QR(:);
for i = 1:9
    bin_num = matrix2vec(1+(i-1)*4:(i*4));
    ID(i) = bi2de(bin_num.','left-msb');
end
end

function bin = img2QR(img)         
bin = zeros(6);   
for i=1:6
    for j=1:6
        temp = img(1+(i-1)*200:(i*200), 1+(j-1)*200:(j*200));
        bin(i,j) = mean(temp(:));
    end
end
bin(bin>0.5) = 1;
bin(bin<=0.5) = 0;
end


function straight_image = straight_image(img,points,points_fix,trans,str)
trans_image = fitgeotrans(points,points_fix,trans);
straight_image = imwarp(img, trans_image, 'OutputView', imref2d([1200 1200]));
figure;
imshow(straight_image); title([str , trans] );
end

function norm_img = img_norm(img)
norm_img = double(rgb2gray(img));
norm_img = (norm_img - min(norm_img(:)))/(max(norm_img(:)) - min(norm_img(:)));
end
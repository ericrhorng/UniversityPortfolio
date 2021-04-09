close all
clc

Image1=imread('im1.jpg');
Image2=imread('im3.jpg');
Image3=[Image1 Image2];
Sift1=importdata('im1.sift');
Sift2=importdata('im3.sift');
ClosestMatch=ones(length(Sift1),1);

figure;
title('Image 1 and 3')
subplot(2,1,1)
imshow(Image3)
hold on
plot(Sift1(:,1),Sift1(:,2),'rx')
plot(Sift2(:,1)+640,Sift2(:,2),'rx')

%Find Closest Match
for x=1:length(Sift1)
    KeyPoint1=Sift1(x,5:end);
    DistanceArray=linspace(1,length(Sift2),length(Sift2))';
    for y=1:length(Sift2)
        DistanceArray(y)=norm(KeyPoint1-Sift2(y,5:end));
    end
    [Distances, Indexes]=sort(DistanceArray);
    if (Distances(1)/Distances(2))<0.5
        ClosestMatch(x)=Indexes(1);
    else
        ClosestMatch(x)=0;
    end
end
Image3=[Image1 Image2];
subplot(2,1,2)
imshow(Image3);
%Draw some lines
for x=1:length(ClosestMatch) %length(ClosestMatch)
   if ClosestMatch(x)~=0
       X1=round(Sift1(x,1));
       X2=round(Sift2(ClosestMatch(x),1));
       Y1=round(Sift1(x,2));
       Y2=round(Sift2(ClosestMatch(x),2));
       
       L=line([X1 X2+640],[Y1,Y2]);
       set(L,'Color','green')
   end
end

%%
Image1=imread('im1.jpg');
Image2=imread('im4.jpg');
Image3=[Image1 Image2];
Sift1=importdata('im1.sift');
Sift2=importdata('im4.sift');
ClosestMatch=ones(length(Sift1),1);

figure;
subplot(2,1,1)
imshow(Image3)
hold on
plot(Sift1(:,1),Sift1(:,2),'rx')
plot(Sift2(:,1)+640,Sift2(:,2),'rx')

%Find Closest Match
for x=1:length(Sift1)
    KeyPoint1=Sift1(x,5:end);
    DistanceArray=linspace(1,length(Sift2),length(Sift2))';
    for y=1:length(Sift2)
        DistanceArray(y)=norm(KeyPoint1-Sift2(y,5:end));
    end
    [Distances, Indexes]=sort(DistanceArray);
    if (Distances(1)/Distances(2))<0.5
        ClosestMatch(x)=Indexes(1);
    else
        ClosestMatch(x)=0;
    end
end
Image3=[Image1 Image2];
title('Image 1 and 4')
subplot(2,1,2)
imshow(Image3);
%Draw some lines
for x=1:length(ClosestMatch) %length(ClosestMatch)
   if ClosestMatch(x)~=0
       X1=round(Sift1(x,1));
       X2=round(Sift2(ClosestMatch(x),1));
       Y1=round(Sift1(x,2));
       Y2=round(Sift2(ClosestMatch(x),2));
       
       L=line([X1 X2+640],[Y1,Y2]);
       set(L,'Color','green')
   end
end

close all
clc

Image1=imread('im1.jpg');
Image2=imread('im2.jpg');
Image3=[Image1 Image2];
Sift1=importdata('im1.sift');
Sift2=importdata('im2.sift');

%{
for x=1:length(Sift1)
    %For Image 1
    XCoord=round(Sift1(x,1));
    YCoord=round(Sift1(x,2));
    Image3(YCoord-2:YCoord+2,XCoord,1)=255;
    Image3(YCoord:YCoord,XCoord-2:XCoord+2,1)=255;
    %For Image 2
    XCoord=round(Sift2(x,1))+640;%Offset
    YCoord=round(Sift2(x,2));
    Image3(YCoord-2:YCoord+2,XCoord,1)=255;
    Image3(YCoord:YCoord,XCoord-2:XCoord+2,1)=255;
    
end
%}

imshow(Image3);
hold on
plot(Sift1(:,1),Sift1(:,2),'rx')
plot(Sift2(:,1)+640,Sift2(:,2),'rx')
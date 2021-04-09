clc
close all

LeftImage=imread('left.jpg');
RightImage=imread('right.jpg');

%Create Large Image
FinalImage=LeftImage; %Automatically scale to avoid black bars
H=[1.6010 -0.0300 -317.9341; 0.1279 1.5325 -22.5847; 0.0007 0 1.2865];

%Get difference in brightness
%======================================================================
%Get points from both images, find the percentage difference and scale the
%pixel values
%======================================================================
Points=[506 508 505 506 507 503 503 503 503 503 503 492 492 492;8 32 79 235 331 42 111 65 300 150 240 65 38 13;1 1 1 1 1 1 1 1 1 1 1 1 1 1];
PointsRight=H*Points;
for x=1:size(Points,2)
PointsRight(1:2,x)=PointsRight(1:2,x)/PointsRight(3,x);
end
BiIntResults=zeros(1,size(Points,2));
for x=1:size(Points,2)
   CurrentPixelX=PointsRight(1,x);
   CurrentPixelY=PointsRight(2,x);
   LeftU=double(RightImage(floor(CurrentPixelY),floor(CurrentPixelX)));
   RightU=double(RightImage(floor(CurrentPixelY),ceil(CurrentPixelX)));
   LeftL=double(RightImage(ceil(CurrentPixelY),floor(CurrentPixelX)));
   RightL=double(RightImage(ceil(CurrentPixelY),ceil(CurrentPixelX)));
   BiIntResults(x)=BiInt(CurrentPixelX,CurrentPixelY,LeftU,RightU,LeftL,RightL);
end

PercentageDifference=[];
for x=1:size(Points,2)
   LeftPoint=double(LeftImage(Points(2,x),Points(1,x)));
   RightPoint=BiIntResults(x);
   PercentageDifference(x)=LeftPoint/RightPoint;
end

%RightImage=RightImage*mean(PercentageDifference); 
RightImage=RightImage*0.9;
LeftImage=imadjust(LeftImage);
RightImage=RightImage-7;

%Perform Homograpgy and stitching
%================================================================
for x=380:1024
    for y=1:384
        RightCoord=H*[x;y;1];
        RightCoord(1:2)=RightCoord(1:2)/RightCoord(3);        
        
        if RightCoord(1)<=512 && RightCoord(1)>=1 && RightCoord(2)<=384 && RightCoord(2)>=1
            CurrentPixelX=RightCoord(1);
            CurrentPixelY=RightCoord(2);
            LeftU=RightImage(floor(CurrentPixelY),floor(CurrentPixelX));
            RightU=RightImage(floor(CurrentPixelY),ceil(CurrentPixelX));
            LeftL=RightImage(ceil(CurrentPixelY),floor(CurrentPixelX));
            RightL=RightImage(ceil(CurrentPixelY),ceil(CurrentPixelX));
            
            Intensity=BiInt(CurrentPixelX,CurrentPixelY,LeftU,RightU,LeftL,RightL);
            FinalImage(y,x)=uint8(Intensity);
        end
    end
end

%3x3 Gaussian Blur the seam
%Gaussian=[1/16 1/8 1/16; 1/8 1/4 1/8; 1/16 1/8 1/16];
Gaussian=[1/4 1/2 1/4];
%Gaussian=(1/159)*[2 4 5 4 2; 4 9 12 9 4; 5 12 15 12 5; 4 9 12 9 4; 2 4 5 4 2];

%FinalImage(3:382,502:522)=conv2(FinalImage(1:382,501:523),Gaussian,'valid');
FinalImage=imadjust(FinalImage);

%Crop image a bit to remove residual black edges
FinalImage=FinalImage(3:381,1:780);
imshow(FinalImage)
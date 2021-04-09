clc
close all

LeftImage=imread('left.jpg');
RightImage=imread('right.jpg');

CrossSize=5;
ColorLeft=repmat(LeftImage,[1 1 3]);
ColorRight=repmat(RightImage,[1 1 3]);

%Perform Homography Transform
H=[1.6010 -0.0300 -317.9341; 0.1279 1.5325 -22.5847; 0.0007 0 1.2865];
Points=[338 468 253 263 242;197 290 170 256 136;1 1 1 1 1];
PointsRight=H*Points;

%Transform to rounded 2D pixel Coordinates for crosshairs
for x=1:5
PointsRight(1:2,x)=PointsRight(1:2,x)/PointsRight(3,x);
end

BiIntResults=zeros(1,5);
for x=1:5
   CurrentPixelX=PointsRight(1,x);
   CurrentPixelY=PointsRight(2,x);
   LeftU=double(RightImage(floor(CurrentPixelY),floor(CurrentPixelX)));
   RightU=double(RightImage(floor(CurrentPixelY),ceil(CurrentPixelX)));
   LeftL=double(RightImage(ceil(CurrentPixelY),floor(CurrentPixelX)));
   RightL=double(RightImage(ceil(CurrentPixelY),ceil(CurrentPixelX)));
   
   BiIntResults(x)=BiInt(CurrentPixelX,CurrentPixelY,LeftU,RightU,LeftL,RightL);
end
BiIntResults


PointsRight=round(PointsRight);
%Draw Crosshairs
for x=1:5
   ColorRight(PointsRight(2,x),PointsRight(1,x)-CrossSize:PointsRight(1,x)+CrossSize,1)=256; 
   ColorRight(PointsRight(2,x)-CrossSize:PointsRight(2,x)+CrossSize,PointsRight(1,x),1)=256;
   ColorLeft(Points(2,x),Points(1,x)-CrossSize:Points(1,x)+CrossSize,1)=256; 
   ColorLeft(Points(2,x)-CrossSize:Points(2,x)+CrossSize,Points(1,x),1)=256;
end

subplot(1,2,1)
imshow(ColorLeft)
subplot(1,2,2)
imshow(ColorRight)
close all
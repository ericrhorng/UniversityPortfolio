clc
close all

LeftImage=imread('left.jpg');
RightImage=imread('right.jpg');
CrossSize=5;
ColorLeft=repmat(LeftImage,[1 1 3]);

Points=[338 468 253 263 242;197 290 170 256 136;1 1 1 1 1];

%Draw Crosshairs
for x=1:5
   ColorLeft(Points(2,x),Points(1,x)-CrossSize:Points(1,x)+CrossSize,1)=256; 
   ColorLeft(Points(2,x)-CrossSize:Points(2,x)+CrossSize,Points(1,x),1)=256;
end

imshow(ColorLeft)
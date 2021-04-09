clc
close all

LeftImage=imread('left.jpg');
RightImage=imread('right.jpg');

%Create Large Image
FinalImage=uint8(zeros(384,1024));
FinalImage(1:384,1:512)=LeftImage;
H=[1.6010 -0.0300 -317.9341; 0.1279 1.5325 -22.5847; 0.0007 0 1.2865];

for x=513:1024
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

imshow(FinalImage)
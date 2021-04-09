clear all
close all
clc

TestImage=imread('test05.jpg');
Gaussian=(1/159)*[2 4 5 4 2; 4 9 12 9 4; 5 12 15 12 5; 4 9 12 9 4; 2 4 5 4 2];

ImageDim=size(TestImage);
GradientOri=zeros(ImageDim(1:2));
Canny=zeros(ImageDim(1:2));

TestImage=My_Conv(TestImage,Gaussian);

%Find X and Y derivatives and calculate magnitude
SobelX=1/8*[-1 0 1; -2 0 2; -1 0 1];
SobelY=1/8*[-1 -2 -1; 0 0 0; 1 2 1];
GradientX=My_Conv(TestImage,SobelX);
GradientY=My_Conv(TestImage,SobelY);
GradientMag=sqrt(GradientX.^2+GradientY.^2);

%Calculate Orientation
for x = 1:size(TestImage,1)
    for y=1:size(TestImage,2)
        Temp=atan(GradientX(x,y)./GradientY(x,y));
        Temp=Temp*(180/pi);
        if Temp<0
            Temp=Temp+360;
        end
        Temp=floor(Temp/45)*45;
        
        if isfinite(Temp)
            GradientOri(x,y)=Temp;
        else
            GradientOri(x,y)=0;
        end
    end
end

%Non-Maxima Suppression
for x=1:size(TestImage,1)
    for y=1:size(TestImage,2)
        AdjVector=GetAdj(x,y,GradientOri(x,y));
        if (AdjVector(1)>0) && (AdjVector(2)>0) && (AdjVector(1)<ImageDim(1)) && (AdjVector(2)<ImageDim(2))
            if GradientMag(x,y)>GradientMag(AdjVector(1),AdjVector(2))
                GradientMag(AdjVector(1),AdjVector(2))=0;
            end
        end
        
        if (AdjVector(3)>0) && (AdjVector(4)>0) && (AdjVector(3)<ImageDim(1)) && (AdjVector(4)<ImageDim(2))
            %AdjVector
            if GradientMag(x,y)>GradientMag(AdjVector(3),AdjVector(4))
                GradientMag(AdjVector(3),AdjVector(4))=0;
            end
        end
        
    end
end

GradientMag(GradientMag>8)=255;
GradientMag(GradientMag<15)=0;


imshow(uint8(GradientMag),[])
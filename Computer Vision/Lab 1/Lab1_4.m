clear all
close all
clc

TestImage=imread('test05.jpg');
Gaussian=(1/159)*[2 4 5 4 2; 4 9 12 9 4; 5 12 15 12 5; 4 9 12 9 4; 2 4 5 4 2];

ImageDim=size(TestImage);
GradientOri=zeros(ImageDim(1:2));
OriColorCoded=zeros(ImageDim(1),ImageDim(2),3);

TestImage=My_Conv(TestImage,Gaussian);

%Find X and Y derivatives and calculate magnitude
SobelX=1/8*[-1 0 1; -2 0 2; -1 0 1];
SobelY=1/8*[-1 -2 -1; 0 0 0; 1 2 1];
GradientX=My_Conv(TestImage,SobelX);
GradientY=My_Conv(TestImage,SobelY);
GradientMag=sqrt(GradientX.^2+GradientY.^2);


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
        
        %Color coding orientations
        switch GradientOri(x,y)
            case 0
                OriColorCoded(x,y,1)=255;
            case 45
                OriColorCoded(x,y,2)=255;
                OriColorCoded(x,y,3)=255;
            case 90
                OriColorCoded(x,y,1)=255;
                OriColorCoded(x,y,3)=255;
            case 135
                OriColorCoded(x,y,1)=255;
                OriColorCoded(x,y,2)=255;
            case 180
                OriColorCoded(x,y,2)=255;
                OriColorCoded(x,y,3)=255;
            case 225
                OriColorCoded(x,y,3)=255;
                OriColorCoded(x,y,1)=255;
            case 270
                OriColorCoded(x,y,1)=255;
                OriColorCoded(x,y,2)=51;
                OriColorCoded(x,y,3)=153;
            case 315    
                OriColorCoded(x,y,2)=255;
                
        end
    end
end


imshow(uint8(GradientMag),[])
figure;
imshow(uint8(OriColorCoded))
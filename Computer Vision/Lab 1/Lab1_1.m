clear all
close all
clc

TestImage=imread('test05.jpg');

Gaussian=(1/159)*[2 4 5 4 2; 4 9 12 9 4; 5 12 15 12 5; 4 9 12 9 4; 2 4 5 4 2];

ImageDim=size(TestImage);
FinalImage=zeros(ImageDim(1:2));
imshow(TestImage)


%Go through every pixel
for y = 1:ImageDim(2)
    for x=1:ImageDim(1)
        Sum=0;
        %Perform Convolution
        for j=1:size(Gaussian,1)
            for i=1:size(Gaussian,2)
                %Convert Gaussian to Kernal Coordinates
                iNew=i-ceil(size(Gaussian,2)/2);
                jNew=j-ceil(size(Gaussian,1)/2);
                %Make sure indexes are within range
                if ((x-iNew)> 0 && (x-iNew) < ImageDim(1))
                    if  ((y-jNew)> 0 && (y-jNew)< ImageDim(2))
                        %[x iNew x-iNew y jNew y-jNew]
                        Sum=Sum+TestImage(x-iNew,y-jNew)*Gaussian(i,j);
                    end
                end
            end
        end
        FinalImage(x,y)=Sum;
    end
end

%{
figure;
FinalImageConv=conv2(TestImage,Gaussian,'same');
imshow(uint8(FinalImageConv))
%}

%Show result
figure;
imshow(uint8(FinalImage),[])

function Convolution=My_Conv(Image,Mask)

ImageDim=size(Image);
Convolution=zeros(ImageDim(1:2));

%Go through every pixel
for y = 1:ImageDim(2)
    for x=1:ImageDim(1)
        Sum=0;
        %Perform Convolution
        for j=1:size(Mask,1)
            for i=1:size(Mask,2)
                %Convert Gaussian to Kernal Coordinates
                iNew=i-ceil(size(Mask,2)/2);
                jNew=j-ceil(size(Mask,1)/2);
                %Make sure indexes are within range
                if ((x-iNew)> 0 && (x-iNew) < ImageDim(1))
                    if  ((y-jNew)> 0 && (y-jNew)< ImageDim(2))
                        %[x iNew x-iNew y jNew y-jNew]
                        Sum=Sum+Image(x-iNew,y-jNew)*Mask(i,j);
                    end
                end
            end
        end
        Convolution(x,y)=Sum;
    end
end

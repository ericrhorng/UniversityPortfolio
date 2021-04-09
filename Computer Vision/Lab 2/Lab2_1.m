close all
clc

Image=imread('mandrill.jpg');
k=4;

rng('shuffle');
RandomIdx=randi(size(Image,1),1,k);%Get Random Row
RandomIdy=randi(size(Image,2),1,k); %Get Random Column
NewMean=impixel(Image,RandomIdy,RandomIdx)

Label=zeros(size(Image,1),size(Image,2));
Distance=zeros(1,k);
ExitFlag=0;

while(ExitFlag==0)
    %Find closest mean
    for x=1:size(Image,1)
        for y=1:size(Image,2)
            for i=1:k
                Distance(i)=sqrt((NewMean(i,1)-double(Image(x,y,1)))^2+(NewMean(i,2)-double(Image(x,y,2)))^2+(NewMean(i,3)-double(Image(x,y,3)))^2);
            end
            [ClosestMean, ClosestIdx]=min(Distance);
            Label(x,y)=ClosestIdx;
        end
    end
    
    %Sum up all close RGB values
    for x=1:size(Image,1)
        for y=1:size(Image,2)
            NewMean(Label(x,y),1)=NewMean(Label(x,y),1)+double(Image(x,y,1));
            NewMean(Label(x,y),2)=NewMean(Label(x,y),2)+double(Image(x,y,2));
            NewMean(Label(x,y),3)=NewMean(Label(x,y),3)+double(Image(x,y,3));
        end
    end
    
    %Divide by number of close RGB values
    for i=1:k
        NewMean(i,:)=NewMean(i,:)/sum(sum(Label==i));
    end
    
    %Assign new color means to pixels
    NewImage=zeros(size(Image));
    for x=1:size(NewImage,1)
        for y=1:size(NewImage,2)
            NewImage(x,y,:)=NewMean(Label(x,y),:);
        end
    end
    
    NewMean
    
    imshow(uint8(NewImage))
    In=input('Press Enter to continue else press 1: ');
    if In==1
       ExitFlag=1; 
    end
end

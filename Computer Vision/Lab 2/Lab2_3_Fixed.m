close all
clc

Image=imread('mandrill.jpg');
k=10;

rng('shuffle');
RandomIdx=randi(size(Image,1),1,k);%Get Random Row
RandomIdy=randi(size(Image,2),1,k); %Get Random Column
NewMean=impixel(Image,RandomIdy,RandomIdx); %Get First Random Means

Label=zeros(size(Image,1),size(Image,2));
Distance=zeros(1,k);
ExitFlag=0;
History=NewMean;
Iterations=0;

while(ExitFlag==0)
    Iterations=Iterations+1;
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
    History=[History ; NewMean];
    
    %Plot Mandrill color data
    figure(1)
    subplot(1,2,1)
    RedScatter=reshape(Image(:,1:100:end,1),[],1);
    GreenScatter=reshape(Image(:,1:100:end,2),[],1);
    BlueScatter=reshape(Image(:,1:100:end,3),[],1);
    InitialScatter=scatter3(RedScatter,GreenScatter,BlueScatter,3,'filled');
    InitialScatter.CData=uint8([RedScatter GreenScatter BlueScatter]);
    xlabel('Red')
    ylabel('Green')
    zlabel('Blue')
    hold on
    
    %Plot Mean color data
    subplot(1,2,1)
    RedScatter=reshape(NewImage(:,1:100:end,1),[],1);
    GreenScatter=reshape(NewImage(:,1:100:end,2),[],1);
    BlueScatter=reshape(NewImage(:,1:100:end,3),[],1);
    Iteration_Scatter=scatter3(RedScatter,GreenScatter,BlueScatter,100,'filled','d','MarkerEdgeColor',[0 0 0]);
    Iteration_Scatter.CData=uint8([RedScatter GreenScatter BlueScatter]);
    hold off
    
    subplot(1,2,2)
    imshow(uint8(NewImage))
    In=input('Press Enter to continue else press 1: ');
    if In==1
        ExitFlag=1;
    end
end

close all

%Generate Final scatter matrix with convergences
figure(3)
RedScatter=reshape(Image(:,1:100:end,1),[],1);
GreenScatter=reshape(Image(:,1:100:end,2),[],1);
BlueScatter=reshape(Image(:,1:100:end,3),[],1);
InitialScatter=scatter3(RedScatter,GreenScatter,BlueScatter,3,'filled');
InitialScatter.CData=uint8([RedScatter GreenScatter BlueScatter]);
hold on

RedScatter=reshape(History(:,1),[],1);
GreenScatter=reshape(History(:,2),[],1);
BlueScatter=reshape(History(:,3),[],1);
My_Scatter=scatter3(RedScatter,GreenScatter,BlueScatter,100,'filled','d','MarkerEdgeColor',[0 0 0]);
My_Scatter.CData=uint8([RedScatter GreenScatter BlueScatter]);
xlabel('Red')
ylabel('Green')
zlabel('Blue')

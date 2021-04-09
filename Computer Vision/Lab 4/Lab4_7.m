close all
clc

Image1=imread('im1.jpg');
Image2=imread('im4.jpg');
Image3=[Image1 Image2];
Sift1=importdata('im1.sift');
Sift2=importdata('im4.sift');
ClosestMatch=ones(length(Sift1),1);

%Find Closest Match
for x=1:length(Sift1)
    KeyPoint1=Sift1(x,5:end);
    DistanceArray=linspace(1,length(Sift2),length(Sift2))';
    for y=1:length(Sift2)
        DistanceArray(y)=norm(KeyPoint1-Sift2(y,5:end));
    end
    [Distances, Indexes]=sort(DistanceArray);
    if (Distances(1)/Distances(2))<0.5
        ClosestMatch(x)=Indexes(1);
    else
        ClosestMatch(x)=0;
    end
end

%Disparity for Image 2=2.6797
%Disparity for Image 3=6.6586
%Dispartiy for Image 4=9.8892

%Calculate Depth
ImageCentre=640/2; %U0
DepthData=zeros(length(Sift1),3);
Disparity=zeros(length(Sift1),1);
for x=1:length(Sift1)
    if (ClosestMatch(x)>0)
        u1=Sift1(x,1)-ImageCentre;
        u2=Sift2(ClosestMatch(x),1)-ImageCentre;
        Disparity(x)=abs(u1-u2);
        DepthData(x,3)=abs(9.8892/(u1-u2)); %Calculate Depth
        DepthData(x,1)=Sift1(x,1);
        DepthData(x,2)=Sift1(x,2);
    end
end

%Find all non-zero depths and disparities
DepthData=DepthData(~all(DepthData==0,2),:);
Disparity=Disparity(find(Disparity>0));

%Add calculated disparity onto Image 1 points
DepthData(:,1)=DepthData(:,1)+(6.6586/median(DepthData(:,3)));

imshow(imread('im3.jpg'))
hold on
plot(DepthData(:,1),DepthData(:,2),'rx')


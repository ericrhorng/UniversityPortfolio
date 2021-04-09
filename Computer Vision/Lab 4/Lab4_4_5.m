close all
clc

Image1=imread('im1.jpg');
Image2=imread('im2.jpg');
Image3=[Image1 Image2];
Sift1=importdata('im1.sift');
Sift2=importdata('im2.sift');
ClosestMatch=ones(length(Sift1),1);

%Find Closest Match
for x=1:length(Sift1)
    KeyPoint1=Sift1(x,5:end);
    DistanceArray=linspace(1,length(Sift2),length(Sift2))';
    for y=1:length(Sift2)
        DistanceArray(y)=norm(KeyPoint1-Sift2(y,5:end));
    end
    [Distances Indexes]=sort(DistanceArray);
    if (Distances(1)/Distances(2))<0.5
        ClosestMatch(x)=Indexes(1);
    else
        ClosestMatch(x)=0;
    end
end

imshow(Image3);
%Draw some lines
for x=1:length(ClosestMatch) %length(ClosestMatch)
   if ClosestMatch(x)~=0
       X1=(Sift1(x,1));
       X2=(Sift2(ClosestMatch(x),1));
       Y1=(Sift1(x,2));
       Y2=(Sift2(ClosestMatch(x),2));
       
       L=line([X1 X2+640],[Y1,Y2]);
       set(L,'Color','green')
   end
end
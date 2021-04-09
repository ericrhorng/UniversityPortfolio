clear all
close all
clc

TestImage=imread('test05.jpg');
Gaussian=(1/159)*[2 4 5 4 2; 4 9 12 9 4; 5 12 15 12 5; 4 9 12 9 4; 2 4 5 4 2];

TestImage=My_Conv(TestImage,Gaussian);

SobelX=1/8*[-1 0 1; -2 0 2; -1 0 1];
SobelY=1/8*[-1 -2 -1; 0 0 0; 1 2 1];

GradientX=My_Conv(TestImage,SobelX);
GradientY=My_Conv(TestImage,SobelY);

GradientMag=sqrt(GradientX.^2+GradientY.^2);


imshow(uint8(GradientX),[])
clear all
clc

%=========================Servo Specifications
MG90SWeight=13.4; %Grams
MG90STorque=2.2; %kg.cm

MG996RWeight=55;
MG996RTorque=11;

DS3218MGWeight=60;
DS3218MGTorque=21.5;
%==========================Weights
MinLoad=50; %Requirement
GripperWeight=50; %Gripper Assembly Estimated Weight
ServoBracket=50; %Plastic Mounting Bracket
%=========================Distance between pivots
BeamALength=27; %Length between MG90S wrist pivot and 15kg elbow pivot
BeamBLength=24; %Length between 15kg elbow pivot and 20kg shoulder pivot
%========================Weight of 3D printed beams
PlasticDensity=1.05; %g/cm^3
BeamWidth=5;
BeamHeight=3;
PercentageEmpty=0.5;
BeamAWeight=BeamALength*BeamWidth*BeamHeight*(1-PercentageEmpty);
BeamBWeight=BeamBLength*BeamWidth*BeamHeight*(1-PercentageEmpty);
%=========================Lengths
L1=1;
L2=1;
L3=4;
L4=BeamALength;
L5=BeamBLength;
%=========================Moments
%Torque at each pivot/servo
T_MG90SGripper=MinLoad*L1;
T_Wrist_1=(MinLoad*(L1+L2))+(GripperWeight)*L2;
T_Wrist_2=(MinLoad*(L1+L2+L3))+(GripperWeight)*(L2+L3)+(MG90SWeight+ServoBracket)*L3;
T_Elbow=(MinLoad*(L1+L2+L3+BeamALength))+(GripperWeight)*(L2+L3+BeamALength)+(MG90SWeight+ServoBracket)*(L3+BeamALength)+(MG90SWeight+ServoBracket)*BeamALength+BeamAWeight*(BeamALength/2);
T_Shoulder=(MinLoad*(L1+L2+L3+BeamALength+BeamBLength))+(GripperWeight)*(L2+L3+BeamALength+BeamBLength)+(MG90SWeight+ServoBracket)*(L3+BeamALength+BeamBLength)+(MG90SWeight+ServoBracket)*(BeamALength+BeamBLength)+(BeamAWeight*((BeamALength/2)+BeamBLength))+MG996RWeight*BeamBLength+BeamBWeight*(BeamBLength/2);

%Converting to kg.cm
T_Wrist_1=T_Wrist_1/1000
T_Wrist_2=T_Wrist_2/1000
T_Elbow=T_Elbow/1000
T_Shoulder=T_Shoulder/1000

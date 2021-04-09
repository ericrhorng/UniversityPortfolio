function Result=BiInt(OriPixelX,OriPixelY,LeftU,RightU,LeftL,RightL)
Int1=(ceil(OriPixelX)-OriPixelX)/(ceil(OriPixelX)-floor(OriPixelX));
Int2=(OriPixelX-floor(OriPixelX))/(ceil(OriPixelX)-floor(OriPixelX));

Temp1=Int1*LeftU + Int2*RightU;
Temp2=Int1*LeftL + Int2*RightL;

Result=((ceil(OriPixelY)-OriPixelY)/(ceil(OriPixelY)-floor(OriPixelY)))*Temp1 + (((OriPixelY-floor(OriPixelY))/(ceil(OriPixelY)-floor(OriPixelY)))*Temp2);
end
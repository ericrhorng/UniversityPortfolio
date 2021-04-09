function GetAdjacent=GetAdj(x,y,Ori)
Adj1X=0;
Adj1Y=0;
Adj2X=0;
Adj2Y=0;

switch Ori
    case 0
        Adj1X=x-1;
        Adj2X=x+1;
        Adj1Y=y;
        Adj2Y=y;
        
    case 45
        Adj1X=x-1;
        Adj2X=x+1;
        Adj1Y=y-1;
        Adj2Y=y+1;
        
    case 90
        Adj1X=x;
        Adj2X=x;
        Adj1Y=y-1;
        Adj2Y=y+1;
        
    case 135
        Adj1X=x+1;
        Adj2X=x-1;
        Adj1Y=y-1;
        Adj2Y=y+1;
        
    case 180
        Adj1X=x-1;
        Adj2X=x+1;
        Adj1Y=y;
        Adj2Y=y;
        
    case 225
        Adj1X=x-1;
        Adj2X=x+1;
        Adj1Y=y-1;
        Adj2Y=y+1;
        
    case 270
        Adj1X=x-1;
        Adj2X=x+1;
        Adj1Y=y;
        Adj2Y=y;
        
    case 315
        Adj1X=x+1;
        Adj2X=x-1;
        Adj1Y=y-1;
        Adj2Y=y+1;
end

GetAdjacent=[Adj1X Adj1Y Adj2X Adj2Y];
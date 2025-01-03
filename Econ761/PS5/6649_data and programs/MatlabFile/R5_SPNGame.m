function Sim= R5_SPNGame(const,XKM,XWM,XSM,nei_id,nei_dist,Sm78)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Solve the entry & network problem               %%%%%%%%%%%%%%%%%%%
%%% Using Km/Wm's exact best response
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% const.dpar=[dkw;dk;dks;  dwk;dw;dws;  dsk;dsw;dss];
% const.N: sample size;
% const.NR: number of repetitions
% const.maxsm: maximum number of small stores
% Sm78 is the model predicted no of small stores in 78, which contains info
% about unobserved market error (epsilon_m) in 78

dpar=const.dpar; N=const.N; NR=const.NR; maxsm=const.maxsm;
oneRM=ones(1,maxsm);    %replicate columns

dkw=dpar(1); dk=dpar(2); dks=dpar(3);
dwk=dpar(4); dw=dpar(5); dws=dpar(6);
dsk=dpar(7); dsw=dpar(8);

Sim = zeros(N,10);
aa=15;
nk=15; nw=20;

for j=1:NR
    %%%%%%%%%%===========The equilibrium most profitable for Kmart =========%%%%%%%%%%%
    k=1;
    KU=ones(N,1); WL=zeros(N,1);
    
    tXSM=XSM(:,(j-1)*maxsm+1:j*maxsm);
    while k<aa
        K0=KU; W0=WL;
        
        %%% find out the number of small firms given WL/KU
        t=tXSM + dsk + dsw*WL(:,oneRM);      %conditioning on Km in
        Sm = sum(t>0,2);        %number of small stores if Km in
        XK = XKM(:,j) + dkw*WL + dks*log(Sm+1);
        KU= T63_SPNExactKU(dk,XK,KU,nei_id,nei_dist,nk);  %Use KU as starting pt for each iteration
        if K0==KU & k>1
            break
        end;

        t=tXSM + dsk*KU(:,oneRM) + dsw;      %conditioning on Wm in
        Sm = sum(t>0,2);        %number of small stores if Wm in
        XW = XWM(:,j) + dwk*KU + dws*log(Sm+1);
        WL= T63_SPNExactWL(dw,XW,WL,nei_id,nei_dist,nw);
        if W0==WL
            break
        end;
        k=k+1;
    end;
    
    %%% Refine the result, solving for Km/Wm's exact best response until converge
    k=1; KBR=KU; WBR=WL;
    while k<aa
        K0=KBR; W0=WBR;

        t=tXSM + dsk + dsw*WBR(:,oneRM);      %conditioning on Km in
        Sm = sum(t>0,2);        %number of small stores if Km in
        
        XK = XKM(:,j) + dkw*WBR + dks*log(Sm+1);
        KBR = T63_SPNKBR(dk,XK,KBR,nei_id,nei_dist,nk);
        if K0==KBR & k>1
            break
        end;
        
        t=tXSM + dsk*KBR(:,oneRM) + dsw;      %conditioning on Wm in
        Sm = sum(t>0,2);        %number of small stores if Wm in
        
        XW = XWM(:,j) + dwk*KBR + dws*log(Sm+1);
        WBR= T63_SPNWBR(dw,XW,WBR,nei_id,nei_dist,nw);
        if W0==WBR
            break
        end;
        k=k+1;
    end;
    
    %%% Solve for the number of small firms
    t=tXSM + dsk*KBR(:,oneRM) + dsw*WBR(:,oneRM);
    SimS = sum(t>0,2);        %small stores in if profit>0

    %%% Solve for the number of neighbors
    tmp=[KBR;0];    
    NeiK=tmp(nei_id);       
    NeiK=sum(NeiK.*nei_dist,2);

    tmp=[WBR;0];    
    NeiW=tmp(nei_id);       
    NeiW=sum(NeiW.*nei_dist,2);
    
    K0W1=(KBR==0).*WBR; SbSb=Sm78(:,j).* SimS; % covariance of Sm78 and Sm88
    tmp=[KBR, WBR, K0W1, SimS, KBR.*SimS, WBR.*SimS, K0W1.*SimS, NeiK, NeiW, SbSb];
    Sim=Sim+tmp;
end;
Sim=Sim/NR;

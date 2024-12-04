function [epoly2] = makeEPOLY2(epoly,cwidth)

% cwidth = 6;
% epoly2 = [epoly(40,:); epoly(39,:); mean(epoly([38 39],:))-[cwidth 0]; mean(epoly([21 22],:))-[cwidth 0]; epoly(21,:); epoly(24,:); epoly(23,:); mean(epoly([23 22],:))+[0 cwidth]; mean(epoly([30 29],:))+[0 cwidth]; epoly(29,:); epoly(28,:); epoly(27,:);...
%     mean(epoly([27 30],:))+[cwidth 0]; mean(epoly([33 34],:))+[cwidth 0]; epoly(33,:); epoly(32,:); epoly(31,:); mean(epoly([31 34],:))-[0 cwidth]; mean(epoly([38 37],:))-[0 cwidth]; epoly(37,:); epoly(36,:);...
%     NaN NaN; epoly(38,:); mean(epoly([38 39],:))+[cwidth 0]; mean(epoly([21 22],:))+[cwidth 0]; epoly(22,:); mean(epoly([23 22],:))-[0 cwidth]; mean(epoly([30 29],:))-[0 cwidth]; epoly(26,:);...
%     mean(epoly([27 30],:))-[cwidth 0]; mean(epoly([33 34],:))-[cwidth 0]; epoly(34,:); mean(epoly([31 34],:))+[0 cwidth]; mean(epoly([38 37],:))+[0 cwidth]; epoly(38,:)];

epoly2 = [epoly(40,:); % bottom left
        mean(epoly([40 39],:))-[0 cwidth]; mean(epoly([40 39],:))-[cwidth cwidth]; mean(epoly([40 39],:))-[cwidth -cwidth]; mean(epoly([40 39],:))-[0 -cwidth]; % Box D left fake door
    epoly(39,:); mean(epoly([38 39],:))-[cwidth 0]; mean(epoly([21 22],:))-[cwidth 0]; epoly(21,:); 
        mean(epoly([21 24],:))-[0 cwidth]; mean(epoly([21 24],:))-[cwidth cwidth]; mean(epoly([21 24],:))-[cwidth -cwidth]; mean(epoly([21 24],:))-[0 -cwidth]; % Box A left fake door
    epoly(24,:); 
        mean(epoly([24 23],:))-[cwidth 0]; mean(epoly([24 23],:))-[cwidth -cwidth]; mean(epoly([24 23],:))+[cwidth cwidth]; mean(epoly([24 23],:))+[cwidth 0]; % Box A top fake door
    epoly(23,:); mean(epoly([23 22],:))+[0 cwidth]; mean(epoly([30 29],:))+[0 cwidth]; epoly(29,:); 
        mean(epoly([29 28],:))-[cwidth 0]; mean(epoly([29 28],:))-[cwidth -cwidth]; mean(epoly([29 28],:))+[cwidth cwidth]; mean(epoly([29 28],:))+[cwidth 0]; % Box B top fake door
    epoly(28,:); 
        mean(epoly([27 28],:))+[0 cwidth]; mean(epoly([27 28],:))+[cwidth cwidth]; mean(epoly([27 28],:))-[-cwidth cwidth]; mean(epoly([27 28],:))-[0 cwidth]; % Box B right fake door
    epoly(27,:); mean(epoly([27 30],:))+[cwidth 0]; mean(epoly([33 34],:))+[cwidth 0]; epoly(33,:); 
        mean(epoly([33 32],:))+[0 cwidth]; mean(epoly([33 32],:))+[cwidth cwidth]; mean(epoly([33 32],:))-[-cwidth cwidth]; mean(epoly([33 32],:))-[0 cwidth]; % Box C right fake door
    epoly(32,:); 
        mean(epoly([31 32],:))+[cwidth 0]; mean(epoly([31 32],:))+[cwidth -cwidth]; mean(epoly([31 32],:))+[-cwidth -cwidth]; mean(epoly([31 32],:))+[-cwidth 0]; % Box C bottom fake door
    epoly(31,:); mean(epoly([31 34],:))-[0 cwidth]; mean(epoly([38 37],:))-[0 cwidth]; epoly(37,:); 
        mean(epoly([37 36],:))+[cwidth 0]; mean(epoly([37 36],:))+[cwidth -cwidth]; mean(epoly([37 36],:))+[-cwidth -cwidth]; mean(epoly([37 36],:))+[-cwidth 0]; % Box D bottom fake door
    epoly(36,:); 
    NaN NaN; 
    epoly(38,:); mean(epoly([38 39],:))+[cwidth 0]; mean(epoly([21 22],:))+[cwidth 0]; epoly(22,:); mean(epoly([23 22],:))-[0 cwidth]; mean(epoly([30 29],:))-[0 cwidth]; epoly(26,:); % inner section
        mean(epoly([27 30],:))-[cwidth 0]; mean(epoly([33 34],:))-[cwidth 0]; epoly(34,:); mean(epoly([31 34],:))+[0 cwidth]; mean(epoly([38 37],:))+[0 cwidth]; epoly(38,:)];



% 
% figure
% plot(epoly2(:,1),epoly2(:,2),'k')
% daspect([1 1 1])
% axis off tight xy
% set(gcf,'InvertHardCopy','off'); % this stops white lines being plotted black      
% set(gcf,'color','w'); % makes the background colour white




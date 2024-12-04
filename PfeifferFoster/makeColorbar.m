function c = makeColorbar(ax, ticks, ticklabels, label, height)
    origPos = ax.Position;
    if isempty(ticklabels)
        ticklabels = strsplit(num2str(ticks,2));
    end
    c = colorbar('eastoutside','Ticks',ticks,'Ticklabels',ticklabels, 'Fontsize', 10);
    c.Position = [origPos(1) + origPos(3), origPos(2) + origPos(4)*(1-height)/2, c.Position(3), origPos(4)*height];
    c.Label.String = label;
    c.Label.Position(1) = c.Label.Position(1) * 0.5;

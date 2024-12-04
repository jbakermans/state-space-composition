function loc = getHomeWell(pos, speed)
    % The home well position is different in each session, because the arena gets rebuilt each time
    % Find the home well from the peak of the dwellmap of when animal is immobile
    % Also exclude any positions within 10% of the border: some rats really like following walls
    include = speed < 2 & all(pos > 20,2) & all(pos < 180,2);
    dwellmap = getDwellmap(pos(include,:), 1);
    % Find maximum location in dwellmap
    [~, i] = max(dwellmap(:));
    [x,y] = ind2sub(size(dwellmap),i);
    x = x*2; y = y*2; 
    % Find distance to well for each location
    wellDist = sqrt(sum((pos - [x,y]).^2, 2));
    % Then return the home well as the average of immobile locations near the peak
    loc = mean(pos(include & wellDist < 2,:), 1);
    % If you want to plot the resulting location:
    % figure(); hold on; imagesc(dwellmap', 'XData', [1,199], 'YData', [1, 199]); plot(pos(include,1), pos(include,2), 'r'); scatter(loc(1), loc(2),'g');    
end
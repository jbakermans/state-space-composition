function loc = getWellLoc(pos, speed, id)
    % The home well position is different in each session, because the arena gets rebuilt each time
    % Set the seed location for the requested well
    seedLoc = 200/6*[mod(id-1, 6) + 0.5, floor(id/6)+0.5];
    seedDist = sqrt(sum((seedLoc - pos).^2,2));    
    % Only include posititions at very low speed and near the guessed distance of the well
    include = speed < 2 & seedDist < 10;
    dwellmap = getDwellmap(pos(include,:), 1);
    % Find maximum location in dwellmap
    [~, i] = max(dwellmap(:));
    [x,y] = ind2sub(size(dwellmap),i);
    x = x*2-1; y = y*2-1; 
    % Find distance to well for each location
    wellDist = sqrt(sum((pos - [x,y]).^2, 2));
    % Then return the home well as the average of immobile locations near the peak
    loc = mean(pos(speed < 2 & wellDist < 2,:), 1);
    % If you want to plot the resulting location:
    % figure(); hold on; imagesc(dwellmap', 'XData', [1,199], 'YData', [1, 199]); plot(pos(include,1), pos(include,2), 'r'); scatter(loc(1), loc(2),'g');    
end
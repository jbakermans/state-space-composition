function dwellmap = getDwellmap(pos, dt)
    % Calculate dwellmap: total time spent in each location
    % Multiply by dt so the values mean "total time spent here in seconds"
    dwellmap = histcounts2(pos(:, 1), pos(:, 2), 0:2:200, 0:2:200)*dt;
    dwellmap = imgaussfilt(dwellmap, 2); % 4cm smoothing, so 2 bins
    dwellmap(dwellmap < 0.01) = 0;
    % To plot the dwellmap with the behaviour that generated it (notice transpose):
    % figure(); hold on; imagesc(dwellmap', 'XData', [1,199], 'YData', [1, 199]); plot(pos(:,1), pos(:,2), 'r');
    % That means finding a location's coordinates on dwell/ratemap looks like:
    % [m, i] = max(dwellmap(:)); [x,y] = ind2sub(size(dwellmap),i); x = x*2; y = y*2;
end

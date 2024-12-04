function [diffMap, diffPos] = getRippleRatemapChange(currRipple, ripples, cells, coords, doPlot, doExcludeJumps, doThreshold, currCells)
    % Final arguments are optional, set defaults here
    if nargin < 8
        doPlot = false;
        doExcludeJumps = false;
        doThreshold = false;
        currCells = 1:size(cells,1);
    end
    % Get times 
    pos = coords.behaviour.pos;
    speed = coords.behaviour.speed;
    time = coords.behaviour.time;
    dt = coords.behaviour.dt;
    times = coords.spikes.time;
    binWidth = coords.spikes.dt;
    % For this particular ripple: find dwell map before and after ripple
    beforePos = speed > 5 & time' < ripples.tpeak(currRipple) - 1;
    afterPos = speed > 5 & time' > ripples.tpeak(currRipple) + 1;
    dwellBefore = getDwellmap(pos(beforePos,:), dt);
    dwellAfter =  getDwellmap(pos(afterPos,:), dt);
    diffMap = cell(size(cells,1),1);
    % Optional argument: position of a spike, which defines location of red x
    if nargout > 1
        diffPos = cell(size(cells,1),1);
    end
    for currCell = currCells
        % For this particular cell: find spikes before and after ripple
        beforeSpikes = cells.spikes{currCell}.speed > 5 & cells.spikes{currCell}.spike < ripples.tpeak(currRipple) - 1;
        afterSpikes = cells.spikes{currCell}.speed > 5 & cells.spikes{currCell}.spike > ripples.tpeak(currRipple) + 1;
        duringSpikes = cells.spikes{currCell}.spike(cells.spikes{currCell}.spike > times(ripples.start(currRipple)) & cells.spikes{currCell}.spike < times(ripples.end(currRipple)));
        % Only continue if there were any spikes during this replay
        if isempty(duringSpikes)
            continue;
        end
        % Build ratemaps and create a map of where they change
        rateBefore = getRatemap(cells.spikes{currCell}.pos(beforeSpikes, :), dwellBefore);                
        rateAfter = getRatemap(cells.spikes{currCell}.pos(afterSpikes, :), dwellAfter);                        
        rateDiff = rateAfter - rateBefore;
        % Optionally only include positive rate map changes (new firing fields only) - may be biased
        if doThreshold
            rateDiff(rateDiff < 0) = 0;
        end
        % Then decode this ripple *without* this particular cell
        windowStep = round(5e-3 / binWidth);
        windowSize = round(20e-3 / binWidth);
        windowCenters = ripples.start(currRipple):windowStep:(ripples.end(currRipple)+windowStep);
        % Create ratemaps and spike counts *without* this cell
        currRatemaps = permute(cat(3, cells.ratemap{[1:(currCell-1), (currCell+1):end]}), [3,1,2]);
        currSum = sum(currRatemaps,1);    
        currLog = log(currRatemaps); 
        currLog(currLog == -inf) = -999;
        currCounts = cells.counts([1:(currCell-1), (currCell+1):end],:);
        % And start decoding
        decoded = nan(length(windowCenters), size(cells.ratemap{1},1), size(cells.ratemap{1},2));
        for currWindow = 1:length(windowCenters)
            windowBins = (windowCenters(currWindow) - floor(windowSize/2) + 1):(windowCenters(currWindow) + ceil(windowSize/2));
            decoded(currWindow, :, :) = decodePosition(currSum, currLog, sum(currCounts(:,windowBins), 2), length(windowBins)*binWidth);
        end
        % Determine peak posterior probability at every step
        [~, peakIndex] = max(reshape(decoded,[size(decoded,1), size(decoded,2)*size(decoded,3)]), [], 2);
        [peakRow, peakCol] = ind2sub([size(decoded,2), size(decoded,3)], peakIndex);
        locs = [peakRow, peakCol]*2 - 1; % in cm
        % Then interpolate where this cell spiked in the current trajectory
        spikePos = interp1(times(ripples.start(currRipple)) + 5e-3*(0:(size(locs,1)-1))', locs, duringSpikes);
        % Optionally only include spikes if the interpolation wasn't over very long distance (these may be noisy: big jumps in decoded pos)        
        if doExcludeJumps
            % Also get the previous and next decoded location
            spikePosPrevious = interp1(times(ripples.start(currRipple)) + 5e-3*(0:(size(locs,1)-1))', locs, duringSpikes, 'previous');
            spikePosNext = interp1(times(ripples.start(currRipple)) + 5e-3*(0:(size(locs,1)-1))', locs, duringSpikes, 'next');
            spikePos = spikePos(sqrt(sum((spikePosNext - spikePosPrevious).^2,2)) < 25,:);
        end
        % Finally: collect difference map centred at the spike position
        diffMap{currCell} = nan(2*size(decoded,2), 2*size(decoded,3), size(spikePos,1));
        for currPos = 1:size(spikePos,1)
            diffMap{currCell}(size(decoded,2) - ceil(spikePos(currPos,1)/2) + (1:size(decoded,2)),size(decoded,3) - ceil(spikePos(currPos,2)/2) + (1:size(decoded,3)),currPos) = rateDiff;
        end
        % And story spike positions, if required
        if nargout > 1
            diffPos{currCell} = spikePos;
        end        
        if doPlot
            % Plot results for this cell
            f = figure('Position', [10, 10, 450, 150]); 
            
            % Load rat icon
            [ratIcon, ~, ratAlpha] = imread('icons/rat.png');
            
            ax = subplot(1,3,1);
            colormap(ax, 'parula');            
            hold on;
            imagesc([1, 199], [1, 199], rateBefore', 'AlphaData', ~isnan(rateBefore'));
            %plot(pos(beforePos, 1), pos(beforePos, 2), 'g-');
            %scatter(cells.spikes{currCell}.pos(beforeSpikes, 1), cells.spikes{currCell}.pos(beforeSpikes, 2), 'r.');
            %scatter(ripples.pos(currRipple,1), ripples.pos(currRipple,2), 200, 'go', 'filled');            
            %imagesc(ripples.pos(currRipple,1) + [-20, 20], ripples.pos(currRipple,2) + [-20, 20], flipud(255-ratIcon), 'AlphaData', flipud(ratAlpha));
            hold off;
            xlim([0 200]);
            ylim([0 200]);
            axis square;     
            axis off;
            makeColorbar(ax, [min(rateBefore(:)), max(rateBefore(:))], [], 'Rate (Hz)', 0.6);
            title('Before ripple');

            ax = subplot(1,3,2);
            %colormap(ax, brewermap([],'-Spectral'));                        
            colormap(ax, 'parula');            
            hold on;
            imagesc([1, 199], [1, 199], rateAfter', 'AlphaData', ~isnan(rateAfter'));
            %plot(pos(afterPos, 1), pos(afterPos, 2), 'g-');
            %scatter(cells.spikes{currCell}.pos(afterSpikes, 1), cells.spikes{currCell}.pos(afterSpikes, 2), 'r.');
            %scatter(ripples.pos(currRipple,1), ripples.pos(currRipple,2), 200, 'go', 'filled');            
            %imagesc(ripples.pos(currRipple,1) + [-20, 20], ripples.pos(currRipple,2) + [-20, 20], flipud(255-ratIcon), 'AlphaData', flipud(ratAlpha));            
            hold off;
            xlim([0 200]);
            ylim([0 200]);
            axis square;    
            axis off;
            makeColorbar(ax, [min(rateAfter(:)), max(rateAfter(:))], [], 'Rate (Hz)', 0.6);
            title('After ripple');

            ax = subplot(1,3,3);
            colormap(ax, [[linspace(0.3,1,128)', linspace(0.3,1,128)', ones(128,1)]; ...
                    [ones(128,1), linspace(1,0.3,128)', linspace(1,0.3,128)']]);
            replay_cm = colormap(ax, brewermap(floor(size(ripples.path{currRipple},1)*1.5), '-Greens'));                
            colormap(ax, brewermap([],"-RdBu"));                
            %colormap(ax, 'parula');                            
            hold on;
            imagesc([1, 199], [1, 199], rateDiff', 'AlphaData', ~isnan(rateDiff'), [-max(abs(rateDiff(:))), max(abs(rateDiff(:)))]);
            %imagesc([1, 199], [1, 199], rateDiff' .* (rateDiff' > 0), 'AlphaData', ~isnan(rateDiff'));
            %scatter(ripples.pos(currRipple,1), ripples.pos(currRipple,2), 200, 'go', 'filled'); 
            imagesc(ripples.pos(currRipple,1) + [-20, 20], ripples.pos(currRipple,2) + [-20, 20], flipud(ratIcon), 'AlphaData', flipud(ratAlpha));            
            for currStep = 2:(size(ripples.path{currRipple},1)-1)
                if sqrt(sum(diff(ripples.path{currRipple}([(currStep-1), currStep],:),1,1).^2,2)) < 50 && sqrt(sum(diff(ripples.path{currRipple}([(currStep+1), currStep],:),1,1).^2,2)) < 50
                    plot(ripples.path{currRipple}([(currStep-1), currStep],1), ripples.path{currRipple}([(currStep-1), currStep],2), 'Color', replay_cm(currStep,:), 'LineWidth',3);
                end
            end
            scatter(mean(spikePos(:,1)), mean(spikePos(:,2)), 300, 'rx', 'LineWidth', 2);
            xlim([0 200]);
            ylim([0 200]);
            makeColorbar(ax, [-max(abs(rateDiff(:))), 0, max(abs(rateDiff(:)))], [], 'Change (Hz)', 0.6);
            axis square;    
            axis off;
            title('Ratemap change and replay');        

            %sgtitle(['Ripple ' num2str(currRipple) ', cell ' num2str(currCell)]);

            %saveas(f, ['Figs_Aligned/Ripple' num2str(currRipple) '_Cell' num2str(currCell) '.png']);            
            %close(f);            
            %keyboard;
        end
        disp(['Finished ripple ' num2str(currRipple) ', cell ' num2str(currCell) ' aligned difference']);
    end
end
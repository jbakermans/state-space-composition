% Collect stats
allStats = cell(4,2);
allCorrs = cell(4,1);
for rat = 1:4
    for session = 1:2
        currBase = ['output/R' num2str(rat) 'S' num2str(session) '_'];
        if exist([currBase 'stats.mat'], 'file') == 2
            stats = load([currBase 'stats.mat']); stats = stats.stats;
        else
            stats = getSessionStats(rat, session);
            save([currBase 'stats.mat'], 'stats');
        end
        allStats{rat, session} = stats;
    end
    if exist([currBase 'corrs.mat'], 'file') == 2
        corrs = load([currBase 'corrs.mat']); corrs = corrs.corrs;
    else
        corrs = getSessionCorrs(rat);
        save([currBase 'corrs.mat'], 'corrs');
    end
    allCorrs{rat} = corrs;    
end

%% Collect test, controls, and matched for each rat and session
% These are the actual stats I want to calculate
test = cell(4,2);
control = cell(4,2);
matched = cell(4,2);
% I'll also get averaged change maps for each of them
testMaps = cell(4,2);
controlMaps = cell(4,2);
matchedMaps = cell(4,2);
% And which replay spikes are included
testInclude = cell(4,2);
controlInclude = cell(4,2);
matchedInclude = cell(4,2);
% This is to check if it's not just changes near home that contribute to stats
distThreshold = 0; % Increase this to 25 to see distant spike contribution
pos = cell(4,2);
home = cell(4,2);
% These are for additional sets of position data to build heatmaps across animals 
ratPos = cell(4,2);
ripplePos = cell(4,2);
spikePos = cell(4,2);
nCells = zeros(4,2);
% Also collect some specific previous-home aligned maps (specify rat, cell pairs)
anticorrExamples = [1,79; 2,108];
% And some specific rat, session, ripple, cell pairs 
rippleChangeExamples = [1,1,14,43;1,1,24,146; 1,1,27,76]; 
% Now run through sessions and calculate data
for rat = 1:4
    for session = 1:2
        % Select current stats
        currStats = allStats{rat, session};
        % Load current ripple data
        rats = {'Janni','Harpy','Imp','Naga'};
        sessions = {'Open1','Open2'};
        datDir = fullfile('/Users/jbakermans/Documents/Data/PfeifferFoster/DataForBehrensBakermans/',rats{rat}, sessions{session});
        ripples = load(fullfile(datDir, 'Output', 'ripples.mat')); ripples = ripples.ripples;
        cells = load(fullfile(datDir, 'Output', 'cells.mat')); cells = cells.cells; nCells(rat, session) = size(cells,1);
        coords = load(fullfile(datDir, 'Output', 'coords.mat')); coords = coords.coords;        
        diffMap = cell(size(ripples,1),1);
        diffPos = cell(size(ripples,1),1);
        for currRipple = 1:size(ripples,1)
            currFile = fullfile(datDir, 'Output', ['diff_' num2str(currRipple) '.mat']); 
            if exist(currFile, 'file') == 2
                currDat = load(currFile);
                diffMap{currRipple} = currDat.diffMap;
                diffPos{currRipple} = currDat.diffPos;
            end
            disp(['Finished loading diff map ' num2str(currRipple) '/' num2str(size(ripples,1))]);
        end
        diffmaps = getDiffMaps(diffMap, diffPos);        
        % Calculate some ripple summary that will determine that ripple's inclusion
        nearestToLoc = cellfun(@(x, y) min(sqrt(sum((x-y).^2,2))), ripples.path, num2cell(ripples.pos,2));
        meanToLoc = cellfun(@(x, y) mean(sqrt(sum((x-y).^2,2))), ripples.path, num2cell(ripples.pos,2)); 
        if session == 1
            home{rat, session} = getWellLoc(coords.behaviour.pos, coords.behaviour.speed, 15);
        elseif session == 2
            prevHome = home{rat, session-1};
            home{rat, session} = getWellLoc(coords.behaviour.pos, coords.behaviour.speed, 29);
        end
        nearestToHome = cellfun(@(x) min(sqrt(sum((home{rat, session} - x).^2,2))), ripples.path);
        % Get inclusion for test and control
        testRipples =  find(sqrt(sum((ripples.pos-home{rat, session}).^2,2))<10 & nearestToLoc < 30 & meanToLoc > 40);
        controlRipples = find(sqrt(sum((ripples.pos-home{rat, session}).^2,2))>10 & nearestToHome > 30 & nearestToLoc < 30 & meanToLoc > 40);
        % Get matched sample
        targetBins = linspace(coords.behaviour.time(1), coords.behaviour.time(end), 6);
        targetCounts = histcounts(ripples.tpeak(testRipples), targetBins);
        bestCorr = 0;
        bestSample = [];
        for iteration = 1:10000
            sample = controlRipples(randperm(length(controlRipples)));
            sample = sample(1:length(testRipples));
            sampleCounts = histcounts(ripples.tpeak(sample), targetBins);
            sampleCorr = corr(targetCounts', sampleCounts');
            if sampleCorr > bestCorr
                bestCorr = sampleCorr;
                bestSample = sample;
            end
        end
        matchedRipples = bestSample;
        % Get replay spikes to include
        testInclude{rat, session} = ismember(currStats.ripple, testRipples) & sqrt(sum((currStats.spikepos - currStats.home).^2,2)) > distThreshold;
        controlInclude{rat, session} = ismember(currStats.ripple, controlRipples);
        matchedInclude{rat, session} = ismember(currStats.ripple, matchedRipples);        
        % Finally: add selected rows to stats array        
        test{rat, session} = table2array(currStats(testInclude{rat, session}, {'roi','rad'}));
        control{rat, session} = table2array(currStats(controlInclude{rat, session}, {'roi','rad'}));
        matched{rat, session} = table2array(currStats(matchedInclude{rat, session}, {'roi','rad'}));
        pos{rat, session} = currStats.spikepos(testInclude{rat, session}, :) - home{rat, session};
        % Collect averaged aligned maps across selected spikes
        testMaps{rat, session} = squeeze(nanmean(diffmaps.map(testInclude{rat, session},:,:), 1));
        controlMaps{rat, session} = squeeze(nanmean(diffmaps.map(controlInclude{rat, session},:,:), 1));
        matchedMaps{rat, session} = squeeze(nanmean(diffmaps.map(matchedInclude{rat, session},:,:), 1));
        % Collect some additional position data to build heatmaps
        ratPos{rat, session} = coords.behaviour.pos;
        ripplePos{rat, session} = ripples.pos;
        spikePos{rat, session} = currStats.spikepos; 
        % Print progress
        disp(['Finished rat ' num2str(rat) ', session ' num2str(session)]);
    end
end
% Bit dumb but makes life easier: reload the ripples, cells, coords from the first rat & sess for examples
% Now I can create the replay spike example plots below, but only if they're both from rat 1, sess 1
datDir = fullfile('/Users/jbakermans/Documents/Data/PfeifferFoster/DataForBehrensBakermans/',rats{1}, sessions{1});
ripples = load(fullfile(datDir, 'Output', 'ripples.mat')); ripples = ripples.ripples;
cells = load(fullfile(datDir, 'Output', 'cells.mat')); cells = cells.cells;
coords = load(fullfile(datDir, 'Output', 'coords.mat')); coords = coords.coords;        

%% Plot all results
doSave = true;

% 1. Heatmaps of rat, ripple, decoded spike positions across cells in both sessions
plotPos = {cat(1, ratPos{:,1}), cat(1, ripplePos{:,1}), cat(1, spikePos{:,1});...
    cat(1, ratPos{:,2}), cat(1, ripplePos{:,2}), cat(1, spikePos{:, 2})}';
plotNames = {'Behaviour', 'Ripples', 'Decoded spikes'};
valNames = {'Time (s)', 'Ripples (1)', 'Spikes (1)'};
textNames = {'N_{cells} = ', 'N_{ripples} = ', 'N_{spikes} = '};

f = figure('Position',[10, 10, 300, 450]);
cmap = brewermap([],"-Spectral");
colormap(f, cmap);  
saveDat = cell(size(plotPos));
for currRow = 1:size(plotPos,1)
    for currCol = 1:size(plotPos,2)
        ax = subplot(size(plotPos,1), size(plotPos, 2), (currRow - 1)*size(plotPos,2) + currCol);
        currDat = histcounts2(plotPos{currRow, currCol}(:, 1), plotPos{currRow, currCol}(:, 2), 0:2:200, 0:2:200);
        currDat = imgaussfilt(currDat, 2);
        if currRow == 1
            currDat = currDat * coords.behaviour.dt;
        end
        saveDat{currRow, currCol} = currDat;
        hold on;
        imagesc([1, 199], [1, 199], currDat');
        set(ax,'YDir','normal');
        if currRow == 1
            text(0, 199, [textNames{currRow} num2str(sum(nCells(:,currCol)))], ...
                'HorizontalAlignment','left', 'VerticalAlignment','top','color',[1,1,1])
        else
            text(0, 199, [textNames{currRow} num2str(size(plotPos{currRow,currCol},1))], ...
                'HorizontalAlignment','left', 'VerticalAlignment','top','color',[1,1,1])
        end
        hold off;
        axis square;
        box on;
        xticks([]);
        yticks([]);
        makeColorbar(ax, [0, max(currDat(:))], [], valNames{currRow}, 0.8);          
        title([plotNames{currRow} ' day ' num2str(currCol)]);
    end
end
if doSave
    saveas(f, 'Figs_Final/4b_maps.png');
    saveas(f, 'Figs_Final/4b_maps.svg');
    save('Figs_Data/4b.mat', 'saveDat');
end

% 2. Examples of ratemap change and replay spike - this assumes they're both from rat 1, sess 1
for currExample = 1:size(rippleChangeExamples,1)
    currRipple = rippleChangeExamples(currExample, 3);
    currCell = rippleChangeExamples(currExample, 4);
    [currDatMap, currDatPos] = getRippleRatemapChange(currRipple, ripples, cells, coords, true, false, false, currCell);
    if doSave
        saveas(gcf, ['Figs_Final/4e_cells_' num2str(currExample) '.png']);
        saveas(gcf, ['Figs_Final/4e_cells_' num2str(currExample) '.svg']);
        saveDat = {currDatMap{currCell}, currDatPos{currCell}};
        save(['Figs_Data/4e_' num2str(currExample) '.mat'], 'saveDat');    
    end    
end  

% 3. Aligned ratemap change averaged across all animals, for home/elsewhere/control replays
alignedPlot = {testMaps, controlMaps, matchedMaps};
nReplaySpikes = {cellfun(@(x) size(x,1), test(:)), cellfun(@(x) size(x,1), control(:)), cellfun(@(x) size(x,1), matched(:))};
plotNames = {'Replay from home', 'Replay elsewhere', 'Matched control'};
roi = false(size(testMaps{1},1), size(testMaps{1},2));
radius = 50;
center = floor(size(testMaps{1},2)/2);
for row = (center-radius):(center+radius)
    for col = ceil(center-sqrt(radius^2 - (center-row)^2)):floor(center+sqrt(radius^2 - (center-row)^2))
        roi(row, col) = true;
    end
end
for currMap = 1:length(alignedPlot)
    % Average across animals, but make sure its weighted by nr of replay spikes
    currMaps = cat(3, alignedPlot{currMap}{:});
    currMaps = permute(currMaps, [3,1,2]);
    currMaps = reshape(currMaps, [size(currMaps,1), size(currMaps,2)*size(currMaps,3)]);
    currMaps = sum(currMaps .* repmat(nReplaySpikes{currMap}, [1, size(currMaps,2)]),1) / sum(nReplaySpikes{currMap});
    currMaps = reshape(currMaps, size(alignedPlot{currMap}{1},1), size(alignedPlot{currMap}{1},2));
    currMaps(~roi) = nan;
    alignedPlot{currMap} = currMaps;
end
maplims = cat(1, alignedPlot{:});
maplims = [min(0,min(maplims(:))), max(maplims(:))];

f = figure('Position',[10, 10, 450, 150]);
for currMap = 1:length(alignedPlot)
    ax = subplot(1,length(alignedPlot),currMap);
    hold on;
    imagesc([-(size(alignedPlot{currMap},1)-1), (size(alignedPlot{currMap},1)-1)], [-(size(alignedPlot{currMap},1)-1), (size(alignedPlot{currMap},1)-1)], alignedPlot{currMap}, 'AlphaData', ~isnan(alignedPlot{currMap}), maplims);
    scatter(0, 0, 300, 'rx', 'LineWidth', 2);
    xlim([-100,100]);
    ylim([-100,100]);
    hold off;
    axis square;
    xlabel('dx from spike (cm)');
    if currMap == 1
        ylabel('dy from spike (cm)');
    end
    if currMap == 3
        makeColorbar(ax, maplims, [], 'Change (Hz)', 0.7);
    end
    text(-100,100,['N_{spikes} = ' num2str(sum(nReplaySpikes{currMap}))], ...
        'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', 'color', [0,0,0]);
    title(plotNames{currMap});
end
if doSave
    saveas(f, 'Figs_Final/4h_aligned.png');
    saveas(f, 'Figs_Final/4h_aligned.svg');
    saveDat = alignedPlot;
    save(['Figs_Data/4h_' num2str(currExample) '.mat'], 'saveDat');        
end

% 4. Statistics of replay-spike-aligned changemaps, within roi and along radial spokes
acrossTest = {nanmean(cat(1, test{:}), 1), nanstd(cat(1, test{:}), 0, 1) / sqrt(size(cat(1, test{:}),1))};
acrossControl = {nanmean(cat(1, control{:}), 1), nanstd(cat(1, control{:}), 0, 1) / sqrt(size(cat(1, control{:}),1))};
acrossMatched = {nanmean(cat(1, matched{:}), 1), nanstd(cat(1, matched{:}), 0, 1) / sqrt(size(cat(1, matched{:}),1))};

f = figure('Position', [10, 10, 300, 150]);
subplot(1,2,1);
hold on;
bar([acrossTest{1}(1), acrossControl{1}(1), acrossMatched{1}(1)]);
errorbar([acrossTest{1}(1), acrossControl{1}(1), acrossMatched{1}(1)], [acrossTest{2}(1), acrossControl{2}(1), acrossMatched{2}(1)], 'k.');
hold off;
xticks(1:3);
xticklabels({'Home','Elsewhere','Matched'});
ylabel('Ratemap change (Hz)');
title('Within central ROI');
ax = subplot(1,2,2);
errorbar(repmat(2*(0:49)', [1, 3]), [acrossTest{1}(2:end); acrossControl{1}(2:end); acrossMatched{1}(2:end)]', [acrossTest{2}(2:end); acrossControl{2}(2:end); acrossMatched{2}(2:end)]');
legend({'Home','Elsewhere','Matched'}, 'location', 'none','units', 'normalized', 'position',[ax.Position(1) + ax.Position(3)*0.2, ax.Position(2) + ax.Position(3)*0.3, ax.Position(3)*0.8, ax.Position(4)*0.3]);
ylabel('Ratemap change (Hz)');
xlabel('Radial distance (cm)');
title('Along radial spokes');
if doSave
    saveas(f, 'Figs_Final/4i_stats.png');
    saveas(f, 'Figs_Final/4i_stats.svg');
    saveDat = {cat(1, test{:}), cat(1, control{:}), cat(1, matched{:})};
    saveDat = {saveDat{1}(:,1), saveDat{2}(:,1), saveDat{3}(:,1); saveDat{1}(:,2:end), saveDat{2}(:,2:end), saveDat{3}(:,2:end)};        
    save(['Figs_Data/4i_' num2str(currExample) '.mat'], 'saveDat');            
end

% 5. Examples of home-previous home aligned change maps for cells with high anticorrelation
for currMap = 1:size(anticorrExamples,1)
    currDat = squeeze(allCorrs{anticorrExamples(currMap,1)}.curr(anticorrExamples(currMap,2),:,:));
    prevDat = squeeze(allCorrs{anticorrExamples(currMap,1)}.prev(anticorrExamples(currMap,2),:,:));

    % Load icons for home and previous home
    [homeIcon, ~, homeAlpha] = imread('icons/home.png');
    [prevIcon, ~, prevAlpha] = imread('icons/prev_home.png');
    
    f = figure('Position', [10,10,300,150]);
    % Blue neg, red pos colour map
    colormap(f, flipud(brewermap([],"RdBu")));  
    
    ax = subplot(1,2,1); 
    hold on; 
    %imagesc([-199,199], [-199,199], currDat', 'AlphaData', ~isnan(currDat'), 0.3*[-max(abs(prevDat(:))), max(abs(prevDat(:)))]);
    imagesc([-199,199], [-199,199], ((currDat>0).*(currDat/max(currDat(:))) + (currDat<0).*(currDat/-min(currDat(:))))', 'AlphaData', ~isnan(currDat'), [-1, 1]);
    scatter(0, 0, 200, 'kx', 'LineWidth', 2);
    imagesc(flipud(homeIcon),'xdata',[-40,40],'ydata',[20,100]-120*(currMap-1), 'AlphaData', flipud(homeAlpha))
    xlim([-150,110]);
    ylim([-150,110]);
    axis equal;
    legend('Current home', 'location', 'none','units', 'normalized', 'position',[ax.Position(1) + ax.Position(3)*0.1, ax.Position(2) + ax.Position(3)*2.1, ax.Position(3)*0.8, ax.Position(4)*0.1]);
    xlabel('dx from home (cm)');
    ylabel('dy from home (cm)');
    title('Total ratemap change');
    
    ax = subplot(1,2,2); 
    hold on; 
    imagesc([-199,199], [-199,199], ((prevDat>0).*(prevDat/max(prevDat(:))) + (prevDat<0).*(prevDat/-min(prevDat(:))))', 'AlphaData', ~isnan(prevDat'),[-1,1]); % [-max(abs(prevDat(:))), max(abs(prevDat(:)))]
    scatter(0, 0,  200,  0.5*[1,1,1], 'x', 'LineWidth', 2);
    imagesc(flipud(255-(0.5*(255-homeIcon))),'xdata',[-40,40],'ydata',[20,100]-120*(currMap-1), 'AlphaData', flipud(prevAlpha))    
    xlim([-150,110]);
    ylim([-150,110]);    
    axis equal;
    makeColorbar(ax, [-1,0,1], {num2str(min(prevDat(:)),2), '0', num2str(max(prevDat(:)),2)}, 'Change (Hz)', 0.7);
    
    legend('Previous home', 'location', 'none','units', 'normalized', 'position',[ax.Position(1) + ax.Position(3)*0.1, ax.Position(2) + ax.Position(3)*2.1, ax.Position(3)*0.8, ax.Position(4)*0.1]);
    xlabel('dx from home (cm)');
    title(['Correlation: ' num2str(allCorrs{anticorrExamples(currMap,1)}.corr(anticorrExamples(currMap,2)), 2)]);
    if doSave
        saveas(f, ['Figs_Final/5f_cells_' num2str(currMap) '.png']);
        saveas(f, ['Figs_Final/5f_cells_' num2str(currMap) '.svg']);
        saveDat = {currDat, prevDat};
        save(['Figs_Data/5f_' num2str(currMap) '.mat'], 'saveDat');           
    end    
end

% 6. Distribution of home-previous home aligned change map correlations
acrossCorrs = cellfun(@(x) x.corr, allCorrs, 'UniformOutput', false);
corrLim = max(abs(cat(1, acrossCorrs{:})));

f = figure('Position',[10, 10, 150, 150]);
histogram(cat(1, acrossCorrs{:}), linspace(-corrLim,corrLim, 50));
[h, p] = ttest(cat(1, acrossCorrs{:}), zeros(size(cat(1, acrossCorrs{:}))), 'tail', 'left');
text(-corrLim,100,['N_{cells} = ' num2str(size(cat(1, acrossCorrs{:}),1))], ...
    'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', 'color', [0,0,0]);
title(['C < 0: p = ' num2str(p,2)]);
ylabel('Cell count (1)');
xlabel('Change correlation (1)');
xlim([-corrLim, corrLim]);
if doSave
    saveas(f, 'Figs_Final/5g_distribution.png');
    saveas(f, 'Figs_Final/5g_distribution.svg');
    saveDat = cat(1, acrossCorrs{:});
    save('Figs_Data/5g.mat', 'saveDat');    
end

% 7. Overlap between replay spikes and ratemap changes, compared to home-prevhome correlation
spikeHist = cell(size(allStats,1),1);
spikeChange = cell(size(spikeHist));
spikeCorr = cell(size(spikeHist));
spikeCorrNorm = cell(size(spikeHist));
for currRat = 1:size(spikeHist,1)
    currCells = nCells(currRat, 2);
    spikeHist{currRat} = nan(currCells, 100,100);
    spikeChange{currRat} = nan(currCells, 100,100);
    spikeCorr{currRat} = nan(currCells,1);
    spikeCorrNorm{currRat} = nan(currCells,1);
    for currCell = 1:currCells
        % Create a histogram of where the replay spikes occured
        spikeHist{currRat}(currCell,:,:) = histcounts2(...
            allStats{currRat,2}.spikepos(allStats{currRat,2}.cell == currCell & testInclude{currRat,2}, 1), ...
            allStats{currRat,2}.spikepos(allStats{currRat,2}.cell == currCell & testInclude{currRat,2}, 2), 0:2:200, 0:2:200);        
        spikeHist{currRat}(currCell,:,:) = imgaussfilt(squeeze(spikeHist{currRat}(currCell,:,:)), 2);
        % Cut-out the 200x200 cm arena from the whole-session change maps
        spikeChange{currRat}(currCell,:,:) = squeeze(allCorrs{currRat}.curr(currCell, (0:99)+100-ceil(home{currRat,2}(2)/2), (0:99)+100-ceil(home{currRat,2}(1)/2)));         
        % Stick both in 1-d column vectors
        currChanges = abs(reshape(spikeChange{currRat}(currCell,:,:), [], 1));
        currSpikes = reshape(spikeHist{currRat}(currCell,:,:), [], 1);
        include = ~isnan(currChanges) & ~isnan(currSpikes);
        if any(include)
            spikeCorr{currRat}(currCell) = mean(currChanges(include) .* currSpikes(include));
            spikeCorrNorm{currRat}(currCell) = corr(currChanges(include), currSpikes(include)); 
        end
    end
end
x = cat(1,acrossCorrs{:});
y = cat(1,spikeCorr{:});
include = ~isnan(x) & ~isnan(y);
nBins = 10;
xBins = linspace(min(x(include)), max(x(include)), nBins + 1);
yBinned = nan(nBins,2);
for currBin = 1:nBins
    select = x >= xBins(currBin) & x <= xBins(currBin+1);
    yBinned(currBin, 1) = mean(y(select));
    yBinned(currBin, 2) = std(y(select)) / sqrt(sum(select));
end

f = figure('Position',[10, 10, 150, 150]);
hold on;
scatter(x(include), y(include)*1e3, [], [0.5, 0.5, 0.5], '.');
errorbar(xBins(2:end) - 0.5*(xBins(2)-xBins(1)),yBinned(:,1)*1e3, yBinned(:,2)*1e3, 'LineWidth', 2);
hold off;
[r, p] = corr(x(include), y(include), 'tail', 'left');
title(['R = ' num2str(r,2) ', p = ' num2str(p,2)]);
xlabel('Change correlation (1)');
ylabel('Replay overlap (mHz)');
xlim([-corrLim, corrLim]);
if doSave
    saveas(f, 'Figs_Final/5h_overlap.png');
    saveas(f, 'Figs_Final/5h_overlap.svg');
    saveDat = [x(include), y(include)];
    save('Figs_Data/5h.mat', 'saveDat');    
end

% 8. Overlap between normalised replay spikes and ratemap changes, compared to home-prevhome correlation
x = cat(1,acrossCorrs{:});
y = cat(1,spikeCorrNorm{:});
include = ~isnan(x) & ~isnan(y);
nBins = 10;
xBins = linspace(min(x(include)), max(x(include)), nBins + 1);
yBinned = nan(nBins,2);
for currBin = 1:nBins
    select = x >= xBins(currBin) & x <= xBins(currBin+1);
    yBinned(currBin, 1) = mean(y(select & include));
    yBinned(currBin, 2) = std(y(select & include)) / sqrt(sum(select & include));
end

f = figure('Position',[10, 10, 150, 150]);
hold on;
scatter(x(include), y(include), [], [0.5, 0.5, 0.5], '.');
errorbar(xBins(2:end) - 0.5*(xBins(2)-xBins(1)),yBinned(:,1), yBinned(:,2), 'LineWidth', 2);
hold off;
[r, p] = corr(x(include), y(include), 'tail', 'left');
title(['R = ' num2str(r,2) ', p = ' num2str(p,2)]);
xlabel('Change correlation (1)');
ylabel('Replay correlation (1)');
xlim([-corrLim, corrLim]);
if doSave
    saveas(f, 'Figs_Final/5i_normalised.png');
    saveas(f, 'Figs_Final/5i_normalised.svg');
    saveDat = [x(include), y(include)];
    save('Figs_Data/5i.mat', 'saveDat');        
end

% 9. Number of home replays, compared to home-prevhome correlation
roiVsCorr = cell(size(acrossCorrs));
for currRat = 1:size(roiVsCorr,1)
    roiVsCorr{currRat} = nan(size(acrossCorrs{currRat},1), 2);
    for currCell = 1:size(acrossCorrs{currRat},1)
        roiVsCorr{currRat}(currCell,1) = acrossCorrs{currRat}(currCell);        
        roiVsCorr{currRat}(currCell,2) = sum(allStats{currRat,2}.cell == currCell & testInclude{currRat,2});
    end
end
roiVsCorr = cat(1, roiVsCorr{:});
include = ~any(isnan(roiVsCorr),2);
x = roiVsCorr(include,1);
y = roiVsCorr(include,2);

nBins = 10;
xBins = linspace(min(x(:)), max(x(:)), nBins + 1);
yBinned = nan(nBins,2);
for currBin = 1:nBins
    select = x >= xBins(currBin) & x <= xBins(currBin+1);
    yBinned(currBin, 1) = mean(y(select));
    yBinned(currBin, 2) = std(y(select)) / sqrt(sum(select));
end

f = figure('Position',[10, 10, 150, 150]);
hold on;
scatter(x, y, [], [0.5, 0.5, 0.5], '.');
errorbar(xBins(2:end) - 0.5*(xBins(2)-xBins(1)),yBinned(:,1), yBinned(:,2), 'LineWidth', 2);
hold off;
[r,p] = corr(x, y, 'tail', 'left');
title(['R = ' num2str(r,2) ', p = ' num2str(p,2)]);
xlabel('Change correlation (1)');
ylabel('Home replays (1)');
xlim([-corrLim, corrLim]);
if doSave
    saveas(f, 'Figs_Final/5j_replay.png');
    saveas(f, 'Figs_Final/5j_replay.svg');
    saveDat = [x,y];
    save('Figs_Data/5j.mat', 'saveDat');            
end

% 11. Plot simulation results, loaded in matlab for figure style consistency
sim = load(fullfile('output','sim.mat'));
% 11a. Aligned change maps
alignedPlot = {sim.changes, sim.changes_control};
plotNames = {'Replay from home', 'Replay elsewhere'};
roi = false(size(alignedPlot{1},2), size(alignedPlot{1},3));
radius = 25;
center = floor(size(alignedPlot{1},2)/2);
for row = (center-radius):(center+radius)
    for col = ceil(center-sqrt(radius^2 - (center-row)^2)):floor(center+sqrt(radius^2 - (center-row)^2))
        roi(row, col) = true;
    end
end
for currMap = 1:length(alignedPlot)
    currMaps = squeeze(nanmean(alignedPlot{currMap},1));
    currMaps(~roi) = nan;
    alignedPlot{currMap} = currMaps;
end
maplims = cat(1, alignedPlot{:});
maplims = [0, max(maplims(:))];
f = figure('Position',[10, 10, 300, 150]);
for currMap = 1:length(alignedPlot)
    ax = subplot(1,length(alignedPlot),currMap);
    hold on;
    imagesc([-(size(alignedPlot{currMap},1)*2-1), (size(alignedPlot{currMap},1)*2-1)], [-(size(alignedPlot{currMap},1)*2-1), (size(alignedPlot{currMap},1)*2-1)], alignedPlot{currMap}, 'AlphaData', ~isnan(alignedPlot{currMap}), maplims);
    scatter(0, 0, 300, 'rx', 'LineWidth', 2);
    xlim([-radius*4,radius*4]);
    ylim([-radius*4,radius*4]);
    hold off;
    axis square;
    xlabel('dx from spike (cm)');
    if currMap == 1
        ylabel('dy from spike (cm)');
    end
    if currMap == 2
        c = makeColorbar(ax, maplims, strsplit(num2str(maplims,1)), 'Change (Hz)', 0.7);
    end
    title(plotNames{currMap});
end
if doSave
    saveas(f, 'Figs_Final/4f_sim_aligned.png');
    saveas(f, 'Figs_Final/4f_sim_aligned.svg');
end
% 11b. Change stats
f = figure('Position', [10, 10, 300, 150]);
subplot(1,2,1);
hold on;
bar([nanmean(sim.stats_roi), nanmean(sim.stats_roi_control)]);
errorbar([nanmean(sim.stats_roi), nanmean(sim.stats_roi_control)], [nanstd(sim.stats_roi', 0, 1) / sqrt(size(sim.stats_roi',1)),nanstd(sim.stats_roi_control', 0, 1) / sqrt(size(sim.stats_roi_control',1))], 'k.');
hold off;
xticks(1:2);
xticklabels({'Home','Elsewhere'});
ylabel('Ratemap change (Hz)');
title('Within central ROI');
ax = subplot(1,2,2);
errorbar(repmat(4*(0:24)', [1, 2]), [nanmean(sim.stats_angle,1); nanmean(sim.stats_angle_control,1)]', [nanstd(sim.stats_angle, 0, 1) / sqrt(size(sim.stats_angle,1)); nanstd(sim.stats_angle_control, 0, 1) / sqrt(size(sim.stats_angle_control,1))]');
l = legend({'Home','Elsewhere'});
l.Position(1) = l.Position(1) + 0.03;
ylabel('Ratemap change (Hz)');
xlabel('Radial distance (cm)');
title('Along radial spokes');
if doSave
    saveas(f, 'Figs_Final/4g_sim_stats.png');
    saveas(f, 'Figs_Final/4g_sim_stats.svg');
end
if false
    simdata = {sim.stats_roi', sim.stats_roi_control'};
    figure('Position', [10, 10, 450, 150])
    subplot(1,4,1); hold on; bar([nanmean(sim.stats_roi), nanmean(sim.stats_roi_control)]); errorbar([nanmean(sim.stats_roi), nanmean(sim.stats_roi_control)], [nanstd(sim.stats_roi', 0, 1) / sqrt(size(sim.stats_roi',1)),nanstd(sim.stats_roi_control', 0, 1) / sqrt(size(sim.stats_roi_control',1))], 'k.');
    hold off; ylim([-0.1, 0.7]);
    subplot(1,4,2); hold on; scatter(1 + 0.2*randn(size(simdata{1})), simdata{1},'.'); scatter(2 + 0.2*randn(size(simdata{2})), simdata{2},'.');ylim([-3, 3]);
    subplot(1,4,3); boxchart([ones(size(simdata{1}));2*ones(size(simdata{2}))], cat(1,simdata{:}));ylim([-3, 3]);
    subplot(1,4,4); violin(simdata);ylim([-3, 3]);
end
% 11c. Home-away ratemap example
currDat = nan(100);
prevDat = nan(100);
currDat((50-floor(home{1,2}(2)/4)):(100-floor(home{1,2}(2)/4)-1),...
    (50-floor(home{1,2}(1)/4)):(100-floor(home{1,2}(1)/4)-1)) = squeeze(sim.aligned_changes(4,1,:,:))';
prevDat((50-floor(home{1,1}(2)/4)):(100-floor(home{1,1}(2)/4)-1),...
    (50-floor(home{1,1}(1)/4)):(100-floor(home{1,1}(1)/4)-1)) = squeeze(sim.aligned_changes(4,1,:,:))';
[homeIcon, ~, homeAlpha] = imread('icons/home.png');
[prevIcon, ~, prevAlpha] = imread('icons/prev_home.png');
f = figure('Position', [10,10,300,150]);
colormap(f, flipud(brewermap([],"RdBu")));  
ax = subplot(1,2,1); 
hold on; 
imagesc([-199,199], [-199,199],  ((currDat>0).*(currDat/max(currDat(:))) + (currDat<0).*(currDat/-min(currDat(:))))', 'AlphaData', ~isnan(currDat'), [-1, 1]);
scatter(0, 0, 200, 'kx', 'LineWidth', 2);
imagesc(flipud(homeIcon),'xdata',[-40,40],'ydata',[20,100]-120*0, 'AlphaData', flipud(homeAlpha))
xlim([-150,120]);
ylim([-150,120]);
axis equal;
legend('Current home', 'location', 'none','units', 'normalized', 'position',[ax.Position(1) + ax.Position(3)*0.1, ax.Position(2) + ax.Position(3)*2.1, ax.Position(3)*0.8, ax.Position(4)*0.1]);
xlabel('dx from home (cm)');
ylabel('dy from home (cm)');
title('Total ratemap change');
ax = subplot(1,2,2); 
hold on; 
imagesc([-199,199], [-199,199],  ((prevDat>0).*(prevDat/max(prevDat(:))) + (prevDat<0).*(prevDat/-min(prevDat(:))))', 'AlphaData', ~isnan(prevDat'),[-1,1]); % [-max(abs(prevDat(:))), max(abs(prevDat(:)))]
scatter(0, 0,  200,  0.5*[1,1,1], 'x', 'LineWidth', 2);
imagesc(flipud(255-(0.5*(255-homeIcon))),'xdata',[-40,40],'ydata',[20,100]-120*0, 'AlphaData', flipud(prevAlpha))    
xlim([-150,120]);
ylim([-150,120]);    
axis equal;
makeColorbar(ax, [-1,0,1], {num2str(min(prevDat(:)),2), '0', num2str(max(prevDat(:)),2)}, 'Change (Hz)', 0.7);
legend('Previous home', 'location', 'none','units', 'normalized', 'position',[ax.Position(1) + ax.Position(3)*0.1, ax.Position(2) + ax.Position(3)*2.1, ax.Position(3)*0.8, ax.Position(4)*0.1]);
xlabel('dx from home (cm)');
currCorr = corrcoef(currDat(~isnan(currDat) & ~isnan(prevDat)), prevDat(~isnan(currDat) & ~isnan(prevDat)));
title(['Correlation: ' num2str(currCorr(1,2), 2)]);
if doSave
    saveas(f, ['Figs_Final/5d_sim_map.png']);
    saveas(f, ['Figs_Final/5d_sim_map.svg']);
end    
% 11d. Correlation histograms
corrLim = max(abs([sim.aligned_corr_landmark, sim.aligned_corr_place]));
corrBins = linspace(-corrLim, corrLim, 50);
dBins = 0.5*(corrBins(2)-corrBins(1));
countsBoth = histcounts([sim.aligned_corr_landmark, sim.aligned_corr_place], corrBins);
countsLandmark = histcounts(sim.aligned_corr_landmark, corrBins);
f = figure('Position',[10, 10, 300, 150]);
subplot(1,2,1);
bar(corrBins(2:end) - dBins, countsBoth, 1);
[h, p] = ttest([sim.aligned_corr_landmark, sim.aligned_corr_place], zeros(size([sim.aligned_corr_landmark, sim.aligned_corr_place])), 'tail', 'left');
title(['C < 0: p = ' num2str(p,2)]);
ylabel('Cell count (1)');
xlabel('Change correlation (1)');
xlim([-corrLim, corrLim]);
ylim([0, 1.2*max(countsBoth)]);
subplot(1,2,2);
hold on;
bar(corrBins(2:end) - dBins, countsBoth, 1);
bar(corrBins(2:end) - dBins, countsLandmark, 1);
ylabel('Cell count (1)');
xlabel('Change correlation (1)');
xlim([-corrLim, corrLim]);
ylim([0, 1.2*max(countsBoth)]);
l = legend('Place cells', 'Landmark cells');
l.Position(2) = l.Position(2)+0.1;
l.Position(1) = l.Position(1)+0.05;
if doSave
    saveas(f, 'Figs_Final/5e_sim_distribution.png');
    saveas(f, 'Figs_Final/5e_sim_distribution.svg');
end

% Supplementary figure 1: stats while excluding replay spikes near home
distances = [10, 25, 50];
allpos = cat(1,pos{:});
alltest = cat(1, test{:});
distanceBars = zeros(length(distances),2); % mean, standard error
for d = 1:length(distances)
    distanceThreshold = distances(d);
    currInclude = sqrt(sum(allpos.^2,2)) > distanceThreshold;

    % A. Included replay spike locations
    f = figure('Position',[10, 10, 250, 150]);
    subplot(1,2,1);
    scatter(allpos(currInclude,1), allpos(currInclude,2), 2, '.');
    text(-150,200,['N_{spikes} = ' num2str(size(allpos(currInclude,:),1))], ...
        'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', 'color', [0,0,0]);
    xlabel('dx from home (cm)');    
    ylabel('dy from home (cm)')
    xlim([-150, 150]);
    ylim([-200, 200]);
    title(['> ' num2str(distanceThreshold) ' cm from home']);    
    axis equal;
    subplot(1,2,2);
    currBins = linspace(0,200,25);
    hold on;
    h = histogram(sqrt(sum(allpos(currInclude & alltest(:,1) > prctile(alltest(currInclude,1), 90),:).^2,2)), currBins, 'Normalization', 'probability');
    hc_baseline = histcounts(sqrt(sum(allpos(currInclude,:).^2,2)), currBins, 'Normalization', 'probability');
    plot(currBins(2:end)-0.5*(currBins(2)-currBins(1)), hc_baseline, '--', 'linewidth', 2);
    ylim([0, 1.5*max(h.Values)]);
    hold off;
    l = legend('Top 10%', 'All spikes');
    l.Position(1) = l.Position(1) + 0.075;
    xlabel('Distance from home (cm)');
    ylabel('Probability (1)');
    xlim([0, 200]);
    title('Ratemap changes');
    if doSave
        saveas(f, ['Figs_Final/S7a_include_' num2str(distanceThreshold) '.png']);
        saveas(f, ['Figs_Final/S7a_include_' num2str(distanceThreshold) '.svg']);
        saveDat = {allpos(currInclude,:), h.Data};
        save(['Figs_Data/S7a_' num2str(distanceThreshold) '.mat'], 'saveDat');        
    end    

    % B. Statistics of replay-spike-aligned changemaps, within roi and along radial spokes
    currAcrossTest = {nanmean(alltest(currInclude,:), 1), nanstd(alltest(currInclude,:), 0, 1) / sqrt(size(alltest(currInclude,:),1))};
    distanceBars(d,:) = [currAcrossTest{1}(1), currAcrossTest{2}(1)];
    f = figure('Position', [10, 10, 300, 150]);
    subplot(1,2,1);
    hold on;
    bar([currAcrossTest{1}(1), acrossControl{1}(1), acrossMatched{1}(1)]);
    errorbar([currAcrossTest{1}(1), acrossControl{1}(1), acrossMatched{1}(1)], [currAcrossTest{2}(1), acrossControl{2}(1), acrossMatched{2}(1)], 'k.');
    hold off;
    xticks(1:3);
    xticklabels({'Home','Elsewhere','Matched'});
    ylabel('Ratemap change (Hz)');
    title('Within central ROI');
    ax = subplot(1,2,2);
    errorbar(repmat(2*(0:49)', [1, 3]), [currAcrossTest{1}(2:end); acrossControl{1}(2:end); acrossMatched{1}(2:end)]', [currAcrossTest{2}(2:end); acrossControl{2}(2:end); acrossMatched{2}(2:end)]');
    legend({'Home','Elsewhere','Matched'}, 'location', 'none','units', 'normalized', 'position',[ax.Position(1) + ax.Position(3)*0.2, ax.Position(2) + ax.Position(3)*0.3, ax.Position(3)*0.8, ax.Position(4)*0.3]);
    ylabel('Ratemap change (Hz)');
    xlabel('Radial distance (cm)');
    title('Along radial spokes');
    if doSave
        saveas(f, ['Figs_Final/S7b_stats_' num2str(distanceThreshold) '.png']);
        saveas(f, ['Figs_Final/S7b_stats_' num2str(distanceThreshold) '.svg']);
        saveDat = {alltest(currInclude,:), cat(1, control{:}), cat(1, matched{:})};
        saveDat = {saveDat{1}(:,1), saveDat{2}(:,1), saveDat{3}(:,1); saveDat{1}(:,2:end), saveDat{2}(:,2:end), saveDat{3}(:,2:end)};        
        save(['Figs_Data/S7b_' num2str(distanceThreshold) '.mat'], 'saveDat');                
    end    
end

% One more bar plot with all the distances in (actually, discard the first, because it's only 10cm)
f = figure('Position', [10, 10, 150, 150]);
hold on;
bar([acrossTest{1}(1), distanceBars(2,1), distanceBars(3,1), acrossControl{1}(1), acrossMatched{1}(1)]);
errorbar([acrossTest{1}(1), distanceBars(2,1), distanceBars(3,1), acrossControl{1}(1), acrossMatched{1}(1)], [acrossTest{2}(1), distanceBars(2,2), distanceBars(3,2), acrossControl{2}(1), acrossMatched{2}(1)], 'k.');
hold off;
xticks(1:5);
xticklabels({'All home', ['> ' num2str(distances(2)) 'cm'], ['> ' num2str(distances(3)) 'cm'],'Elsewhere','Matched'});
ylabel('Ratemap change (Hz)');
title('Within central ROI');
if doSave
    saveas(f, ['Figs_Final/5k_distances.png']);
    saveas(f, ['Figs_Final/5k_distances.svg']);
    saveDat = [acrossTest{1}(1), distanceBars(2,1), distanceBars(3,1), acrossControl{1}(1), acrossMatched{1}(1); acrossTest{2}(1), distanceBars(2,2), distanceBars(3,2), acrossControl{2}(1), acrossMatched{2}(1)];
    save('Figs_Data/5k.mat', 'saveDat');                
end    
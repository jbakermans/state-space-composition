function stats = getSessionStats(rat, session)
    rats = {'Janni','Harpy','Imp','Naga'};
    % Define sessions
    sessions = {'Open1','Open2'};
    % Load pre-calculated data derivatives
    datDir = fullfile('/Volumes/My Passport for Mac/PfeifferFoster_data/DataForBehrensBakermans/', ...
        rats{rat}, sessions{session});
    cells = load(fullfile(datDir, 'Output', 'cells.mat')); cells = cells.cells;
    coords = load(fullfile(datDir, 'Output', 'coords.mat')); coords = coords.coords;
    ripples = load(fullfile(datDir, 'Output', 'ripples.mat')); ripples = ripples.ripples;
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
    % Build big table with all diffmaps
    diffmaps = getDiffMaps(diffMap, diffPos);

    % Exctract home well location
    if session == 1
        home = getWellLoc(coords.behaviour.pos, coords.behaviour.speed, 15);
    elseif session == 2
        home = getWellLoc(coords.behaviour.pos, coords.behaviour.speed, 29);
    end    
    
    % Make a massive table that matches diffmaps table, but with stats
    stats = table();
    stats.cell = diffmaps.cell;
    stats.ripple = diffmaps.ripple;
    stats.spikepos = diffmaps.pos;
    stats.animalpos = ripples.pos(stats.ripple,:);
    stats.time = ripples.tpeak(stats.ripple);
    stats.home = repmat(home,[size(stats,1), 1]);
    
    % Prepare stats ingredients
    roi = false(size(diffmaps.map,2), size(diffmaps.map,3));
    radius = 4;
    center = floor(size(diffmaps.map,2)/2);
    for row = (center-radius):(center+radius)
        for col = ceil(center-sqrt(radius^2 - (center-row)^2)):floor(center+sqrt(radius^2 - (center-row)^2))
            roi(row, col) = true;
        end
    end
    angles = linspace(0,2*pi,21);
    angles = angles(2:end); % Because linspace includes end value
    %steps = linspace(0,50,20); % This is in bins, not cm! How about making it *actual* bins?
    steps = 0:49;
    
    % Prepare the stats arrays
    stats.roi = nan(size(stats,1), 1);
    stats.rad = nan(size(stats,1), length(steps));
    
    % Now calculate stats for each diffmap
    for currDiffMap = 1:size(diffmaps,1)
        currmap = squeeze(diffmaps.map(currDiffMap,:,:));
        % Roi stats: mean within circular roi at center of map
        stats.roi(currDiffMap) = nanmean(currmap(roi));
        % Radial stats: mean across radial lines in all directions
        currLines = nan(length(angles), length(steps));
        for currLine = 1:length(angles)
            currLines(currLine, :) = interp2(1:size(currmap,1), 1:size(currmap,2), currmap, center + steps*cos(angles(currLine)), center + steps*sin(angles(currLine)));
        end
        stats.rad(currDiffMap,:) = nanmean(currLines,1);  
        % Display progress
        if mod(currDiffMap, 100) == 1
            disp(['Finished spike ' num2str(currDiffMap) '/' num2str(size(diffmaps,1))]);
        end
    end    
end
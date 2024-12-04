function corrs = getSessionCorrs(rat)
    rats = {'Janni','Harpy','Imp','Naga'};
    sessions = {'Open1','Open2'};
    session = 2;
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
    % Get current home well
    home = getWellLoc(coords.behaviour.pos, coords.behaviour.speed, 29);
        
    % Get previous home well location
    prevDatDir = fullfile('/Volumes/My Passport for Mac/PfeifferFoster_data/DataForBehrensBakermans/', ...
        rats{rat}, sessions{session-1});
    prevCoords = load(fullfile(prevDatDir, 'Output', 'coords.mat')); prevCoords = prevCoords.coords;        
    prevHome = getWellLoc(prevCoords.behaviour.pos, prevCoords.behaviour.speed, 15);

    % Re-align difference maps to home well and previous home well
    homeMaps = getRealignedMaps(diffMap, diffPos, home);
    prevHomeMaps = getRealignedMaps(diffMap, diffPos, prevHome);

    % The make total ratemap change per cell for home and prevhome
    cellHomeMaps = nan(200, 200, size(cells,1));
    cellPrevHomeMaps = nan(200, 200, size(cells,1));
    for currCell = 1:size(cells,1)
        currHomeMap = nan(200);
        currPrevHomeMap = nan(200);
        rippleCount = 0;
        for currRipple = 1:size(ripples,1)
            if ~isempty(homeMaps{currRipple}{currCell})
                newHomeMap = nanmean(homeMaps{currRipple}{currCell},3);
                currHomeMap(isnan(currHomeMap) & ~isnan(newHomeMap)) = 0;
                currHomeMap(~isnan(newHomeMap)) = currHomeMap(~isnan(newHomeMap)) + newHomeMap(~isnan(newHomeMap));
                newPrevHomeMap = nanmean(prevHomeMaps{currRipple}{currCell},3);
                currPrevHomeMap(isnan(currPrevHomeMap) & ~isnan(newPrevHomeMap)) = 0;
                currPrevHomeMap(~isnan(newPrevHomeMap)) = currPrevHomeMap(~isnan(newPrevHomeMap)) + newPrevHomeMap(~isnan(newPrevHomeMap));
                rippleCount = rippleCount + 1;
            end
        end
        cellHomeMaps(:,:,currCell) = currHomeMap / rippleCount;
        cellPrevHomeMaps(:,:,currCell) = currPrevHomeMap / rippleCount;
        disp(['Finished cell ' num2str(currCell) '/' num2str(size(cells,1))]);
    end

    % Correlate the two
    homePrevHome = nan(size(cells,1),1);
    for currCell = 1:length(homePrevHome)
        currHomeMap = cellHomeMaps(:,:, currCell);
        currPrevHomeMap = cellPrevHomeMaps(:,:, currCell);
        include = ~isnan(currHomeMap) & ~isnan(currPrevHomeMap);
        if any(include(:))
            homePrevHome(currCell) = corr(currHomeMap(include), currPrevHomeMap(include));
        end
    end
    
    % Stick them in a table
    corrs = table();
    corrs.corr = homePrevHome;
    corrs.curr = permute(cellHomeMaps, [3,1,2]);
    corrs.prev = permute(cellPrevHomeMaps, [3,1,2]);
end
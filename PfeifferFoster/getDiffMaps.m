function diffmaps = getDiffMaps(diffMap, diffPos)
    % Find out what the total number of diffmaps is across ripples and cells
    nMaps = sum(cellfun(@(x) sum(~cellfun(@isempty, x)), diffMap));
    % The table will have one row per diffmap
    diffmaps = table();
    % Create rows for maps, positions, cells and ripples
    diffmaps.cell = nan(nMaps,1);
    diffmaps.ripple = nan(nMaps,1);
    diffmaps.pos = nan(nMaps,2);
    diffmaps.map = nan(nMaps, 200, 200);
    % Build table row by row, for every cell and ripple
    currRow = 1;
    for currRipple = 1:size(diffMap,1)
        for currCell = 1:size(diffMap{currRipple},1)
            if ~isempty(diffMap{currRipple}{currCell})
                diffmaps.map(currRow,:,:) = nanmean(diffMap{currRipple}{currCell},3);
                diffmaps.pos(currRow,:) = mean(diffPos{currRipple}{currCell},1);
                diffmaps.cell(currRow) = currCell;
                diffmaps.ripple(currRow) = currRipple;
                currRow = currRow + 1;
            end
        end
        disp(['Finished ripple ' num2str(currRipple) '/' num2str(size(diffMap,1))]);
    end
end
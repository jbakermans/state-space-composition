function getRippleRatemapChangeFile(currRipple, ripplesFile, cellsFile, coordsFile, outDir, doPlot, doExcludeJumps, doThreshold)
    % Final arguments are optional, set defaults here
    if nargin < 7
        doPlot = false;
        doExcludeJumps = false;
        doThreshold = false;
    end
    % Load ripples, cells, coords
    ripples = load(ripplesFile); ripples = ripples.ripples;
    cells = load(cellsFile); cells = cells.cells;
    coords = load(coordsFile); coords = coords.coords;
    % Run ripple detection
    [diffMap, diffPos] = getRippleRatemapChange(currRipple, ripples, cells, coords, doPlot, doExcludeJumps, doThreshold, 1:size(cells,1));
    % Save result
    save(fullfile(outDir, ['diff_' num2str(currRipple)]), 'diffMap', 'diffPos');
end
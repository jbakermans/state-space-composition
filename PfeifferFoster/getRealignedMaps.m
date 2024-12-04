function alignedMaps = getRealignedMaps(diffMap, diffPos, alignPos)
    alignedMaps = cell(size(diffMap));
    for currRipple = 1:length(alignedMaps)
        alignedMaps{currRipple} = cell(size(diffMap{currRipple}));
        for currCell = 1:length(alignedMaps{currRipple})
            alignedMaps{currRipple}{currCell} = nan(size(diffMap{currRipple}{currCell}));
            if ~isempty(alignedMaps{currRipple}{currCell})
                for currMap = 1:size(diffMap{currRipple}{currCell}, 3)
                    % Find what location the current center of the map is at
                    currPos = diffPos{currRipple}{currCell}(currMap,:);
                    % Find how to shift it to get it to the right place
                    shift = round((currPos - alignPos)/2);
                    % Grab the currently relevant map
                    mapFrom = diffMap{currRipple}{currCell}(:,:,currMap);
                    % And copy it over, after shifting, to the new map
                    alignedMaps{currRipple}{currCell}(...
                        max(1, 1 + shift(1)):min(size(mapFrom,1), size(mapFrom,1) + shift(1)), ...
                        max(1, 1 + shift(2)):min(size(mapFrom,2), size(mapFrom,2) + shift(2)), currMap) = ...
                        mapFrom(...
                        max(1, 1 - shift(1)):min(size(mapFrom,1), size(mapFrom,1) - shift(1)), ...
                        max(1, 1 - shift(2)):min(size(mapFrom,2), size(mapFrom,2) - shift(2)));
                    % To check results: plot both
%                     figure(); 
%                     subplot(1,2,1); hold on; imagesc([1 399], [1 399], mapFrom'); scatter(currPos(1)+200-currPos(1), currPos(2)+200-currPos(2), 'rx'); scatter(alignPos(1)+200-currPos(1), alignPos(2)+200-currPos(2), 'gx');
%                     subplot(1,2,2); hold on; imagesc([1 399], [1 399], alignedMaps{currRipple}{currCell}(:,:,currMap)'); scatter(currPos(1)+200-alignPos(1), currPos(2)+200-alignPos(2), 'rx'); scatter(alignPos(1)+200-alignPos(1), alignPos(2)+200-alignPos(2), 'gx');
                end
            end
        end
        disp(['Finished ripple ' num2str(currRipple) '/' num2str(size(diffMap,1))]);
    end    
end
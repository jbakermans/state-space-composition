function [ratemap, spikemap] = getRatemap(spikePos, dwellmap)
    % Now make spike map: number of spikes in every bin
    spikemap = histcounts2(spikePos(:,1), spikePos(:,2), 0:2:200, 0:2:200);
    spikemap = imgaussfilt(spikemap, 2); % 4cm smoothing, so 2 bins
    % The dwellmap describes time spend in bin in seconds, 
    % spikemap number of spikes in that bin, so ratemap is rate in Hz
    ratemap = nan(size(spikemap));
    ratemap(dwellmap > 0) = spikemap(dwellmap>0) ./ dwellmap(dwellmap>0);    
end
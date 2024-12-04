function ripples = getRipples(Ripple_Events, Spike_Data, cells, coords)
    % Extract values from coords struct
    speed = coords.behaviour.speed;
    pos = coords.behaviour.pos;
    time = coords.behaviour.time;
    binWidth = coords.spikes.dt;
    bins = coords.spikes.bin;
    times = coords.spikes.time;
    % Ripple data has columns [start, end, peak, power, all-data z-scored power, epoch z-scored power]
    % Make a table of all ripples
    % Filter out all ripples between start and end of session
    Ripple_Events = Ripple_Events(Ripple_Events(:,1) > time(1) & Ripple_Events(:,2) < time(end),:);
    ripples = table();
    % Copy values from columns
    ripples.tstart = Ripple_Events(:,1);
    ripples.tend = Ripple_Events(:,2);
    ripples.tpeak = Ripple_Events(:,3);
    ripples.power = Ripple_Events(:,4);
    ripples.z = Ripple_Events(:,6);
    % Convert times to bins
    ripples.peak = ceil((ripples.tpeak - times(1))/(times(2)-times(1)));
    % Get position and speed for each ripple
    ripples.pos = zeros(size(ripples,1),2);
    ripples.speed = zeros(size(ripples,1),1);
    for currRipple = 1:size(ripples,1)
        ripples.pos(currRipple,:) = interp1(time, pos, ripples.tpeak(currRipple));
        ripples.speed(currRipple) = interp1(time, speed, ripples.tpeak(currRipple));
    end
    % Set whether this was a home well ripple (within 10 cm of home well)
    ripples.home = sqrt(sum((ripples.pos - getHomeWell(pos, speed)).^2,2)) < 10;

    % Ripples usually go with high population spiking activity
    % To determine the start and the end, I'll smooth the population activity, 
    % then find where that population activity passes through the mean activity
    % Set kernel for spike histogram smoothing with 10ms standard deviation
    kernel = gausswin(100, (100-1)/(2 * 10e-3 / binWidth));  
    kernel = kernel/sum(kernel);
    % Get histogram of spikes when speed < 5 cm/s
    counts = histcounts(Spike_Data(interp1(time, speed, Spike_Data(:,1)) < 5,1), bins);
    rate = conv(counts, kernel, 'same') / binWidth;
    % Get mean rate
    meanRate = mean(rate(rate>1e-12));
    % Find for each ripple peak where the population rate crosses the mean before and after
    for currRipple = 1:size(ripples,1)
        ripples.start(currRipple) = ripples.peak(currRipple) + 1 - find(rate(ripples.peak(currRipple):-1:1) < meanRate, 1, 'first');
        ripples.end(currRipple) = ripples.peak(currRipple) - 1 + find(rate(ripples.peak(currRipple):end) < meanRate, 1, 'first');
        disp(['Finished ripple ' num2str(currRipple) '/' num2str(size(ripples,1)) ' definitions']);    
    end

    % We hypothesise that replay changes ratemaps. For each ripple, calculate ratemaps before & after for each cell
    % ripples.dwell_before = cell(size(ripples,1),1);
    % ripples.dwell_after = cell(size(ripples,1),1);
    % ripples.rate_before = cell(size(ripples,1),size(cells,1));
    % ripples.rate_before = cell(size(ripples,1),size(cells,1));
    % ripples.rate_diff = cell(size(ripples,1),size(cells,1));
    % for currRipple = 1:size(ripples,1)
    %     before = speed > 5 & time' < ripples.tpeak(currRipple) - 1;
    %     after = speed > 5 & time' > ripples.tpeak(currRipple) + 1;
    %     ripples.dwell_before{currRipple} = getDwellmap(pos(before,:), time(2)-time(1));
    %     ripples.dwell_after{currRipple} =  getDwellmap(pos(after,:), time(2)-time(1));
    %     for currCell = 1:size(cells,1)
    %         before = cells.spikes{currCell}.speed > 5 & cells.spikes{currCell}.spike < ripples.tpeak(currRipple) - 1;
    %         after = cells.spikes{currCell}.speed > 5 & cells.spikes{currCell}.spike > ripples.tpeak(currRipple) + 1;
    %         [ripples.rate_before{currRipple, currCell}, ~] = getRatemap(cells.spikes{currCell}.pos(before, :), ripples.dwell_before{currRipple});                
    %         [ripples.rate_after{currRipple, currCell}, ~] = getRatemap(cells.spikes{currCell}.pos(after, :), ripples.dwell_after{currRipple});                        
    %         ripples.rate_diff{currRipple, currCell} = ripples.rate_after{currRipple, currCell} - ripples.rate_before{currRipple, currCell};
    %     end
    %     disp(['Finished ripple ' num2str(currRipple) '/' num2str(size(ripples,1)) ' ratemaps']);
    % end

    % And the way that replay changes ratemaps, is by path-integrating there. Decode trajectories during ripples
    ripples.decoded = cell(size(ripples,1),1);
    ripples.path = cell(size(ripples,1),1);
    ripples.posterior = cell(size(ripples,1),1);
    % Progress through ripple at 5 ms steps; calculate corresponding bins
    windowStep = round(5e-3 / binWidth);
    % Decode in 20 ms windows; calculate corresponding bins
    windowSize = round(20e-3 / binWidth);
    % Prepare 3d matrix of ratemaps, to feed into decoding
    ratemaps = permute(cat(3, cells.ratemap{:}), [3,1,2]);
    % Then precalculate the log of that, because decoding will need log against underflow
    log_ratemaps = log(ratemaps); 
    log_ratemaps(log_ratemaps == -inf) = -999;
    % Also pre-calculate the sum of ratemaps which occurs in the decoding exponent: it's always the same
    sum_ratemaps = sum(ratemaps,1);    
    % Now run through all trajectories and decode positions in each
    for currRipple = 1:size(ripples,1)
        % Make list of bin centers for this event
        windowCenters = ripples.start(currRipple):windowStep:ripples.end(currRipple);
        % Then decode position in windows around each of these
        decoded = nan(length(windowCenters), size(cells.ratemap{1},1), size(cells.ratemap{1},2));
        for currWindow = 1:length(windowCenters)
            windowBins = (windowCenters(currWindow) - floor(windowSize/2) + 1):(windowCenters(currWindow) + ceil(windowSize/2));
            decoded(currWindow, :, :) = decodePosition(sum_ratemaps, log_ratemaps, sum(cells.counts(:,windowBins), 2), length(windowBins)*binWidth);
        end
        % Determine peak posterior probability at every step
        [~, peakIndex] = max(reshape(decoded,[size(decoded,1), size(decoded,2)*size(decoded,3)]), [], 2);
        [peakRow, peakCol] = ind2sub([size(decoded,2), size(decoded,3)], peakIndex);
        % Convert decoded bins to x,y positions
        locs = [peakRow, peakCol]*2; % in cm
        % Update table
        ripples.path{currRipple} = locs;
        ripples.decoded{currRipple} = decoded;
        ripples.posterior{currRipple} = squeeze(nanmean(decoded, 1));    
        disp(['Finished ripple ' num2str(currRipple) '/' num2str(size(ripples,1)) ' decoding']);    
    end
end
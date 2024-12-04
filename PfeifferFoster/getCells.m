function [cells, coords] = getCells(Position_Data, Spike_Data, Excitatory_Neurons, binWidth)
    % Set default spike histogram bin width to 1 ms
    if nargin == 3
        binWidth = 1e-3;
    end
    % Get position and time 
    pos_orig = Position_Data(:,[2,3]); % in cm
    time_orig = Position_Data(:,1); % in s
    % Resample position to regular times (framerate 1/30) so time-bins are identical
    % Thas useful 1) for determining the bin of a given spike (no search required),
    % and 2) for ratemaps, so that each bin has equal size
    dt = 1/30; % in s
    % Define new regular timescale; drop the first timepoint, because step is usually < 1/30
    time = time_orig(1):dt:time_orig(end);
    % Interpolate position along this timescale
    pos = interp1(time_orig,pos_orig,time);
    % But sometimes there are gaps in tracking, so the steps exceed the time step. Exclude those
    exclude = abs(time - interp1(time_orig, time_orig, time, 'nearest')) > dt;
    pos(exclude,:) = nan;
    disp(['Excluded ' num2str(sum(exclude)) '/' num2str(size(pos,1)) ' timesteps because of failed tracking']);
    % Calculate speed
    speed = [0; sqrt(sum(diff(pos).^2,2)) / dt]; % in cm/s    
    % Create time bins for spike histogram: 1ms bins between start and stop
    bins = time(1):binWidth:time(end);
    times = bins(2:end) - 0.5*binWidth;
    % Create coords structure: all info about behavioural and spike timings
    coords = struct();
    coords.behaviour = struct();
    coords.behaviour.pos = pos;
    coords.behaviour.speed = speed;
    coords.behaviour.time = time;
    coords.behaviour.dt = dt;
    coords.spikes = struct();
    coords.spikes.bin = bins;
    coords.spikes.time = times;
    coords.spikes.dt = binWidth;
    % Set kernel for spike histogram smoothing with 10ms standard deviation
    % Standard deviation sigma = (width-1)/(2*alpha) = 10ms, 
    % so for a 100-sized window, set alpha to (width-1)/(2 * 10ms / binWidth) = 99/20
    kernel = gausswin(100, (100-1)/(2 * 10e-3 / binWidth));  
    kernel = kernel/sum(kernel);
    % Get dwellmap
    dwellmap = getDwellmap(pos(speed > 5, :), dt);
    % Get all neurons
    neurons = unique(Spike_Data(:,2));
    % Make a big table across neurons
    cells = table();
    cells.ids = neurons;
    cells.spikes = cell(size(neurons));
    %cells.histogram = cell(size(neurons));
    cells.spikemap = cell(size(neurons));
    cells.dwellmap = cell(size(neurons));
    cells.ratemap = cell(size(neurons));
    for currNeuronIdx = 1:length(neurons)
        currNeuron = neurons(currNeuronIdx);
        % Create sub-table of spikes
        cells.spikes{currNeuronIdx} = table();
        cells.spikes{currNeuronIdx}.spike = Spike_Data(Spike_Data(:,2)==currNeuron,1);
        % Calculate for each spike which bin it belongs to
        cells.spikes{currNeuronIdx}.bin = ceil((cells.spikes{currNeuronIdx}.spike - time(1))/dt);
        % Set any bins outside the range to nan
        cells.spikes{currNeuronIdx}.bin(cells.spikes{currNeuronIdx}.bin < 1 | cells.spikes{currNeuronIdx}.bin > length(time)) = nan;
        % Initialise time, pos, and speed as nan
        cells.spikes{currNeuronIdx}.time(:) = nan;
        cells.spikes{currNeuronIdx}.pos = nan(size(cells.spikes{currNeuronIdx}.spike,1),2);
        cells.spikes{currNeuronIdx}.speed(:) = nan;
        % Then copy over all entries from non-nan bin
        notnan = ~isnan(cells.spikes{currNeuronIdx}.bin);
        cells.spikes{currNeuronIdx}.time(notnan) = time(cells.spikes{currNeuronIdx}.bin(notnan));
        cells.spikes{currNeuronIdx}.pos(notnan,:) = pos(cells.spikes{currNeuronIdx}.bin(notnan),:);
        cells.spikes{currNeuronIdx}.speed(notnan) = speed(cells.spikes{currNeuronIdx}.bin(notnan));    
        % Calculate spike histogram - not needed for now, so comment to make it faster
%         data.histogram{currNeuronIdx} = table();
%         data.histogram{currNeuronIdx}.time = bins(1:(end-1)) + binWidth * 0.5;
%         data.histogram{currNeuronIdx}.counts = histcounts(data.spikes{currNeuronIdx}.spike, bins);
%         data.histogram{currNeuronIdx}.rate = conv(data.histogram{currNeuronIdx}.counts, kernel, 'same') / binWidth;
        % But let's keep the counts - will make later decoding much faster
        cells.counts(currNeuronIdx,:) = histcounts(cells.spikes{currNeuronIdx}.spike, bins);
        % Calculate ratemap 
        [ratemap, spikemap] = getRatemap(cells.spikes{currNeuronIdx}.pos(notnan & cells.spikes{currNeuronIdx}.speed > 5, :), dwellmap);        
        cells.spikemap{currNeuronIdx} = spikemap;
        cells.dwellmap{currNeuronIdx} = dwellmap;
        cells.ratemap{currNeuronIdx} = ratemap;
        % Set whether neuron is excitatory
        cells.excitatory(currNeuronIdx) = ismember(currNeuron, Excitatory_Neurons);
        % Set whether neuron has place field: excitatory and peak > 1Hz
        cells.place(currNeuronIdx) = cells.excitatory(currNeuronIdx) & ...
            max(cells.ratemap{currNeuronIdx}(dwellmap > 0)) > 1;
        % Display progress
        disp(['Finished cell ' num2str(currNeuronIdx) ' / ' num2str(length(neurons))]);
    end    
end
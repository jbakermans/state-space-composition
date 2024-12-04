function [door_spikes, ratemap_changes] = getReplayMapChange(rat, date, behaviour, spikes, partn, flip)
% Get behavioural info
% Find the right behaviour row
brow = find(contains(behaviour.rat, rat) & behaviour.date == date & behaviour.partn == partn);
% Also hard-code the behaviour row which contains the closed doors
brow_close = find(contains(behaviour.rat, rat) & behaviour.date == date & behaviour.partn == 3);
% Get positions and times for this session
sess_pos = behaviour.positions{brow}(:,1:2);
sess_time = behaviour.positions{brow}(:,3);
dt = sess_time(2)-sess_time(1);
% Get door properties
doors = {'AB', 'BA', 'BC', 'CB', 'CD', 'DC', 'DA', 'AD'};
door_pos = [ ...
mean(behaviour.epoly{brow}([23 22 30 29],:));
mean(behaviour.epoly{brow}([27 30 33 34],:));...
mean(behaviour.epoly{brow}([31 34 38 37],:));...
mean(behaviour.epoly{brow}([38 39 21 22],:))]; % Order: 1 (AB), 2 (BC), 3 (CD), 4 (DA)
% Get the currently closed doors
curr_closed = zeros(size(doors));
for curr_door = 1:length(curr_closed)
    curr_closed(curr_door) = contains(behaviour.closed_doors{brow_close}, doors{curr_door});
end
curr_closed = find(curr_closed);
% Get world coordinates
epoly = behaviour.epoly{brow};
epoly2 = makeEPOLY2(epoly,9);
map_limits = [min(epoly(:,1))-5 max(epoly(:,1))+5; min(epoly(:,2))-5 max(epoly(:,2))+5]; 
% Then there is a 2cm padding on both sides,
% and then the limits are exactly 170 cm apart, so 85 x 2cm bins!
map_limits = map_limits + [-2, 2; -2, 2];

% Get spike info
% Find all relevant rows
srow = find(contains(spikes.rat, rat) & spikes.date == date & spikes.partn == partn);
% Get the corresponding cell id: each cell has 5 parts/sessions
cells = floor(srow/5) + 1;
% Get matrix of spike times
spike_times = spikes.spike_times(srow);

% Find replay spikes and ratemap changes
% Create cell arrays to hold results for this rat on this date
% Columns: door spikes, non-local door spikes, all spikes, all non-local spikes, new place fields
door_spikes = nan(length(cells),5);
% Columns: absolute ratemap difference, absolute ratemap sum, positive ratemap difference, total rate after
ratemap_changes = nan(length(cells),2);

% Select particular cell to track
for curr_cell = 1:length(cells)
    % Only look for replay if this is a place cell
    if ~strcmp(spikes.cell_type_name(srow), 'place_cell')
        disp(['Rat ' rat ' date ' num2str(date) ' cell ' num2str(curr_cell) ': not a place cell'])        
        continue;
    end    

    % Exclude cells that never fire
    if max(spikes.frate(max(1,(srow(curr_cell)-1)):min(size(spikes,1),(srow(curr_cell)+1)))) == 0
        disp(['Rat ' rat ' date ' num2str(date) ' cell ' num2str(curr_cell) ': no spikes'])        
        continue;
    end                

    % Find all firing fields to make sure that spikes are non-local
    if partn==1 || (partn==5 && flip)
        before_chulls={};
    else
        before_chulls = spikes.place_field_data_chulls{srow(curr_cell) - 1 + 2*flip};
    end
    if partn==5 || (partn==1 && flip)
        after_chulls={};
    else
        after_chulls = spikes.place_field_data_chulls{srow(curr_cell) + 1 - 2*flip};
    end
    % If there are no fields before: set it to empty cell array, the expected format
    if isempty(before_chulls)
        before_chulls = {};
    end
    if isempty(before_chulls)
        after_chulls = {};
    end
    all_chulls = [before_chulls; after_chulls];
    % Convert all convex hull coordinates (in ratemap pixels) to real-world position
    % Also convert them to polygon shape objects
    for curr_hull = 1:length(all_chulls)
        % Simplify convex hull: it contains unnecessary vertices 
        % (e.g. on a straight line between two points)
        all_chulls{curr_hull} = ...
            all_chulls{curr_hull}(convhull(all_chulls{curr_hull}, 'Simplify', true),:);
        % Convert coordinates from ratemap pixels to real world positions
        all_chulls{curr_hull} = pixToPos(...
            [1, size(spikes.aligned_all_ratemap{srow(curr_cell)-1},1);...
            1, size(spikes.aligned_all_ratemap{srow(curr_cell)-1},2)], ...
            map_limits, all_chulls{curr_hull});
        % Create a polyshape for further processing (e.g. overlaps)
        all_chulls{curr_hull} = polyshape(all_chulls{curr_hull});
    end                              
    % Find new chulls
    do_overlap = false(length(after_chulls), 1);
    for curr_after = 1:length(after_chulls)
        for curr_before = 1:length(before_chulls)            
            % Find if the current *after* hull overlaps with any *before*
            curr_overlap = intersect(all_chulls{curr_before}, all_chulls{length(before_chulls) + curr_after});
            if curr_overlap.NumRegions > 0
                % There only needs to be a single overlap, so break when found
                do_overlap(curr_after) = true;
                break;
            end
        end
    end
    new_fields = find(~do_overlap);
        
    % Get map before and after door discovery session
    if partn==1
        before = nan(size(spikes.aligned_forage_ratemap{srow(curr_cell)}));
    else
        before = spikes.aligned_forage_ratemap{srow(curr_cell)-1};
    end
    if partn==5
        after = nan(size(spikes.aligned_forage_ratemap{srow(curr_cell)}));
    else
        after = spikes.aligned_forage_ratemap{srow(curr_cell)+1};
    end
    % Keep only valid locations, included in both maps
    valid = ~isnan(before) & ~isnan(after);
    % Get the absolute difference in firing rate
    difference = after(valid) - before(valid);

    % Find the location of each spike
    spike_time_ids = zeros(size(spike_times{curr_cell}));
    for curr_spike = 1:length(spike_time_ids)
        spike_time_ids(curr_spike) = find((sess_time + dt)>spike_times{curr_cell}(curr_spike),1);
    end
    spike_locs = behaviour.positions{brow}(spike_time_ids,1:2);
    % Get distance to closed door for each spike location
    spike_door_dist = zeros(size(spike_locs,1), length(doors));
    for curr_door = 1:length(doors)
        spike_door_dist(:,curr_door) = sqrt(sum((spike_locs - door_pos(ceil(curr_door/2),:)).^2,2));
    end
    % Find spikes with minimum distance to any closed door below threshold
    spike_at_door = min(spike_door_dist(:, curr_closed),[],2) < 10;
    % Find if this spike occured within any firing field
    if isempty(all_chulls) || isempty(spike_locs)
        spike_is_local = false(size(spike_at_door));
    else
        spike_is_local = isinterior(union([all_chulls{:}]), spike_locs(:,1),spike_locs(:,2));             
    end

    % Append current value to results cell arrays
    door_spikes(curr_cell, 1) = sum(spike_at_door);
    %door_spikes(curr_cell, 2) = sum(~spike_at_door & ~spike_is_local & behaviour.speed{brow}(spike_time_ids,:) < 2);
    door_spikes(curr_cell, 2) = sum(spike_at_door & ~spike_is_local & behaviour.speed{brow}(spike_time_ids,:) < 2);
    door_spikes(curr_cell, 3) = length(spike_time_ids);        
    door_spikes(curr_cell, 4) = sum(~spike_is_local);        
    door_spikes(curr_cell, 5) = mean(spikes.frate((srow(curr_cell)-partn+1):(srow(curr_cell)-partn+5))); % mean(spikes.frate((srow(curr_cell)-1):(srow(curr_cell)+1)));        
    
    ratemap_changes(curr_cell, 1) = sum(abs(difference));         
    ratemap_changes(curr_cell, 2) = sum(abs(after(valid) + before(valid)));         
    ratemap_changes(curr_cell, 3) = sum(difference(difference > 0));
    ratemap_changes(curr_cell, 4) = sum(after(valid));
    ratemap_changes(curr_cell, 5) = sum(~do_overlap); 

    % Print summary
    if false
        disp(['Rat ' rat ' date ' num2str(date) ' cell ' num2str(curr_cell) ': ' ...
            num2str(door_spikes{curr_rat}{curr_date}(curr_cell, 2)) ' non-local door spikes (' num2str(100 * door_spikes{curr_rat}{curr_date}(curr_cell, 2) / door_spikes{curr_rat}{curr_date}(curr_cell, 3)) '%) , ' ...
            num2str(ratemap_changes{curr_rat}{curr_date}(curr_cell, 1)) ' ratemap change (' num2str(100 * ratemap_changes{curr_rat}{curr_date}(curr_cell, 1) / ratemap_changes{curr_rat}{curr_date}(curr_cell, 2)) '%).'])
    end
end
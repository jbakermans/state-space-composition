%% Load spikes and behaviour if not loaded already
if exist('spikes','var') == 0
    spikes = load('/Users/jbakermans/Documents/Data/DuvelleSpiers/Data/Four_Room_sdata_big.mat');
    spikes = spikes.sdata_big;
end
if exist('behaviour','var') == 0
    behaviour = load('/Users/jbakermans/Documents/Data/DuvelleSpiers/Data/Four_Room_bdata_big.mat');
    behaviour = behaviour.bdata_big;
end

%% Plot one example animal, one example session, with goal + foraging behaviour
brows = [32, 33, 34, 107,108,109]; % r35, 17012020 and r39, 2032020
figure('Position', [10, 10, 600, 300])
for currSess = 1:length(brows)
    brow = brows(currSess);
    % Get world coordinates
    epoly = behaviour.epoly{brow};
    epoly2 = makeEPOLY2(epoly,9);
    doors = {'AB', 'BA', 'BC', 'CB', 'CD', 'DC', 'DA', 'AD'};
    door_pos = [ ...
        mean(behaviour.epoly{brow}([23 22 30 29],:));
        mean(behaviour.epoly{brow}([27 30 33 34],:));...
        mean(behaviour.epoly{brow}([31 34 38 37],:));...
        mean(behaviour.epoly{brow}([38 39 21 22],:))]; % Order: 1 (AB), 2 (BC), 3 (CD), 4 (DA)
    curr_closed = zeros(size(doors));
    for curr_door = 1:length(curr_closed)
        curr_closed(curr_door) = contains(behaviour.closed_doors{brow}, doors{curr_door});
    end
    curr_closed = find(curr_closed);    
    % Get traces to plot
    foraging = behaviour.positions{brow}(:,1:2);
    foraging(behaviour.behaviour_index{brow}==1) = nan;
    goal = behaviour.positions{brow}(:,1:2);
    goal(behaviour.behaviour_index{brow}==2) = nan;
    % Plot both foraging and goal-directed behaviour
    subplot(2,3,currSess);
    hold on;
    plotDoors(door_pos, curr_closed);
    plot(foraging(:,1), foraging(:,2), 'Color', [0.4940 0.1840 0.5560]);
    plot(goal(:,1), goal(:,2),'Color', [0.4660 0.6740 0.1880]);
    plot(epoly2(:,1), epoly2(:,2), 'k');    
    hold off;
    axis equal;
    axis off;
    % Set title if on first row
    if currSess == 1
        title('Before session');
        legend('Foraging', 'Goal-directed');
    elseif currSess == 2
        title('Door closing session');
    elseif currSess == 3
        title('After session');        
    end
end

%% Plot two example cells that replay first, then have new place fiels
srow = find(contains(spikes.rat, 'r37') & spikes.date == 4022020 & spikes.partn == 3);
srows(1) = srow(1); % r37, 4022020, 1
srow = find(contains(spikes.rat, 'r39') & spikes.date == 2032020 & spikes.partn == 3);
srows(2) = srow(19); % r39, 2032020, 19
srow = find(contains(spikes.rat, 'r39') & spikes.date == 2032020 & spikes.partn == 3);
srows(3) = srow(29); % r39, 2032020, 19
brows = [48, 108, 108];
fields = [1,1,1];

srows = srows(1:2);
figure('Position', [10, 10, 600, 300])
for currRow = 1:length(srows)
    srow = srows(currRow);
    brow = brows(currRow);
    
    % Get ratemaps
    before = spikes.aligned_forage_ratemap{srow-1};
    after = spikes.aligned_forage_ratemap{srow+1};
    
    % Get coordinates
    epoly = behaviour.epoly{brow};
    epoly2 = makeEPOLY2(epoly,9);  
    map_limits = [min(epoly(:,1))-5 max(epoly(:,1))+5; min(epoly(:,2))-5 max(epoly(:,2))+5]; 
    map_limits = map_limits + [-2, 2; -2, 2];    
    sess_time = behaviour.positions{brow}(:,3);
    dt = sess_time(2)-sess_time(1);    
    
    % Get door properties
    doors = {'AB', 'BA', 'BC', 'CB', 'CD', 'DC', 'DA', 'AD'};
    door_pos = [ ...
        mean(behaviour.epoly{brow}([23 22 30 29],:));
        mean(behaviour.epoly{brow}([27 30 33 34],:));...
        mean(behaviour.epoly{brow}([31 34 38 37],:));...
        mean(behaviour.epoly{brow}([38 39 21 22],:))]; % Order: 1 (AB), 2 (BC), 3 (CD), 4 (DA)
    curr_closed = zeros(size(doors));
    for curr_door = 1:length(curr_closed)
        curr_closed(curr_door) = contains(behaviour.closed_doors{brow}, doors{curr_door});
    end
    curr_closed = find(curr_closed);
    
    
    % Find any new firing fields: convex hulls that don't overlap with previous
    before_chulls = spikes.place_field_data_chulls{srow- 1};
    after_chulls = spikes.place_field_data_chulls{srow + 1};
    % If there are no fields before: set it to empty cell array, the expected format
    if isempty(before_chulls)
        before_chulls = {};
    end
    % Convert all convex hull coordinates (in ratemap pixels) to real-world position
    % Also convert them to polygon shape objects
    for curr_hull = 1:length(before_chulls)
        % Simplify convex hull: it contains unnecessary vertices 
        % (e.g. on a straight line between two points)
        before_chulls{curr_hull} = ...
            before_chulls{curr_hull}(convhull(before_chulls{curr_hull}, 'Simplify', true),:);
        % Convert coordinates from ratemap pixels to real world positions
        before_chulls{curr_hull} = pixToPos(...
            [1, size(spikes.aligned_all_ratemap{srow-1},1);...
            1, size(spikes.aligned_all_ratemap{srow-1},2)], ...
            map_limits, before_chulls{curr_hull});
        % Create a polyshape for further processing (e.g. overlaps)
        before_chulls{curr_hull} = polyshape(before_chulls{curr_hull});
    end
    for curr_hull = 1:length(after_chulls)
        after_chulls{curr_hull} = ...
            after_chulls{curr_hull}(convhull(after_chulls{curr_hull}, 'Simplify', true),:);        
        after_chulls{curr_hull} = pixToPos(...
            [1, size(spikes.aligned_all_ratemap{srow+1},1);...
            1, size(spikes.aligned_all_ratemap{srow+1},2)], ...
            map_limits, after_chulls{curr_hull});
        after_chulls{curr_hull} = polyshape(after_chulls{curr_hull});        
    end
    % Find for each *after* convex hull if overlaps with any *before* convex hull
    do_overlap = false(length(after_chulls), 1);
    for curr_after = 1:length(after_chulls)
        for curr_before = 1:length(before_chulls)            
            % Find if the current *after* hull overlaps with any *before*
            curr_overlap = intersect(before_chulls{curr_before}, after_chulls{curr_after});
            if curr_overlap.NumRegions > 0
                % There only needs to be a single overlap, so break when found
                do_overlap(curr_after) = true;
                break;
            end
        end
    end
    % Now we know which fields have appeared after the session
    new_fields = find(~do_overlap);
    % Find the location of each spike
    spike_time_ids = zeros(size(spikes.spike_times{srow}));
    for curr_spike = 1:length(spike_time_ids)
        spike_time_ids(curr_spike) = find((sess_time + dt)>spikes.spike_times{srow}(curr_spike),1);
    end
    spike_locs = behaviour.positions{brow}(spike_time_ids,1:2);
    spike_speeds = behaviour.speed{brow}(spike_time_ids);
    % Get distance to closed door for each spike location
    spike_door_dist = zeros(size(spike_locs,1), length(doors));
    for curr_door = 1:length(doors)
        spike_door_dist(:,curr_door) = sqrt(sum((spike_locs - door_pos(ceil(curr_door/2),:)).^2,2));
    end
    % Find all local spikes: spikes that occured while animal ran through firing field
    spike_is_local = isinterior(union([before_chulls{:}, after_chulls{:}]), spike_locs(:,1),spike_locs(:,2));
    % Then find first local spike for each new field    
    first_in_field = zeros(length(new_fields));
    for curr_new = 1:length(new_fields)
        curr_first = find(isinterior(after_chulls{new_fields(curr_new)},spike_locs(:,1),spike_locs(:,2)),1);
        % Now if there is simply no local spike: set it to end of session
        if isempty(curr_first)
            curr_first = size(spike_locs,1);
        end
        first_in_field(curr_new) = curr_first;
    end
    % Final question: was there a non-local spike near the door before first local spike?
    replay = false(length(new_fields),1);
    replay_spikes = cell(size(replay));
    for curr_new = 1:length(new_fields)
        replay_spikes{curr_new} = find(min(spike_door_dist(1:first_in_field(curr_new), curr_closed),[],2) < 10 & ...
            ~spike_is_local(1:first_in_field(curr_new)) & spike_speeds(1:first_in_field(curr_new)) < 5);
        replay(curr_new) = ~isempty(replay_spikes{curr_new});
    end
    % Set cutoff for plotting: could be first local spike, or last replay spike
    cutoff = replay_spikes{fields(currRow)}(end); % first_in_field(fields(currRow))    
    
    % First subplot: ratemap before
    subplot(length(srows),4,(currRow-1)*4+1);
    hold on;
    plot(epoly2(:,1), epoly2(:,2));    
    if numel(unique(before(~isnan(before)))) == 1
        imagesc(before, 'AlphaData',~isnan(before),'XData', map_limits(1,:), 'YData', map_limits(2,:),...
            [min(before(~isnan(before))), min(before(~isnan(before)))+1]);
    else
        imagesc(before,'AlphaData',~isnan(before),'XData', map_limits(1,:), 'YData', map_limits(2,:));
    end
    %plot([before_chulls{:}]);
    axis square; axis off;
    if currRow == 1
        title('Session before');
    end

    % Second subplot: behaviour + spikes *before* and *after* first local field spike
    ax = subplot(length(srows),4,(currRow-1)*4+2);    
    hold on;
    % Plot arena
    plot(epoly2(:,1), epoly2(:,2));
    plotDoors(door_pos, curr_closed);        
    % Plot firing field
    plot(after_chulls{new_fields(fields(currRow))});                                
    % Plot behaviour up to and after first local spike of this field
    plot(behaviour.positions{brow}(1:spike_time_ids(cutoff),1),behaviour.positions{brow}(1:spike_time_ids(cutoff),2),'Color', [0.5 0.5 0.5]);
    %plot(behaviour.positions{brow}(1:spike_time_ids(first_in_field(fields(currRow))),1),behaviour.positions{brow}(1:spike_time_ids(first_in_field(curr_new)),2),'Color', [0.5 0.5 0.5]);
    % Plot spikes up to and after first local spike of this field
    scatter(spike_locs(1:cutoff,1),spike_locs(1:cutoff,2), 100, [0 0.4470 0.7410], '.');
    %scatter(spike_locs(1:first_in_field(fields(currRow)),1),spike_locs(1:first_in_field(fields(currRow)),2), 100, [0 0.4470 0.7410], '.');
    % Plot non-local door spikes before first local spike of this field
    scatter(spike_locs(replay_spikes{fields(currRow)},1),spike_locs(replay_spikes{fields(currRow)},2), 150, [1,1,1], 'x', 'LineWidth',4);    
    scatter(spike_locs(replay_spikes{fields(currRow)},1),spike_locs(replay_spikes{fields(currRow)},2), 100, [0 0.4470 0.7410], 'x', 'LineWidth',2);
    hold off;
    axis equal; axis off;    
    if currRow == 1
        title('Before replay');
    end    
    
    % Second subplot: behaviour + spikes *before* and *after* first local field spike
    ax = subplot(length(srows),4,(currRow-1)*4+3);    
    hold on;
    % Plot arena
    plot(epoly2(:,1), epoly2(:,2));
    plotDoors(door_pos, curr_closed);        
    % Plot firing field
    plot(after_chulls{new_fields(fields(currRow))});                                
    % Plot behaviour up to and after first local spike of this field
    plot(behaviour.positions{brow}(spike_time_ids(cutoff):end,1),behaviour.positions{brow}(spike_time_ids(cutoff):end,2),'Color', [0.5 0.5 0.5]);
    % Plot spikes up to and after first local spike of this field
    scatter(spike_locs(cutoff:end,1),spike_locs(cutoff:end,2), 100,[0.8500 0.3250 0.0980], '.');    
    hold off;
    axis equal; axis off;   
    if currRow == 1
        title('After replay');
    end        
    
    % Fourth subplot: ratemap after    
    subplot(length(srows),4,(currRow-1)*4+4);
    hold on;
    plot(epoly2(:,1), epoly2(:,2));    
    imagesc(after,'AlphaData',~isnan(after),'XData', map_limits(1,:), 'YData', map_limits(2,:));
    %plot([after_chulls{:}]);
    axis square; axis off;
    if currRow == 1
        title('Session after');
    end
end

%% Plot stats

% Create list of rats
rats = {'r35';'r37';'r38';'r39';'r44'};
% And of all dates for each rat
dates = {[6012020;7012020;9012020;10012020;16012020;17012020;20012020;21012020;31122019], ...
    [4022020;5022020;6022020;7022020;16022020], ...
    [2022020;3022020;10022020;11022020;12022020;27012020], ...
    [2032020;3032020;4032020;8032020;29022020], ...
    [14032020;17032020;18032020;19032020;20032020]};
% Do the following for each session
all_spikes = cell(6,1);
all_changes = cell(6,1);
for session = 1:6
    % Store door spikes and ratemap change per rat, per session, per cell
    door_spikes = cell(size(rats));
    ratemap_changes = cell(size(rats));
    % Select one animal on one day
    for curr_rat = 1:length(rats)
        % Create cell arrays to hold results for this rat
        door_spikes{curr_rat} = cell(length(dates{curr_rat}),1);
        ratemap_changes{curr_rat} = cell(length(dates{curr_rat}),1);
        for curr_date = 1:length(dates{curr_rat})     
            % Select session
            rat = rats{curr_rat};
            date = dates{curr_rat}(curr_date);
            % Populate door spikes and ratemap changes
            if session == 6
                [curr_spikes, curr_changes] = getReplayMapChange(rat, date, behaviour, spikes, 3, true);
            else
                [curr_spikes, curr_changes] = getReplayMapChange(rat, date, behaviour, spikes, session, false);
            end
            door_spikes{curr_rat}{curr_date} = curr_spikes;
            ratemap_changes{curr_rat}{curr_date} = curr_changes;
            % Display progress
            disp(['Finished rat ' num2str(curr_rat) '/' num2str(length(rats)) ...
                ', session ' num2str(curr_date) '/' num2str(length(dates{curr_rat}))]);
        end
    end
    all_spikes{session} = door_spikes;
    all_changes{session} = ratemap_changes;
    disp(['FINISHED SESSION ' num2str(session)]);
end

%% Number of replay spikes vs number of new place fields
sessions = [2,3,6];
saveData = cell(length(sessions),2);
for currSessIdx = 1:length(sessions)
    currSess = sessions(currSessIdx);
    all_data  = [];
    for curr_rat = 1:length(rats)
        for curr_date = 1:length(dates{curr_rat})     
            rat = rats{curr_rat};
            date = dates{curr_rat}(curr_date);
            x = all_spikes{currSess}{curr_rat}{curr_date};
            y = all_changes{currSess}{curr_rat}{curr_date};
            valid = ~isnan(x(:,2)) & ~isnan(y(:,5));
            all_data = [all_data; x(valid,2), y(valid,5), x(valid,5)];
        end
    end
    counts = zeros(2);
    counts(1,1) = sum(all_data(:,1) == 0 & all_data(:,2) == 0); % No replay, no new fields
    counts(1,2) = sum(all_data(:,1) == 0 & all_data(:,2) > 0); % No replay, new fields
    counts(2,1) = sum(all_data(:,1) >0 & all_data(:,2) == 0); % Replay, no new fields
    counts(2,2) = sum(all_data(:,1) >0 & all_data(:,2) > 0); % Replay, new fields
    counts = counts / sum(counts(:));
    
    figure('Position', [10, 10, 200, 200])
    imagesc(counts, [0 max(counts(:))]);
    for x = 1:2
        for y = 1:2
            text(y,x,[num2str(100*counts(x,y),2) ' %'], 'FontSize', 16, 'HorizontalAlignment', 'Center', 'VerticalAlignment', 'Middle')
        end
    end
    xticks([1,2]); 
    if currSess == 6
        xticklabels({'No fields gone','Fields gone'}); 
    else
        xticklabels({'No new fields','New fields'}); 
    end
    yticks([1,2]); 
    yticklabels({'No replay','Replay'});
    ytickangle(90);
    % Significance: these all give the same p-vals
    glm = fitglm([all_data(:,1)>0, all_data(:,3)], all_data(:,2) > 0);
    if currSess == 6
        title({'Erased fields ~ replay + base rate.', ['p_{replay} = ' num2str(glm.Coefficients.pValue(2),2) ' (N_{cells} = ' num2str(size(all_data,1)) ')']}); 
    else
        title({'New fields ~ replay + base rate.', ['p_{replay} = ' num2str(glm.Coefficients.pValue(2),2) ' (N_{cells} = ' num2str(size(all_data,1)) ')']}); 
    end    
    
    % Store figure data: both the counts array and the GLM input
    saveData{currSessIdx,1} = counts;
    saveData{currSessIdx,2} = [all_data(:,1)>0, all_data(:,3), all_data(:,2) > 0];
end
% Reorder data so they correspond to paper order (S8d, S9a, S9b)
saveData = saveData([2,1,3],:);
% Save figure source data: one for session 3 (Fig s8) and one for the others (Fig s9)
save('s89.mat', 'saveData');
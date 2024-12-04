function getRippleRatemapChangePrep(rat, session, outDir, datDir)
    % Define rats
    rats = {'Janni','Harpy','Imp','Naga'};
    % Define sessions
    sessions = {'Open1','Open2'};

    % Load files
    if nargin == 3
        datDir = '/Volumes/My Passport for Mac/PfeifferFoster_data/DataForBehrensBakermans/';
    end
    datDir = fullfile(datDir, ...
        rats{rat}, sessions{session});
    spikeDat = load(fullfile(datDir, 'Spike_Data.mat'));
    posDat = load(fullfile(datDir, 'Position_Data.mat'));
    rippleDat = load(fullfile(datDir, 'Ripple_Events.mat'));
    % In rat 1, session 2, I found two series where for about a minute the position is totally frozen
    % I suspect something goes wrong with tracking at those times: 
    disp([num2str(size(posDat.Position_Data)) ', ' num2str(size(spikeDat.Spike_Data)) ', ' num2str(size(spikeDat.Excitatory_Neurons))]);
    % Get data table that has info about each neuron in this session
    [cells, coords] = getCells(posDat.Position_Data, spikeDat.Spike_Data, spikeDat.Excitatory_Neurons);
    % Only keep cells classified as place field for following analyses
    cells = cells(cells.place,:);
    spikeDat.Spike_Data = spikeDat.Spike_Data(ismember(spikeDat.Spike_Data(:,2), cells.ids),:);
    % Then continue to find ripples
    ripples = getRipples(rippleDat.Ripple_Events, spikeDat.Spike_Data, cells, coords);

    % Save files
    save(fullfile(outDir, 'cells.mat'), 'cells', '-v7.3');
    save(fullfile(outDir, 'coords.mat'), 'coords', '-v7.3');
    save(fullfile(outDir, 'ripples.mat'), 'ripples', '-v7.3');
    fid = fopen(fullfile(outDir, 'ripples.txt'),'wt');
    fprintf(fid, num2str(size(ripples,1)));
    fclose(fid);
end
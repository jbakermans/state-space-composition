% Set data base directory
baseDir='/Users/jbakermans/Documents/Data/PfeifferFoster/DataForBehrensBakermans/';
% Define rats
rats = {'Janni','Harpy','Imp','Naga'};
% Define sessions
sessions = {'Open1','Open2'};
% Create all required files
for rat = 1:4
    for session = 1:2
        % Setup directories
        datDir = fullfile(baseDir, rats{rat}, sessions{session});
        outDir = fullfile(datDir, 'Output');
        if ~isfolder(outDir)
        	mkdir(outDir);
        end
        % Prep difference files
        getRippleRatemapChangePrep(rat, session, outDir, baseDir);
        % Read text that has number of ripples
        ripples = str2double(fileread(fullfile(outDir, 'ripples.txt')));
        % Run ripple analysis for each
        parfor r = 1:ripples
            getRippleRatemapChangeFile(r, fullfile(outDir, 'ripples.mat'), fullfile(outDir, 'cells.mat'), fullfile(outDir, 'coords.mat'), outDir)
        end
        % Print progress
        disp(['Finished rat ' num2str(rat) '/4, session ' num2str(session) '/2']);
    end
end
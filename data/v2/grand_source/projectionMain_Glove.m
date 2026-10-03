%%% Main code for analyzing semantic projections in GloVe vs. human ratings %%

%% Parameters %%
nPerms = 10000;      % for permutation tests
         
%% Compute semantic subspaces, projections, GloVe-MTurk correlations, significance %%
clear projectionTest
load categoriesGloVe
load featuresGloVe
load ratingData
load catFeatPairs

featNames = featDef.Dimension;
for f = 1:length(featNames)
    disp(['Feature: ', featNames{f}]);

    %% Get GloVe subspace %%
    [sideA, sideB, theSubspace] = getSemanticSubspace(featGlove, featDef, featNames{f});
    gloveSubspace.(featNames{f}) = theSubspace;

    %% Loop through categories for current feature %%
    featInds = strcmp(catFeatPairs(:,2), featNames{f});
    currCats = catFeatPairs(featInds,1);     
    
    for c = 1:length(currCats)
        disp(['  Category: ', currCats{c}]);
        
        %% Create main semantic projection (in GloVe) %%
        disp('    Organizing data');
        gloveItems = items.(currCats{c});
        [~,catInds] = sort(gloveItems);   % alphabetize items, in order to later organize the corresponding vectors accordingly        
        gloveVec = catGlove.(currCats{c});
        gloveVec = gloveVec(catInds,:);
        projDataGlove.([currCats{c}, '_', featNames{f}]) = ...
            (gloveSubspace.(featNames{f})*(gloveVec'))/(norm(gloveSubspace.(featNames{f})));

        %% Organize human ratings %%
        turkRatings = ratings.([currCats{c}, '_', featNames{f}]);
        ratingItems = turkRatings.Properties.RowNames;
        [~,ratingInds] = sort(ratingItems);         % alphabetize items, in order to later organize ratings accordingly
        
        turkRatings = table2array(turkRatings);
        turkRatings = turkRatings(ratingInds,:);
        turkData.([currCats{c}, '_', featNames{f}]) = turkRatings;  
        turkGroup = mean(turkRatings,2);
        
        
        %% Compute main correlation between GloVe and human ratings %%
        disp('    Computing statistics');
        
        glove = projDataGlove.([currCats{c}, '_', featNames{f}]);
        if size(glove,2) > 1                        % in case glove is a row vector, turn into a columns vector
            glove = glove';
        end        
        turk = turkData.([currCats{c}, '_', featNames{f}]);        
        tMain = computeCriticalMeasures(glove, turkRatings, nPerms);       
        
        %% Insert results into table %%
        if ~exist('projectionTest', 'var')
            rowNames = tMain.Properties.RowNames;
            critMeasTypes = tMain.Properties.VariableNames; % types of critical measure: Pearson's r (Fisher-transformed), percent of consistently ordered pairs, Kendall's tau (A)
            projectionTest.main = struct;
            projectionTest.ctrl = struct;            
            projectionTest.extremes = struct;
        end
        
        tMain = table2array(tMain);
        for cm = 1:numel(critMeasTypes)
            tableMain = array2table((tMain(:,cm))', 'variableNames', rowNames);
            tableFeatCat = table(currCats(c), featNames(f), 'variableNames', {'Category', 'Feature'});
            tableAll = [tableFeatCat, tableMain];
            if ~isfield(projectionTest.main, critMeasTypes{cm})
                projectionTest.main.(critMeasTypes{cm}) = tableAll;
            else
                projectionTest.main.(critMeasTypes{cm}) = [projectionTest.main.(critMeasTypes{cm}); tableAll];
            end
            
            %% For control tests & extreme-removal below %%
            ctrlRow = {currCats{c}, featNames{f}, 'main', tMain(1,cm)};
            extremesRows = {currCats{c}, featNames{f}, 0, 'DV', tMain(1,cm);
                currCats{c}, featNames{f}, 0, 'reliability_OVR', tMain(strcmp(rowNames, 'reliability_OVR'),cm); 
                currCats{c}, featNames{f}, 0, 'reliability_SH', tMain(strcmp(rowNames, 'reliability_SH'),cm)};
            if ~isfield(projectionTest.ctrl, critMeasTypes{cm})
                projectionTest.ctrl.(critMeasTypes{cm}) = ctrlRow;                   % will be turned to table at the end of the feature and category loops
                projectionTest.extremes.(critMeasTypes{cm}) = extremesRows;          % will be turned to table at the end of the feature and category loops
            else
                projectionTest.ctrl.(critMeasTypes{cm})(end+1,:) = ctrlRow;
                projectionTest.extremes.(critMeasTypes{cm})(end+1:end+3,:) = extremesRows;                
            end            
        end
        
        
        %% Create control data: projection, cosine distance, and Euclidean distances of each item from each side of the feature axes %%
        ctrlData.Proj = [(sideA*(gloveVec'))/((norm(sideA))^2);
            (sideB*(gloveVec'))/((norm(sideB))^2)];            
        ctrlData.Cosine = [pdist2(sideA, gloveVec, 'cosine');
            pdist2(sideB, gloveVec, 'cosine')];
        ctrlData.Euclid = [pdist2(sideA, gloveVec, 'euclidean');
            pdist2(sideB, gloveVec, 'euclidean')];
               
        %% Compute correlation between control data and human ratings, and insert into table %%
        disp('    Running control tests');
        clear tCtrl
        if (f==1) && (c==1)
            ctrlTypes = fieldnames(ctrlData);
        end
        for ctrlInd = 1:numel(ctrlTypes)
            maxVals = -Inf*ones(numel(critMeasTypes),1);
                % this will store the maximum correlation with behavioral data, from among the two sides (A & B)
            for sideInd = 1:2
                ctrl = ctrlData.(ctrlTypes{ctrlInd})(sideInd,:);
                if size(ctrl,2) > 1         % in case ctrl is a row vector, turn into a columns vector
                    ctrl = ctrl';
                end            
                tCtrl = computeCriticalMeasures(ctrl, turkGroup, 0);
                tCtrl = table2array(tCtrl);

                for cm = 1:numel(critMeasTypes)
                    if tCtrl(1,cm) > maxVals(cm)
                        maxVals(cm) = tCtrl(1,cm);
                    end
                end
            end

            for cm = 1:numel(critMeasTypes)
                ctrlRow = {currCats{c}, featNames{f}, ctrlTypes{ctrlInd}, maxVals(cm)};
                projectionTest.ctrl.(critMeasTypes{cm})(end+1,:) = ctrlRow;
            end
        end        
        
        
        %% Remove (up to 10) extreme items, re-compute critical measures, and insert into table%%
        disp('    Removing extremes');
        clear tExtremes
        inds = 1:numel(turkGroup);          % will keep track of which data points are still in the sample after removing extremes
        for ii = 1:5
            inds = (turkGroup < max(turkGroup(inds))) & (turkGroup > min(turkGroup(inds)));        % at every iteration, two more extreme items will be removed
            tExtremes = computeCriticalMeasures(glove(inds), turkRatings(inds,:), 0);
            tExtremesRowNames = tExtremes.Properties.RowNames;
            tExtremes = table2array(tExtremes);
            
            for cm = 1:numel(critMeasTypes)            
                extremesRows = {currCats{c}, featNames{f}, ii*2, 'DV', tExtremes(1,cm);
                    currCats{c}, featNames{f}, ii*2, 'reliability_OVR', tExtremes(strcmp(tExtremesRowNames, 'reliability_OVR'),cm);
                    currCats{c}, featNames{f}, ii*2, 'reliability_SH', tExtremes(strcmp(tExtremesRowNames, 'reliability_SH'),cm)};  
                projectionTest.extremes.(critMeasTypes{cm})(end+1:end+3,:) = extremesRows;            
            end
        end        
        
    end
end


%% Find category-feature pairs for which there is low inter-rater reliability %%
badInds = find(projectionTest.main.r_Fisher.reliability_OVR < 0.25);
badCatFeatPairs = cell(numel(badInds),2);
for b = 1:numel(badInds)
    badCatFeatPairs{b,1} = projectionTest.main.r_Fisher.Category{badInds(b)};
    badCatFeatPairs{b,2} = projectionTest.main.r_Fisher.Feature{badInds(b)};
end

%% Turn control cells into table %%
for cm = 1:numel(critMeasTypes)
    projectionTest.ctrl.(critMeasTypes{cm}) = cell2table(projectionTest.ctrl.(critMeasTypes{cm}), ...
       'variableNames', {'Category', 'Feature', 'Type', 'DV'});
    t = projectionTest.ctrl.(critMeasTypes{cm});
    badRows = false(size(t,1),1);
    for b = 1:size(badCatFeatPairs,1)
        badRows(strcmp(t.Category, badCatFeatPairs{b,1}) & strcmp(t.Feature, badCatFeatPairs{b,2})) = true;
    end
    t = t(~badRows,:);
    writetable(t, ['ControlTest_', critMeasTypes{cm}, '.txt'], 'delimiter', '\t');
end

%% Save %%
disp('Projections and ratings of items are ordered alphabetically');
save projections_Glove gloveSubspace projDataGlove turkData projectionTest badCatFeatPairs

%% FDR-correction of p-values %%
load projections_Glove
fields = fieldnames(projectionTest.main);

for f = 1:numel(fields)
    data = projectionTest.main.(fields{f});
    pVals = data.pValuePermutationTest;
    [q,~] = FDRcorrect(pVals);
    q = table(q, 'variableNames', {'pValueFDR'});
    projectionTest.main.(fields{f}) = [projectionTest.main.(fields{f}), q];
end
save projections_Glove projectionTest -append
   
%% Find category-feature pairs for which p > 0.05 %%
nsInds = find(projectionTest.main.r_Fisher.pValueFDR > 0.05);
nsCatFeatPairs = [badCatFeatPairs; projectionTest.main.r_Fisher.Category(nsInds), projectionTest.main.r_Fisher.Feature(nsInds)];

%% Turn extreme cells into table %%
for cm = 1:numel(critMeasTypes)
    projectionTest.extremes.(critMeasTypes{cm}) = cell2table( projectionTest.extremes.(critMeasTypes{cm}), ...
       'variableNames', {'Category', 'Feature', 'N_Removed', 'Type', 'DV'});
    t = projectionTest.extremes.(critMeasTypes{cm});
    nsRows = false(size(t,1),1);
    for b = 1:size(nsCatFeatPairs,1)
        nsRows(strcmp(t.Category, nsCatFeatPairs{b,1}) & strcmp(t.Feature, nsCatFeatPairs{b,2})) = true;
    end
    t = t(~nsRows,:);
    writetable(t, ['ExtremesRemovalTest_', critMeasTypes{cm}, '.txt'], 'delimiter', '\t');    
end

%% Plot %%
figSaveDir = pwd;
projectionPlotFig3(projectionTest, ubField, figSaveDir);
projectionPlotFig4(projectionTest, projDataGlove, turkData, ubField, figSaveDir);
projectionPlotFig5(projectionTest, ubField, figSaveDir);
projectionPlotFig6(projectionTest, ubField, figSaveDir);
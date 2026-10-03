function [dvMat, dvRemMat, catNames, featNames] = projectionPlotExtractData(projTest, ubField, reportFlag)
    
%%% helper function for all the plotting functions

%%% INPUT:
%%% projTest = a structure of the kind projectionTest.main created by projectionMain_Glove.m
%%% ubfield = "upper bound field", name of column in the projTest tables storing inter-rater reliability
%%% reportFlag = boolean, whether to report category-feature pairs with low (vs. high) reliability

%%% OUTPUT:
%%% dvMat = 3 dimensional matrix;
%%%         rows: category-feature pairs
%%%         columns: 1 = value of dependent variable for evaluating semantic projection (e.g., Pearson's correlation)
%%%                  2 = upper bound for dependent variable (based on inter-rater reliability)
%%%                  3 = p-value of dependent variable
%%%         layers = different dependent variables
%%% dvRemMat = same as above, but for data calculated after removing items with extreme values
%%%            (returns NaN if the projTest structure doesn't have an "extremes" field)
%%% catNames, featNames = columnn cells with the category/feature names for each row in dvMat (and dvRemMat)

if isfield(projTest, 'main')
    data = projTest.main;
else
    data = projTest;
end

dvNames = fieldnames(data);
nExpts = size(data.(dvNames{1}),1);                           % number of experiments
catNames = data.(dvNames{1}).Category;
featNames = data.(dvNames{1}).Feature;

%% Extract data %%
dvMat = zeros(nExpts, 3, numel(dvNames));                        % columns: 1 = dv, 2 = upper bound, 3 = sig
isSig = true(size(data.(dvNames{1}), 1), 1);
for dvInd = 1:numel(dvNames)
    isSig = isSig & data.(dvNames{dvInd}).pValueFDR < 0.05;
end

for dvInd = 1:numel(dvNames)
    dvMat(:,1,dvInd) = data.(dvNames{dvInd}).value;
    dvMat(:,2,dvInd) = data.(dvNames{dvInd}).(ubField);
    dvMat(:,3,dvInd) = data.(dvNames{dvInd}).pValueFDR;
end

%% Extract data after removal of extreme values %%
if isfield(projTest, 'extremes')
    data = projTest.extremes;
    nRemoved = unique(data.(dvNames{1}).N_Removed);                % #of extreme values removed in each iteration
    dvRemMat = zeros(nExpts, 2, numel(nRemoved), numel(dvNames));  % for "removed" data; columns: same as dvMat

    for dvInd = 1:numel(dvNames)    
        for r = 1:numel(nRemoved)
            for ii = 1:numel(catNames)
                row1 = strcmp(data.(dvNames{dvInd}).Category, catNames{ii}) & ...
                    strcmp(data.(dvNames{dvInd}).Feature, featNames{ii}) & ...
                    strcmp(data.(dvNames{dvInd}).Type, 'DV') & ...
                    data.(dvNames{dvInd}).N_Removed == nRemoved(r);
                row2 = strcmp(data.(dvNames{dvInd}).Category, catNames{ii}) & ...
                    strcmp(data.(dvNames{dvInd}).Feature, featNames{ii}) & ...
                    strcmp(data.(dvNames{dvInd}).Type, ubField) & ...
                    data.(dvNames{dvInd}).N_Removed == nRemoved(r);

                dvRemMat(ii,1,r,dvInd) = data.(dvNames{dvInd}).DV(row1);
                dvRemMat(ii,2,r,dvInd) = data.(dvNames{dvInd}).DV(row2);
            end
        end
    end
else
    dvRemMat = nan;
end

%% Exclude bad category/feature pairs: those with low inter-subject raliability %%
badInds = [];
for dvInd = 1:numel(dvNames)
    badInds = union(badInds, find(zscore(dvMat(:,2,dvInd))<=-2.5));
end

if reportFlag
    disp(' ');
    disp('Category/feature pairs with noisy human estimates: ');
    for b = 1:length(badInds)
        disp([num2str(b), '. ', catNames{badInds(b)}, '  by  ', featNames{badInds(b)}]);
    end
    disp(' ');
    for dvInd = 1:numel(dvNames)
        disp(['All ', dvNames{dvInd}, ' values < ', num2str(max(dvMat(badInds,2,dvInd)))]);
    end
    disp(' ');
end

goodInds = setdiff(1:size(dvMat,1),badInds);
dvMat = dvMat(goodInds,:,:);
if ~isnan(dvRemMat)
    dvRemMat = dvRemMat(goodInds,:,:,:);
end
catNames = catNames(goodInds);
featNames = featNames(goodInds);

if reportFlag
    disp('Remaining pairs: ');
    for dvInd = 1:numel(dvNames)
        if strcmp(dvNames{dvInd}, 'r_Fisher')
            M = tanh(mean(dvMat(:,1:2,dvInd),1));
            med = prctile(tanh(dvMat(:,1:2,dvInd)),50,1);
            theDV = 'r (no Fisher)';
        else
            M = mean(dvMat(:,1:2,dvInd),1);
            med = prctile(dvMat(:,1:2,dvInd),50,1);
            theDV = dvNames{dvInd};
        end
        SD = std(dvMat(:,1:2,dvInd),[],1);

        disp([theDV, ': M=', num2str(M(1)), ', SD=', num2str(SD(1)), ', median=', num2str(med(1))]);
        disp(['        (', ubField, ': M=', num2str(M(2)), ', SD=', num2str(SD(2)), ', median=', num2str(med(2))]);
    end        
end
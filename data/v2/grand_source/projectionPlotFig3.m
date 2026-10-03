function projectionPlotFig3(projTest, ubField, savePath)
    
%%% This function plots a matrix of categories-by-features, with each
%%% entry colored by the fit of semantic projection to human data

%%% INPUT:
%%% projTest = a structure of the kind projectionTest.main created by projectionMain_Glove.m
%%% ubfield = "upper bound field", name of column in the projTest tables storing inter-rater reliability
%%% savePath = string of full path to where the results are saved 

if isfield(projTest, 'main')
    projTest = projTest.main;
end
rVals = projTest.r_Fisher;
ocpVals = projTest.OCp;

catVals = sort(unique(rVals.Category));
nC = numel(catVals);
featVals = sort(unique(rVals.Feature));
nF = numel(featVals);

tVals = {'rVals', 'ocpVals'};
tNames = {'r', 'OCp'};
tLims = {[-1 1], [0 100]};
tTrans = {'tanh(t.value)', '100*t.value'};
tTicks = {-1:0.25:1, 0:20:100};

%% Remove bad category/feature pairs %%
badInds = [];
for tInd = 1:numel(tVals)
    eval(['t = ', tVals{tInd}, ';']);      
    badInds = union(badInds, find(zscore(t.(ubField))<=-2.5));
end

goodInds = setdiff(1:size(t,1),badInds);
for tInd = 1:numel(tVals)
    eval(['t = ', tVals{tInd}, ';']);
    t = t(goodInds, :);
    eval([tVals{tInd}, ' = t;']);    
end

%% Find significant categure/feature pairs %%
for tInd = 1:numel(tVals)
    eval(['t = ', tVals{tInd}, ';']);          
    if tInd == 1
        sigInds = true(size(t,1),1);
    end
    sigInds = sigInds & (t.pValueFDR < 0.05);
end

%% Organize data %%
valMat = nan(nC, nF, numel(tVals));
sigMat = false(nC, nF);
for tInd = 1:numel(tVals)
    eval(['t = ', tVals{tInd}, ';']);
    eval(['t.value = ', tTrans{tInd}, ';']);

    for ii = 1:size(t,1)
        cCurr = find(strcmp(t.Category(ii), catVals));
        fCurr = find(strcmp(t.Feature(ii), featVals));
        valMat(cCurr, fCurr, tInd) = t.value(ii);
        sigMat(cCurr, fCurr) = sigInds(ii);
    end
end

%% Update names of two features %%
featVals = strrep(featVals, 'loudness', 'volume');
featVals = strrep(featVals, 'political', 'partisanship');
[featVals,ord] = sort(featVals);
valMat = valMat(:,ord,:);
sigMat = sigMat(:,ord);

%% Make figure %%
clr = colormap('jet');
for fig = 1:size(valMat,3)
    figure(fig)
    clf reset
    set(gca, 'xlim', [0.5, nF+0.5], 'xtick', [], 'ylim', [0.5 nC+0.5], 'ytick', [], 'linewidth', 1, ...
        'units', 'centimeters', 'position', [2 2 7 3.7]);   
    set(gcf, 'Units', 'centimeters', 'PaperPosition', [0, 0, 11 7.7]);

    hold on
    
    clrLims = linspace(tLims{fig}(1), tLims{fig}(2), size(clr,1)+1);
    
    %% Fill in colors %%
    for c = 1:nC
        for f = 1:nF
            currVal = valMat(c,f,fig);
            if ~isnan(currVal)
                clrInd = find(valMat(c,f,fig) < clrLims, 1, 'first') - 1;
                currClr = clr(clrInd,:);
            else
                currClr = [200 200 200]/255;
            end
            
            fill([f-0.5, f+0.5, f+0.5, f-0.5, f-0.5], ...
                [(nC-c+1)-0.5, (nC-c+1)-0.5, (nC-c+1)+0.5, (nC-c+1)+0.5, (nC-c+1)-0.5], ...
                currClr, 'linestyle', 'none');    
        end
    end
    
    %% Make contours %%
    for c = 1:nC
        for f = 1:nF
            if sigMat(c,f)
                w = 2;
            else
                w = 1;
            end
            
            plot([f-0.5, f+0.5, f+0.5, f-0.5, f-0.5], ...
                [(nC-c+1)-0.5, (nC-c+1)-0.5, (nC-c+1)+0.5, (nC-c+1)+0.5, (nC-c+1)-0.5], ...
                'k', 'linewidth', w);    
        end
    end    
    
    %% Median values across features / categories %%
    for c = 1:nC
        currVals = valMat(c,:,fig);
        med = prctile(currVals(~isnan(currVals)), 50);
        if strcmp(tNames{fig}, 'r')
            med = round(med*100)/100;
        else
            med = round(med);
        end
        text(nF+1, nC-c+1, num2str(med), 'horizontalalignment', 'left');
    end
    for f = 1:nF
        currVals = valMat(:,f,fig);
        med = prctile(currVals(~isnan(currVals)), 50);
        if strcmp(tNames{fig}, 'r')
            med = round(med*100)/100;
        else
            med = round(med);
        end        
        text(f, nC+1, num2str(med), 'horizontalalignment', 'center', 'verticalalignment', 'top');
    end
    
    %% Cosmetics %%
    set(gca, 'xtick', 1:nF, 'xticklabels', featVals, 'xticklabelrotation', 30, ...
        'ytick', 1:nC, 'yticklabels', flip(catVals,1));
    colormap('jet');
    caxis([tTicks{fig}(1), tTicks{fig}(end)]);
    colorbar('ticks', tTicks{fig}, 'linewidth', 1);
    
    print(fullfile(savePath, ['Figure03_', tNames{fig}, '.eps']), '-depsc', '-r300');     
end
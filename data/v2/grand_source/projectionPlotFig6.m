function projectionPlotFig6(projTest, ubField, savePath)
    
%%% This function plots the results of the analysis where category items
%%% with extreme ratings are gradually removed

%%% INPUT:
%%% projTest = a structure of the kind projectionTest.main created by projectionMain_Glove.m
%%% ubfield = "upper bound field", name of column in the projTest tables storing inter-rater reliability
%%% savePath = string of full path to where the results are saved 

dvNames = fieldnames(projTest.main);
nRemoved = unique(projTest.extremes.(dvNames{1}).N_Removed);           % #of extreme values removed in each iteration
[dvMat, dvRemMat, catNames, featNames] = projectionPlotExtractData(projTest, ubField, false);

isSig = true(size(catNames));
for dvInd = 1:numel(dvNames)
    isSig = isSig & dvMat(:,3,dvInd) < 0.05;
end

dvRemMat = dvRemMat(isSig,:,:,:);
catNames = catNames(isSig);
featNames = featNames(isSig);

nFeats = length(unique(featNames));
featNums = cellfun(@(x)(find(strcmp(x,unique(featNames)),1)), featNames);

figure
clf reset
colorsFig6 = colormap('jet');
inds = 1:floor(length(colorsFig6)/nFeats):length(colorsFig6); % one color per feature
colorsFig6 = colorsFig6(inds,:);                              % keep only nFeat colors
colorsFig6 = colorsFig6*0.8; %%% alternatively: +0.5;

subplot(numel(dvNames),1,1)
l = plot(zeros(3,nFeats), zeros(3,nFeats));
for f = 1:nFeats
    set(l(f), 'color', colorsFig6(f,:));
end
legend(unique(featNames));
hold on

colorsFig6 = colorsFig6(featNums,:);                          % each experiment gets its color
xLims = [-0.01*min(nRemoved), max(1.01*nRemoved)];
for dvInd = 1:numel(dvNames)
    dv1 = permute(dvRemMat(:,1,:,dvInd),[3 1 2 4]);       % columns are different lines (experiments)    
    dv2 = permute(dvRemMat(:,2,:,dvInd),[3 1 2 4]);
    yLims = [min(dv1(:)), max(max(dv1(:)),max(dv2(:)))];
    yLims = [yLims(1)-0.05*(yLims(2)-yLims(1)), yLims(2)+0.05*(yLims(2)-yLims(1))];
        
    switch dvNames{dvInd}
        case 'r_Fisher'
            dv1 = tanh(dv1);
            dv2 = tanh(dv2);
            yLims = tanh(yLims);
            yTicks = (floor(10*yLims(1))/10):0.2:(ceil(10*yLims(2))/10);
        case 'OCp'
            dv1 = 100*dv1;
            dv2 = 100*dv2;
            yLims = 100*yLims;
            yTicks = floor(yLims(1)):5:ceil(yLims(2));
        case 'tau'
            yTicks = (floor(10*yLims(1))/10):0.2:(ceil(10*yLims(2))/10);
    end
    m1 = prctile(dv1,50,2);
    m2 = prctile(dv2,50,2);
    
    subplot(numel(dvNames), 1, dvInd)
    hold on
    xVals = (repmat(nRemoved, 1, size(dv1,2)));    
    l = plot(xVals, dv1, '-', 'linewidth', 1);  % lines
    size(dv1)
    for exptInd = 1:size(dv1,2)
        l(exptInd).Color = [colorsFig6(exptInd,:) 0.5];  % the 4th value sets the alpha
    end

    for n = 1:size(dv1,1)
        s = scatter(xVals(n,:), dv1(n,:), 36, colorsFig6, 'filled');
        alpha(s, 0.5);
    end
        
    plot(nRemoved, m1, '-s', 'linewidth', 3, 'color', 'k', 'markerfacecolor','k', 'markersize', 6);
    plot(nRemoved, m2, '--s', 'linewidth', 3, 'color','k', 'markerfacecolor', 'k', 'markersize', 6);
    set(gca,'xlim',xLims,'xtick',nRemoved,'xticklabels',nRemoved, ...
        'ylim',yLims,'ytick',yTicks);
    box off
    title(regexprep(dvNames{dvInd},'Fisher','{Fisher}'));
end
print(fullfile(savePath, ['Figure06_', datestr(now,'yyyymmdd'), '.eps']),'-depsc');
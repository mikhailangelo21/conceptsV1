function projectionPlotFig4(projTest, projData, turkData, ubField, savePath)
    
%%% This function plots, for each category-feature pair, a scatterplot of
%%% human ratings vs. semantic projection per item

%%% INPUT:
%%% projTest = a structure of the kind projectionTest created by projectionMain_Glove.m
%%% projData = a structure of the kind projDataGlove created by projectionMain_Glove.m
%%% turkData =  a structure of the kind turjData created by projectionMain_Glove.m
%%% ubfield = "upper bound field", name of column in the projTest tables storing inter-rater reliability
%%% savePath = string of full path to where the results are saved 

if isfield(projTest, 'main')
    projTest = projTest.main;
end

dvNames = fieldnames(projTest);
[dvMat, ~, catNames, featNames] = projectionPlotExtractData(projTest, ubField, false);

nExpts = size(dvMat,1);
isSig = all(dvMat(:,3,:) < 0.05, 3);

figure
clf reset
for exptInd = 1:nExpts    
    currCat = catNames{exptInd};
    currFeat = featNames{exptInd};
    y = zscore(mean(turkData.([currCat, '_', currFeat]),2));
    x = zscore(projData.([currCat, '_', currFeat]));
    
    currDV = permute(dvMat(exptInd,1,:), [1 3 2]);
    for dvInd = 1:numel(dvNames)
        switch dvNames{dvInd}
            case 'r_Fisher'
                r = round(1000*tanh(currDV(dvInd)))/1000;
            case 'OCp'
                pp = round(100*currDV(dvInd));
            case 'tau'
                tau = round(1000*currDV(dvInd))/1000;
        end
    end
    
    row = find(strcmp(unique(catNames), currCat),1);
    catInds = strcmp(catNames, currCat);
    col = find(strcmp(featNames(catInds), currFeat),1);
    subplot(9,8,(row-1)*8+col);
    hold on
    if isSig(exptInd) == 0
        set(subplot(9,8,(row-1)*8+col), 'color', 0.8*[1 1 1]);
    end
    
    plot(x,y,'.k','markersize',5);
    xLine = [min(x), max(x)];
    yLine = r*xLine;
    
    plot(xLine, yLine, '-', 'linewidth', 2, 'color', [0 0 max(r*0.75,0)]);
    xExtra = 0.03*(max(x)-min(x));
    yExtra = 0.03*(max(y)-min(y));
    set(gca, 'xlim', [min(x)-xExtra, max(x)+xExtra], ...
        'ylim', [min(y)-yExtra, max(y)+yExtra], ...
        'xtick', [], 'ytick', []);
    if col == 1
        currCat(1) = upper(currCat(1));        
        ylabel(currCat);
    end
    currFeat(1) = upper(currFeat(1));
    title(currFeat);
    
    tau = num2str(tau);
    pp = num2str(pp);
    if r > 0
        r = num2str(round(1000*r)/1000);
        xlabel(['r\it=.', r(3:end), ', {OC}\it_{p}=', pp, '%'], ...
        'fontsize', 8);
    else
        r = num2str(round(1000*r)/1000);        
        xlabel(['r\it=-.', r(4:end), ', {OC}\it_{p}=',pp, '%'], ...
            'fontsize', 8);
    end
    box on
end

print(fullfile(savePath, ['Figure04_', datestr(now,'yyyymmdd'), '.eps']),'-depsc');
saveas(gcf, fullfile(savePath, ['Figure04_', datestr(now,'yyyymmdd'), '.png']), 'png');
function projectionPlotFig5(projTest, ubField, savePath)
    
%%% This function plots the overall distribution of evaluation measures
%%% (e.g., Pearson's correlations) across category-feature pairs

%%% INPUT:
%%% projTest = a structure of the kind projectionTest created by projectionMain_Glove.m
%%% ubfield = "upper bound field", name of column in the projTest tables storing inter-rater reliability
%%% savePath = string of full path to where the results are saved 

if isfield(projTest, 'main')
    projTest = projTest.main;
end
dvNames = fieldnames(projTest);
[dvMat, ~, ~, ~] = projectionPlotExtractData(projTest, ubField, false);

figure
clf reset
set(gcf,'units','normalized','outerposition',[0 0 0.8 0.4],'color','white');

for dvInd = 1:numel(dvNames)
    xUnadjusted = dvMat(:,1,dvInd);
    xAdjusted = dvMat(:,2,dvInd);      % currently this stores the upper bound, but will be changed below   
    switch dvNames{dvInd}
        case 'r_Fisher'
            xUnadjusted = tanh(xUnadjusted);
            xAdjusted = tanh(xAdjusted);
            xAdjusted = sign(xUnadjusted).*min(sqrt((xUnadjusted.^2)./(xAdjusted.^2)),0.9999);
            xVals = (-1:0.01:1)';   % for plotting the empirical PDF
            tickUnit = 0.2;           
            xMean = [mean(atanh(xUnadjusted)), mean(atanh(xAdjusted))];
            xSD = [std(atanh(xUnadjusted),1), std(atanh(xAdjusted),1)];           
        case 'OCp'
            xUnadjusted = 100*xUnadjusted;
            xAdjusted = 100*xAdjusted;            
            xAdjusted = 100*min(xUnadjusted./xAdjusted,1);
            xVals = (0:1:100)';
            tickUnit = 10;        
            xMean = [mean(xUnadjusted), mean(xAdjusted)];
            xSD = [std(xUnadjusted,1), std(xAdjusted,1)];           
        case 'tau'
            xVals = (-1:0.01:1)';
            xAdjusted = min(xUnadjusted./xAdjusted,1);
            tickUnit = 0.2;
            xMean = [mean(xUnadjusted), mean(xAdjusted)];
            xSD = [std(xUnadjusted,1), std(xAdjusted,1)];           
    end
    medUnadjusted = prctile(xUnadjusted,50);    
    medAdjusted = prctile(xAdjusted,50);
    xMed = [medUnadjusted, medAdjusted];
    xQ1 = [prctile(xUnadjusted,25), prctile(xAdjusted,25)]; % first quartile
    xQ3 = [prctile(xUnadjusted,75), prctile(xAdjusted,75)]; % third quartile
    
    %% Compute empirical bootstrap distribution for the median %%
    n = size(xUnadjusted,1);
    meanArray = zeros(10^5, 2);
    medArray = zeros(10^5, 2);
    for perm = 1:(10^5)
        inds = ceil(n*rand(n,1));
        medArray(perm,1) = prctile(xUnadjusted(inds),50);
        medArray(perm,2) = prctile(xAdjusted(inds),50);
        switch dvNames{dvInd}
            case 'r_Fisher'
                meanArray(perm,:) = [mean(atanh(xUnadjusted(inds))), mean(atanh(xAdjusted(inds)))];
            case 'OCp'
                meanArray(perm,:) = [mean(xUnadjusted(inds)); mean(xAdjusted(inds))];
            case 'tau'
                meanArray(perm,:) = [mean(xUnadjusted(inds)); mean(xAdjusted(inds))];                
        end
    end        
    medCI = [prctile(medArray,2.5,1);
        prctile(medArray,97.5,1)];
    meanCI = [prctile(meanArray,2.5,1);
        prctile(meanArray,97.5,1)];
    
    disp([dvNames{dvInd}, ' (all pairs, not just significant ones):']);
    
    disp(['M = ', num2str(xMean(1)), ' (95% CI: ' num2str(meanCI(1,1)), '-', num2str(meanCI(2,1)), '), ' ...
        'SD = ', num2str(xSD(1)), ...
        ', Med = ', num2str(xMed(1)), ' (95% CI: ', num2str(medCI(1,1)), '-', num2str(medCI(2,1)), '), ', ...
        'IQR = ', num2str(xQ1(1)), '-', num2str(xQ3(1))]);
    
    disp(['Adjusted: M = ', num2str(xMean(2)), ' (95% CI: ' num2str(meanCI(1,2)), '-', num2str(meanCI(2,2)), '), ' ...
        'SD = ', num2str(xSD(2)), ...
        ', Med = ', num2str(xMed(2)), ' (95% CI: ', num2str(medCI(1,2)), '-', num2str(medCI(2,2)), '), ', ...
        'IQR = ', num2str(xQ1(2)), '-', num2str(xQ3(2))]);
    disp(' ');
   
    %% Estimate empirical PDF %%
    pdfUnadjusted = fitdist(xUnadjusted, 'kernel');
    yValsUnadjusted = pdf(pdfUnadjusted, xVals);
    
    pdfAdjusted = fitdist(xAdjusted, 'kernel');
    yValsAdjusted = pdf(pdfAdjusted, xVals);
    
    xMin1 = xVals(yValsUnadjusted<(10^-4) & cumsum(yValsUnadjusted) < 0.5);
            % the second part of the logical expression is to avoid choosing values on the right tail
    xMin2 = xVals(yValsAdjusted<(10^-4) & cumsum(yValsAdjusted) < 0.5);
    xMin = min(xMin1(end), xMin2(end));
    
    %% Plot %%
    subplot(1, numel(dvNames), dvInd);
    hold on
    plot(xVals, yValsAdjusted, '-', 'linewidth', 1.5, 'color', [0.7 0.7 0.7]);
    plot(xVals, yValsUnadjusted, '-', 'linewidth', 1.5, 'color', [0 0 0]);
    
    xAll = [xUnadjusted, xAdjusted];
    yMaxAll = max(max(yValsAdjusted), max(yValsUnadjusted));
    yLimsUnadjusted = [0; 0.12*yMaxAll];
    yLimsAdjusted = [0.17*yMaxAll; 0.29*yMaxAll];
    yLims = [yLimsUnadjusted, yLimsAdjusted];
    colorsFig5 = [0 0 0; 0.6 0.6 0.6];
    
    for ii = 1:2
        fill([medCI(1,ii), medCI(2,ii), medCI(2,ii), medCI(1,ii), medCI(1,ii)], ...
            [repmat(yLims(1,ii),1,2), repmat(yLims(2,ii),1,2), yLims(1,ii)], ...
            colorsFig5(ii,:), 'facealpha', 0.3, 'edgecolor', 'none');
        plot(xAll(:,ii), yLims(1,ii)+(yLims(2,ii)-yLims(1,ii))*rand(size(xAll(:,ii))), 'o', ...
            'markerfacecolor', colorsFig5(ii,:), 'markeredgecolor', [1 1 1]);    
        plot([xMed(ii); xMed(ii)], yLims(:,ii), '-', 'linewidth', 3, 'color', colorsFig5(ii,:));
    end
    
    %% Cosmetics %%
    set(gca,'fontname','palatino','fontsize',18);
    legend({[regexprep(dvNames{dvInd}, 'Fisher', '{Fisher}'), '-adjusted'], regexprep(dvNames{dvInd}, 'Fisher', '{Fisher}')}, ...
        'location','northwest','box','off');
    xLims = [xMin, max(xVals)];
    yLims = [-0.05*yMaxAll, 1.05*yMaxAll];
    set(gca, 'xlim', xLims, 'xtick', fliplr(max(xVals):(-tickUnit):xMin), 'ylim', yLims, 'ytick', []);
    box off    
end
print(fullfile(savePath, ['Figure05_', datestr(now,'yyyymmdd'), '.eps']),'-depsc');

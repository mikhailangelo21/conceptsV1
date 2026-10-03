function [aVec,bVec,diffVec] = getSemanticSubspace(featData, featDef, featName)

%%% INPUT:
%%% featData = structure array with DSM vectors of the different features;
%%% featDef = structure array with feature-value names ("definitions");
%%% featName = name of feature to extract

%%% OUTPUT:
%%% posVec = average of vectors from one extreme ("end A")
%%% negVec = average of vectors from the other extreme ("end B")
%%% diffVec = average difference between each end-A vector
%%%       and each end-B vectors

featInd = find(strcmp(featName, featDef.Dimension),1);
theFields = setdiff(fieldnames(featData), 'Dimension');
nA = sum(~cellfun(@isempty, strfind(theFields,'Positive')));   % number of "positive"-extreme (end A) adjectives
nB = sum(~cellfun(@isempty, strfind(theFields,'Negative')));   % number of "negative"-extreme (end B) adjectives

for a = 1:nA
    aVecCurr = featData.(['Positive', num2str(a)])(featInd,:);
    if a == 1
        aVec = zeros(size(aVecCurr));
    end
    aVec = aVec + aVecCurr;
    
    for b = 1:nB
        bVecCurr = featData.(['Negative', num2str(b)])(featInd,:);
        if a == 1
            if b == 1
                diffMat = zeros(nA*nB, length(aVecCurr));
                bVec = zeros(size(bVecCurr));
            end
            bVec = bVec + bVecCurr;
        end
        diffMat((a-1)*nB+b,:) = aVecCurr - bVecCurr;
    end
end
aVec = aVec / nA;
bVec = bVec / nB;
diffVec = mean(diffMat,1);
function [qVals, reject] = FDRcorrect(pVals)

% This script performs FDR-correction for a group of p-values
%
% INPUT:
% pVals: a vector of p-Values
%
% OUTPUT:
% qVals: a vector of FDR-corrected p-values, called q-values
%        (reject the null hypotheses when q < your-alpha)
%        uses the Benjamini-Hochberg-Yekutieli method
% reject: a vector with 0s and 1s for hypotheses that are 
%         non-rejected and rejected, respectively (for q < 0.05)

%% Main %%
if size(pVals,1)==1     % convert data to column form
    isRow = 1;
    pVals = pVals';
else
    isRow = 0;
end

constant = sum(ones(size(pVals))./(1:length(pVals))');
[pSorted, inds] = sort(pVals, 'ascend');
qSorted = (pSorted*(constant*length(pVals)))./(1:length(pVals))';
qSorted(qSorted>1) = 1;
ind = length(qSorted)-find(flipud(qSorted)<=0.05,1)+1;
if isempty(ind)
    ind = 0;
end
reject = (1:length(qSorted))'<=ind;
qVals(inds,1) = qSorted;
reject(inds,1) = reject;

if isRow == 1
    qVals = qVals';
    reject = reject';
end
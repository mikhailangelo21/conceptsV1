function t = computeCriticalMeasures(vecs1,vecs2,nPerms)

% INPUT:
% vecs1 = Nx1 matrix (e.g., ratings of N items in a DSM space)
% vecs2 = NxK matrix (K >= 1) (e.g., ratings of the same N items and by K MTurk workers)
% nPerms = number of permutations for significance testing (can be 0)
%
% OUTPUT:
% t = results table, with the following columns:
%     (1) Pearson correlation (r) between vecs1 and vecs2 (or its average across columns)
%     (2) Percentage of consistently ordered pairs (pp) between them
%     (3) Kendall's Tau correlation (type A)
%     Rows of t correspond to:
%     value, mean of permuted values, SD of permuted values, permutation p-value,
%     one-vs-rest reliability of vecs2 (across columns, if K>1),
%     split-half reliability of vecs2

measTypes = {'r', 'OCp', 'tau'}; % types of critical measures
n = length(vecs1);
k = size(vecs2,2);
t = nan(6,numel(measTypes));

%% Critical measure %%
for m = 1:numel(measTypes)
    t(1,m) = computeCriticalMeasures_helper(vecs1,mean(vecs2,2),measTypes{m});
end

%% Permutations %%
if nPerms > 0
    permVals = nan(nPerms,numel(measTypes));
    for p = 1:nPerms
        inds = randperm(n);                         % shuffle vecs1 randomsly
        for m = 1:numel(measTypes)
            permVals(p,m) = computeCriticalMeasures_helper(vecs1(inds),mean(vecs2,2),measTypes{m});
        end
    end    
    
    t(2,:) = mean(permVals,1);
    t(3,:) = std(permVals,[],1);
    
    %% p-values %%
    for m = 1:numel(measTypes)
        %% Plot the empirical null distribution (to evaluate normality) %%
%         figure(1)
%         hist(permVals(:,m),100)
%         title(measTypes{m});
%         pause(0.5)
        
        params = fitdist(permVals(:,m),'normal');
        t(4,m) =  1 - normcdf(t(1,m), params.mu, params.sigma);
        
        %% If the distribution is not normal, use this instead of the two lines above %%
        % t(4,m) = (sum(permVals(:,m) > t(1,m)) + 0.5*sum(permVals(:,m)==t(1,m)))/nPerms;
    end
end

%% Reliability %%
if k > 1
    ISC_OVR = zeros(k,numel(measTypes)); % one-vs-rest
    for kInd = 1:k
        vecs2_A = vecs2(:,kInd);
        vecs2_B = vecs2(:,setdiff(1:k,kInd));
        for m = 1:numel(measTypes)
            ISC_OVR(kInd,m) = computeCriticalMeasures_helper(vecs2_A, mean(vecs2_B,2), measTypes{m});
        end
    end
    t(5,:) = mean(ISC_OVR,1);
            
    SH_splitInds = zeros(k,k);        % split-half; each row will be a different way of splitting k columns into 2 groups
    for ii = 1:floor(k/2)
        SH_splitInds(ii,:) = mod(1:k,2*ii) <= (ii-1);       % e.g., mod(1:k,2) <= 0; mod(1:k,4)<=1; mod(1:k,6)<=2; etc.
    end        
    for ii = (floor(k/2)+1):k
        SH_splitInds(ii,randperm(k,round(k/2))) = 1;
    end
    ISC_SH = zeros(k,numel(measTypes));
    for kInd = 1:k
        nOnes = sum(SH_splitInds(kInd,:)==1, 2);
        nDiff = nOnes - round(k/2);            
        if sign(nDiff) == 1
            oneInds = find(SH_splitInds(kInd,:)==1, nDiff);
            SH_splitInds(kInd,oneInds) = 0;
        elseif sign(nDiff) == -1
            zeroInds = find(SH_splitInds(kInd,:)==0, abs(nDiff));
            SH_splitInds(kInd,zeroInds) = 1;
        end
        vecs2_A = vecs2(:,SH_splitInds(kInd,:)==1);
        vecs2_B = vecs2(:,SH_splitInds(kInd,:)==0);
        for m = 1:numel(measTypes)
            ISC_SH(kInd,m) = computeCriticalMeasures_helper(mean(vecs2_A,2), mean(vecs2_B,2), measTypes{m});
        end
    end
    t(6,:) = mean(ISC_SH,1);
end

measTypes = regexprep(measTypes, 'r', 'r_Fisher');
t = array2table(t, 'variablenames', measTypes, 'rownames', {'value', 'premutation_M', 'premutation_SD', 'pValuePermutationTest', 'reliability_OVR', 'reliability_SH'});
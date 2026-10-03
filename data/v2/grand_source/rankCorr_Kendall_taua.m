function taua = rankCorr_Kendall_taua(a,b,varargin)

%%% Computes the Kendall's tau a correlation coefficient between the input vectors (a,b)

%%% INPUT:
%%% a,b: vectors
%%% pairs (optional): Nx2 array of numbers denoting item pairs to compare across a and b
%%%                   (in case only a subset of the pairs should be compared)

%%% OUTPUT:
%%% taua = Kendall's tau (type a) correlation

%%% Idan Blank, Feb 20, 2017; based on Kriegeskorte's RSA toolbox
%%%                           (with changes to variable names)

%%% SANITY CHECKED %%%


%% Compute Kendall rank correlation coefficient tau-a %%
if isempty(varargin)
    n = size(a,1);
    denom = n*(n-1)/2;
    K = 0;
    for k = 1:n-1
        pairRelations_a=sign(a(k)-a(k+1:n));            % for entries above k: higher than entry k = -1, lower than entry k = 1 
        pairRelations_b=sign(b(k)-b(k+1:n));
        K = K + sum(pairRelations_a.*pairRelations_b);  % concordant entry pairs will have a product of 1, otherwise -1
    end
    taua = K/denom;                                     % normalise by the total number of pairs     
else
    pairs = varargin{1};
    denom = size(pairs,1);
    K = 0;
    for k = 1:size(pairs,1)
        K = K + sign(a(pairs(k,1))-a(pairs(k,2)))*sign(b(pairs(k,1))-b(pairs(k,2)));
    end
    taua = K/denom;
end
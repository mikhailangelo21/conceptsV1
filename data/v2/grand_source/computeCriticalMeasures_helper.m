function val = computeCriticalMeasures_helper(vec1, vec2, type)
    
% INPUT:
% vec1,2 = Nx1 vectors
% type = string, either 'r' (Pearson's correlation, Fisher transformed),
%        'OCp' (percent of consistently ordered pairs),
%         or 'tau' (Kendall's tau)
%
% OUTPUT:
% val = value of type 'type' denoting co-variation between vec1 and vec2

switch type
    case 'r'
        val = corr(vec1,vec2);
        val = max(min(val,0.9999),-0.9999);
        val = atanh(val);
    case 'OCp'
        n = numel(vec1);
        vec1_repRows = repmat(vec1,1,n);                % copying vec1 into n columns
        vec1_repCols = repmat(vec1',n,1);               % copying vec1 into n rows
        vec1_bigger = vec1_repRows > vec1_repCols;      % comparing each pair of ratings

        vec2_repRows = repmat(vec2,1,n);                % copying vec2 into n columns
        vec2_repCols = repmat(vec2',n,1);               % copying vec2 into n rows
        vec2_bigger = vec2_repRows > vec2_repCols;      % comparing each pair of ratings
        
        val = sum(sum(vec1_bigger & vec2_bigger)) / nchoosek(n,2);
    case 'tau'
        val = rankCorr_Kendall_taua(vec1, vec2);
end
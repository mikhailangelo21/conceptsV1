"""Train-only transforms; ordinal metrics and concept-group uncertainty."""
from __future__ import annotations
import numpy as np
from scipy.stats import rankdata
from sklearn.decomposition import PCA


def spearman(y,p):
    y=rankdata(y); p=rankdata(p); y=y-y.mean(); p=p-p.mean()
    denom=np.linalg.norm(y)*np.linalg.norm(p)
    return float(y@p/denom) if denom else float('nan')


def difference_direction(x,y,categories):
    differences=[]; used=[]; skipped=[]; groups={}
    for cat in sorted(set(categories)):
        idx=np.flatnonzero(categories==cat); n=len(idx); k=max(1,n//3)
        values=np.sort(y[idx]); low,high=values[k-1],values[-k]
        if n<3 or low>=high:
            skipped.append(cat); continue
        small=idx[y[idx]<=low]; large=idx[y[idx]>=high]
        diff=x[large].mean(0)-x[small].mean(0)
        differences.append(diff); used.append(cat); groups[cat]={'low_indices':small.tolist(),'high_indices':large.tolist(),'low_cut':float(low),'high_cut':float(high)}
    if not differences: raise ValueError('No categories with distinct lower/upper size groups')
    differences=np.stack(differences); direction=differences.mean(0)
    norm=np.linalg.norm(direction)
    if not np.isfinite(norm) or norm<=1e-12: raise ValueError('Degenerate mean-difference direction')
    direction=direction/norm; mu=x.mean(0)
    if np.dot((x-mu)@direction,y-y.mean())<0: direction=-direction
    scores=(x-mu)@direction; scale=float(scores.std(ddof=1))
    if scale<=1e-12: raise ValueError('Degenerate training projection variance')
    unit=differences/np.maximum(np.linalg.norm(differences,axis=1,keepdims=True),1e-12)
    return dict(mu=mu,direction=direction,scale=scale,category_differences=differences,
                categories=used,skipped_categories=skipped,category_groups=groups,
                category_cosines=unit@unit.T,category_alignment=unit@direction)


def ridge_grid(x,y,validation_x,validation_y,alphas):
    """Centered dual ridge, no coordinate scaling. Validation chooses alpha only."""
    mu=x.mean(0); ym=float(y.mean()); z=x-mu
    eigen, vectors=np.linalg.eigh(z@z.T)
    eigen=np.maximum(eigen,0); rhs=vectors.T@(y-ym); candidates=[]
    for alpha in sorted(alphas):
        coef=z.T@(vectors@(rhs/(eigen+alpha)))
        score=spearman(validation_y,(validation_x-mu)@coef+ym)
        candidates.append((score if np.isfinite(score) else -np.inf,float(alpha),coef))
    score,alpha,coef=max(candidates,key=lambda a:(a[0],-a[1]))
    return dict(coef=coef,intercept=ym,alpha=alpha,mu=mu,validation_rho=score,
                grid=[dict(alpha=a,rho=s) for s,a,_ in candidates])


def fit_layer(x,y,categories,vx,vy,alphas,training_ids):
    direction=difference_direction(x,y,categories)
    ridge=ridge_grid(x,y,vx,vy,alphas)
    rank=int(np.linalg.matrix_rank(x-x.mean(0)))
    pca=PCA(n_components=min(rank,len(x)-1,x.shape[1]),whiten=False,svd_solver='full').fit(x)
    return dict(**direction,ridge_coef=ridge['coef'],ridge_intercept=ridge['intercept'],ridge_alpha=ridge['alpha'],
                ridge_grid=ridge['grid'],pca_mean=pca.mean_,pca_components=pca.components_,pca_variance=pca.explained_variance_ratio_,
                training_ids=list(training_ids),validation_rho=spearman(vy,(vx-direction['mu'])@direction['direction']))


def choose_layer(fits):
    def key(item):
        layer,fit=item; s=fit['validation_rho']
        return (s if np.isfinite(s) else -np.inf,-layer)
    return max(fits.items(),key=key)[0]


def grouped_bootstrap_indices(groups,rng):
    unique=np.unique(groups); draws=rng.choice(unique,len(unique),replace=True)
    return np.concatenate([np.flatnonzero(np.asarray(groups)==g) for g in draws])


def bootstrap_rho(y,p,reps,seed):
    """Input is one aggregated observation per independent synonym group."""
    rng=np.random.default_rng(seed); indices=rng.integers(len(y),size=(reps,len(y)))
    a=rankdata(np.asarray(y)[indices],axis=1); b=rankdata(np.asarray(p)[indices],axis=1)
    a-=a.mean(1,keepdims=True); b-=b.mean(1,keepdims=True)
    denom=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
    scores=np.divide((a*b).sum(1),denom,out=np.full(reps,np.nan),where=denom>0)
    good=scores[np.isfinite(scores)]
    return dict(low=float(np.quantile(good,.025)) if len(good) else None,high=float(np.quantile(good,.975)) if len(good) else None,
                valid_replicates=len(good),degenerate_replicates=int(reps-len(good)))


def bootstrap_mean(values,reps,seed):
    values=np.asarray(values); rng=np.random.default_rng(seed)
    means=values[rng.integers(len(values),size=(reps,len(values)))].mean(1)
    return dict(mean=float(values.mean()),low=float(np.quantile(means,.025)),high=float(np.quantile(means,.975)),n_groups=len(values))


def shuffle_within_category(y,categories,rng):
    out=y.copy()
    for cat in np.unique(categories):
        mask=np.flatnonzero(categories==cat); out[mask]=rng.permutation(y[mask])
    return out

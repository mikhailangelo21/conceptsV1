"""Optional CPU-only nonlinear follow-up; never silently enabled by a profile."""
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from .util import atomic_json


def nonlinear_exploration(run, suite, activation, selections, models):
    cfg=suite['config']['nonlinear']; outcomes=[]; stability=[]
    if not cfg['enabled']:
        atomic_json(run/'nonlinear.json',dict(status='disabled_by_configuration',restarts=[]))
        return
    from .suite_analysis import _eligible_fit, _target_rows, _xy
    blocks=suite['config']['extraction_block_numbers']
    for quality, rows in _target_rows(suite['cases']).items():
        if quality not in selections: continue
        train=[r for r in rows if _eligible_fit(r)]
        valid=[r for r in rows if r['entity_split']=='validation' and r['template_split']=='train' and r['combination_split']=='train']
        test=[r for r in rows if r['entity_split']=='test']
        if not train or not valid or not test: continue
        s=selections[quality]; li=blocks.index(s['block']); ro=s['readout']
        x,y=_xy(train,activation,ro,li,quality); vx,vy=_xy(valid,activation,ro,li,quality); tx,ty=_xy(test,activation,ro,li,quality)
        scaler=StandardScaler().fit(x); x=scaler.transform(x); vx=scaler.transform(vx); tx=scaler.transform(tx)
        # Train-only truncated SVD keeps this optional CPU fit bounded.
        _,_,vh=np.linalg.svd(x,full_matrices=False); basis=vh[:min(32,len(train)-1,x.shape[1])]
        x=x@basis.T; vx=vx@basis.T; tx=tx@basis.T
        baseline_x,_=_xy(valid,activation,ro,li,quality)
        baseline=float(np.square(models[quality].predict(baseline_x)-vy).mean())
        restarts=[]; projectors=[]
        import copy
        for seed in cfg['restart_seeds'][:cfg['maximum_restarts']]:
            probe=MLPRegressor(hidden_layer_sizes=(16,),activation='tanh',solver='adam',random_state=seed,batch_size=min(32,len(x)))
            best=None; stale=0; history=[]
            for epoch in range(100):
                probe.partial_fit(x,y)
                loss=float(np.square(probe.predict(vx)-vy).mean()); history.append(loss)
                if best is None or loss < best[0]-1e-6: best=(loss,copy.deepcopy(probe),epoch); stale=0
                else: stale+=1
                if stale>=10: break
            loss,fit,epoch=best
            row=dict(quality=quality,seed=seed,validation_mse=loss,linear_validation_mse=baseline,epochs=len(history),best_epoch=epoch,
                     stopped_early=stale>=10,validation_history=history)
            outcomes.append(row); restarts.append((loss,seed,fit,row))
            u,sv,_=np.linalg.svd(fit.coefs_[0],full_matrices=False); rank=min(4,np.linalg.matrix_rank(fit.coefs_[0]))
            projectors.append((seed,u[:,:rank]@u[:,:rank].T))
        chosen=min(restarts,key=lambda r:(r[0],r[1])); loss,seed,fit,row=chosen
        row.update(selected=True,test_mse=float(np.square(fit.predict(tx)-ty).mean()),
                   validation_improvement_over_linear=baseline-loss,
                   escalation_allowed=loss<baseline)
        for i,(seed,a) in enumerate(projectors):
            for seed2,b in projectors[i+1:]: stability.append(dict(quality=quality,seed_a=seed,seed_b=seed2,projector_distance=float(np.linalg.norm(a-b))))
    atomic_json(run/'nonlinear.json',dict(status='complete',restarts=outcomes,subspace_stability=stability,
                interpretation='Optimizer restarts are not local semantic regions. Test results cannot select a restart. No automatic escalation.'))

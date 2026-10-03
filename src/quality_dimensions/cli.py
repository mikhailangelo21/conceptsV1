import argparse
import json
from pathlib import Path
from .pipeline import configuration,prepare,execute
from .model import environment
from .report import report
from .data import import_data


def main():
    parser=argparse.ArgumentParser(prog='qd',description='Ordinal size accessibility, context sensitivity and additive steering')
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('doctor')
    sae=sub.add_parser('sae',help='Qwen-Scope SAE collection (separate versioned runs)')
    sae_sub=sae.add_subparsers(dest='sae_command',required=True)
    for name in ['plan','collect','extract','encode','validate']:
        p=sae_sub.add_parser(name)
        p.add_argument('--config',default='configs/sae_scope_v1.json')
        p.add_argument('--profile',choices=['smoke','laptop','extended'],default='laptop')
        p.add_argument('--layers',help='Comma-separated zero-based SAE layer indices; default 0..27 (smoke defaults to 3)')
        p.add_argument('--trace-count',type=int)
        p.add_argument('--run-dir')
        p.add_argument('--resume',action='store_true')
        p.add_argument('--max-model-minutes',type=float)
        p.add_argument('--max-encoding-minutes',type=float)
    sae_analysis=sub.add_parser('sae-analyze',help='Analyze existing Qwen-Scope SAE collection without model extraction')
    sae_analysis_sub=sae_analysis.add_subparsers(dest='analysis_command',required=True)
    p=sae_analysis_sub.add_parser('run')
    p.add_argument('--config',default='configs/sae_analysis_v1.json')
    p.add_argument('--profile',choices=['smoke','core','extended'],default='core')
    p.add_argument('--run-dir')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--stage',choices=['all','audit','methods','linear','nonlinear','tokens','extended'],default='all')
    p=sae_analysis_sub.add_parser('report')
    p.add_argument('--run-dir',required=True)
    p=sae_analysis_sub.add_parser('enrich')
    p.add_argument('--run-dir',required=True)
    p=sae_analysis_sub.add_parser('optional-methods')
    p.add_argument('--run-dir',required=True)
    p=sae_analysis_sub.add_parser('context')
    p.add_argument('--run-dir',required=True)
    p=sae_analysis_sub.add_parser('reconstruction')
    p.add_argument('--run-dir',required=True)
    p=sae_analysis_sub.add_parser('baselines')
    p.add_argument('--run-dir',required=True)
    geometry=sub.add_parser('geometry',help='Covariance geometry extension of V3 (separate versioned runs)')
    geometry_sub=geometry.add_subparsers(dest='geometry_command',required=True)
    p=geometry_sub.add_parser('metric'); p.add_argument('--config',default='configs/quality_geometry_v1.json'); p.add_argument('--vocabulary',choices=['smoke','full'],default='full')
    p=geometry_sub.add_parser('run'); p.add_argument('--config',default='configs/quality_geometry_v1.json'); p.add_argument('--profile',choices=['smoke','laptop'],default='smoke'); p.add_argument('--vocabulary',choices=['smoke','full']); p.add_argument('--run-dir'); p.add_argument('--resume',action='store_true'); p.add_argument('--stage',choices=['all','prepare','analyze'],default='all')
    p=geometry_sub.add_parser('report'); p.add_argument('--run-dir',required=True)
    p=geometry_sub.add_parser('steer'); p.add_argument('--run-dir',required=True); p.add_argument('--case-id',required=True); p.add_argument('--quality',required=True); p.add_argument('--strength',type=float,required=True); p.add_argument('--measurement-reviewed',action='store_true')
    suite=sub.add_parser('suite',help='Versioned quality-suite v3 commands')
    suite_sub=suite.add_subparsers(dest='suite_command',required=True)
    for name in ['plan','run']:
        p=suite_sub.add_parser(name); p.add_argument('--config',default='configs/quality_suite_v3.json'); p.add_argument('--profile',choices=['smoke','laptop','extended'],default=None); p.add_argument('--run-dir'); p.add_argument('--resume',action='store_true'); p.add_argument('--max-model-minutes',type=float,default=None)
        if name=='run': p.add_argument('--stage',choices=['all','extract','behavior','analyze'],default='all')
    p=suite_sub.add_parser('report'); p.add_argument('--run-dir',required=True)
    p=suite_sub.add_parser('import-ratings'); p.add_argument('--input',required=True); p.add_argument('--output',required=True); p.add_argument('--suite-root',default='data/quality_suite_v3')
    p=sub.add_parser('audit'); p.add_argument('--config',default='configs/v2.yaml')
    p=sub.add_parser('v2'); p.add_argument('--config',default='configs/v2.yaml'); p.add_argument('--stage',choices=['all','behavior','context','precision','causal','report'],default='all'); p.add_argument('--resume',action='store_true'); p.add_argument('--max-model-minutes',type=float); p.add_argument('--run-dir')
    for name in ['prepare-data','run','extract','analyze','intervene']:
        p=sub.add_parser(name); p.add_argument('--config',required=True); p.add_argument('--run-dir'); p.add_argument('--resume',action='store_true'); p.add_argument('--max-model-minutes',type=float)
    p=sub.add_parser('report'); p.add_argument('--run-dir',required=True)
    p=sub.add_parser('import-data'); p.add_argument('--input',required=True); p.add_argument('--output',required=True); p.add_argument('--references',required=True); p.add_argument('--pairs',required=True); p.add_argument('--ratings')
    args=parser.parse_args()
    if args.command=='sae-analyze':
        if args.analysis_command=='report':
            from .sae_analysis_report import build_report
            print(build_report(args.run_dir))
        elif args.analysis_command=='enrich':
            from .sae_analysis_enrich import enrich
            print(enrich(args.run_dir))
        elif args.analysis_command=='optional-methods':
            from .sae_analysis_optional import optional_methods
            print(optional_methods(args.run_dir))
        elif args.analysis_command=='context':
            from .sae_analysis_context import context_checks
            print(context_checks(args.run_dir))
        elif args.analysis_command=='reconstruction':
            from .sae_analysis_reconstruction import reconstruction_check
            print(reconstruction_check(args.run_dir))
        elif args.analysis_command=='baselines':
            from .sae_analysis_baselines import lexical_baselines
            print(lexical_baselines(args.run_dir))
        else:
            from .sae_analysis import run_analysis
            print(run_analysis(args.config,args.profile,args.run_dir,args.resume,args.stage))
        return
    if args.command=='sae':
        from .sae_collection import prepare as sae_prepare, plan as sae_plan, extract as sae_extract, encode as sae_encode, validate as sae_validate
        if args.max_model_minutes is not None and args.max_model_minutes<=0: parser.error('Model budget must be positive')
        if args.max_encoding_minutes is not None and args.max_encoding_minutes<=0: parser.error('Encoding budget must be positive')
        try: layers=[int(s) for s in args.layers.split(',')] if args.layers else None
        except ValueError: parser.error('--layers must be comma-separated integers')
        run=sae_prepare(args.config,args.profile,layers,args.trace_count,args.run_dir,args.resume)
        print(run,flush=True)
        if args.sae_command=='plan':
            print(json.dumps(sae_plan(run),indent=2)); return
        if not (run/'plan.json').exists(): print(json.dumps(sae_plan(run),indent=2),flush=True)
        if args.sae_command=='collect':
            before=json.loads((run/'state.json').read_text())
            if before['model_seconds']==0 and before['encoding_seconds']<600 and not any((run/'raw/main').glob('*.npz')):
                limit=args.max_encoding_minutes or json.loads((run/'config.json').read_text())['max_encoding_minutes']
                sae_encode(run,min(limit,10))
            sae_extract(run,args.max_model_minutes)
            sae_encode(run,args.max_encoding_minutes)
        elif args.sae_command=='extract': sae_extract(run,args.max_model_minutes)
        elif args.sae_command=='encode': sae_encode(run,args.max_encoding_minutes)
        result=sae_validate(run)
        saved=json.loads((run/'state.json').read_text())
        sae_config=json.loads((run/'config.json').read_text())
        resume=f'qd sae collect --config {args.config} --profile {args.profile} --run-dir {run} --resume'
        if args.layers: resume+=f' --layers {args.layers}'
        if args.trace_count is not None: resume+=f' --trace-count {args.trace_count}'
        model_cap=args.max_model_minutes or sae_config['max_model_minutes']
        encode_cap=args.max_encoding_minutes or sae_config['max_encoding_minutes']
        if saved['model_seconds']>=model_cap*60-1: resume+=f' --max-model-minutes {int(saved["model_seconds"]//60)+30}'
        elif args.max_model_minutes is not None: resume+=f' --max-model-minutes {args.max_model_minutes}'
        if saved['encoding_seconds']>=encode_cap*60-1: resume+=f' --max-encoding-minutes {int(saved["encoding_seconds"]//60)+30}'
        elif args.max_encoding_minutes is not None: resume+=f' --max-encoding-minutes {args.max_encoding_minutes}'
        layer_counts={key:f"{value['completed']}/{value['expected']}" for key,value in result['layers'].items() if value['completed']}
        print(json.dumps({'status':result['status'],'completed_layer_counts':layer_counts,
                          'manifest':str(run/'manifest.json'),'resume_command':resume},indent=2))
        if args.sae_command=='collect' and result['status']!='complete': raise SystemExit(2)
        return
    if args.command=='geometry':
        if args.geometry_command=='metric':
            from .geometry_pipeline import prepare_geometry_metric
            prepare_geometry_metric(args.config,args.vocabulary)
        elif args.geometry_command=='run':
            from .geometry_pipeline import run_geometry
            run_geometry(args.config,args.profile,args.run_dir,args.vocabulary,args.stage,args.resume)
        elif args.geometry_command=='steer':
            from .geometry_pipeline import run_final_steering
            print(run_final_steering(args.run_dir,args.case_id,args.quality,args.strength,args.measurement_reviewed))
        else:
            from .geometry_report import geometry_report
            print(geometry_report(args.run_dir))
        return
    if args.command=='suite':
        if args.suite_command=='report':
            from .suite_report import suite_report
            print(suite_report(args.run_dir)); return
        if args.suite_command=='import-ratings':
            from .suite_data import import_human_ratings
            print(import_human_ratings(args.input,args.output,args.suite_root)); return
        from .suite_data import load_suite,suite_plan
        from .suite_pipeline import prepare_suite,run_suite
        profile=args.profile or json.loads(Path(args.config).read_text())['default_profile']
        if args.max_model_minutes is not None and args.max_model_minutes<=0: parser.error('Budget must be positive')
        if args.suite_command=='plan':
            run,spec=prepare_suite(args.config,profile,args.run_dir,args.resume)
            print(run); print(json.dumps(suite_plan(spec),indent=2)); return
        run=run_suite(args.config,profile,args.run_dir,args.resume,args.max_model_minutes,args.stage)
        if args.stage=='all' and json.loads((run/'state.json').read_text())['status']!='complete': raise SystemExit(2)
        return
    if args.command in ['audit','v2']:
        from .v2config import load_config
        config=load_config(args.config)
        if args.command=='audit':
            from .audit import run_audit
            run_audit(config)
        else:
            from .refinement import execute_v2
            execute_v2(config,args.stage,args.run_dir,args.resume,args.max_model_minutes)
        return
    if args.command=='doctor': print(json.dumps(environment(),indent=2)); return
    if args.command=='report': print(report(args.run_dir)); return
    if args.command=='import-data': import_data(args.input,args.output,args.references,args.pairs,args.ratings); return
    config=configuration(args.config)
    if args.max_model_minutes is not None and args.max_model_minutes<=0: parser.error('Budget must be positive')
    if args.command=='prepare-data':
        run,*_=prepare(config,args.run_dir,args.resume,preparing=True); print(run); print((run/'plan.json').read_text()); report(run)
    else:
        run=execute(config,args.command,args.run_dir,args.resume,args.max_model_minutes)
        state=json.loads((run/'state.json').read_text())
        if state['status']=='blocked' or state.get('error_type'): raise SystemExit(2)
        if state['status']=='partial' and state.get('reason','').startswith(('ValueError:','RuntimeError:','FloatingPointError:','OSError:')): raise SystemExit(2)

if __name__=='__main__': main()

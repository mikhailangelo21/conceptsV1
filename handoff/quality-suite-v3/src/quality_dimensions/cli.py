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

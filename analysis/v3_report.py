"""Build the source-backed native report manifest from reviewed V3 tables."""
import json
import sqlite3
import copy
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/quality_suite_v3_findings'
data=json.loads((OUT/'datasets.json').read_text())
title='What the V3 quality experiments show'
manifest=dict(version=1,surface='report',title=title,description='Detailed interpretation of the bounded laptop screen, with matched controls and test denominators.',blocks=[],charts=[],tables=[],sources=[])
snapshot=dict(version=1,status='ready',datasets={})
db=sqlite3.connect(':memory:')

def materialize(id, frame):
    name='derived_'+id
    frame.to_sql(name,db,index=False,if_exists='replace')
    sql='SELECT * FROM "'+name+'"'
    result=pd.read_sql_query(sql,db)
    snapshot['datasets'][id]=json.loads(result.to_json(orient='records'))
    return dict(sql=sql,language='sql',engine='sqlite',tables_used=[name],description='Reads the complete derived table, materialized by analysis/v3_report.py from the reviewed CSV. No top-N sampling or test-based selection.')

def chart(id,title,frame,x,y,group,source,ylab,subtitle):
    query=materialize(id,frame)
    sid=id+'_source'
    query['description'] += ' '+subtitle
    manifest['sources'].append(dict(id=sid,label=source,path='reports/quality_suite_v3_findings/'+source,query=query))
    enc=dict(x=dict(field=x,type='nominal',label=x.replace('_',' ').title()),y=dict(field=y,type='quantitative',label=ylab))
    if group:enc['color']=dict(field=group,type='nominal',label=group.replace('_',' ').title())
    enc['tooltip']=[dict(field=c,type='quantitative' if pd.api.types.is_numeric_dtype(frame[c]) else 'nominal',label=c.replace('_',' ')) for c in frame.columns if c not in [x,y,group]]
    manifest['charts'].append(dict(id=id,title=title,subtitle=subtitle,type='bar',dataset=id,sourceId=sid,encodings=enc,settings=dict(orientation='horizontal',groupMode='grouped',categoryLabelPolicy='wrap'),labels=dict(values='all'),legend=dict(sort='spec')))
    return dict(id=id+'_block',type='chart',chartId=id)

def table(id,title,frame,source,sort):
    query=materialize(id,frame)
    sid=id+'_source';manifest['sources'].append(dict(id=sid,label=source,path='reports/quality_suite_v3_findings/'+source,query=query))
    manifest['tables'].append(dict(id=id,title=title,dataset=id,sourceId=sid,defaultSort=dict(field=sort,direction='asc'),columns=[dict(field=c,label=c.replace('_',' ').title()) for c in frame.columns]))
    return dict(id=id+'_block',type='table',tableId=id)

g=pd.DataFrame(data['graded']);g=g[(g.method=='ridge') & g.wording.isin(['plain','paraphrase'])].copy()
g['wording']=g.wording.map({'plain':'Familiar','paraphrase':'Held-out paraphrase'})
g=g[['quality','wording','rho','mae','n_rows','n_groups']]
figs={}
figs['1.']=[chart('graded_order','Graded test ordering by property and wording',g,'quality','rho','wording','graded.csv','Spearman rho','Ten rows and two entity families per property and wording; ordering can survive numerical miscalibration.'),chart('graded_error','Graded test error by property and wording',g,'quality','mae','wording','graded.csv','MAE on 0–1 design scale','Lower is better. Unconstrained predictions can exceed the target range.')]
b=pd.DataFrame(data['binding']);b=b[b.method.isin(['ridge','static_embedding'])].copy();b['series']=b.method.map({'ridge':'Hidden state','static_embedding':'Static average'})+' / '+b.wording.map({'plain':'familiar','paraphrase':'paraphrase'})
b=b[(b.method=='ridge') | (b.wording=='plain')].copy()
b['series']=b.apply(lambda r:'Static average' if r['method']=='static_embedding' else ('Probe: familiar' if r['wording']=='plain' else 'Probe: paraphrase'),axis=1)
figs['2.']=[chart('binding_error','Binding test error by property and method',b[['quality','series','mae','n_rows','n_groups']],'quality','mae','series','binding.csv','MAE on binary 0–1 targets','Eight rows from one test entity family per property and wording; static averages cannot distinguish same-word assignments.')]
p=pd.DataFrame(data['paired']);p=p[(p.quality==p.measured_quality)&p.kind.isin(['adjacent_increase','clause_order_invariant','irrelevant_invariant']) & p.template_id.isin(['plain','paraphrase'])]
p=p.groupby(['kind','template_id'],as_index=False).agg(mean_absolute_delta=('mean_absolute_delta','mean'),properties=('quality','nunique'),total_pairs=('n_pairs','sum'),groups_per_property=('n_groups','min'))
p['kind']=p.kind.map({'adjacent_increase':'Genuine level increase','clause_order_invariant':'Clause reordering','irrelevant_invariant':'Irrelevant edit'})
figs['2.'].append(chart('nuisance','Probe response to meaningful and invariant changes',p,'kind','mean_absolute_delta','template_id','paired.csv','Mean absolute prediction change','Equal-weight mean over 14 properties. Invariant edits ideally produce zero change; these are distinct contrast families.'))
l=pd.DataFrame(data['lexical']);figs['3.']=[chart('lexical_order','Lexical test ordering by property and wording',l[['quality','wording','rho','n_rows','n_groups']],'quality','rho','wording','lexical.csv','Spearman rho','Only three to seven independent test concepts per property; labels are provisional authored ratings.')]
m=pd.DataFrame(data['matched_behavior']);m=m.melt(id_vars=['quality_id','template_id','n_comparisons','n_groups'],value_vars=['natural_accuracy','ab_accuracy'],var_name='format',value_name='accuracy');m['series']=m['format'].map({'natural_accuracy':'Natural','ab_accuracy':'A/B'})+' / '+m.template_id.map({'plain':'familiar','paraphrase':'paraphrase'})
figs['4.']=[]
for wording,label in [('plain','familiar wording'),('paraphrase','held-out paraphrase')]:
    subset=m[m.template_id==wording].copy()
    subset['series']=subset['format'].map({'natural_accuracy':'Natural','ab_accuracy':'A/B'})
    figs['4.'].append(chart('answer_format_'+wording,'Matched test accuracy: '+label,subset,'quality_id','accuracy','series','matched_behavior.csv','Accuracy (0–1)','Four underlying comparisons per property from one family; each A/B score averages four counterbalances.'))
n=pd.DataFrame(data['numeric_overall']);figs['5.']=[chart('numeric_error','Physical prediction error by property and method',n,'quality','range_normalized_mae','method','numeric_overall.csv','MAE / physical training range','One test family per property, 9–22 cases; challenge roles differ across properties. Lower is better.')]
f=pd.DataFrame(data['factorial']);f=f[f.quality!='hue'];fc=f[f.partition.isin(['entity_test|factorial|numeric','entity_test+template|factorial|reordered'])].copy();fc['wording']=fc.partition.map({'entity_test|factorial|numeric':'Familiar numerical','entity_test+template|factorial|reordered':'Held-out reordered'})
figs['6.']=[chart('factorial_error','Joint probe error by output and wording',fc[['domain','quality','wording','mae','rho','n_rows','n_groups','rank']],'quality','mae','wording','factorial.csv','MAE on 0–1 design scale','Held-out entity, with training-eligible combinations in both series; one test family per domain.'),table('factorial_detail','All factorial test partitions',f[['domain','quality','partition','rank','n_rows','n_groups','rho','mae']],'factorial.csv','domain')]
h=pd.DataFrame(data['hue']);h['condition']=h.partition.map({'entity_test|hue|numeric':'Hue: numerical description','entity_test|hue|plain':'Hue: colour words','entity_test|factorial|numeric':'Factorial: familiar wording / seen combinations','entity_test+template|factorial|reordered':'Factorial: new wording / seen combinations','combination+entity_test|factorial|numeric':'Factorial: familiar wording / new combinations','combination+entity_test+template|factorial|reordered':'Factorial: new wording / new combinations'})
figs['7.']=[chart('hue_error','Circular hue prediction error across test conditions',h[['condition','circular_mae_degrees','n_rows','n_groups']],'condition','circular_mae_degrees',None,'hue.csv','Circular MAE (degrees)','All displayed conditions use block 16, final-token ridge selected on validation; lower is better.'),table('hue_neighbors','Hue neighbourhood checks at the selected site',pd.DataFrame(data['hue_neighborhood']),'hue_neighborhood.csv','template_id')]
text=(OUT/'findings.md').read_text()
# Presentation-only refinements: preserve reviewed values and all categories.
# Canonical charts consume `settings`, not the legacy ignored `options` object.
chart_map=[]
for section, blocks in list(figs.items()):
    revised=[]
    for block in blocks:
        if block['type']!='chart':
            revised.append(block); continue
        c=next(c for c in manifest['charts'] if c['id']==block['chartId'])
        frame=pd.DataFrame(snapshot['datasets'][c['dataset']])
        x=c['encodings']['x']['field']; y=c['encodings']['y']['field']
        group=c['encodings'].get('color',{}).get('field')
        c['encodings']['x']['label']=''
        c['referenceLines']=[]
        def reference(value,label):
            c['referenceLines'].append(dict(axis='x',value=value,label=label,color='gray',lineStyle='dashed'))
        if y=='rho':
            c['encodings']['y']['label']='Ordering score (higher is better)'
            reference(1,'Perfect order')
            if c['id']=='lexical_order':reference(-1,'Reversed order')
            c['subtitle']+=' Score: 1 = perfect order, 0 = no monotonic relationship.'
        elif y=='accuracy':
            c['valueFormat']='percent';c['encodings']['y']['label']='Correct answers (%)'
            reference(.5,'50% guess');reference(1,'100%')
        elif y=='range_normalized_mae':
            c['valueFormat']='percent';c['encodings']['y']['label']='Error as % of training range (lower is better)'
            frame['method']=frame.method.map({'physical_ridge':'Hidden-state probe','numeric_literal':'Number-text baseline'})
        elif y=='mae':
            c['encodings']['y']['label']='Average error (lower is better)'
            if c['id']=='binding_error':
                frame=frame[frame.series!='Static average'].copy()
                reference(.5,'Word-only baseline')
                c['subtitle']+=' Dashed line: static word-average error = 0.5.'
            else:
                reference(.25,'One level')
                c['subtitle']+=' Dashed line: one 0.25 step on the design scale.'
        elif y=='circular_mae_degrees':
            c['encodings']['y']['label']='Average angular error (degrees; lower is better)'
            reference(90,'Uniform guess')
            c['subtitle']+=' 90° is the expected error of a uniform random angle, not a tested null.'
            frame['condition']=frame.condition.map({'Hue: numerical description':'Hue / numbers','Hue: colour words':'Hue / words','Factorial: familiar wording / seen combinations':'Known wording + known mix','Factorial: new wording / seen combinations':'New wording + known mix','Factorial: familiar wording / new combinations':'Known wording + new mix','Factorial: new wording / new combinations':'New wording + new mix'})
        else:
            c['encodings']['y']['label']='Change in predicted level'
        if group:
            frame[group]=frame[group].replace({'plain':'Familiar','paraphrase':'Reworded','Held-out paraphrase':'Reworded','Probe: familiar':'Familiar','Probe: paraphrase':'Reworded','Familiar numerical':'Familiar','Held-out reordered':'Reworded'})
        if x in ['quality','quality_id']:frame[x]=frame[x].str.capitalize()
        if c['id']=='factorial_error':
            frame[x]=frame.domain.str.replace('_',' ').str.title()+' · '+frame[x]
        # Alphabetical order keeps adjacent charts directly comparable.
        categories=sorted(frame[x].unique()) if x in ['quality','quality_id'] else list(frame[x].unique())
        series_order={'Familiar':0,'Reworded':1,'Natural':0,'A/B':1,'Hidden-state probe':0,'Number-text baseline':1}
        frame['_category']=frame[x].map({v:i for i,v in enumerate(categories)})
        frame['_series']=frame[group].map(series_order).fillna(0) if group else 0
        frame=frame.sort_values(['_category','_series']).drop(columns=['_category','_series'])
        chunk_labels=None
        if c['id']=='numeric_error':
            chunks=[categories[:4],categories[4:]]
        elif c['id']=='factorial_error':
            domain_sets=[['Affect','Colour'],['Extent Mass','Motion Thermal','Sound'],['Shape','Surface']]
            chunks=[[v for v in categories if v.split(' · ')[0] in domains] for domains in domain_sets]
            chunk_labels=['affect and colour','extent, motion and sound','shape and surface']
        else:
            chunks=[categories[i:i+7] for i in range(0,len(categories),7)]
        manifest['charts'].remove(c)
        snapshot['datasets'].pop(c['dataset'])
        for i,chunk in enumerate(chunks):
            new=copy.deepcopy(c);suffix='' if len(chunks)==1 else '_'+str(i+1)
            new['id']=c['id']+suffix;new['dataset']=new['id']
            if suffix:new['title']+=' — '+(chunk_labels[i] if chunk_labels else chunk[0]+' to '+chunk[-1])
            subset=frame[frame[x].isin(chunk)]
            query=materialize(new['id'],subset)
            oldsource=next(s for s in manifest['sources'] if s['id']==c['sourceId'])
            new['sourceId']=new['id']+'_reviewed_source'
            manifest['sources'].append(dict(id=new['sourceId'],label=oldsource['label'],path=oldsource['path'],query=query))
            manifest['charts'].append(new)
            revised.append(dict(id=new['id']+'_block',type='chart',chartId=new['id']))
            chart_map.append(dict(id=new['id'],question=new['title'],comparison=new['subtitle'],family='horizontal grouped bar',rows=len(subset),categories=len(chunk),metric=y,source=oldsource['path'],palette='Native comparison palette; consistent series order. Legend, paired position, values and reference lines also encode the comparison.'))
    figs[section]=revised
(OUT/'chart_map.json').write_text(json.dumps(chart_map,indent=2))
text=text.replace('The first two figures therefore show','The ordering and error figures therefore show')
parts=text.split('\n## ')
for i,part in enumerate(parts):
    body='# '+title if i==0 else '## '+part
    # Long immutable identifiers remain complete in source metadata; abbreviate
    # prose identifiers so inline code cannot force mobile horizontal overflow.
    body=body.replace('`ea980cb0a6c2ae4b936e82123acc929f1cec04c1`','`ea980cb0a6c2…`')
    body=body.replace('`quality_suite_v3_laptop-88fad55b46b0`','the laptop run ending `88fad55b46b0`')
    manifest['blocks'].append(dict(id='section_'+str(i),type='markdown',body=body,sourceId='run'))
    if i:
        prefix=part.split(' ')[0]
        manifest['blocks'].extend(figs.get(prefix,[]))
        if part.startswith('How to read'):
            manifest['blocks'].append(table('coverage','Independent entity coverage by family and split',pd.DataFrame(data['counts']),'counts.csv','family'))
manifest['sources'].insert(0,dict(id='run',label='Completed laptop run and reproducible interpretation',path='runs/quality_suite_v3_laptop-88fad55b46b0',query=dict(description='Primary artifacts: linear_metrics.csv, linear_predictions.csv, paired_contrasts.csv, behavior_scores.csv, numeric_challenges.csv, multioutput_metrics.csv, circular_neighborhoods.csv and model selections. Reproduction and input hashes: reports/quality_suite_v3_findings/provenance.json; derivations: analysis/v3_findings.py. All evidence is synthetic screening evidence from one pinned model.')))
payload=dict(surface='report',manifest=manifest,snapshot=snapshot)
(OUT/'artifact.json').write_text(json.dumps(payload,indent=2,allow_nan=False))
print(f'{len(manifest["charts"])} charts, {len(manifest["tables"])} tables, {len(manifest["blocks"])} blocks; artifact.json written')

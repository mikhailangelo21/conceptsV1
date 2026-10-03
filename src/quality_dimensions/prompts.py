from .util import digest

TEMPLATES = {
 't0': 'Context: The subject is the {lemma}. {context}\nTask: Name the subject.\nAnswer:',
 't1': 'The object under discussion is the {lemma}. {context}\nWhich object is the subject?\nAnswer:',
 't2': 'Read this description of the {lemma}, which is our subject. {context}\nGive the name of the subject.\nAnswer:'}
CONDITIONS = {
 'neutral':'The subject has its ordinary size.',
 'irrelevant':'The subject has its ordinary size and the observation takes place on a Tuesday.',
 'shrunk':'The subject has been shrunk to one hundredth of its ordinary size.',
 'enlarged':'The subject has been enlarged to one hundred times its ordinary size.',
 'distractor_shrunk':'Another object has been shrunk to one hundredth of its ordinary size; the subject is unchanged and has its ordinary size.',
 'distractor_enlarged':'Another object has been enlarged to one hundred times its ordinary size; the subject is unchanged and has its ordinary size.'}


def geometry_prompts(concepts,splits,config):
    rows=[]
    for r in concepts.to_dict('records'):
        for tid in config['seen_templates']+[config['heldout_template']]:
            for condition in CONDITIONS:
                if condition!='neutral' and tid not in config['context_templates']: continue
                text=TEMPLATES[tid].format(lemma=r['lemma'],context=CONDITIONS[condition])
                rows.append(dict(prompt_id=digest([r['concept_id'],tid,condition,text])[:24],concept_id=r['concept_id'],synonym_group=r['synonym_group'],condition=condition,template_id=tid,split=splits[r['concept_id']],text=text))
    return rows


def semantic_labels(polarity, swapped):
    yes,no=('B','A') if swapped else ('A','B')
    return (yes,no) if polarity=='larger' else (no,yes)


def behavioral_prompts(concepts, references, pairs, splits, subjects, config):
    cs=concepts.set_index('concept_id'); rs=references.set_index('concept_id'); rows=[]
    for pair in pairs.itertuples():
        cid=pair.target
        if splits[cid] not in ['validation','test']: continue
        lemma=cs.loc[cid,'lemma']; reference=rs.loc[pair.reference,'lemma']
        for template in config['behavior_templates']:
            for polarity in ['larger','smaller']:
                for swap in [False,True]:
                    larger,smaller=semantic_labels(polarity,swap)
                    options='A = no; B = yes.' if swap else 'A = yes; B = no.'
                    question=f'Is the subject physically {polarity} than the reference?' if template=='b0' else f'Compared with the reference, is the subject physically {polarity}?'
                    text=f'Context: The subject is the {lemma} of ordinary size.\nReference: The {reference} of ordinary size.\nQuestion: {question}\nOptions: {options}\nAnswer:'
                    rows.append(dict(concept_id=cid, task='size', template_id=template, pair_id=pair.pair_id, reference=pair.reference,polarity=polarity,swapped=swap,text=text,larger_label=larger,correct_label=larger if pair.expected_relation=='larger' else smaller))
            # Color is deterministic from seed + ID, independent of size labels.
            blue=int(digest([config['seed'],'tag',cid])[:8],16)%2==0
            for task in ['identity','tag']:
                for swap in [False,True]:
                    choices=[lemma,reference] if task=='identity' else ['blue','red']
                    correct=0 if task=='identity' or blue else 1
                    if swap: choices.reverse(); correct=1-correct
                    text=f'Context: The subject is the {lemma} of ordinary size. It has a {"blue" if blue else "red"} tag.\nReference: The {reference} of ordinary size.\nQuestion: '+('Which object is the subject?' if task=='identity' else 'What color is the subject\'s tag?')+f'\nOptions: A = {choices[0]}; B = {choices[1]}.\nAnswer:'
                    rows.append(dict(concept_id=cid,task=task,template_id=template,pair_id=pair.pair_id,reference=pair.reference,polarity='na',swapped=swap,text=text,larger_label='',correct_label='AB'[correct]))
    for r in rows:
        r['split']=splits[r['concept_id']]; r['synonym_group']=cs.loc[r['concept_id'],'synonym_group']; r['prompt_id']=digest(r)[:24]
    return rows


def tokenize_prompt(tokenizer,row,max_tokens):
    text=row['text']
    if not text.endswith('Answer:'): raise ValueError('Missing final Answer: marker')
    ids=tokenizer.encode(text,add_special_tokens=False)
    # The preceding newline can merge with punctuation in byte-level BPE.
    # Assert the marker itself, whose suffix is stable in the full boundary.
    marker=tokenizer.encode('Answer:',add_special_tokens=False)
    if ids[-len(marker):]!=marker: raise ValueError('Inconsistent readout-marker tokenization')
    if len(ids)>max_tokens: raise ValueError(f'{row["prompt_id"]}: {len(ids)} tokens exceeds {max_tokens}; truncation prohibited')
    return dict(row,token_ids=ids,token_count=len(ids),readout_index=len(ids)-1)


def answer_encoding(tokenizer,rows):
    for prefix in [' ', '']:
        selected=None
        for row in rows:
            ids=row['token_ids']; pair=[]
            for label in 'AB':
                full=tokenizer.encode(row['text']+prefix+label,add_special_tokens=False)
                if full[:len(ids)]!=ids or len(full)!=len(ids)+1: break
                pair.append(full[-1])
            if len(pair)!=2 or pair[0]==pair[1]: break
            if selected is not None and selected!=pair: break
            selected=pair
        else:
            if selected: return dict(prefix=prefix,token_ids=dict(zip('AB',selected)))
    raise ValueError('No common verified single-token A/B encoding; multi-token scoring is intentionally unsupported')

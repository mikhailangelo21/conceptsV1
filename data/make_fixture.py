"""Regenerate the transparent engineering fixture, never experimental ground truth."""
import csv
from pathlib import Path
ROOT = Path(__file__).parent
CATEGORIES = {
 'animals': [('garden ant',1),('housefly',2),('house mouse',4),('domestic rabbit',6),('domestic cat',6),('Labrador dog',7),('domestic sheep',8),('adult horse',9),('adult African elephant',10),('adult blue whale',12)],
 'produce': [('poppy seed',1),('garden pea',2),('blueberry',3),('cherry',3),('strawberry',4),('apple',4),('orange',4),('grapefruit',5),('cantaloupe melon',6),('watermelon',7)],
 'household': [('sewing bead',2),('shirt button',3),('thimble',3),('teacup',4),('coffee mug',5),('dinner plate',6),('bedside lamp',6),('dining chair',7),('two-seat sofa',9),('double bed',9)],
 'tools': [('sewing needle',3),('paper clip',3),('door key',4),('teaspoon',5),('screwdriver',5),('claw hammer',6),('hand saw',7),('garden spade',8),('wheelbarrow',8),('extension ladder',10)],
 'transport': [('roller skate',6),('skateboard',7),('kick scooter',8),('adult bicycle',8),('motorcycle',9),('compact car',10),('delivery van',10),('city bus',11),('railway passenger carriage',12),('Boeing 747 airliner',13)],
 'instruments': [('guitar pick',3),('kazoo',4),('harmonica',4),('soprano recorder',6),('tambourine',6),('trumpet',7),('violin',7),('acoustic guitar',8),('cello',9),('concert grand piano',10)]}
FIELDS = ['concept_id','lemma','synonym_group','category','size_score','label_kind','label_source','notes','aliases']
def write(name, rows, fields):
 with (ROOT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
rows=[]
for cat, items in CATEGORIES.items():
 for i,(lemma,score) in enumerate(items):
  cid=f'{cat}_{i:02d}'
  rows.append(dict(concept_id=cid,lemma=lemma,synonym_group=cid,category=cat,size_score=score,label_kind='demo_ordinal',label_source='Assistant-authored provisional engineering fixture; no raters or measurements',notes='Ordinary adult/full-sized object; approximate overall extent. Ordinal only.',aliases=''))
write('concepts.csv',rows,FIELDS)
refs=[dict(concept_id='ref_salt',lemma='grain of table salt',label_source='Assistant-authored obvious ordinal comparison'),dict(concept_id='ref_container',lemma='standard forty-foot shipping container',label_source='Assistant-authored obvious ordinal comparison')]
write('references.csv',refs,list(refs[0]))
pairs=[dict(pair_id='pair_'+r['concept_id'],target=r['concept_id'],reference='ref_container' if r['size_score']<=6 else 'ref_salt',expected_relation='smaller' if r['size_score']<=6 else 'larger',provenance='demo_ordinal: assistant-authored qualitative relation, not measured') for r in rows]
write('pairs.csv',pairs,list(pairs[0]))

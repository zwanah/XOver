"""Validate and import existing XOver selections/generated candidates; never import gold answers."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from xover_demo.config import Config,LANGUAGES,PARAMETERS
from xover_demo.retrieval import Retriever
from backend.cache import DemoStore
from pipeline.c2_retrieval.profile_kernel import ProfileKernelProvider
import numpy as np

c=Config.from_environment();c.assets.mkdir(parents=True,exist_ok=True)
profile=c.assets/'profile.npy'
if not profile.exists():
    base=c.taco/'TACO/caculate_score/scores/Influence_score'
    files=[base/f'qwen3_qwen3_influence_ml_8x_euclidean{suffix}_dev_scores.pkl' for suffix in ('','_ZH','_JA')]
    provider=ProfileKernelProvider.from_influence_pkls([str(p) for p in files],6180)
    np.save(profile,provider._U,allow_pickle=False)
    print('Prepared original multilingual profile kernel',flush=True)
store=DemoStore(c.cache);retriever=Retriever(c,store)
counts={'selections':0,'candidate_batches':0,'skipped':0}
showcases=[]
for lang in LANGUAGES:
    folder='fewshot_profile' if lang=='en' else f'fewshot_{lang}_profile'
    for split in ('dev','test'):
        path=c.research/'data'/folder/f'questions_{split}_k6_a0.3_profile_lam0.5_M20.json'
        if not path.exists():continue
        cell=json.loads(path.read_text());args=cell['args']
        if any(args.get(k)!=v for k,v in {'k':6,'alpha':.3,'lambda':.5,'M':20,'dedup_prefix_len':80,'method':'dpp','kernel_mode':'profile'}.items()):raise ValueError('Incompatible research export')
        by_index={}
        for row in cell['questions']:
            indices=row['selected']
            if len(indices)!=6 or len(set(indices))!=6 or any(type(i) is not int or not 0<=i<6180 for i in indices):raise ValueError('Invalid selected pool indices')
            payload=retriever.payload(row['nl_question'],indices,precomputed=True)
            if payload['prompt']!=row['prompt']:raise ValueError('Research prompt differs from current pool/assembly')
            identity=retriever.identity(row['nl_question'],lang)
            existing=store._get('xover-selection',identity)
            if existing is None or existing.value!=payload:store._put('xover-selection',identity,payload)
            counts['selections']+=1;by_index[row['query_index']]=row
        candidate_path=c.research/'data'/f'diag_ml_{split}'/lang/'candidates_a0.3_lam0.5_k8.json'
        if not candidate_path.exists():continue
        saved=json.loads(candidate_path.read_text())
        if saved['args']['model']!='deepseek-v4-flash' or saved['args']['cell']!=path.name:raise ValueError('Candidate model/prompt provenance mismatch')
        for record in saved['records']:
            row=by_index.get(record['query_index'])
            if row is None or row['nl_question']!=record['nl_question'] or len(record['candidates'])<3:
                counts['skipped']+=1;continue
            queries=record['candidates'][:3]
            if not all(isinstance(q,str) and len(q)<=12000 for q in queries):counts['skipped']+=1;continue
            payload={'queries':queries,'bbox':record.get('bbox','none'),'model':saved['args']['model'],'source':'precomputed'}
            identity=retriever.identity(row['nl_question'],lang)
            existing=store._get('xover-research-candidates',identity)
            if existing is None or existing.value!=payload:store._put('xover-research-candidates',identity,payload)
            counts['candidate_batches']+=1
            if split=='test' and record['query_index']==926:
                showcases.append({'language':lang,'question':row['nl_question'],'label':'Cafés in Martina Franca','center':[40.705,17.339],'zoom':14})
    print(lang,counts,flush=True)
(c.root/'frontend/showcases.json').write_text(json.dumps(showcases,ensure_ascii=False,indent=2))
print(json.dumps(counts),flush=True)

"""Run the original XOver C2 selector with multilingual ml_8x query embeddings."""
import json
import pickle
import threading
from dataclasses import replace
from pathlib import Path
from copy import deepcopy
import numpy as np

from backend.retrieval.config import RetrievalConfig
from backend.retrieval.encoders import QwenInfluenceEncoder
from pipeline.c2_retrieval.config import PoolConfig
from pipeline.c2_retrieval.pool_map import PoolMap
from pipeline.c2_retrieval.scores import QueryScores
from pipeline.c2_retrieval.select import ProfileDppSelector
from pipeline.c2_retrieval.profile_kernel import ProfileKernelProvider
from pipeline.c2_retrieval.assemble import assemble_prompt
from .config import PARAMETERS


def revisions(paths):
    return [[str(p.resolve()),p.stat().st_size,p.stat().st_mtime_ns] for p in paths]


def similarity(q, vectors):
    distances=np.sqrt(np.sum((vectors-np.asarray(q,dtype=np.float32).reshape(1,-1))**2,axis=1))
    span=distances.max()-distances.min()
    return np.ones_like(distances) if span < 1e-12 else 1-(distances-distances.min())/span


class Retriever:
    def __init__(self, config, store):
        self.config,self.store=config,store
        self.lock=threading.RLock()
        self.pool=None;self.encoder=None;self.base=None;self.influence=None;self.kernel=None
        paths=[config.assets/'profile.npy',config.checkpoint/'adapter_config.json',config.checkpoint/'adapter_model.safetensors']
        for split in ('train','syn'):
            paths += [config.data/f'{split}_final.json',config.taco/'embeddings'/f'qwen3_{split}_embeddings.pkl',config.taco/'embeddings_influence/qwen3_influence_ml_8x'/f'{split}_embeddings.pkl']
        self.revision=revisions(paths)
        self.memory={}

    def identity(self,question,language,k_shot=6):
        return {'version':1,'question':question,'language':language,'parameters':{**PARAMETERS,'k':k_shot},'artifacts':self.revision}

    def load_pool(self):
        if self.pool is None:
            c=PoolConfig(str(self.config.data),'',{'train':('train_final.json',''),'syn':('syn_final.json','')},6180)
            self.pool=PoolMap.build(c)
        return self.pool

    def _load(self):
        pool=self.load_pool()
        if self.base is not None:return
        base=[];influence=[]
        for split in ('train','syn'):
            with (self.config.taco/'embeddings'/f'qwen3_{split}_embeddings.pkl').open('rb') as f:b=pickle.load(f)
            with (self.config.taco/'embeddings_influence/qwen3_influence_ml_8x'/f'{split}_embeddings.pkl').open('rb') as f:i=pickle.load(f)
            entries=[e for e in pool.pool if e.split==split]
            for saved in (b,i):
                if saved.get('nl') is not None and list(saved['nl']) != [e.nl_question for e in entries]:raise ValueError('Embedding/pool text alignment failed')
                if len(saved['nl_embeddings']) != len(entries):raise ValueError('Embedding/pool size mismatch')
            base.append(np.asarray(b['nl_embeddings'],dtype=np.float32));influence.append(np.asarray(i['nl_embeddings'],dtype=np.float32))
        self.base=np.vstack(base);self.influence=np.vstack(influence)
        self.kernel=ProfileKernelProvider(np.load(self.config.assets/'profile.npy',allow_pickle=False))
        if self.kernel._U.shape[0]!=len(pool.pool):raise ValueError('Profile/pool mismatch')
        c=replace(RetrievalConfig(),influence_checkpoint=self.config.checkpoint,influence_model='qwen3_influence_ml_8x')
        self.encoder=QwenInfluenceEncoder(c)

    def payload(self,question,indices,k_shot=6,**extra):
        pool=self.load_pool()
        examples=[{'nl':pool.entry(i).nl_question,'ovq':pool.entry(i).overpassql,'example_index':i,'rank':rank} for rank,i in enumerate(indices,1)]
        return {'examples':examples,'selected':indices,'prompt':assemble_prompt(indices,pool,question) if indices else f'/* Answer the following: {question} */','parameters':{**PARAMETERS,'k':k_shot},**extra}

    def historical(self,question,language,k_shot):
        folder='fewshot_profile' if language=='en' else f'fewshot_{language}_profile'
        for split in ('dev','test'):
            path=self.config.research/'data'/folder/f'questions_{split}_k{k_shot}_a0.3_profile_lam0.5_M20.json'
            if not path.is_file():continue
            cell=json.loads(path.read_text())
            expected={'k':k_shot,'alpha':.3,'lambda':.5,'M':20,'dedup_prefix_len':80,'method':'dpp','kernel_mode':'profile'}
            if any(cell['args'].get(k)!=v for k,v in expected.items()):continue
            for row in cell['questions']:
                if row['nl_question']!=question:continue
                indices=row['selected']
                if len(indices)!=k_shot or len(set(indices))!=k_shot or any(type(i) is not int or not 0<=i<6180 for i in indices):continue
                result=self.payload(question,indices,k_shot,precomputed=True,cache_source='precomputed')
                if result['prompt']==row['prompt']:return result
        return None

    def select(self,question,language,k_shot=6):
        if type(k_shot) is not int or not 0<=k_shot<=12:raise ValueError('K-shot must be an integer from 0 to 12')
        with self.lock:
            identity=self.identity(question,language,k_shot);key=(question,language,k_shot)
            if key in self.memory:return {**deepcopy(self.memory[key]),'cache_source':'memory'}
            hit=self.store._get('xover-selection',identity)
            if hit:
                result=hit.value
                result={**result,'cache_source':'precomputed' if result.get('precomputed') else 'persistent'}
            else:
                result=self.historical(question,language,k_shot) if k_shot else self.payload(question,[],0,cache_source='computed')
                if result is not None:
                    self.store._put('xover-selection',identity,result)
                    self.memory[key]=deepcopy(result)
                    if len(self.memory)>128:self.memory.pop(next(iter(self.memory)))
                    return result
                self._load()
                b,i=self.encoder.encode_pair(question)
                sim=similarity(b,self.base);inf=similarity(i,self.influence)
                scores=QueryScores(-1,tuple(range(len(self.pool.pool))),tuple(sim),tuple(inf))
                selector=ProfileDppSelector(alpha=.3,dedup_prefix_len=80,M=20,lam=.5,phi_mode='linear',kernel_provider=self.kernel)
                selected=selector.select(scores,self.pool,k_shot).selected
                result=self.payload(question,selected,k_shot,cache_source='computed')
                for example in result['examples']:
                    n=example['example_index'];example.update(similarity=float(sim[n]),influence=float(inf[n]))
                self.store._put('xover-selection',identity,result)
            self.memory[key]=deepcopy(result)
            if len(self.memory)>128:self.memory.pop(next(iter(self.memory)))
            return result

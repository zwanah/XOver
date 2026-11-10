"""XOver C2 -> candidate execution/C3 -> conservative C4, with explicit provenance."""
import asyncio
import re
from time import perf_counter
from openai import AsyncOpenAI
from pipeline.c3_selection.denotation import signature
from pipeline.c3_selection.confidence import select_confidence
from pipeline.c4_revision.hygiene import clean_output, is_truncated
from pipeline.c4_revision.checkers import SyntaxChecker
from backend.protection import ProtectionDenied
from .retrieval import Retriever
from .config import PARAMETERS

SYSTEM = ("You are an expert in OpenStreetMap and the Overpass Query Language (OverpassQL). "
"You are given several worked examples, each pairing a question with its correct query "
"on a line beginning '#OverpassQL:'. Answer the final question by outputting ONLY a line "
"'#OverpassQL:' followed by the query — no reasoning, explanation, or commentary. The "
"query MUST start with '[out:' and be directly executable on an Overpass API endpoint. "
"Use only tags, values, and areas grounded in the question; do not invent constraints.")


def extract(raw):
    if '#OverpassQL' in raw:raw=raw.rsplit('#OverpassQL',1)[1].lstrip(': \n\t')
    result=clean_output(raw)
    if not result.startswith('[out:') or len(result)>12000 or is_truncated(result):raise ValueError('Incomplete candidate')
    return result


def denotation(result):
    elements=result.get('elements',[])
    count=next((e for e in elements if e.get('type')=='count'),None)
    if count is not None:
        n=int(count.get('tags',{}).get('total',0));return signature(f'count_{n}',n)
    ids=[f"{e['type']}_{e['id']}" for e in elements if 'id' in e and e.get('type') in ('node','way','relation')]
    return signature(ids,len(ids))


class Engine:
    def __init__(self,config,settings,store,overpass,protection):
        self.config,self.settings,self.store,self.overpass,self.protection=config,settings,store,overpass,protection
        self.retriever=Retriever(config,store)

    async def model(self,messages,client):
        await asyncio.to_thread(self.protection.reserve_model,client)
        async with AsyncOpenAI(api_key=self.settings.openai_api_key,timeout=60,max_retries=0) as api:
            response=await api.responses.create(model=self.settings.model,input=messages,reasoning={'effort':'low'},max_output_tokens=2200,store=False)
        return response.output_text, {'input_tokens':response.usage.input_tokens,'output_tokens':response.usage.output_tokens}

    async def run(self,question,language,bbox,client,emit):
        started=perf_counter();timings={};usage=[]
        identity={**self.retriever.identity(question,language),'bbox':bbox,'model':self.settings.model,'candidates':3,'pipeline_version':1,'overpass':list(self.settings.execution_urls)}
        complete=await asyncio.to_thread(self.store._get,'xover-answer',identity)
        if complete:
            result={**complete.value,'answer_cache':True,'model_usage':[], 'timings':{'server_ms':round((perf_counter()-started)*1000,2)}}
            await emit({'stage':'complete','data':result});return result
        t=perf_counter();await emit({'stage':'selection','message':'C2 · Selecting cross-lingual examples'})
        selected=await asyncio.to_thread(self.retriever.select,question,language)
        timings['selection_ms']=round((perf_counter()-t)*1000,2)
        await emit({'stage':'selection_done','source':selected['cache_source'],'elapsed_ms':timings['selection_ms'],'examples':selected['examples'],'parameters':PARAMETERS})
        t=perf_counter();await emit({'stage':'generation','message':'Generating or loading 3 candidate queries'})
        precomputed=await asyncio.to_thread(self.store._get,'xover-research-candidates',self.retriever.identity(question,language))
        batch=None
        if precomputed:
            saved=precomputed.value
            scoped=any(re.search(r'\{\{\s*(bbox|center)\s*\}\}',q) for q in saved['queries'])
            if saved['bbox']==bbox or (saved['bbox']=='none' and not scoped):
                batch={**saved,'source':'precomputed'}
        candidate_key={**identity,'prompt':selected['prompt'],'system':SYSTEM}
        if batch is None:
            hit=await asyncio.to_thread(self.store._get,'xover-candidates',candidate_key)
            if hit:batch={**hit.value,'source':'persistent'}
        if batch is None:
            prompt=selected['prompt']+f'\n\nCurrent map bbox (south,west,north,east): {bbox}. Use {{{{bbox}}}} only when the question refers to the current view. Return geometry suitable for a map.'
            queries=[]
            # Three independently decoded candidates. The public demo deliberately
            # bounds this to three, versus the research's eight sampled candidates.
            for n in range(3):
                text,tokens=await self.model([{'role':'system','content':SYSTEM},{'role':'user','content':prompt}],client)
                usage.append(tokens);queries.append(text)
                await emit({'stage':'candidate_generated','index':n+1,'total':3})
            batch={'queries':queries,'source':'generated','model':self.settings.model}
            await asyncio.to_thread(self.store._put,'xover-candidates',candidate_key,batch)
        timings['generation_ms']=round((perf_counter()-t)*1000,2)
        await emit({'stage':'generation_done','source':batch['source'],'model':batch['model'],'elapsed_ms':timings['generation_ms']})
        await asyncio.to_thread(self.protection.admit,client,'execute')
        await emit({'stage':'execution','message':'Executing candidate queries against the project map service'})
        t=perf_counter();executed=[]
        for index,raw in enumerate(batch['queries'][:3]):
            entry=await self.execute_candidate(raw,bbox)
            executed.append(entry)
            await emit({'stage':'candidate_executed','index':index+1,'ok':entry['ok'],'count':entry.get('count',0),'cache_hit':entry.get('cache',{}).get('hit',False)})
        timings['execution_ms']=round((perf_counter()-t)*1000,2)
        await emit({'stage':'selection_candidates','message':'C3 · Comparing execution results'})
        candidates=[{'sig':e['signature'],'ex':0} for e in executed]
        # ex is an unused diagnostic field in the original selector; no test gold
        # enters a live decision. No judge is claimed: this is nonempty voting.
        decision=select_confidence(candidates,comparator=None,prefer_nonempty=True)
        picked=decision.picked_index
        await emit({'stage':'revision','message':'C4 · Checking output hygiene and execution'})
        repaired=False
        if picked<0:
            # Only parser/incomplete-output failures justify a syntax repair.
            # Timeouts/upstream failures must not trigger semantic rewrites.
            failed=next((i for i,e in enumerate(executed) if e.get('repairable')),None)
            if failed is None:raise ValueError('All candidates failed to execute; retry when the map service is available.')
            raw=batch['queries'][failed]
            checker=SyntaxChecker(None,None,max_repair_attempts=1)
            messages=checker._build_repair_prompt(raw,'The query is incomplete or has an OverpassQL syntax error.',question)
            text,tokens=await self.model(messages,client);usage.append(tokens)
            revised=await self.execute_candidate(text,bbox)
            if not revised['ok']:raise ValueError('Candidates and the bounded syntax repair could not execute. Try a more specific question.')
            executed[failed]=revised;picked=failed;repaired=True
        chosen=executed[picked]
        summaries=[{'index':i+1,'query':e['query'],'ok':e['ok'],'count':e.get('count',0),'selected':i==picked,'cache':e.get('cache'), 'hygiene_changed':e.get('hygiene_changed',False)} for i,e in enumerate(executed)]
        timings['server_ms']=round((perf_counter()-started)*1000,2)
        result={'question':question,'language':language,'query':chosen['query'],'executed_query':chosen['executed_query'],'result':chosen['result'],'cache':chosen['cache'],
                'selection':selected,'candidates':summaries,'generation':{'source':batch['source'],'model':batch['model']},
                'candidate_selection':{'method':'execution nonempty vote','route':decision.route,'groups':decision.n_groups,'agreement':decision.top1_consistency},
                'revision':{'syntax_repaired':repaired,'hygiene_changed':any(e.get('hygiene_changed') for e in executed)},'model_usage':usage,'answer_cache':False,'timings':timings}
        await asyncio.to_thread(self.store._put,'xover-answer',identity,result)
        await emit({'stage':'complete','data':result});return result

    async def execute_candidate(self,raw,bbox):
        try:
            query=extract(raw)
        except ValueError:
            return {'query':clean_output(raw)[:12000],'ok':False,'signature':('error','incomplete'),'repairable':True}
        try:
            async with self.protection.slot('execute'):
                expanded=await self.overpass.expand_macros(query,bbox)
                result,cache=await self.overpass.execute_with_cache(expanded,raw_query=query,bbox=bbox)
            sig=denotation(result)
            count=next((int(e.get('tags',{}).get('total',0)) for e in result.get('elements',[]) if e.get('type')=='count'),len(result.get('elements',[])))
            return {'query':query,'executed_query':expanded,'ok':True,'signature':sig,'result':result,'count':count,'cache':cache,'hygiene_changed':query!=raw.strip()}
        except ProtectionDenied:raise
        except Exception as exc:
            detail=str(exc).lower()
            return {'query':query,'ok':False,'signature':('error','execution'),'repairable':any(x in detail for x in ('parse error','parse_error','syntax error'))}

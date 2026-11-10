import asyncio
import json
import logging
from contextlib import suppress
from fastapi import HTTPException,Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel,Field,field_validator
from backend.cache import DemoStore
from backend.protection import ProtectionDenied
from .config import Config,LANGUAGES,PARAMETERS
from .engine import Engine

log=logging.getLogger(__name__)
class Query(BaseModel):
    question:str=Field(min_length=1,max_length=2000)
    language:str='en'
    bbox:str=Field(max_length=150)
    @field_validator('question')
    @classmethod
    def nonempty(cls,v):
        if not v.strip():raise ValueError('Question is required')
        return v.strip()
    @field_validator('language')
    @classmethod
    def known_language(cls,v):
        if v not in LANGUAGES:raise ValueError('Unsupported language')
        return v
    @field_validator('bbox')
    @classmethod
    def valid_bbox(cls,v):
        import math
        values=[float(x) for x in v.split(',')]
        if len(values)!=4 or not all(math.isfinite(x) for x in values):raise ValueError('Invalid map bounds')
        s,w,n,e=values
        if not -90<=s<n<=90 or not -180<=w<e<=180:raise ValueError('Invalid map bounds')
        return ','.join(str(x) for x in values)


def install(app,settings,*,engine=None):
    config=Config.from_environment()
    if engine is None:
        engine=Engine(config,settings,DemoStore(config.cache),app.state.overpass,app.state.protection)
    app.state.xover=engine

    @app.get('/api/xover/config')
    async def public_config():
        return {'name':'XOver Demo','languages':LANGUAGES,'parameters':PARAMETERS,'candidates':3,'model':settings.model,
                'generation_available':bool(settings.openai_api_key),'turnstile':app.state.turnstile.public_config(),
                'protection':app.state.protection.public_config(),'pipeline':['multilingual ml_8x retrieval','profile DPP','three candidates','execution voting','hygiene and bounded syntax repair']}

    @app.post('/api/xover/query')
    async def query(body:Query,request:Request):
        try:await app.state.turnstile.verify(request,'generate')
        except ProtectionDenied as exc:raise HTTPException(exc.status_code,exc.detail,headers={'Retry-After':str(exc.retry_after)}) from None
        async def events():
            queue=asyncio.Queue()
            async def emit(event):await queue.put(event)
            async def run():
                try:
                    await emit({'stage':'verified','message':'Human verification complete'})
                    async with app.state.protection.slot('generate'):
                        await asyncio.wait_for(engine.run(body.question,body.language,body.bbox,getattr(request.state,'protection_client','local'),emit),timeout=240)
                except ProtectionDenied as exc:await emit({'stage':'error','detail':exc.detail,'status':exc.status_code})
                except ValueError as exc:await emit({'stage':'error','detail':str(exc),'status':422})
                except asyncio.TimeoutError:await emit({'stage':'error','detail':'The query timed out. Please try again; saved stages can be reused.','status':504})
                except Exception:
                    log.exception('XOver query failed')
                    await emit({'stage':'error','detail':'XOver could not complete this query. Please try again.','status':502})
                finally:await queue.put(None)
            task=asyncio.create_task(run())
            try:
                while True:
                    event=await queue.get()
                    if event is None:break
                    yield json.dumps(event,ensure_ascii=False)+'\n'
            finally:
                task.cancel()
                with suppress(asyncio.CancelledError):await task
        return StreamingResponse(events(),media_type='application/x-ndjson',headers={'Cache-Control':'no-store, no-transform','X-Accel-Buffering':'no'})

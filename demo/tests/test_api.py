import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock
from types import SimpleNamespace
from fastapi.testclient import TestClient
from backend.main import create_app
from backend.config import Settings
from backend.protection import ProtectionConfig
from backend.turnstile import TurnstileConfig
from xover_demo.api import install

class ApiTests(unittest.TestCase):
 def test_query_protection_and_single_verification(self):
  with tempfile.TemporaryDirectory() as tmp:
   settings=Settings(openai_api_key='test',frontend_dir=Path(tmp)/'no-frontend',protection=ProtectionConfig(enabled=True,state_path=Path(tmp)/'limits.db'),turnstile=TurnstileConfig(site_key='real-site',secret_key='real-secret'))
   app=create_app(settings)
   async def run(q,l,b,c,emit,k_shot=6):await emit({'stage':'complete','data':{'question':q}})
   engine=SimpleNamespace(run=AsyncMock(side_effect=run));install(app,settings,engine=engine)
   client=TestClient(app);body={'question':'咖啡馆','language':'zh','bbox':'40,17,41,18'};headers={'Origin':'https://wanzhuoyue.com'}
   self.assertEqual(client.post('/api/xover/query',json=body,headers=headers).status_code,403)
   engine.run.assert_not_awaited()
   app.state.turnstile.verify=AsyncMock()
   response=client.post('/api/xover/query',json={**body,'k_shot':3},headers=headers)
   self.assertEqual(response.status_code,200);self.assertIn('complete',response.text)
   self.assertEqual(app.state.turnstile.verify.await_count,1)
   self.assertEqual(engine.run.await_args.kwargs['k_shot'],3)
   self.assertEqual(client.post('/api/xover/query',json=body,headers={'Origin':'https://evil.example'}).status_code,403)

 def test_k_bounds_and_types(self):
  from xover_demo.api import Query
  from pydantic import ValidationError
  for k in (-1,13,2.5,True,'3'):
   with self.assertRaises(ValidationError):Query(question='q',language='en',bbox='1,2,3,4',k_shot=k)
  for k in (0,1,6,12):self.assertEqual(Query(question='q',bbox='1,2,3,4',k_shot=k).k_shot,k)

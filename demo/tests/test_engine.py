import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from backend.cache import DemoStore
from backend.config import Settings
from backend.protection import DemoProtection,ProtectionConfig
from xover_demo.engine import Engine,extract,denotation

class EngineTests(unittest.IsolatedAsyncioTestCase):
 async def test_empty_is_valid_and_full_answer_reused(self):
  with tempfile.TemporaryDirectory() as tmp:
   e=Engine.__new__(Engine);e.settings=Settings(model='test');e.store=DemoStore(Path(tmp)/'c.db');e.protection=DemoProtection(ProtectionConfig())
   selected={'prompt':'example prompt','examples':[],'cache_source':'computed'}
   e.retriever=SimpleNamespace(identity=lambda q,l:{'question':q,'language':l},select=lambda q,l:selected)
   query='[out:json];node(1);out;'
   e.model=AsyncMock(return_value=(query,{'input_tokens':1,'output_tokens':1}))
   e.overpass=SimpleNamespace(expand_macros=AsyncMock(return_value=query),execute_with_cache=AsyncMock(return_value=({'elements':[]},{'hit':False})))
   emit=AsyncMock()
   result=await e.run('question','en','1,2,3,4','local',emit)
   self.assertEqual(e.model.await_count,3)
   self.assertFalse(result['revision']['syntax_repaired'])
   self.assertEqual(result['candidate_selection']['groups'],1)
   second=await e.run('question','en','1,2,3,4','local',emit)
   self.assertTrue(second['answer_cache']);self.assertEqual(second['model_usage'],[])
   self.assertEqual(e.model.await_count,3)
   await e.run('question','en','2,3,4,5','local',emit)
   self.assertEqual(e.model.await_count,6)

 def test_hygiene_and_incomplete_query(self):
  self.assertEqual(extract('```overpassql\n[out:json];node(1);out;\n```'),'[out:json];node(1);out;')
  with self.assertRaises(ValueError):extract('[out:json];node(')
  self.assertEqual(denotation({'elements':[]}),('empty',))
  self.assertNotEqual(denotation({'elements':[{'type':'node','id':1}]}),denotation({'elements':[{'type':'way','id':1}]}))

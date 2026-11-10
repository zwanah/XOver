"""Build curated multilingual cards from the existing XOver test questions."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from xover_demo.config import Config,LANGUAGES
CARDS=[
 {'id':'accessible-transit','query_index':675,'label':'Accessible stations & stops','detail':'New York · accessibility filters','place':'New York, USA','category':'Accessible transit','icon':'transit','center':[40.71427,-74.00597],'zoom':12},
 {'id':'libraries','query_index':570,'label':'Libraries, minus “Miejska”','detail':'Wrocław · name exclusions','place':'Wrocław, Poland','category':'Libraries · filtered by name','icon':'library','center':[51.108,17.038],'zoom':12},
 {'id':'cafes','query_index':926,'label':'Cafés in Martina Franca','detail':'Martina Franca · city-wide search','place':'Martina Franca, Italy','category':'Cafés','icon':'cafe','center':[40.705,17.339],'zoom':14},
]
def build(config):
 result=[]
 for lang in LANGUAGES:
  folder='fewshot_profile' if lang=='en' else f'fewshot_{lang}_profile'
  path=config.research/'data'/folder/'questions_test_k6_a0.3_profile_lam0.5_M20.json'
  rows={r['query_index']:r for r in json.loads(path.read_text())['questions']}
  for card in CARDS:
   result.append({**card,'language':lang,'question':rows[card['query_index']]['nl_question']})
 (config.root/'frontend/showcases.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 return len(result)
if __name__=='__main__':print('Built multilingual showcase cards:',build(Config.from_environment()))

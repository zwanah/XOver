from dataclasses import dataclass
from pathlib import Path
import os

LANGUAGES = {'en':'English','zh':'中文','yue':'粵語','fr':'Français','de':'Deutsch','ja':'日本語','ko':'한국어','ru':'Русский'}
DATA_LANGUAGES = dict(zip(LANGUAGES, ['english','mandarin_simplified','cantonese','french','german','japanese','korean','russian']))
PARAMETERS = {'k':6,'alpha':0.3,'lambda':0.5,'M':20,'dedup_prefix_len':80,'method':'profile-dpp','phi':'linear'}

@dataclass(frozen=True)
class Config:
    root: Path = Path(__file__).resolve().parents[1]
    research: Path = Path('/data/russ/LLM4Geo/MOsmNL/final')
    taco: Path = Path('/data/russ/LLM4Geo/Prompt_budget')
    data: Path = Path('/data/russ/LLM4Geo/dataset/OsmNL/Final')
    @property
    def assets(self): return self.root/'assets'
    @property
    def cache(self): return self.root/'outputs/cache.sqlite3'
    @property
    def checkpoint(self): return self.taco/'train_influence/results/influence_model_ckpt/qwen3e-regression_cosine1.0_angle1.0_ibn0.0_ml_8x'
    @classmethod
    def from_environment(cls):
        return cls(root=Path(os.getenv('XOVER_DEMO_ROOT',str(cls.root))),research=Path(os.getenv('XOVER_RESEARCH_ROOT',str(cls.research))),taco=Path(os.getenv('XOVER_TACO_ROOT',str(cls.taco))),data=Path(os.getenv('XOVER_POOL_ROOT',str(cls.data))))

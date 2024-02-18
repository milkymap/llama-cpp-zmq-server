
from typing import List, Any, Dict  

from pydantic import BaseModel

class StrategyArgsModel(BaseModel):
    args:List[Any]=[]
    kwargs:Dict[str, Any]={}
from typing import List, Optional, Dict 
from pydantic import BaseModel

from ...strategy.schema import StrategyArgsModel

class ListModelsResSchema(BaseModel):
    llama_model_names:List[str]

class DownloadModelReqSchema(BaseModel):
    name_of_model:str 
    url2model:str 
    bytes_monitor_rate:int=65536

class DownloadModelResSchema(BaseModel):
    download_status:bool=False  
    error_message:Optional[str]=None   

class LaunchModelReqSchema(BaseModel):
    llama_model_name:str 
    context_size:int=2048
    n_threads:int=8
    verbose:bool=True 
    n_gpu_layers:int=0
  
class LaunchModelResSchema(BaseModel):
    llama_model_name:str 
    status:bool 
    message:str 

class LlmGenerationReqSchema(BaseModel):
    prompt:str 
    max_tokens:int=512 
    stop:List[str]=["###"]
    temperature:float=0.1 
    top_p:float=0.2 
    top_k:int=7
    
class InspectModelsResSchema(BaseModel):
    model_name2worker_ids:Dict[str, List[int]]
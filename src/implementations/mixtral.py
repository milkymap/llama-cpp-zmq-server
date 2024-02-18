import re 

from typing import Any, Dict, List, Tuple 
from llama_cpp import Llama

from ..strategy import ABCStrategy
from os import path 

from ..server.routers.schema import LlmGenerationReqSchema

class MixtralEngine(ABCStrategy):
    def __init__(self, llama_model_name:str, context_size:int=2048, n_threads:int=8, n_gpu_layers:int=-1, verbose=True):
        self.model = Llama(
            model_path=llama_model_name, n_ctx=context_size, n_threads=n_threads, 
            n_gpu_layers=n_gpu_layers, verbose=verbose 
        )        
    
    def process_task(self, encoded_task: LlmGenerationReqSchema) -> Any:
        streaming = self.model(**encoded_task.model_dump(), stream=True)
        return streaming
    
import zmq 
import zmq.asyncio as azmq 
import asyncio 

import json 

import pickle 
from os import path 
from time import sleep 
from asyncio import Lock 
import httpx 

import aiofiles

import wget 
from typing import Tuple, Optional, Dict, List, Generator, AsyncGenerator 

from os import listdir
from fastapi import APIRouter, HTTPException, Body
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse, JSONResponse

from .schema import (
    ListModelsResSchema, DownloadModelReqSchema, DownloadModelResSchema, 
    LaunchModelReqSchema, LaunchModelResSchema, LlmGenerationReqSchema, 
    InspectModelsResSchema
)
from ...log import logger 
from ...schema.worker import WorkerStatus
from ...schema.task import TaskResponse

class LlamaRouter:
    def __init__(self, path2models:str, ctx:azmq.Context, outer_address:str, nb_llm_workers:int):
        self.path2models = path2models
        self.ctx = ctx 
        self.outer_address = outer_address
        self.router = APIRouter()

        self.shared_mutex:Lock = Lock()
        self.worker_id2schema_hmap:Dict[int, Optional[LaunchModelReqSchema]] = {}
        self.target_worker_id:int = 0
        
        for worker_id in range(nb_llm_workers):
            self.worker_id2schema_hmap[worker_id] = None 

        self.router.add_api_route('/list_models', self.list_models, response_model=ListModelsResSchema)
        self.router.add_api_route('/download_model', self.download_model, response_class=StreamingResponse, methods=['POST'])
        self.router.add_api_route('/launch_model/{worker_id:int}', self.launch_model, response_model=LaunchModelResSchema, methods=['POST'])
        self.router.add_api_route('/inspect_worker/{worker_id:int}', self.inspect_background_worker, response_model=Optional[LaunchModelReqSchema], methods=['GET'])
        self.router.add_api_route('/inspect_models', self.inspect_models, methods=['GET'], response_model=InspectModelsResSchema)

        self.router.add_api_route('/generate', self.generate, methods=['POST'])
    
    async def _aio_get_model(self, url2model:str, model_path:str, bytes_monitor_rate:int): 
        async with aiofiles.open(file=model_path, mode='wb') as afp: 
            async with httpx.AsyncClient() as client:
                res = await client.get(url2model)
                cdn_link = res.headers.get('Location', None)
                if cdn_link is None:
                   raise HTTPException(status_code=500, detail='can not download, no cdn_link was found')  
                async with client.stream('GET', cdn_link) as resp:
                    content_length = resp.headers.get('Content-Length')
                    if content_length is None:
                        raise HTTPException(status_code=500, detail='can not find the stream size')
                    stream_size = int(content_length)
                    stream_step = 0
                    async for chunk in resp.aiter_bytes(chunk_size=32768):
                        await afp.write(chunk)
                        chunk_len = len(chunk)
                        stream_step += chunk_len 
                        json_fmt_data = json.dumps(
                            {
                                'stream_size': stream_size,
                                'stream_step': stream_step 
                            }
                        ) + '\n'
                        logger.info(f'{json_fmt_data}')
                        if stream_step % bytes_monitor_rate == 0:
                            yield json_fmt_data

                    json_fmt_data = json.dumps(
                        {
                            'stream_size': stream_size,
                            'stream_step': stream_step 
                        }
                    ) + '\n'
                    yield json_fmt_data

    async def list_models(self):
        model_names = listdir(self.path2models)
        return ListModelsResSchema(llama_model_names=model_names)
    
    async def download_model(self, incoming_req:DownloadModelReqSchema):
        model_path = path.join(self.path2models, incoming_req.name_of_model)
        if path.isfile(model_path):
            raise HTTPException(status_code=500, detail=f'{incoming_req.name_of_model} was already downlaoded')
        
        return StreamingResponse(
            status_code=200,
            content=self._aio_get_model(
                url2model=incoming_req.url2model, 
                model_path=model_path, 
                bytes_monitor_rate=incoming_req.bytes_monitor_rate
            ),
            media_type='application/x-ndjson'
        )

    async def inspect_models(self):
        model_name2worker_ids:Dict[str, int] = {}
        async with self.shared_mutex:
            for worker_id, launched_model_schema in self.worker_id2schema_hmap.items():
                if launched_model_schema is None:
                    continue
                target_workers:List[int] = model_name2worker_ids.get(launched_model_schema.llama_model_name, [])
                target_workers.append(worker_id)
                model_name2worker_ids[launched_model_schema.llama_model_name] = target_workers
        return InspectModelsResSchema(model_name2worker_ids=model_name2worker_ids)
    
    async def inspect_background_worker(self, worker_id:int):
        current_worker_schema:Optional[LaunchModelReqSchema] = None 
        async with self.shared_mutex:
            worker_ids = list(self.worker_id2schema_hmap.keys())
            current_worker_schema = self.worker_id2schema_hmap.get(worker_id, ValueError(f'worker {worker_id} is not a valid id, please choose one of : {worker_ids}'))
        
        if isinstance(current_worker_schema, ValueError):
            raise HTTPException(
                status_code=500,
                detail=str(current_worker_schema)
            )
        
        return current_worker_schema
    
    async def launch_model(self, worker_id:int, reset_worker:bool=False, incoming_req:LaunchModelReqSchema=Body()):
        current_worker_schema:Optional[LaunchModelReqSchema] = None 
        async with self.shared_mutex:
            worker_ids = list(self.worker_id2schema_hmap.keys())
            current_worker_schema = self.worker_id2schema_hmap.get(worker_id, ValueError(f'worker {worker_id} is not a valid id, please choose one of : {worker_ids}'))

        if isinstance(current_worker_schema, ValueError):
            raise HTTPException(
                status_code=500,
                detail=str(current_worker_schema)
            )

        if current_worker_schema is not None and not reset_worker:
            raise HTTPException(
                status_code=500, 
                detail=f'WARNING : {worker_id} hold a schema : {current_worker_schema.model_dump()}, please force reset_worker to true!'
            )
        
        socket:azmq.Socket = self.ctx.socket(zmq.DEALER)
        socket.connect(self.outer_address)

        incoming_req.llama_model_name = path.join(self.path2models, incoming_req.llama_model_name)
        
        fmt_worker_id = f'{worker_id:05d}'
        await socket.send_multipart([b'', b'LAUNCH_LLM', fmt_worker_id.encode()], flags=zmq.SNDMORE)
        await socket.send_pyobj(incoming_req)

        task_response:Optional[TaskResponse]=None 
        while True:
            try:
                socket_event_type = await socket.poll(timeout=1000)
                if socket_event_type != zmq.POLLIN:
                    continue

                _, _, encoded_task_response = await socket.recv_multipart()
                task_response = pickle.loads(encoded_task_response)
                break 
            except asyncio.CancelledError:
                break 
            except Exception as e:
                logger.error(e)
                break 
        
        socket.close(linger=0)
        if task_response is None:
            raise HTTPException(status_code=500, detail=f'can not launch the model : {incoming_req.llama_model_name}')
        
        if not task_response.task_status:
            raise HTTPException(status_code=500, detail=f'can not launch the model : {incoming_req.llama_model_name} => {task_response.error_message}')
        
        async with self.shared_mutex:
            incoming_req.llama_model_name = path.split(incoming_req.llama_model_name)[-1]
            self.worker_id2schema_hmap[worker_id] = incoming_req
            
        return LaunchModelResSchema(
            llama_model_name=incoming_req.llama_model_name,
            status=True,
            message=f'{incoming_req.llama_model_name} was lauched...'
        )
    
    async def _background_generator(self, worker_ids:List[int], llama_model_name:str ,incoming_req:LlmGenerationReqSchema):
        
        socket:azmq.Socket = self.ctx.socket(zmq.DEALER)
        socket.connect(self.outer_address)

        async with self.shared_mutex:
            fmt_worker_id = f'{worker_ids[self.target_worker_id % len(worker_ids)]:05d}'
            logger.info(f'target worker for llm inference {llama_model_name} => {fmt_worker_id}')
            await socket.send_multipart([b'', b'MAKE_PREDICTION', fmt_worker_id.encode()], flags=zmq.SNDMORE)
            await socket.send_pyobj(incoming_req)
            self.target_worker_id = (self.target_worker_id + 1) % len(self.worker_id2schema_hmap)

        task_response:Optional[TaskResponse]=None 
        while True:
            try:
                socket_event_type = await socket.poll(timeout=1000)
                if socket_event_type != zmq.POLLIN:
                    continue

                _, worker_status, encoded_task_response = await socket.recv_multipart()
                task_response = pickle.loads(encoded_task_response)
                if not task_response.task_status:
                    socket.close(linger=0)
                    raise HTTPException(status_code=500, detail=f'llm generation error => {task_response.error_message}')
                
                stream_data = task_response.task_content
                print(stream_data['choices'][0]['text'], end='', flush=True)
                yield json.dumps({
                    'stream_data': stream_data,
                    'more_stream': worker_status != WorkerStatus.DONE  
                }) + '\n'

                if worker_status == WorkerStatus.DONE:
                    break 
            except asyncio.CancelledError:
                break 
            except Exception as e:
                logger.error(e)
                break 
        
        yield json.dumps({
            'stream_data': None,
            'more_stream': False   
        }) + '\n'

        socket.close(linger=0)
        
    async def generate(self, llama_model_name:str, incoming_req:LlmGenerationReqSchema):
        worker_ids:List[int] = []
        async with self.shared_mutex:
            for worker_id, schema_val in self.worker_id2schema_hmap.items():
                if schema_val is None:
                    continue
                if llama_model_name != schema_val.llama_model_name:
                    continue
                worker_ids.append(worker_id)
        
        if len(worker_ids) == 0:
            raise HTTPException(
                status_code=500,
                detail=f'{llama_model_name} was not found, there is no worker holding this model')

        return StreamingResponse(
            status_code=200,
            content=self._background_generator(worker_ids, llama_model_name, incoming_req),
            media_type='application/x-ndjson'
        )
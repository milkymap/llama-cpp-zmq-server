import zmq 
import zmq.asyncio as azmq 

import multiprocessing as mp 

import signal 
import asyncio

from typing import Generator

from fastapi import FastAPI

import uvicorn

from os import path 

from ..log import logger 
from ..settings.server import AppConfig
from .routers.llama import LlamaRouter

from ..backend import LLMBroker
from ..implementations import MixtralEngine

class APIServer:
    def __init__(self, app_config:AppConfig):
        assert path.isdir(app_config.path2models), f"{app_config.path2models} must be a valid dir"

        self.cfg = app_config
        self.ctx:azmq.Context 
        self.app = FastAPI(**app_config.model_dump(), lifespan=self.lifespan_handler()) 
        self.config = uvicorn.Config(app=self.app, host=app_config.host, port=app_config.port)
        self.server = uvicorn.Server(config=self.config)
        
    def add_routers(self, app:FastAPI):
        llama_router = LlamaRouter(
            path2models=self.cfg.path2models, 
            ctx=self.ctx, 
            outer_address=self.cfg.outer_address, 
            nb_llm_workers=self.cfg.nb_llm_workers
        )
        app.include_router(router=llama_router.router, prefix="/models")

    def lifespan_handler(self) -> Generator:
        async def handler(app:FastAPI):
            self.ctx = azmq.Context()
            self.add_routers(app)
            logger.info('llamazmq resources initialized')
            yield 
            self.ctx.term()
            logger.info('llamazmq resources released')
        
        return handler

    def listen(self):
        def server_starter():
            async def _listen():
                await self.server.serve()
            asyncio.run(main=_listen())

        server_process = mp.Process(target=server_starter, args=[])
        server_process.start()

        llm_broker = LLMBroker(strategy_cls=MixtralEngine)
        llm_broker.deploy(
            outer_address=self.cfg.outer_address, 
            inner_address=self.cfg.inner_address, 
            nb_workers=self.cfg.nb_llm_workers
        )

        if not server_process.is_alive():
            exit(0)

        server_process.terminate()
        server_process.join()

if __name__ == '__main__':
    server = APIServer(app_config=AppConfig())
    server.listen()
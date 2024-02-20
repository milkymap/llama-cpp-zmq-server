
from pydantic_settings import BaseSettings

class AppConfig(BaseSettings):
    host:str='0.0.0.0'
    port:int=8000 
    title:str='llama-cpp-zmq-server' 
    version:str='0.0.1' 
    description:str="The 'llama-cpp-zmq-server' module is designed to distribute the capabilities of the llama-cpp-python library through a distributed system architecture, leveraging FastAPI and ZeroMQ. This system is architected with a broker-worker model, enabling it to host multiple LLM (Large Language Models) in the background. By integrating ZeroMQ, it facilitates efficient message passing and task distribution among various components, ensuring scalable and responsive performance. FastAPI is utilized to provide a modern, fast (high-performance) web framework for building APIs, making the system accessible and easy to interface with. Additionally, this system supports streaming of multiple models simultaneously, enhancing its capability to handle diverse and concurrent processing tasks. This makes the llama-cpp-zmq-server an ideal solution for distributed, real-time processing of large language models in various applications." 
    docs_url:str='/'   
    inner_address:str='ipc:///tmp/broker2llms.ipc'
    outer_address:str='ipc:///tmp/client2broker.ipc'
    nb_llm_workers:int=2
    path2models:str 
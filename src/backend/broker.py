import zmq 
import signal 
import multiprocessing as mp 

import pickle 

from sys import exit
from queue import Empty as EmptyQueue 
from contextlib import suppress

from typing import List, Tuple, Dict, Any, Callable, Type, Optional, Generator
from ..strategy import ABCStrategy
from ..log import logger 

from ..schema.task import TaskResponse
from ..schema.worker import WorkerStatus
from ..server.routers.schema import LaunchModelReqSchema, LlmGenerationReqSchema

class ZMQBroker:
    """
    This class implements a distributed task processing system using ZeroMQ. It allows submitting tasks
    to worker processes, which apply a given strategy to each task.

    Attributes:
        strategy_cls (Type[ABCStrategy]): The class of the strategy to be applied to each task.
        strategy_args (List[Any]): Arguments to be passed to the strategy class upon instantiation.
        strategy_kwargs (Dict[str, Any]): Keyword arguments to be passed to the strategy class.
    """

    def __init__(self, strategy_cls:Type[ABCStrategy]):
        """
        Initializes the ZMQBroker instance.

        Args:
            strategy_cls (Type[ABCStrategy]): The class of the strategy to be applied to each task.
            strategy_args (List[Any]): Arguments to be passed to the strategy class upon instantiation.
            strategy_kwargs (Dict[str, Any]): Keyword arguments to be passed to the strategy class.
        """

        self.strategy_cls = strategy_cls
        
    def _apply_strategy(self, encoded_task:Any, strategy:ABCStrategy) -> Tuple[bool, Any, Optional[str]]:
        """
        Applies the given strategy to an encoded task.

        Args:
            encoded_task (bytes): The task to be processed, encoded in bytes.
            strategy (ABCStrategy): The strategy instance to apply to the task.

        Returns:
            TaskResponse: The response from processing the task, encapsulating the result or an error message.
        """
        try:
            task_content = strategy(encoded_task=encoded_task)
            return True, task_content, None 
        except Exception as e:
            logger.warning(e)
        return False, None, str(e) 
    
    def _worker(self, address:str, worker_id:str) -> int:
        """
        The worker process method, responsible for processing tasks using the defined strategy.

        Args:
            address (str): The address to connect to for receiving and sending task data.

        Returns:
            int: Exit code indicating the status upon completion.
        """
        signal.signal(
            signal.SIGTERM,
            lambda signal_n, frame: signal.raise_signal(signal.SIGINT)
        )
        
        ctx = zmq.Context()
        try:
            dealer_socket:zmq.Socket = ctx.socket(zmq.DEALER)
            dealer_socket.setsockopt_string(zmq.IDENTITY, worker_id)
            dealer_socket.connect(address)
        except Exception as e:
            logger.warning(e)
            ctx.term()
            exit(1)

        poller = zmq.Poller()
        poller.register(dealer_socket, zmq.POLLIN)

        dealer_socket.send_multipart([b'', WorkerStatus.FREE, b'', b''])
        strategy:Optional[ABCStrategy] = None 
        logger.debug(f'{worker_id} is running')
        while True:
            try:
                socket_pollstatus_dict:Dict[zmq.Socket, int] = dict(poller.poll(timeout=100))
                dealer_socket_status = socket_pollstatus_dict.get(dealer_socket, None)
                if dealer_socket_status is None:
                    continue

                if dealer_socket_status != zmq.POLLIN:
                    continue

                _, task_id, task_type, encoded_task = dealer_socket.recv_multipart()
                
                if task_type not in [b'LAUNCH_LLM', b'MAKE_PREDICTION']:
                    task_response = TaskResponse(task_content=None, task_status=False ,error_message=f'{task_type} is not a valid task')    
                    dealer_socket.send_multipart([b'', WorkerStatus.DONE, task_id], flags=zmq.SNDMORE)
                    dealer_socket.send_pyobj(task_response)
                    dealer_socket.send_multipart([b'', WorkerStatus.FREE, b'', b''])
                    continue
                
                if task_type == b'LAUNCH_LLM':
                    try:
                        strategy_args:LaunchModelReqSchema = pickle.loads(encoded_task)
                        logger.debug(f'{worker_id} will lunch the llm {strategy_args.llama_model_name}')

                        strategy = self.strategy_cls(**strategy_args.model_dump())
                        task_response = TaskResponse(task_content=None, task_status=True, error_message=None)
                    except Exception as e:
                        error_message = str(e)
                        logger.error(error_message)
                        task_response = TaskResponse(task_content=None, task_status=False ,error_message=error_message)
                    
                    dealer_socket.send_multipart([b'', WorkerStatus.DONE, task_id], flags=zmq.SNDMORE)
                    dealer_socket.send_pyobj(task_response)
                    dealer_socket.send_multipart([b'', WorkerStatus.FREE, b'', b''])
                    continue
                
                if strategy is None:
                    task_response = TaskResponse(task_content=None, task_status=False ,error_message='no llm was launched, please launched llm')
                    dealer_socket.send_multipart([b'', WorkerStatus.DONE, task_id], flags=zmq.SNDMORE)
                    dealer_socket.send_pyobj(task_response)
                    dealer_socket.send_multipart([b'', WorkerStatus.FREE, b'', b''])
                    continue

                # MAKE_PREDICTION 
                plain_task:LlmGenerationReqSchema = pickle.loads(encoded_task)
                task_status, task_content, task_error_message = self._apply_strategy(plain_task, strategy)
                if not task_status:
                    task_response = TaskResponse(
                        task_content=task_content,
                        task_status=task_status,
                        error_message=task_error_message
                    )
                    dealer_socket.send_multipart([b'', WorkerStatus.DONE, task_id], flags=zmq.SNDMORE)
                    dealer_socket.send_pyobj(task_response)

                    dealer_socket.send_multipart([b'', WorkerStatus.FREE, b'', b''])
                    continue
                
                for chunk in task_content:
                    chunk_task_response = TaskResponse(
                        task_content=chunk,
                        task_status=task_status,
                        error_message=task_error_message
                    )
                    dealer_socket.send_multipart([b'', WorkerStatus.MORE, task_id], flags=zmq.SNDMORE)
                    dealer_socket.send_pyobj(chunk_task_response)
                dealer_socket.send_multipart([b'', WorkerStatus.DONE, task_id, b''])
                dealer_socket.send_multipart([b'', WorkerStatus.FREE, b'', b''])
            except KeyboardInterrupt:
                break 
            except Exception as e:
                logger.error(e)
                break 

        with suppress(Exception):
            del strategy

        poller.unregister(dealer_socket)
        dealer_socket.close(linger=0)
        ctx.term()
        logger.debug(f'{worker_id} terminated')
        exit(0)
    
    def deploy(self, outer_address:str, inner_address:str, nb_workers:int) -> None:
        """
        Deploy parallel worker processes and yields their responses.

        Args:
            address (str): The address to bind to for communication with worker processes.
            nb_workers (int): The number of worker processes to spawn.

        """

        ctx = zmq.Context()
        try:
            inner_router_socket:zmq.Socket = ctx.socket(zmq.ROUTER)
            inner_router_socket.bind(inner_address)
        except Exception as e:
            logger.warning(e)
            ctx.term()
            exit(1)

        try:
            outer_router_socket:zmq.Socket = ctx.socket(zmq.ROUTER)
            outer_router_socket.bind(outer_address)
        except Exception as e:
            logger.warning(e)
            inner_router_socket.close(linger=0)
            ctx.term()
            exit(1)

        poller = zmq.Poller()
        poller.register(inner_router_socket, zmq.POLLIN)
        poller.register(outer_router_socket, zmq.POLLIN)

        logger.debug(f'core process will launch {nb_workers} processes')

        background_processes:List[mp.Process] = []
        for idx in range(nb_workers):
            worker_id = f'{idx:05d}'
            process_ = mp.Process(target=self._worker, args=[inner_address, worker_id])
            background_processes.append(process_)
            background_processes[-1].start()

        accumulator:List[bytes] = []
        sigint_was_received = False 
        while True: 
            try:
                if all([ process_.exitcode is not None for process_ in background_processes ]):
                    logger.warning('all background processes stopped')
                    break 
                
                socket_pollstatus_dict:Dict[zmq.Socket, int] = dict(poller.poll(timeout=100))
                inner_router_socket_status = socket_pollstatus_dict.get(inner_router_socket, None)
                if inner_router_socket_status == zmq.POLLIN:
                    worker_id, _, worker_status, source_client_id, worker_task_response = inner_router_socket.recv_multipart()
                    if worker_status == WorkerStatus.FREE:
                        accumulator.append(worker_id)
                    
                    if worker_status in [WorkerStatus.DONE, WorkerStatus.MORE]:
                        outer_router_socket.send_multipart([source_client_id, b'', worker_status, worker_task_response])
                        
                if len(accumulator) == 0:                    
                    continue
                
                outer_router_socket_status = socket_pollstatus_dict.get(outer_router_socket, None)
                if outer_router_socket_status != zmq.POLLIN:
                    continue
                
                client_id, _, client_task_type, target_worker_id, client_message = outer_router_socket.recv_multipart()
                print(client_id, client_task_type, client_message)
                #popped_worker_id = accumulator.pop(0)  # FIFO
                inner_router_socket.send_multipart([target_worker_id, b'', client_id, client_task_type, client_message])
            except KeyboardInterrupt:
                sigint_was_received = True 
                break 
            except Exception as e:
                logger.error(e)
                break 
        
        logger.debug('core process is waiting background_processes to terminate')
        for process_ in background_processes:
            if not process_.is_alive():
                continue
            
            if not sigint_was_received:
                process_.terminate()

            process_.join()

        poller.unregister(outer_router_socket)
        poller.unregister(inner_router_socket)
        inner_router_socket.close(linger=0)
        outer_router_socket.close(linger=0)
        ctx.term()

        logger.debug('core process released all resouces')


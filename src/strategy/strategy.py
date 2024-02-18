
from typing import Any 

from abc import ABC, abstractmethod

class ABCStrategy(ABC):
    @abstractmethod
    def process_task(self, encoded_task:Any) -> Any:
        pass  
    
    def __call__(self, encoded_task:Any) -> Any:
        return self.process_task(encoded_task)
     
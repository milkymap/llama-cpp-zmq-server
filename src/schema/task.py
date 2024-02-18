

from typing import Any, Optional 
from pydantic import BaseModel

class TaskResponse(BaseModel):
    task_status:bool=True 
    task_content:Any=None
    error_message:Optional[str]=None   
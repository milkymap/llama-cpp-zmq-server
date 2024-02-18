

from enum import Enum 

class WorkerStatus(bytes, Enum):
    FREE:bytes=b'FREE'
    DONE:bytes=b'DONE'
    MORE:bytes=b'MORE'
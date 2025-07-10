from abc import ABC, abstractmethod
from pathlib import Path
import threading

from .image import FunctionImage
from ..models.function import FunctionModelInferer

class AbstractFunction(ABC):

    @abstractmethod
    def __init__(self, manager: "FunctionManager", name: str, function_id: str, instance_id: str, image: FunctionImage, model: FunctionModelInferer):
        self.manager: "FunctionManager" = None

        self.created_at: int = None
        self.last_worked_at: int = None

        self.work_lock: threading._RLock = None

        self.name: str = None
        self.function_id: str = None
        self.instance_id: str = None
        self.image: FunctionImage = None

        self.model: FunctionModelInferer = None

        self.cpu: float = None
        self.ram: int = None

        self.is_cold: bool = None

    @property
    @abstractmethod
    def was_recently_active(self) -> bool:
        pass

    @property
    @abstractmethod
    def is_active(self) -> bool:
        pass
    
    @property
    @abstractmethod
    def uptime(self) -> int:
        pass
    
    @property
    @abstractmethod
    def time_since_last_work(self) -> int:
        pass

    @staticmethod
    @abstractmethod
    def create_new(manager: "FunctionManager", function_id: str, image: FunctionImage, model: Path) -> "AbstractFunction":
        pass
    
    @abstractmethod
    def coldstart(self):
        pass

    @abstractmethod
    def timeout(self) -> bytes:
        pass

    @abstractmethod
    def lock(self) -> bool:
        pass
    
    @abstractmethod
    def unlock(self):
        pass

    @abstractmethod
    def work(self, body_size: int, result_bytes: int) -> tuple[bytes, float]:
        pass

    @abstractmethod
    def __eq__(self, value):
        pass
    
    @abstractmethod
    def __hash__(self):
        pass
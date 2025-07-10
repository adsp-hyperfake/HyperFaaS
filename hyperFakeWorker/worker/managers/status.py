import threading
from queue import Queue
from weakref import WeakSet

from ..api.controller.controller_pb2 import StatusUpdate

class StatusManager():

    def __init__(self, update_buffer_size: int):
        self.status_lock = threading.RLock()
        self.status_queues: WeakSet[Queue] = WeakSet()
        
        self.update_buffer_size = update_buffer_size

    def send_status_update(self, update: StatusUpdate):
        if not isinstance(update, StatusUpdate):
            raise TypeError("The sent update must be an actual status update!")
        with self.status_lock:
            for q in self.status_queues:
                q.put(update)
            
    def get_status_updates(self):
        updates_queue: Queue[StatusUpdate] = Queue(maxsize=self.update_buffer_size)
        with self.status_lock:
            self.status_queues.add(updates_queue)
        
        while True:
            update = updates_queue.get()
            yield update
            updates_queue.task_done()
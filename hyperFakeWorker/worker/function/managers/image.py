import threading
from .. import FunctionIdStr
from ..image import FunctionImage
from ...kvstore.client import KVStoreClient


class ImageManager():

    def __init__(self, db_address: str):
        self.image_lock = threading.Lock()
        self.images: dict[FunctionIdStr, FunctionImage] = {}

        self.kvs_client = KVStoreClient(db_address)

    def get_image(self, function_id: FunctionIdStr):
        if self.images.get(function_id) is None:
            with self.image_lock:
                self.images[function_id] = self.kvs_client.get_image(function_id)
        return self.images[function_id]
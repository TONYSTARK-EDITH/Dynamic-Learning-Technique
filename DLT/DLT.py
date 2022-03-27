import numpy as np
from abc import *
import joblib
from .Exceptions import *
from .utils import *


class DLT:
    def __init__(self, model_object: object = None, model_file: str = None, batch: int = BATCH) -> None:
        if model_object is None and model_file is None:
            raise NoArgumentException("Provide either the ml _model or the file_path of the _model")
        elif model_file is not None:
            self.model_file = model_file
            self._model = self.render_model_file()
        else:
            self._model = model_object
        self._batch = batch
        self._sub_hash_map = dict()

    def render_model_file(self) -> object:
        return joblib.load(self.model_file)

    @abstractmethod
    def mini_batches_training(self, src: np.array, targets: np.array, shuffle=False):
        assert src.shape[0] == targets.shape[0]
        batch_size = self.batch_size
        if shuffle:
            indices = np.arange(src.shape[0])
            np.random.shuffle(indices)
        for idx in range(0, src.shape[0] - batch_size + 1, batch_size):
            if shuffle:
                excerpt = indices[idx:idx + batch_size]
            else:
                excerpt = slice(idx, idx + batch_size)

            yield src[excerpt], targets[excerpt]

    @property
    def batch_size(self) -> int:
        return self._batch

    @property
    def hash_map(self) -> dict:
        return self._sub_hash_map

    @property
    def model(self) -> object:
        return self._model

    @batch_size.deleter
    def batch_size(self):
        del self._batch
        self._batch = BATCH

    @hash_map.deleter
    def hash_map(self):
        del self._sub_hash_map

    @model.deleter
    def model(self):
        del self._model

    @batch_size.setter
    def batch_size(self, size):
        self._batch = size

    def split_models_into_batches(self) -> None:
        pass

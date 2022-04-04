import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from .splitter import *
from .utils import Utils


class DLT:
    def __init__(self, x, y, model_object: object = None, model_file: str = None,
                 batch: int = Utils.BATCH.value,
                 shuffle=False) -> None:
        if model_object is None and model_file is None:
            raise NoArgumentException("Provide either the ml _model or the file_path of the _model")
        elif model_file is not None:
            self._model_file = model_file
            self._render_model_file()
        else:
            self._model = model_object
        self._batch_size = batch
        self._sub_hash_map = dict()
        self._shuffle = shuffle
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(x, y,
                                                                                    train_size=Utils.TRAIN_SIZE.value,
                                                                                    random_state=Utils.RANDOM_STATE.value)
        self._accuracy = 0
        self._split_models_into_batches()

    def _render_model_file(self):
        self._model = joblib.load(self._model_file)

    def _mini_batches_training(self):
        src = self.x_train
        targets = self.y_train
        assert src.shape[0] == targets.shape[0]
        batch_size = self.batch_size
        if self.shuffle:
            indices = np.arange(src.shape[0])
            np.random.shuffle(indices)
            for idx in range(0, src.shape[0] - batch_size + 1, batch_size):
                excerpt = indices[idx: idx + batch_size]
                yield src[excerpt], targets[excerpt]
        else:
            for idx in range(0, src.shape[0] - batch_size + 1, batch_size):
                excerpt = slice(idx, idx + batch_size)
                yield src[excerpt], targets[excerpt]

    def _split_models_into_batches(self) -> None:
        obj = Splitter(self.model, self.batch_size)
        self._sub_hash_map = dict(obj.batch_list)
        for i in range(self.batch_size):
            for x, y in self._mini_batches_training():
                self.hash_map[i].fit(x, y)
            if obj.model_class == Utils.REGRESSOR:
                self.hash_map[i] = (
                    self.hash_map[i], r2_score(y_true=self.y_test, y_pred=self.hash_map[i].predict(self.x_test))
                )
            else:
                self.hash_map[i] = (
                    self.hash_map[i], accuracy_score(y_true=self.y_test, y_pred=self.hash_map[i].predict(self.x_test))
                )
        print(self.hash_map)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def hash_map(self) -> dict:
        return self._sub_hash_map

    @property
    def model(self) -> object:
        return self._model

    @property
    def shuffle(self):
        return self._shuffle

    @property
    def model_file(self):
        return self._model_file

    @property
    def x_train(self):
        return self._X_train

    @property
    def x_test(self):
        return self._X_test

    @property
    def y_train(self):
        return self._y_train

    @property
    def y_test(self):
        return self._y_test

    @property
    def accuracy(self):
        return self._accuracy

    @batch_size.setter
    def batch_size(self, size):
        self._batch_size = size
        self._split_models_into_batches()

    @batch_size.deleter
    def batch_size(self):
        del self._batch_size
        self._batch_size = Utils.BATCH

    @hash_map.deleter
    def hash_map(self):
        del self._sub_hash_map

    @model.deleter
    def model(self):
        del self._model

    @model_file.deleter
    def model_file(self):
        del self._model_file

    @shuffle.deleter
    def shuffle(self):
        del self._shuffle

    @x_test.deleter
    def x_test(self):
        del self._X_test

    @x_train.deleter
    def x_train(self):
        del self._X_train

    @y_test.deleter
    def y_test(self):
        del self._y_test

    @y_train.deleter
    def y_train(self):
        del self._y_train

    @accuracy.deleter
    def accuracy(self):
        del self._accuracy
        self._accuracy = 0

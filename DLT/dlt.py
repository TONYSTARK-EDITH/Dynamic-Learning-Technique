import joblib
import numpy as np
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split

from .splitter import *

LOGGER = logging.getLogger("log")
LOGGER.setLevel(logging.DEBUG)


class DLT:
    def __init__(self, x, y, model_object: object = None, model_file: str = None,
                 batch: int = Utils.BATCH.value,
                 shuffle: bool = False, verbose: bool = False, is_trained: bool = True) -> None:
        """
        Initialize the DLT object with multiple arguments in it
        :param x: Any object preferable of numpy array
        :param y: Any object preferable of numpy array
        :param model_object: Object Refers to the object of the machine learning object
        :param model_file: String which contains the path to the serialized ML object
        :param batch: Integer which contains the BATCH size of the model - Default would be 10
        :param shuffle: Boolean which contains whether to shuffle the values while training/validations
        :param verbose: Boolean which allows to show the process flow when it is true
        :param is_trained Boolean which shows whether the given model is fitted or not
        """
        self._verbose = verbose
        fh = logging.FileHandler("log", )
        fh.setLevel(logging.DEBUG)
        console = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(levelname)s] - [pid %(process)d] - %(asctime)s : %(message)s in %(module)2s.%("
            "funcName)2s()s at line number : %(lineno)d "
        )
        fh.setFormatter(formatter)
        console.setFormatter(
            logging.Formatter("[%(levelname)s] :\t %(message)s \tin the function %(funcName)2s() at line %(lineno)d"))
        if self.verbose:
            console.setLevel(logging.DEBUG)
        else:
            console.setLevel(logging.ERROR)
        LOGGER.addHandler(fh)
        LOGGER.addHandler(console)
        if model_object is None and model_file is None:
            LOGGER.error("Argument error -- Provide either the ml model or the file_path of the model")
            raise NoArgumentException("Provide either the ml _model or the file_path of the _model")
        elif model_file is not None:
            self._model_file = model_file
            self._render_model_file()
        else:
            self._model = model_object
        self._batch_size = batch
        self._sub_hash_map = dict()
        self._shuffle = shuffle
        self._refined_model = copy.deepcopy(self.model)
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(
            x, y, train_size=Utils.TRAIN_SIZE.value,
            random_state=Utils.RANDOM_STATE.value
        )

        self._accuracy = 0
        self._is_trained = is_trained
        self._split_models_into_batches()
        self._base_acc = 0
        if self.is_trained:
            self._base_acc = r2_score(y_true=self.y_test, y_pred=self.model.predict(
                self.x_test)) if self.model.__class__ in Utils.VALID_REGRESSORS.value else accuracy_score(
                y_true=self.y_test, y_pred=self.model.predict(self.x_test))
            LOGGER.info("Base accuracy has been calculated")

    def _render_model_file(self):
        LOGGER.info("Ml file deserialization process started")
        self._model = joblib.load(self._model_file)
        LOGGER.info("Ml file has been deserialized from the given path")

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
        obj = Splitter(self.model, self.batch_size, LOGGER)
        LOGGER.info("Batch models has been successfully created")
        self._sub_hash_map = dict(obj.batch_list)
        LOGGER.info("Batch models has been stored in the hash map")
        for i in range(self.batch_size):
            LOGGER.info(f"Batch - {i + 1} training has been started")
            for x, y in self._mini_batches_training():
                self.hash_map[i].fit(x, y)
            if obj.model_class == Utils.REGRESSOR:
                self.hash_map[i] = (
                    self.hash_map[i], r2_score(y_true=self.y_test, y_pred=self.hash_map[i].predict(self.x_test))
                )  # Accuracy for each batch
            else:
                self.hash_map[i] = (
                    self.hash_map[i], accuracy_score(y_true=self.y_test, y_pred=self.hash_map[i].predict(self.x_test))
                )  # Accuracy for each batch
            LOGGER.info(f"Batch - {i + 1} accuracy has been calculated")
        self._cal_accuracy()

    def _cal_accuracy(self):
        for batch_model, batch_acc in self.hash_map.values():
            if batch_acc > self.accuracy:
                self._accuracy = batch_acc
                self._refined_model = batch_model

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

    @property
    def verbose(self):
        return self._verbose

    @property
    def refined_model(self):
        return self._refined_model

    @property
    def is_trained(self):
        return self._is_trained

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

    @verbose.deleter
    def verbose(self):
        del self._verbose

    @refined_model.deleter
    def refined_model(self):
        del self._refined_model

    @is_trained.deleter
    def is_trained(self):
        del self._is_trained

    def __str__(self):
        if self.is_trained:
            return f"{'==' * 4}\t {self.model.__class__.__name__} \t{'==' * 4}\n---- \t Base Accuracy : " \
                   f"{self._base_acc} \t ----\n---- \t Refined Accuracy : {self.accuracy} \t ----\n "
        else:
            return f"{'++' * 4}\t {self.model.__class__.__name__} \t{'++' * 4}\n---- \t Accuracy : " \
                   f"{self.accuracy} \t ----\n "

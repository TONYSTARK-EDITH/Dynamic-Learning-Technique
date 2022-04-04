import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from .utils import Utils
from .Exceptions import *
import copy


class Splitter:
    VALID_CLASSIFIERS = (RandomForestClassifier, DecisionTreeClassifier, SVC)
    VALID_REGRESSORS = (RandomForestRegressor, DecisionTreeRegressor, LinearRegression, LogisticRegression, SVR)
    VALID_MODEL = np.array((VALID_REGRESSORS + VALID_CLASSIFIERS))

    def __init__(self, model: object, batch_size: int):
        self._model = model
        self._batch_size = batch_size
        self._batch_list = tuple()
        self._base_model = None
        self._model_class = Utils.REGRESSOR
        self._find_the_base_class()
        self._bootstrap()

    def _batch_model(self, args):
        args["__init__"] = self.model.__init__
        batch = type("Batch", (self.model.__class__,), args)
        self._base_model = batch()

    def _find_the_base_class(self) -> None:
        for i in self.VALID_MODEL:
            if self.model.__class__ is i:
                self._batch_model(vars(self.model))
                self._model_class = Utils.CLASSIFIER if \
                    self.model.__class__ in [j.__class__ for j in self.VALID_CLASSIFIERS] else Utils.REGRESSOR
                return
        raise InvalidMachineLearningModel(
            f"{self.model.__class__} is not a valid ML model for DLT.\nValid models are {self.VALID_MODEL}"
        )

    def _bootstrap(self) -> None:
        self._batch_list = ((batches, copy.deepcopy(self.base_model)) for batches in range(self.batch_size))

    @property
    def model(self):
        return self._model

    @model.deleter
    def model(self):
        del self._model

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.deleter
    def batch_size(self):
        del self._batch_size

    @property
    def base_model(self):
        return self._base_model

    @base_model.deleter
    def base_model(self):
        del self._base_model

    @property
    def batch_list(self):
        return self._batch_list

    @batch_list.deleter
    def batch_list(self):
        del self._batch_list

    @property
    def model_class(self):
        return self._model_class

    @model_class.deleter
    def model_class(self):
        del self._model_class
        self._model_class = Utils.REGRESSOR

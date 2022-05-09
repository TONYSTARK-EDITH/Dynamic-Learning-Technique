import asyncio
import copy
import logging

from .Exceptions import *
from .utils import Utils


class Splitter(object):

    def __init__(self, model: object, batch_size: int, logger: logging):
        self.logger = logger
        self._model = model
        self._batch_size = batch_size
        self._batch_list = tuple()
        self._base_model = None
        self._model_class = Utils.REGRESSOR

    async def run(self):
        await self._find_the_base_class()
        await self._bootstrap()

    async def _batch_model(self, args):
        args["__init__"] = self.model.__init__  # Adding the __init__ of the base_model
        batch = type("Batch", (self.model.__class__,), args)  # Creating a super class with extra methods and attributes
        self.logger.info("Batch model has been created")
        self._base_model = batch()  # Assigning the batch -- function to base_model

    async def _find_the_base_class(self) -> None:
        for i in Utils.VALID_MODEL.value:
            if self.model.__class__ is i:
                await self._batch_model(vars(self.model))
                self.logger.info("Retrieving the attributes and methods from the given model")
                self._model_class = Utils.CLASSIFIER if \
                    self.model.__class__ in [j.__class__ for j in Utils.VALID_CLASSIFIERS.value] else Utils.REGRESSOR
                self.logger.info("Model base class has been identified")
                return
        self.logger.error("There is no valid ml model for DLT")
        raise InvalidMachineLearningModel(
            f"{self.model.__class__} is not a valid ML model for DLT.\nValid models are {Utils.VALID_MODEL.value}"
        )

    async def _bootstrap(self) -> None:
        self._batch_list = ((batches, copy.deepcopy(self.base_model)) for batches in
                            range(self.batch_size))  # Deep copying in order avoid overwriting in all the batches
        self.logger.info("Batch list has been generated with the refined batch models")

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

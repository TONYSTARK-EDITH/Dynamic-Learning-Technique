import unittest
from sklearn.linear_model import Ridge
from DLT import *


class A:
    pass


class MyTestCase(unittest.TestCase):
    SAMPLE_X = [i for i in range(10)]
    SAMPLE_Y = [i ** 2 for i in range(10)]

    def test_exception(self):
        with self.assertRaises(NoArgumentException):
            DLT(self.SAMPLE_X, self.SAMPLE_Y)

        with self.assertRaises(InvalidMachineLearningModel):
            DLT(self.SAMPLE_X, self.SAMPLE_Y, A(), is_trained=False)

        with self.assertRaises(InvalidDatasetProvided):
            DLT([], [], Ridge())

        with self.assertRaises(BatchCountGreaterThanBatchSize):
            DLT(self.SAMPLE_X, self.SAMPLE_Y, Ridge(), batch=10, count_per_batch=11)


if __name__ == '__main__':
    unittest.main()

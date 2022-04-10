import unittest

from sklearn.linear_model import Ridge

from DLT import *


class MyTestCase(unittest.TestCase):
    SAMPLE_X = [i for i in range(100)]
    SAMPLE_Y = [i ** 2 for i in range(100)]

    def test_exception(self):
        with self.assertRaises(NoArgumentException):
            DLT(self.SAMPLE_X, self.SAMPLE_Y)

        with self.assertRaises(InvalidMachineLearningModel):
            DLT(self.SAMPLE_X, self.SAMPLE_Y, Ridge())

        with self.assertRaises(InvalidDatasetProvided):
            DLT([], [], Ridge())


if __name__ == '__main__':
    unittest.main()

import unittest
from DLT import *
from sklearn.ensemble import RandomForestRegressor


class MyTestCase(unittest.TestCase):
    def test_split(self):
        DLT(model_object=RandomForestRegressor())  # add assertion here


if __name__ == '__main__':
    unittest.main()

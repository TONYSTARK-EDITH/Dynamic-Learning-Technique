from DLT import *
from sklearn.datasets import load_iris

if __name__ == "__main__":
    x, y = load_iris(return_X_y=True)

    for i in Utils.VALID_MODEL.value:
        print(DLT(x, y, i(), is_trained=False, verbose=True))
        break

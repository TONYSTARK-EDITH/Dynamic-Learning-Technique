from sklearn.datasets import load_iris

from DLT import *

if __name__ == "__main__":
    x, y = load_iris(return_X_y=True)
    for i in Utils.VALID_MODEL.value:
        refined = DLT(x, y, i(), is_trained=False, verbose=True)
        print(refined.refined_model.predict([[1, 2, 3, 4]]))
        print(refined)

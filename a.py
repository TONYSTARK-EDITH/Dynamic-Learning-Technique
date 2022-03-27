from DLT import *
from sklearn.ensemble import RandomForestRegressor

if __name__ == "__main__":
    DLT(model_object=RandomForestRegressor()).split_models_into_batches()

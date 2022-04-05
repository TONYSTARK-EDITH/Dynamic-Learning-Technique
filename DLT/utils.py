from enum import Enum
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class Utils(Enum):
    BATCH = 10
    RANDOM_STATE = 100
    TRAIN_SIZE = 0.9
    REGRESSOR = "Regression"
    CLASSIFIER = "Classification"
    VALID_CLASSIFIERS = (RandomForestClassifier, DecisionTreeClassifier, SVC)
    VALID_REGRESSORS = (RandomForestRegressor, DecisionTreeRegressor, LinearRegression, LogisticRegression, SVR)
    VALID_MODEL = VALID_REGRESSORS + VALID_CLASSIFIERS

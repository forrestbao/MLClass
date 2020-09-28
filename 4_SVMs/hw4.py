import numpy
import sklearn, sklearn.datasets, sklearn.utils, sklearn.model_selection

def study_C_fix_split(C_range): 


    # load the data
    data = sklearn.datasets.load_breast_cancer()
    X, y = data["data"], data["target"]

    # prepare the training and testing data
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=0)

    # your code here 
    

    return best_C

def study_C_cross_validate(C_range):
    # load the data
    data = sklearn.datasets.load_breast_cancer()
    X, y = data["data"], data["target"]

    # shuffle the data
    X, y = sklearn.utils.shuffle(X, y)

    # your code here

    return best_C 


def study_C_gridCV(C_range):
    # load the data
    data = sklearn.datasets.load_breast_cancer()
    X, y = data["data"], data["target"]

    # shuffle the data
    X, y = sklearn.utils.shuffle(X, y)

    # your code here

    return best_C 


def study_C_and_sigma_gridCV(C_range, sigma_range):
    # load the data
    data = sklearn.datasets.load_breast_cancer()
    X, y = data["data"], data["target"]

    # shuffle the data
    X, y = sklearn.utils.shuffle(X, y)

    # your code here

    return best_C 
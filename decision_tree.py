import numpy as np
from scipy import io
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import DT


def shuffle_and_resize(data):

    labels = data["training_labels"].ravel()
    features = data["training_data"]
    assert len(labels) == len(features)
    #consistent shuffling src: http://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
    rng_state = np.random.get_state()
    np.random.shuffle(labels)
    np.random.set_state(rng_state)
    np.random.shuffle(features)
    return labels, features

def test_impurity():
    print DT.entropy([1, 1, 1, 0])
    print DT.entropy([])
    print DT.entropy([1, 1, 1, 1])
    a = DT.DecisionTree(1)
    print a.impurity_func([1, 1, 0, 1], [0, 0])

data = io.loadmat("./spam-dataset/spam_data.mat")
labels, features = shuffle_and_resize(data)
print features.shape
Dec = DT.DecisionTree()
a = 0
b = 100000
# for feature_vector in features:
#     for item in feature_vector:
#         a = max(a, item)
#         b = min(b, item)
# print 'max_reported feature', a
# print 'min_reported feature', b
Dec.train(features, labels)

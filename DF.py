import numpy as np
from scipy import io
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from random import randint
import DT


class RandomTree(DT.DecisionTree):
    def __init__(self, depth, impurity_func, segmentor_func):
        super(RandomTree, self).__init__(depth, impurity_func, segmentor_func)


class DecisionForest(object):
    # def __init__(self, num_trees=10, num_depth=8, impurity_func=DT.impurity_1, segmentor_func=DT.segmentor_3, num_features_to_use=10):
    def __init__(self, num_trees=50, num_depth=100, impurity_func=DT.impurity_1, segmentor_func=DT.segmentor_3):
        self.trees = [RandomTree(num_depth, impurity_func, segmentor_func) for _ in range(num_trees)]

    def train(self, training_data, training_labels):
        for tree in self.trees:
            tree.train(training_data[:1500, :], training_labels[:1500])
            rng_state = np.random.get_state()
            np.random.shuffle(training_labels)
            np.random.set_state(rng_state)
            np.random.shuffle(training_data)

    def traverse(self, data):
        predictions = [tree.traverse(data) for tree in self.trees]
        return float(sum(predictions)) / float(len(predictions)) > 0.5

    def predict(self, test_data):
        labels = []
        for item in test_data:
            labels.append(self.traverse(item))
        return labels

import numpy as np
from scipy import io
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import math


#assumes binary labels
def score(correct, reported):
    assert len(correct) == len(reported)
    score = 0
    for true_label, reported_label in zip(correct, reported):
        if true_label == reported_label:
            score += 1
    return float(score) / len(correct)


def entropy(distribution):
    samples = len(distribution)
    if not samples:
        return 0
    positives = float(reduce(lambda x, y: x + y, distribution)) / float(samples)
    negatives = 1 - positives
    if not positives or not negatives:
        return 0
    return -1 * (negatives * math.log(negatives)
                 + positives * math.log(positives))


def impurity_1(left_label_hist, right_label_hist):
    left, right = float(len(left_label_hist)), float(len(right_label_hist))
    return left / (left + right) * entropy(left_label_hist) + \
        right / (left + right) * entropy(right_label_hist)

"""you are literally the worst segmentor ever Uses thresholds 0-14 for checking impurity"""
def segmentor_1(data, label, impurity_func):
    split = (0, 0)  # a tuple representing feature, threshold
    # print 'segmentor got data size, label size', data.shape, len(label)
    best_score = float(1)
    for i, feature_column in zip(xrange(data.shape[1]), data.T):
        for j in range(15):
            left = []
            right = []
            for k, kth_row in zip(xrange(data.shape[0]), data):
                # print 'j, k', j, k
                if kth_row[i] <= j:
                    left.append(label[k])
                else:
                    right.append(label[k])
            k_impurity = impurity_func(left, right)
            assert k_impurity < 1.0
            if k_impurity < best_score:
                best_score = k_impurity
                split = (i, j)
    print 'best score ', best_score
    print 'left label, right label sizes:', len(left), len(right)
    return split


def segmentor_2(data, label, impurity_func):
    split = (-1, -1)  # a tuple representing feature, threshold
    # print 'segmentor got data size, label size', data.shape, len(label)
    best_score = float(1)
    for i, feature_column in zip(xrange(data.shape[1]), data.T):
        for j in range(15):
            left = []
            right = []
            for k, kth_row in zip(xrange(data.shape[0]), data):
                # print 'j, k', j, k
                if kth_row[i] <= j:
                    left.append(label[k])
                else:
                    right.append(label[k])
            k_impurity = impurity_func(left, right)
            assert k_impurity < 1.0
            if k_impurity < best_score:
                best_score = k_impurity
                split = (i, j)
    print 'best score ', best_score
    print 'left label, right label sizes:', len(left), len(right)
    return split


class DecisionTree:
    head = None

    def __init__(self, depth=5, impurity_func=impurity_1, segmentor_func=segmentor_1):
        self.depth = depth
        self.impurity_func = impurity_func
        self.segmentor_func = segmentor_func
        self.head = Node()

    def split(self, node, training_data, training_labels, depth):
        if depth == self.depth:
            print '    ' * depth + 'L:', float(sum(training_labels))/len(training_labels) > 0.5, ' (MAX)'
            return LNode((float(sum(training_labels))/len(training_labels)) > 0.5)  # if more than 50% 1s, classify as 1

        node.split_rule = self.segmentor_func(training_data, training_labels, self.impurity_func)
        if node.split_rule == (-1, -1): print 'WARNING: NO SPLIT FOUND, IMPURITY = 1'
        left_set = Data(np.empty([0, 32]), [])
        # left_data = np.empty([0, 32])
        # left_label = []
        right_set = Data(np.empty([0, 32], [])
        right_data = np.empty([0, 32])
        right_label = []
        for i, row in zip(xrange(len(training_data)), training_data):
            if row[node.split_rule[0]] <= node.split_rule[1]:
                # left_data = np.append(left_data, np.reshape(row, (1, 32)), 0)
                # left_label.append(training_labels[i])
                left_set.append(row, training_labels[i])
            else:
                right_data = np.append(right_data, np.reshape(row, (1, 32)), 0)
                right_label.append(training_labels[i])
        if not len(left_set) or not len(right_label):
            print '    ' * depth + 'L:', float(sum(training_labels))/len(training_labels) > 0.5
            return LNode((float(sum(training_labels))/len(training_labels)) > 0.5)  # if more than 50% 1s, classify as 1
        assert right_data.shape[0] == len(right_label)
        print '    ' * depth + 'D:', node.split_rule
        node.left = self.split(Node(), left_set.data, left_set.labels, depth + 1)
        node.right = self.split(Node(), right_data, right_label, depth + 1)
        return node

    def traverse(self, data):
        node = self.head
        while not node.label:
            if data[node.split_rule[0]] <= node.split_rule[1]:
                node = node.left
            else:
                node = node.right
        return node.label

    def train(self, training_data, training_labels):
        self.head = self.split(self.head, training_data, training_labels, 0)

    def predict(self, test_data):
        labels = []
        for item in test_data:
            labels.append(self.traverse(item))


class Node:
    left = None
    right = None
    split_rule = None
    label = None

    def __init__(self):
        pass


class LNode(Node):
    def __init__(self, label):
        self.label = 1 if label else 0


class Data:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        assert data.shape[0] == len(labels)

    def append(self, data, label):
        self.data = np.append(self.data, np.reshape(data, (1, 32)), 0)
        self.labels.append(label)

    def __len__(self):
        return len(self.labels)

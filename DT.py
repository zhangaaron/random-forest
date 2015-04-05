import numpy as np
from scipy import io
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import math

#assumes binary labels
def entropy(distribution):
	samples = len(distribution)
	if not samples:
		return 0
	positives = Float(reduce(lambda x, y: x + y, distribution)) / Float(samples)
	negatives = 1 - positives
	if not positives or not negatives:
		print 'either positives or negatives is 0'
		return 0
	return -1 * (negatives * math.log(negatives) + positives * math.log(positives))
def impurity_1(left_label_hist, right_label_hist):
	left, right = len(left_label_hist), len(right_label_hist)
	return left / (left + right) * entropy(left_label_hist) + right (left + right) * entropy(right_label_hist)
class DecisionTree:
	head = None
	def __init__(self, depth, impurity_func, segmentor_func):
		self.depth = depth
		self.impurity_func = impurity_func
		self.segmentor_func = segmentor_func
		self.head = Node()
	def train(self, training_data, training_labels):
		curr_depth = 0
		self.head = split(self.head, training_data, training_labels, depth)
	def predict(self, test_data):
		labels = []
		for item in test_data:
			labels.append(traverse(item))


class Node:
	left = None
	right = None
	split_rule = None
	depth = 0
	def __init__(self):
		pass
class LNode(Node):
	label = None
	def __init__(self, label):
		left = None
		right = None
		self.label = label






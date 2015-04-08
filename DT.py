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
	positives = float(reduce(lambda x, y: x + y, distribution)) / float(samples)
	negatives = 1 - positives
	if not positives or not negatives:
		return 0
	return -1 * (negatives * math.log(negatives) + positives * math.log(positives))

def impurity_1(left_label_hist, right_label_hist):
	left, right = float(len(left_label_hist)), float(len(right_label_hist))
	return left / (left + right) * entropy(left_label_hist) + right / (left + right) * entropy(right_label_hist)

def segmentor_1(data, label, impurity_func):
	split = (0, 0) # a tuple representing feature, threshold
	best_score = float(1)
	for i in range(data.shape[1]):
		feature_column = data[i,:]
		for j in feature_column:
			left = []
			right = []
			for k in feature_column:
				# print 'j, k', j, k
				if k <= j:
					left.append(label[k])
				else:
					right.append(label[k])
			k_impurity = impurity_func(left, right)
			# print 'left', len(left)
			# print 'right', len(right)
			if k_impurity < best_score:
				best_score = k_impurity
				split = (i, k)
	# print 'best_score', best_score
	return split


class DecisionTree:
	head = None
	def __init__(self, depth = 5, impurity_func = impurity_1, segmentor_func = segmentor_1):
		self.depth = depth
		self.impurity_func = impurity_func
		self.segmentor_func = segmentor_func
		self.head = Node()
	def split(self, node, training_data, training_labels, depth):
		print 'split', depth
		if depth == self.depth:
			if (float(sum(training_labels))/len(training_labels)) > 0.5:
				return LNode(1)
			return LNode(0)
		node.split_rule = self.segmentor_func(training_data, training_labels, self.impurity_func)
		left_data = np.empty([32, 0])
		left_label = []
		right_data = []
		right_label = np.empty([32, 0])
		print 'left data shape ', left_data.shape
		for i in range(len(training_data)):
			print 'training_data to append ', training_data[i].shape
			if training_data[i][node.split_rule[0]] <= node.split_rule[1]:
				# left_data.append(training_data[i])
				left_data = np.append(left_data, training_data[i], 1)
				left_label.append(training_labels[i])
			else:
				right_data = np.append(right_data, training_data[i], 1)
				right_label.append(training_labels[i])
		node.left = self.split(Node(), left_data, left_label, depth + 1)
		node.right = self.split(Node(), right_data, right_label, depth + 1)
		return node
	def traverse(data):
		node = self.head
		while not node.label:
			if data[node.split_rule[0]] <= split_rule[1]:
				node = node.left
			else:
				node  = node.right
		return node.label

	def train(self, training_data, training_labels):
		self.head = self.split(self.head, training_data, training_labels, 0)
	def predict(self, test_data):
		labels = []
		for item in test_data:
			labels.append(traverse(item))


class Node:
	left = None
	right = None
	split_rule = None
	def __init__(self):
		pass
class LNode(Node):
	label = None
	def __init__(self, label):
		left = None
		right = None
		self.label = label
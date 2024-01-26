"""
DecisionTree class
Created by Kamil Kośnik, Kacper Radzikowski
"""

import numpy as np
from itertools import combinations
from node import Node
from collections import Counter

class DecisionTree:
    def __init__(self, min_sample_size=3, max_depth=100, gini_index=False):
        self.min_sample_size = min_sample_size
        self.max_depth = max_depth
        self.gini_index = gini_index
        self.root = None

    def fit(self, data, values):
        self.root = self._create_tree(data, values)

    def predict(self, X):
        results = []
        for x in X:
            result = self._traverse_tree(x, self.root)
            results.append(result)
        return np.array(results)

    def _create_tree(self, data, values, depth=0):
        n_samples, n_attr = data.shape
        n_labels = len(np.unique(values))

        # check if sttoping criteria are met
        if (depth >= self.max_depth or n_labels == 1 or n_samples <= self.min_sample_size):
            leaf_value = self.decide_leaf_value(values)
            return Node(value=leaf_value)

        # find the criteria that creates the best information gain
        best_attribute, best_criteria = self._decide_best_split(data, values, n_attr)

        # create child nodes of the tree
        left_idxs, right_idxs = self._split(data[:, best_attribute], best_criteria)
        left_child = self._create_tree(data[left_idxs, :], values[left_idxs], depth+1)
        right_child = self._create_tree(data[right_idxs, :], values[right_idxs], depth+1)

        return Node(best_attribute, best_criteria, left_child, right_child)

    def decide_leaf_value(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def _decide_best_split(self, X, y, n_attr):
        best_gain = -1
        split_atrtribute, split_criterium = None, None

        for i in range(n_attr):
            X_column = X[:, i]
            possible_attr_values = np.unique(X_column)
            combinations_to_check = len(possible_attr_values) // 2
            possible_criteria = []

            for c in (1, combinations_to_check + 1):
                possible_criteria.extend(list(combinations(possible_attr_values, c)))

            for combination in possible_criteria:
                # calculate information gain
                inf_gain = self._information_gain(y, X_column, combination)

                if inf_gain > best_gain:
                    best_gain = inf_gain
                    split_atrtribute = i
                    split_criterium = combination

        return split_atrtribute, split_criterium

    def _information_gain(self, y, column, criterium):
        if self.gini_index:
            # parent Gini Index
            parent_gini = self._gini_index(y)

            # create children
            left_idxs, right_idxs = self._split(column, criterium)

            if len(left_idxs) == 0 or len(right_idxs) == 0:
                return 0

            # calculate the weighted Gini Index of children
            n = len(y)
            n_left = len(left_idxs)
            n_right = len(right_idxs)

            gini_left = self._gini_index(y[left_idxs])
            gini_right = self._gini_index(y[right_idxs])
            child_gini = (n_left / n) * gini_left + (n_right / n) * gini_right

            information_gain = parent_gini - child_gini

        else:
            # parent entropy
            parent_entropy = self._entropy(y)

            # create children
            left_idxs, right_idxs = self._split(column, criterium)

            if len(left_idxs) == 0 or len(right_idxs) == 0:
                return 0

            # calculate the weighted entropy of children
            n = len(y)
            n_left = len(left_idxs)
            n_right = len(right_idxs)

            entropy_left = self._entropy(y[left_idxs])
            entropy_right = self._entropy(y[right_idxs])
            child_entropy = (n_left / n) * entropy_left + (n_right / n) * entropy_right

            information_gain = parent_entropy - child_entropy

        return information_gain

    def _gini_index(self, y):
        gini_index = 1

        supporting_histogram = np.bincount(y)
        probababilities = supporting_histogram / len(y) # dividing every value in y by number of examples in dataset

        for probability in probababilities:
            if probability > 0:
                to_sum = pow(probability, 2)
                gini_index -= to_sum

        return gini_index

    def _entropy(self, y):
        entropy = 0

        supporting_histogram = np.bincount(y)
        probababilities = supporting_histogram / len(y) # dividing every value in y by number of examples in dataset

        for probability in probababilities:
            if probability > 0:
                to_sum = probability * np.log2(probability)
                entropy -= to_sum

        return entropy

    def _split(self, X_column, criterium_to_split):
        left_idxs = []
        right_idxs = []
        for idx in range(len(X_column)):
            if X_column[idx] in criterium_to_split:
                left_idxs.append(idx)
            else:
                right_idxs.append(idx)

        return left_idxs, right_idxs

    def _traverse_tree(self, x, node):
        if node.isLeafNode():
            return node.getValue()

        if x[node.getFeature()] in node.getCriteria():
            return self._traverse_tree(x, node.getLeftChild())
        else:
            return self._traverse_tree(x, node.getRightChild())

import numpy as np
from itertools import combinations
from node import Node
from collections import Counter

class DecisionTree:
    def __init__(self, min_sample_size, max_depth):
        self.min_sample_size = min_sample_size
        self.max_depth = max_depth
        self.root = None

    def fit(self):
        self.root = self._crete_tree()

    def _create_tree(self, data, values, depth=0):
        n_samples, n_attr = data.shape
        n_labels = len(np.unique(values))

        # check if sttoping criteria are met
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_sample_size):
            leaf_value = self.decide_leaf_value(values)
            return Node(value=leaf_value)

        # find the criteria that creates the best information gain
        best_attribute, best_criteria = self._decide_best_split(data, values, n_attr)

        # create child nodes of the tree

    def decide_leaf_value(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def _decide_best_split(self, X, y, n_attr):
        best_gain = -1
        split_atrtribute, split_criterium = None, None

        for i in range(n_attr):
            X_column = [:, i]
            possible_attr_values = np.unique(X_column)
            combinations_to_check = len(possible_attr_values) // 2
            possible_criteria = []

            for c in range(1, combinations_to_check):
                possible_criteria.append(combinations(possible_attr_values, c))

            for combination in possible_criteria:
                # calculate information gain
                inf_gain = self._information_gain(y, X_column, combination)

                if inf_gain > best_gain:
                    best_gain = inf_gain
                    split_atrtribute = i
                    split_criterium = combination

        return split_atrtribute, split_criterium

    def _information_gain(self, y, column, criterium):
        pass

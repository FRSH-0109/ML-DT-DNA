"""
DecisionTree Node class
Created by Kamil Kośnik, Kacper Radzikowski
"""

class Node:
    def __init__(self, feature=None, criteria=None, left=None, right=None, *, value=None):
        self.feature_ = feature
        self.criteria_ = criteria
        self.left_ = left
        self.right_ = right
        self.value_ = value

    def getFeature(self):
        return self.feature_

    def getCriteria(self):
        return self.criteria_

    def getLeftChild(self):
        return self.left_

    def getRightChild(self):
        return self.right_

    def getValue(self):
        return self.value_

    def isLeafNode(self):
        return self.value_ != None
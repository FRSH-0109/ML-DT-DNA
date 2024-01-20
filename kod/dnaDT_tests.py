import pytest
from loadData import parseDataToTrainDNA
from classTrainDNA import DnaAttrVal, TrainDNA
from node import Node
from decisionTree import DecisionTree

def test_TrainDNA():
    trainDNA1 = TrainDNA(15, 1, "CCTGCCA")
    assert(trainDNA1.getPoint() == 15)
    assert(trainDNA1.getState() == 1)
    assert(trainDNA1.getCode() == "CCTGCCA")
    assert(trainDNA1.getAttributes()[0] == DnaAttrVal.C)
    assert(trainDNA1.getAttributes()[1] == DnaAttrVal.C)
    assert(trainDNA1.getAttributes()[2] == DnaAttrVal.T)
    assert(trainDNA1.getAttributes()[3] == DnaAttrVal.G)
    assert(trainDNA1.getAttributes()[4] == DnaAttrVal.C)
    assert(trainDNA1.getAttributes()[5] == DnaAttrVal.C)
    assert(trainDNA1.getAttributes()[6] == DnaAttrVal.A)

    trainDNA1 = TrainDNA(10, 0, "AACCTTGG")
    assert(trainDNA1.getPoint() == 10)
    assert(trainDNA1.getState() == 0)
    assert(trainDNA1.getCode() == "AACCTTGG")
    assert(trainDNA1.getAttributes()[0] == DnaAttrVal.A)
    assert(trainDNA1.getAttributes()[1] == DnaAttrVal.A)
    assert(trainDNA1.getAttributes()[2] == DnaAttrVal.C)
    assert(trainDNA1.getAttributes()[3] == DnaAttrVal.C)
    assert(trainDNA1.getAttributes()[4] == DnaAttrVal.T)
    assert(trainDNA1.getAttributes()[5] == DnaAttrVal.T)
    assert(trainDNA1.getAttributes()[6] == DnaAttrVal.G)
    assert(trainDNA1.getAttributes()[6] == DnaAttrVal.G)

def test_loadData():
    PATH = "testy/spliceATrainKIS_TEST.dat"
    dna_train_array = parseDataToTrainDNA(PATH)

    assert(len(dna_train_array) == 3)
    assert(dna_train_array[0].getPoint() == 68)
    assert(dna_train_array[0].getState() == 1)
    assert(dna_train_array[0].getAttributes()[0] == DnaAttrVal.G)
    assert(dna_train_array[0].getAttributes()[5] == DnaAttrVal.C)
    assert(dna_train_array[0].getAttributes()[8] == DnaAttrVal.A)

    assert(dna_train_array[2].getPoint() == 68)
    assert(dna_train_array[2].getState() == 0)
    assert(dna_train_array[2].getAttributes()[0] == DnaAttrVal.T)
    assert(dna_train_array[2].getAttributes()[5] == DnaAttrVal.G)
    assert(dna_train_array[2].getAttributes()[8] == DnaAttrVal.A)


def test_node():
    test_node = Node(12, [DnaAttrVal.A, DnaAttrVal.G])
    assert(test_node.getFeature() == 12)
    assert(test_node.getCriteria() == [DnaAttrVal.A, DnaAttrVal.G])
    assert(test_node.getLeftChild() == None)
    assert(test_node.getRightChild() == None)
    assert(test_node.getValue() == None)

def test_nodeWithChildren():
    left_child = Node(14, [DnaAttrVal.C])
    right_child = Node(17, [DnaAttrVal.T])
    test_node = Node(19, [DnaAttrVal.A, DnaAttrVal.T], left_child, right_child)
    assert(test_node.getFeature() == 19)
    assert(test_node.getCriteria() == [DnaAttrVal.A, DnaAttrVal.T])
    assert(test_node.getLeftChild() == left_child)
    assert(test_node.getRightChild() == right_child)
    assert(test_node.getValue() == None)

def test_nodeWithValue():
    test_node = Node(value=1)
    assert(test_node.getFeature() == None)
    assert(test_node.getCriteria() == None)
    assert(test_node.getLeftChild() == None)
    assert(test_node.getRightChild() == None)
    assert(test_node.getValue() == 1)

def test_isLeafNode():
    test_node_1 = Node(12, [DnaAttrVal.A, DnaAttrVal.G])
    test_node_2 = Node(value=0)
    assert(test_node_1.isLeafNode() == False)
    assert(test_node_2.isLeafNode() == True)

def test_decisionTreeClass():
    test_decision_tree = DecisionTree(3, 50)
    assert(test_decision_tree.min_sample_size == 3)
    assert(test_decision_tree.max_depth == 50)
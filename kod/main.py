from loadData import parseDataToTrainDNA
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree, metrics
import classTrainDNA
from decisionTree import DecisionTree
from random import randint

PATH_DTrain = "dane/spliceDTrainKIS.dat"
PATH_ATrain = "dane/spliceATrainKIS.dat"
PATH_ATrain_small = "dane/spliceATrainKIS_small.dat"

test_number = 5

def createPoolTeachAndVerify(data_array, teach_ratio = float):
    data_array_size = len(data_array)
    teach_array = []
    teach_array_szie = int(data_array_size * teach_ratio)
    verify_array = []

    pos_data_elemets = []
    neg_data_elemets = []

    # count ratio of positives examples in data array
    pos_counter = 0
    for exmpl in data_array:
        if(exmpl.getState() == 1):
            pos_counter += 1

    pos_ratio = pos_counter / len(data_array)

    # separate data into positives and negatives array to split data correctly
    for i in range(0, data_array_size):
        if(data_array[i].getState() == 1):
            pos_data_elemets.append(data_array[i])
        else:
            neg_data_elemets.append(data_array[i])

    # add target amount of positives data into teach array
    teach_pos_number = int(pos_ratio * teach_array_szie)
    for i in range(0, teach_pos_number):
        index = randint(0, len(pos_data_elemets)-1)
        teach_array.append(pos_data_elemets[index])
        del pos_data_elemets[index]

    # add rest of positives examples to verify array
    for element in pos_data_elemets:
        verify_array.append(element)

     # add target amount of positives data into teach array
    teach_neg_number = int(teach_array_szie - teach_pos_number)
    for i in range(0, teach_neg_number):
        index = randint(0, len(neg_data_elemets)-1)
        teach_array.append(neg_data_elemets[index])
        del neg_data_elemets[index]

    # add rest of positives examples to verify array
    for element in neg_data_elemets:
        verify_array.append(element)

    return teach_array, verify_array


def main():

    dtrain_array = parseDataToTrainDNA(PATH_DTrain)
    atrain_array = parseDataToTrainDNA(PATH_ATrain)
    atrain_small_array = parseDataToTrainDNA(PATH_ATrain_small)
    conf_matrix_buf = [[0, 0],[0, 0]]
    conf_matrix_buf_skilearn = [[0, 0],[0, 0]]

    for i in range(0, test_number):

        #create teach and verify pool by given data, ratio is teach size to whole array size
        teach_array, verify_array = createPoolTeachAndVerify(dtrain_array, 0.5)

        #prepare data to be passed to decision tree
        verify_array_data = []
        verify_array_values = []
        for element in verify_array:
            verify_array_values.append(element.getState())
            verify_array_data.append(element.getAttributes())

        teach_array_data = []
        teach_array_values = []
        for element in teach_array:
            teach_array_values.append(element.getState())
            teach_array_data.append(element.getAttributes())

        sklearn_teach_array_data = []
        for element in teach_array_data:
            element_to_int = []
            for attr in element:
                attr_translation = {'A': 1, 'C': 2, 'G': 3, 'N': 4, 'S': 5, 'T': 6}
                element_to_int.append(attr_translation[attr.value])
            sklearn_teach_array_data.append(element_to_int)

        sklearn_verify_array_data = []
        for element in verify_array_data:
            element_to_int = []
            for attr in element:
                attr_translation = {'A': 1, 'C': 2, 'G': 3, 'N': 4, 'S': 5, 'T': 6}
                element_to_int.append(attr_translation[attr.value])
            sklearn_verify_array_data.append(element_to_int)

        classifier = DecisionTree()
        classifier.fit(np.array(teach_array_data), np.array(teach_array_values))

        # Testing predciton accuracy
        predicitions = classifier.predict(verify_array_data)
        acc = accuracy(verify_array_values, predicitions)
        print("Own implementation accuracy: " + str(acc))

        # Creating confusion matrix for our implementation of a decision tree algorithm
        confusion_matrix = metrics.confusion_matrix(verify_array_values, predicitions)
        conf_matrix_buf[0][0] += confusion_matrix.T[0][0]
        conf_matrix_buf[0][1] += confusion_matrix.T[0][1]
        conf_matrix_buf[1][0] += confusion_matrix.T[1][0]
        conf_matrix_buf[1][1] += confusion_matrix.T[1][1]

        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=100, min_samples_split=3)
        clf = clf.fit(sklearn_teach_array_data, teach_array_values)
        sklearn_predictions = clf.predict(sklearn_verify_array_data)
        sklearn_acc = accuracy(verify_array_values, sklearn_predictions)
        print("Scikit-learn implementation accuracy: " + str(sklearn_acc))

        sklearn_confusion_matrix = metrics.confusion_matrix(verify_array_values, sklearn_predictions)
        conf_matrix_buf_skilearn[0][0] += sklearn_confusion_matrix.T[0][0]
        conf_matrix_buf_skilearn[0][1] += sklearn_confusion_matrix.T[0][1]
        conf_matrix_buf_skilearn[1][0] += sklearn_confusion_matrix.T[1][0]
        conf_matrix_buf_skilearn[1][1] += sklearn_confusion_matrix.T[1][1]

    conf_matrix_buf[0][0] /= test_number
    conf_matrix_buf[0][1] /= test_number
    conf_matrix_buf[1][0] /= test_number
    conf_matrix_buf[1][1] /= test_number
    confusion_matrix.T[0][0] = conf_matrix_buf[0][0]
    confusion_matrix.T[0][1] = conf_matrix_buf[0][1]
    confusion_matrix.T[1][0] = conf_matrix_buf[1][0]
    confusion_matrix.T[1][1] = conf_matrix_buf[1][1]

    conf_matrix_buf_skilearn[0][0] /= test_number
    conf_matrix_buf_skilearn[0][1] /= test_number
    conf_matrix_buf_skilearn[1][0] /= test_number
    conf_matrix_buf_skilearn[1][1] /= test_number
    sklearn_confusion_matrix.T[0][0] = conf_matrix_buf_skilearn[0][0]
    sklearn_confusion_matrix.T[0][1] = conf_matrix_buf_skilearn[0][1]
    sklearn_confusion_matrix.T[1][0] = conf_matrix_buf_skilearn[1][0]
    sklearn_confusion_matrix.T[1][1] = conf_matrix_buf_skilearn[1][1]

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[0, 1])
    cm_display.plot()
    plt.title(label="Confusion matrix - our implementation")
    cm_figure = plt.gcf()
    cm_figure.savefig('confusion_matrix.pdf', format='pdf')
    plt.show()

    sklearn_cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=sklearn_confusion_matrix, display_labels=[0, 1])
    sklearn_cm_display.plot()
    plt.title(label="Confusion matrix - scikit-learn implementation")
    sklearn_cm_figure = plt.gcf()
    sklearn_cm_figure.savefig('sklearn_confusion_matrix.pdf', format='pdf')
    plt.show()

def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

if __name__ == "__main__":
    main()

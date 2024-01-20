from loadData import parseDataToTrainDNA
import numpy as np
import classTrainDNA

PATH_DTrain = "dane/spliceDTrainKIS.dat"
PATH_ATrain = "dane/spliceATrainKIS.dat"

def main():

    dtrain_array = parseDataToTrainDNA(PATH_DTrain)
    atrain_array = parseDataToTrainDNA(PATH_ATrain)

    dtrain_attribute_val = [dtrain.getAttributes() for dtrain in dtrain_array]
    atrain_attribute_val = [atrain.getAttributes() for atrain in atrain_array]

    possible_attribute_val_d = np.unique(dtrain_attribute_val).tolist()
    possible_attribute_val_a = np.unique(atrain_attribute_val).tolist()

    #print first and last data train from Dtrain
    dtrain_array[0].print()
    dtrain_array[len(dtrain_array)-1].print()

    #print first and last data train from Atrain
    atrain_array[0].print()
    atrain_array[len(atrain_array)-1].print()

    print("Possible attribute values for dtrain_array: " + str(possible_attribute_val_d))
    print("Possible attribute values for atrain_array: " + str(possible_attribute_val_a))

    pass

if __name__ == "__main__":
    main()

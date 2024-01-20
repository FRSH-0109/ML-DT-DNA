from loadData import loadData
import classTrainDNA

PATH_DTrain = "dane/spliceDTrainKIS.dat"
PATH_ATrain = "dane/spliceATrainKIS.dat"
    
def main():

    dtrain_array = loadData(PATH_DTrain)
    atrain_array = loadData(PATH_ATrain)

    #print first and last data train from Dtrain
    dtrain_array[0].print()
    dtrain_array[len(dtrain_array)-1].print()

    #print first and last data train from Atrain
    atrain_array[0].print()
    atrain_array[len(atrain_array)-1].print()

    pass

if __name__ == "__main__":
    main()
    
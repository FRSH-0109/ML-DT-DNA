import numpy as np
from classTrainDNA import TrainDNA


def loadData(path=str):
    data = np.genfromtxt(path,
                    names=None,
                    dtype=None,
                    delimiter=' ')
    
    data_array = []

    point = data[0].decode()
    
    for i in range(1, len(data)-1, 2):
        state = data[i].decode()
        code = data[i+1].decode()
        data_array.append(TrainDNA (point, state, code))

    return data_array
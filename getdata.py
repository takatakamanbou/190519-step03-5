import numpy as np
import cv2
import cifar10

### reading the raw data
#
def getData(path='./cifar-10-batches-py', seed=0):

    c10 = cifar10.CIFAR10(path)
    datL, labL, tL = c10.loadData('L')
    datL /= 255
    datT, labT, tT = c10.loadData('T')
    datT /= 255
    idxL, idxV = c10.genIndexLV(labL, seed=seed)

    return datL[idxL], labL[idxL], tL[idxL], datL[idxV], labL[idxV], tL[idxV], datT, labT, tT


### mini batch indicies for stochastic gradient ascent
#
def makeBatchIndex(N, batchsize):

    idx = np.random.permutation(N)
        
    nbatch = int(np.ceil(float(N) / batchsize))
    idxB = np.zeros(( nbatch, N ), dtype = bool)
    for ib in range(nbatch - 1):
        idxB[ib, idx[ib*batchsize:(ib+1)*batchsize]] = True
    ib = nbatch - 1
    idxB[ib, idx[ib*batchsize:]] = True

    return idxB


if __name__ == '__main__':
    
    datL, labL, tL, datV, labV, tV, datT, labT, tT = getData()
    print(datL.shape, labL.shape, tL.shape)
    print(datV.shape, labV.shape, tV.shape)
    print(datT.shape, labT.shape, tT.shape)


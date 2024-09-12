import cooler
import itertools
import math
from scipy.sparse import coo_matrix,csr_matrix,triu,diags
from scipy.ndimage import convolve
from scipy.signal import argrelmin,find_peaks
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import multiprocessing as mp
import os 
import warnings
import time
import os
from skimage.transform import resize 

import collections, functools, operator
import dask.dataframe as dd
import pickle

from scipy import ndimage

def calcOEMat(mat,maxDis):
    size = mat.shape[0]
    OE   = []
    offsets = []
    #for k in range(size-1):
    for k in range(maxDis+10):
        diag = mat.diagonal(k)
        mean_ = diag.mean()
        if mean_ == 0:
            OE.append(diag)
            offsets.append(k)
        else:
            OE.append(diag/mean_)
            offsets.append(k)
    OE = diags(OE,offsets,format='csr',shape=(size,size))
    return OE

def calcStd(mat,start,end,minDis,maxDis,q):
    Std = np.zeros(mat.shape)
    for i in range(start,end):
        for d in range(minDis,maxDis+1):
            j = i+d
            if j >= mat.shape[0]:
                break
            indices = np.triu_indices(d+1)
            Std[i,j] = mat[i:(j+1),i:(j+1)][indices].std()
    q.put(Std)

def calcNormStd(mat,maxDis):
    '''
      calculate std for entries within given distance d
      Std[d]
    '''
    Std = []
    m = []
    for d in range(maxDis+1):
        if d >= mat.shape[0]:
            break
        m += mat.diagonal(d).tolist()
        Std.append(np.std(m))
    return Std

# def calcFeaturesHorse(tad,oe,label):
def calcEdgeStrengthHorse(tad,oe,):
    edgeStrength = []
    for index,v in tad.iterrows():
        _,i,j = v

        H = oe[i:(i+1),i:(j+1)].mean() - oe[(i-1):i,i:(j+1)].mean()
        V = oe[i:(j+1),(j-1):j].mean() - oe[i:(j+1),(j+1):(j+2)].mean()
        edgeS = H+V
        edgeStrength.append(edgeS) 

    return (edgeStrength)

def calcEdgeStrength(tads,c):
    '''
        tads: dataframe with label T(isTAD) or F(notTAD)
        c: cooler object
    '''
    ncpu = 8

    tads.bin1 = tads.bin1.astype('int')
    tads.bin2 = tads.bin2.astype('int')
    edgeS = []
    #for chrom in c.chromnames:
    for chrom in tads.chrom.unique():
        print(chrom)
        tad = tads[tads.chrom == chrom]
        try:
            mat = csr_matrix(np.nan_to_num(np.triu(c.matrix(balance=False,sparse=False).fetch(chrom))))
        except ValueError:
            mat = csr_matrix(np.nan_to_num(np.triu(c.matrix(balance=False,sparse=False).fetch('chr'+chrom))))

        tad_ = dd.from_pandas(tad,npartitions=2*ncpu)
        horse = functools.partial(calcEdgeStrengthHorse,oe=mat)
        result = tad_.map_partitions(horse).compute()
        for i in result:
            edgeS += i

    return np.array(edgeS)

def filterTADHorse(tad,oe): 
    res = []
    for index,v in tad.iterrows():
        _,i,j = v
        try:
            H = oe[i,i:(j+1)].mean() - oe[(i-1),i:(j+1)].mean()
        except IndexError:
            H = oe[i:(j+1),j].mean() - oe[i:(j+1),(j+1)].mean()
        try:
            V = oe[i:(j+1),j].mean() - oe[i:(j+1),(j+1)].mean()
        except IndexError:
            V = oe[i,i:(j+1)].mean() - oe[(i-1),i:(j+1)].mean()

        if H<0 or V<0:
            res.append(False)
        else:
            res.append(True)
    return res


def calcFeaturesHorseNew(tad,oe,label):
    features = []
    labels = []
    for index,v in tad.iterrows():
        if label :
            _,i,j,label = v
            labels.append(label)
        else:
            _,i,j = v

        summit = oe[i,j]
        vertex = []
        for m in range(25):
            row = i + m//5 - 2
            col = j + m%5 - 2
            try:
                v = summit - oe[row,col]
            except IndexError:
                v = 0
            vertex.append(v)

        vertex.pop(12) # remove the summit itself
        # for m in range(49):
        #     row = i + m//7 - 3
        #     col = j + m%7 - 3
        #     try:
        #         v = summit - oe[row,col]
        #     except IndexError:
        #         v = 0
        #     vertex.append(v)

        # vertex.pop(24) # remove the summit itself
        # for m in range(49):
        #     row = i + m//7 - 3
        #     col = j + m%7 - 3
        #     try:
        #         v = oe[row,col]
        #     except IndexError:
        #         v = 0
        #     vertex.append(v)
        
        if i<=2:
            V = oe[i:(j+1),(j-3):j].mean() - oe[i:(j+1),(j+1):(j+4)].mean()
            H = V
        elif j+4>=oe.shape[0]:
            H = oe[i:(i+3),i:(j+1)].mean() - oe[(i-3):i,i:(j+1)].mean()
            V = H
        else:
            H = oe[i:(i+3),i:(j+1)].mean() - oe[(i-3):i,i:(j+1)].mean()
            V = oe[i:(j+1),(j-3):j].mean() - oe[i:(j+1),(j+1):(j+4)].mean()
            
        # H = oe[i:(i+3),i:(j+1)].mean() - oe[(i-3):i,i:(j+1)].mean()
        # V = oe[i:(j+1),(j-3):j].mean() - oe[i:(j+1),(j+1):(j+4)].mean()
        # try:
        #     H = oe[i,i:(i+5)] - oe[(i-1),i:(i+5)]
        # except IndexError:
        #     H = oe[(j-4):(j+1),j] - oe[(j-4):(j+1),(j+1)]
        # try:
        #     V = oe[(j-4):(j+1),j] - oe[(j-4):(j+1),(j+1)]
        # except IndexError:
        #     V = oe[i,i:(i+5)] - oe[(i-1),i:(i+5)]
        # H = H.tolist()
        # V = V.tolist()

        # try:
        #     H = oe[i,i:(j+1)] - oe[(i-1),i:(j+1)]
        # except IndexError:
        #     H = oe[i:(j+1),j] - oe[i:(j+1),(j+1)]
        # try:
        #     V = oe[i:(j+1),j] - oe[i:(j+1),(j+1)]
        # except IndexError:
        #     V = oe[i,i:(j+1)] - oe[(i-1),i:(j+1)]
        # H = resize(H,(8,),preserve_range=True).tolist()
        # V = resize(V,(8,),preserve_range=True).tolist()

        d = j-i

        midL = math.floor((j+i)/2)
        midR = math.ceil((j+i)/2)

        if i - (j-i) < 0:
            downStream = oe[midR:(j+1),j:(j-midR + j+1)].sum()
            upStream = downStream
        elif j + (j-i) > oe.shape[0] - 1:
            upStream = oe[(i + i - midL):(i+1),i:(midL+1)].sum()
            downStream = upStream
        else:
            upStream = oe[(i + i - midL):(i+1),i:(midL+1)].sum()
            downStream = oe[midR:(j+1),j:(j-midR + j+1)].sum()
        
        diamondScore = oe[i:(midL+1),midR:(j+1)].sum() - (downStream + upStream)/2
        features.append(vertex + [H,V] + [d,diamondScore]) # 42 features # 66 features
    return (features,labels)

def calcFeaturesHorse(tad,oe,label,stdNorm=None):
    features = []
    labels = []
    for index,v in tad.iterrows():
        if label :
            _,i,j,label = v
            labels.append(label)
        else:
            _,i,j = v

        summit = oe[i,j]
        vertex = []
        for m in range(25):
            row = i + m//5 - 2
            col = j + m%5 - 2
            try:
                v = summit - oe[row,col]
            except IndexError:
                # row = i - m//5 + 2
                # col = j - m%5 + 2
                # v = summit - oe[row,col]
                v = 0
            vertex.append(v)

        # vertex = (summit - oe[(i-2):(i+3),(i-2):(i+3)].toarray()).flatten().tolist()
        vertex.pop(12) # remove the summit itself

        # H = oe[i,i:(j+1)].mean() - oe[i-1,i:(j+1)].mean()
        # V = oe[i:(j+1),j].mean() - oe[i:(j+1),j+1].mean()
        H = oe[i:(i+3),i:(j+1)].mean() - oe[(i-3):i,i:(j+1)].mean()
        V = oe[i:(j+1),(j-3):j].mean() - oe[i:(j+1),(j+1):(j+4)].mean()

        d = j-i
        # indices = np.triu_indices(d+1)
        # Std = oe[i:(j+1),i:(j+1)][indices].std()
        # Std = oe[i:(j+1),i:(j+1)][indices].std()/stdNorm[d]

        features.append(vertex + [H,V,d]) # 27 features
        # features.append(vertex + [H,V,Std,d]) # 28 features
        # features.append([L,R,T,B,LT,RT,LB,RB,H,V,Std,d,])
    return (features,labels)

def calcFeatures(tads,c,maxDis,label=False,oeNorm=True,filter=True,medianFilter=True):
    '''
        tads: dataframe with label T(isTAD) or F(notTAD)
        c: cooler object
    '''
    ncpu = 8

    tads.bin1 = tads.bin1.astype('int')
    tads.bin2 = tads.bin2.astype('int')
    features = []
    labels = []
    tads_ = []
    #for chrom in c.chromnames:
    for chrom in tads.chrom.unique():
        print(chrom)
        tad = tads[tads.chrom == chrom]
        print('before filter : ' + str(tad.shape[0]))
        # tads_.append(tad)
        mat = csr_matrix(np.nan_to_num(np.triu(c.matrix(balance=False,sparse=False).fetch(chrom))))
        if oeNorm:
            oe  = calcOEMat(mat,maxDis)
            # stdNorm = calcNormStd(oe,maxDis=maxDis)
        else:
            oe = mat
            # stdNorm = calcNormStd(oe,maxDis=maxDis)

        if medianFilter:
            oe  = calcMedianFilter(oe.toarray())

        if filter:
            tad_ = dd.from_pandas(tad,npartitions=2*ncpu)
            filterHorse = functools.partial(filterTADHorse,oe=oe)
            filterResult = tad_.map_partitions(filterHorse).compute()
            filterIndex = []
            for i in filterResult:
                filterIndex += i
            tad = tad[filterIndex]
            print('after filter : ' + str(tad.shape[0]))
        tads_.append(tad)

        tad_ = dd.from_pandas(tad,npartitions=2*ncpu)
        horse = functools.partial(calcFeaturesHorseNew,oe=oe,label=label,)
        # horse = functools.partial(calcFeaturesHorse,oe=oe,label=label,stdNorm=stdNorm)
        result = tad_.map_partitions(horse).compute()
        for i in result:
            features += i[0]
            labels += i[1]


    tads_ = pd.concat(tads_, ignore_index=True)

    return {'features':np.array(features),'labels':labels,'tads':tads_}
            
            
    

def calcEdgeV(mat,start,end,minDis,maxDis,q):
    '''
        HHHHHHHHHHHHHHHH
        xxxxxxxxxxxxxxxxV
         ##############xV
           ############xV
             ##########xV
               ########xV
                 ######xV
                   ####xV
                     ##xV
                       xV
       H[i,j] = M[i,i:(j+1)].mean() - M[i-1,i:(j+1)].mean()
       V[i,j] = M[i:(j+1),j].mean() - M[i:(j+1),j+1].mean()
    '''
    V = np.zeros(mat.shape)
    for i in range(start,end):
        for d in range(minDis,maxDis+1):
            j = i+d
            if j >= mat.shape[0]-1:
                break

            try:
                V[i,j] = mat[i:(j+1),j].mean() - mat[i:(j+1),j+1].mean()
            except RuntimeWarning:
                V[i,j] = mat[i,i:(j+1)].mean() - mat[i-1,i:(j+1)].mean()
    q.put(V)
    print('put one')

def calcEdgeH(mat,start,end,minDis,maxDis,q):
    '''
        HHHHHHHHHHHHHHHH
        xxxxxxxxxxxxxxxxV
         ##############xV
           ############xV
             ##########xV
               ########xV
                 ######xV
                   ####xV
                     ##xV
                       xV
       H[i,j] = M[i,i:(j+1)].mean() - M[i-1,i:(j+1)].mean()
       V[i,j] = M[i:(j+1),j].mean() - M[i:(j+1),j+1].mean()
    '''
    H = np.zeros(mat.shape)
    for i in range(start,end):
        for d in range(minDis,maxDis+1):
            j = i+d
            if j >= mat.shape[0]-1:
                break

            try:
                H[i,j] = mat[i,i:(j+1)].mean() - mat[i-1,i:(j+1)].mean()
            except RuntimeWarning:
                H[i,j] = mat[i:(j+1),j].mean() - mat[i:(j+1),j+1].mean()
    q.put(H)
    print('put one')

def calcEdge(mat,start,end,minDis,maxDis,q):
    '''
        HHHHHHHHHHHHHHHH
        xxxxxxxxxxxxxxxxV
         ##############xV
           ############xV
             ##########xV
               ########xV
                 ######xV
                   ####xV
                     ##xV
                       xV
       H[i,j] = M[i,i:(j+1)].mean() - M[i-1,i:(j+1)].mean()
       V[i,j] = M[i:(j+1),j].mean() - M[i:(j+1),j+1].mean()
    '''
    H = np.zeros(mat.shape)
    V = np.zeros(mat.shape)
    Std = np.zeros(mat.shape)
    for i in range(start,end):
        for d in range(minDis,maxDis+1):
            j = i+d
            if j >= mat.shape[0]-1:
                break

            indices = np.triu_indices(d+1)
            Std[i,j] = mat[i:(j+1),i:(j+1)][indices].std()

            try:
                H[i,j] = mat[i,i:(j+1)].mean() - mat[i-1,i:(j+1)].mean()
            except RuntimeWarning:
                H[i,j] = mat[i:(j+1),j].mean() - mat[i:(j+1),j+1].mean()
            try:
                V[i,j] = mat[i:(j+1),j].mean() - mat[i:(j+1),j+1].mean()
            except RuntimeWarning:
                V[i,j] = mat[i,i:(j+1)].mean() - mat[i-1,i:(j+1)].mean()
    q.put([H,V,Std])
    print('put one')


def calcNeighbor(mat):
    '''
        this function calculate the difference value between each point(i,j) 
        with its neighbors:
                  x x x
                  x * x
                  x x x
        L:left
        R:right
        T:top
        B:bottom
        LT:left top
        RT:right top
        LB:left bottom
        RB:right bottom
    '''
    shape = mat.shape[0]
    L = np.concatenate((np.zeros((shape,1)),mat[:,:(shape-1)]), axis = 1)
    L = mat - L
    R = np.concatenate((mat[:,1:], np.zeros((shape,1))), axis = 1)
    R = mat -R # should replace the last column by T 
    T = np.concatenate((np.zeros((1,shape)),mat[:(shape-1),:]),axis = 0)
    T = mat - T
    B = np.concatenate((mat[1:,:],np.zeros((1,shape))),axis = 0)
    B = mat - B
    LB = np.concatenate((np.zeros((shape-1,1)),mat[1:,:(shape-1)]), axis = 1)
    LB = np.concatenate((LB,np.zeros((1,shape))),axis = 0)
    LB = mat - LB
    RB = np.concatenate((mat[1:,1:],np.zeros((shape-1,1))), axis = 1)
    RB = np.concatenate((RB,np.zeros((1,shape))),axis = 0)
    RB = mat - RB
    LT = np.concatenate((np.zeros((shape-1,1)),mat[:(shape-1),:(shape-1)]), axis = 1)
    LT = np.concatenate((np.zeros((1,shape)),LT),axis = 0)
    LT = mat - LT
    RT = np.concatenate((mat[:(shape-1),1:],np.zeros((shape-1,1))), axis = 1)
    RT = np.concatenate((np.zeros((1,shape)),RT),axis = 0)
    RT = mat - RT
    # use the other neighbor to replace the out-of-bounds neighbor
    R[:,shape-1]  = T[:,shape-1]
    T[0,:] = R[0,:]
    RT[:,shape-1] = T[:,shape-1]
    RT[0,:] = R[0,:]

    return [L,R,T,B,LT,RT,LB,RB]




def calcConvolve(mat,size=4):
    kernel = np.ones((size,size))
    ori_x = math.ceil(size/2) - 1
    ori_y = -math.floor(size/2)
    return convolve(mat,kernel,mode='mirror',origin=(ori_x,ori_y))

def calcCorner(mat,start,end,shape,minDis,maxDis,q):
    x = []
    y = []
    v = []
    for i in range(start,end):
        # consider set a minDis option
        # j always > i
        for dis in range(minDis,maxDis):
            j = i + dis
            if j >= shape[0]:
                break
            if i-1 < 0:
                right = min(0,mat[i,j+1] - mat[i,j])
                left = max(0,mat[i,j-1] - mat[i,j])
                top  = right
                outter = right
                bottom = min(0,mat[i+1,j] - mat[i,j])
                inner = max(0,mat[i+1,j-1] -mat[i,j])
            elif j+1 >= shape[0]:
                left = max(0,mat[i,j-1] - mat[i,j])
                top  = min(0,mat[i-1,j] - mat[i,j])
                right = top
                bottom = max(0,mat[i+1,j] - mat[i,j])
                outter = top
                inner = max(0,mat[i+1,j-1] -mat[i,j])

            else:
                left = max(0,mat[i,j-1] - mat[i,j])
                right = min(0,mat[i,j+1] - mat[i,j])
                top  = min(0,mat[i-1,j] - mat[i,j])
                bottom = max(0,mat[i+1,j] - mat[i,j])
                outter = min(0,mat[i-1,j+1] - mat[i,j])
                inner = max(0,mat[i+1,j-1] -mat[i,j])
            x.append(i)
            y.append(j)
            v.append(top**2+right**2+outter**2 - left**2-bottom**2-inner**2)
    resMat = csr_matrix((v,(x,y)),shape)
    print('done')
    q.put(resMat) 

def calcDiamond(mat,start,end,shape,minDis,maxDis,q):
    x = []
    y = []
    v = []
    for i in range(start,end):
        # consider set a minDis option
        # j always > i
        for dis in range(minDis,maxDis):
            j = i + dis
            if j >= shape[0]:
                break
            mid = (i+j)/2
            Dsize = int(mid - i + 1)
            v_ = mat[i:(i+Dsize),(j+1-Dsize):(j+1)].mean()
            std_ = mat[i:(i+Dsize),(j+1-Dsize):(j+1)].toarray().std()
            # get the bounds of left diamond and right diamond
            if i-Dsize < 0:
                v_R = mat[(i+Dsize+1):(i+2*Dsize+1),(j+1):(j+1+Dsize)].mean()
                std_R = mat[(i+Dsize+1):(i+2*Dsize+1),(j+1):(j+1+Dsize)].toarray().std()
                v_L = v_R
                std_L = std_R
            elif j+1+Dsize >= shape[0]:
                v_L = mat[(i-Dsize):i,(j+1-2*Dsize):(j+1-Dsize)].mean()
                std_L = mat[(i-Dsize):i,(j+1-2*Dsize):(j+1-Dsize)].toarray().std()
                v_R =v_L
                std_R =std_L
            else:
                v_L = mat[(i-Dsize):i,(j+1-2*Dsize):(j+1-Dsize)].mean()
                v_R = mat[(i+Dsize+1):(i+2*Dsize+1),(j+1):(j+1+Dsize)].mean()
                std_L = mat[(i-Dsize):i,(j+1-2*Dsize):(j+1-Dsize)].toarray().std()
                std_R = mat[(i+Dsize+1):(i+2*Dsize+1),(j+1):(j+1+Dsize)].toarray().std()
                # should consider /0 condition
            #v_LR = v_R+v_L
            #if v_LR == 0:
            #    if v_ == 0:
            #        Dscore = 0
            #    else:
            #        Dscore = 10*v_ #set this to avoid /0 condition
            #else:
                #Dscore = v_/v_LR

            Dscore = v_ - v_L - std_L - v_R - std_R - std_

            x.append(i)
            y.append(j)
            v.append(Dscore)
            #v.append(v_)
    resMat = csr_matrix((v,(x,y)),shape)
    print('done')
    q.put(resMat) 

def calcDiamond_sum(mat,start,end,shape,minDis,maxDis,q):
    x = []
    y = []
    v = []
    for i in range(start,end):
        # consider set a minDis option
        # j always > i
        for dis in range(minDis,maxDis):
            j = i + dis
            if j >= shape[0]:
                break
            mid = (i+j)/2
            Dsize = int(mid - i + 1)
            v_ = mat[i:(i+Dsize),(j+1-Dsize):(j+1)].sum()
            # get the bounds of left diamond and right diamond
            if i-Dsize < 0:
                v_R = mat[(i+Dsize+1):(i+2*Dsize+1),(j+1):(j+1+Dsize)].sum()
                v_L = v_R
            elif j+1+Dsize >= shape[0]:
                v_L = mat[(i-Dsize):i,(j+1-2*Dsize):(j+1-Dsize)].sum()
                v_R =v_L
            else:
                v_L = mat[(i-Dsize):i,(j+1-2*Dsize):(j+1-Dsize)].sum()
                v_R = mat[(i+Dsize+1):(i+2*Dsize+1),(j+1):(j+1+Dsize)].sum()
                # should consider /0 condition
            v_LR = v_R+v_L
            if v_LR == 0:
                if v_ == 0:
                    Dscore = 0
                else:
                    Dscore = 10*v_ #set this to avoid /0 condition
            else:
                Dscore = v_/v_LR

            x.append(i)
            y.append(j)
            #v.append(Dscore)
            v.append(v_)
    resMat = csr_matrix((v,(x,y)),shape)
    print('done')
    q.put(resMat) 

def calcDiamond_(mat,start,end,shape,minDis,maxDis,q):
    x = []
    y = []
    v = []
    for i in range(start,end):
        # consider set a minDis option
        # j always > i
        for dis in range(minDis,maxDis):
            j = i + dis
            if j >= shape[0]:
                break
            mid = (i+j)/2
            midL = math.floor(mid)
            midR = math.ceil(mid)
            v_ = mat[i:(midL+1),midR:(j+1)].sum()
            x.append(i)
            y.append(j)
            v.append(v_)

    resMat = csr_matrix((v,(x,y)),shape)
    print('done')
    q.put(resMat) 




def mergeMpMat_(shape,q,ncpu):
    resMat = csr_matrix(shape)
    for i in range(ncpu):
        mat = q.get()
        resMat += mat
        print('get one')
    return resMat.tocoo()

def mergeMpMat(shape,q,ncpu):
    resMat = np.zeros(shape)
    for i in range(ncpu):
        mat = q.get()
        resMat += mat
        print('get one')
    q.close()
    return coo_matrix(resMat)

def mpRunFunc(mat,target,minDis,maxDis,ncpu):
    shape = mat.shape
    chunksize = int(shape[0]/ncpu) + 1
    i = 0
    diamond_process = []
    q = mp.Queue()
    
    while(i < shape[0]): 
        start = i
        end = i + chunksize
        p = mp.Process(target = target,args=(mat,start,end,minDis,maxDis,q))
        p.start()
        diamond_process.append(p)
        i += chunksize

    resMat = mergeMpMat(shape,q,ncpu)
    return resMat

#def getMatIndex(shape,maxDis):
#    '''
#      get index to flatten a Hi-C matrix
#    '''
#    print('shape: '+str(shape))
#    rows = np.repeat(range(shape),maxDis).tolist()
#    cols = list(itertools.chain.from_iterable([list(range(j,j+maxDis)) for j in range(shape)]))
#    mat = np.array([rows,cols]).T
#    mat = mat[mat[:,1]<shape]
#    return mat

#TODO: should add minDis argument
def flattenHicMat(mat,maxDis):
    cooMat = coo_matrix(mat)
    cooMat.col = cooMat.col - cooMat.row
    return cooMat.toarray()[:,:(maxDis+1)]
def reconstructHicMat(vec,maxDis):
    mat = vec.reshape((-1,(maxDis+1)))
    mat = coo_matrix(mat)
    finalMat = coo_matrix((mat.data,(mat.row,mat.col+mat.row)))
    shape = finalMat.shape[0]
    return finalMat.toarray()[:,:shape]

def create_cooler(bins,mat,outf):
    pixels = {'bin1_id':mat.row,'bin2_id':mat.col,'count':mat.data}
    cooler.create_cooler(outf, bins, pixels, dtypes = {'count':'float64'})


def calcProbsSingleChrom(chrom,c,rfc,outf,minDis,maxDis,ncpu):
    print(chrom)

    mat = csr_matrix(np.nan_to_num(np.triu(c.matrix(balance=True,sparse=False).fetch(chrom))))
    #indexMat = getMatIndex(mat.shape[0],maxDis)
    oeMat = calcOEMat(mat,maxDis)
    #neighbors =[x[indexMat].T for x in calcNeighbor(oeMat.toarray())]
    #neighbors = np.stack(neighbors,axis=1)
    neighbors =[flattenHicMat(x,maxDis) for x in calcNeighbor(oeMat.toarray())]
    print('calculating  ...')
    t = time.time()

    #H = mpRunFunc(oeMat,calcEdgeH,minDis,maxDis,ncpu)[indexMat].T
    #V = mpRunFunc(oeMat,calcEdgeV,minDis,maxDis,ncpu)[indexMat].T
    #Std = mpRunFunc(oeMat,calcStd,minDis,maxDis,ncpu)[indexMat].T
    #d = (indexMat[:,0] - indexMat[:,0]).T
    #featureMat = np.stack([neighbors,H,V,Std,d],axis=1)

    '''
      save tmp file
    '''
    #H = mpRunFunc(oeMat,calcEdgeH,minDis,maxDis,ncpu)
    #H = flattenHicMat(H,maxDis)
    #V = mpRunFunc(oeMat,calcEdgeV,minDis,maxDis,ncpu)
    #V = flattenHicMat(V,maxDis)
    #Std = mpRunFunc(oeMat,calcStd,minDis,maxDis,ncpu)
    #Std = flattenHicMat(Std,maxDis)
    #np.savez('tmp.'+chrom+'.npz',H=H,V = V,Std=Std)
    '''
      load tmp file
    '''
    #load_ = np.load('tmp.'+chrom+'.npz')
    #H = load_['H']
    #V = load_['V']
    #Std = load_['Std']
    #d   = np.tile(list(range(maxDis+1)),(mat.shape[0],1)) # distance

    #featureMat = np.stack(neighbors + [H,V,Std,d],axis=2).reshape((mat.shape[0]*(maxDis+1),12),order = 'C')# dim * (maxDis+1) * 12

    #load_ = np.load('featureMat.'+chrom+'.npz')
    #featureMat = load_['features']

    load_ = np.load('featureMat.'+chrom+'.npz')
    results = load_['results']
    probs = load_['probs']
    
    #results = rfc.predict(featureMat)
    #probs = rfc.predict_proba(featureMat)

    probsHicMat = reconstructHicMat(probs[:,1],maxDis)
    create_cooler(c.bins().fetch(chrom),coo_matrix(probsHicMat),'rfcProbs.cool')

    #np.savez('featureMat.'+chrom+'.npz',features=featureMat,results = results,probs=probs)
    print('cost '+ str(time.time()-t)+ 'seconds')

    #bins = c.bins().fetch(chrom)
    #create_cooler(bins,H,'H.cool')
    #create_cooler(bins,V,'V.cool')
    #create_cooler(bins,Std,'Std.cool')


def calcDIDataFrame(df):
    DI = collections.Counter()
    for v in df.iterrows():
        try:
            bin1,bin2,count,balanced = v[1]
        except:
            bin1,bin2,balanced = v[1] # no balance data, use count as balance variable
        bin1 = int(bin1)
        bin2 = int(bin2)

        if np.isnan(balanced):
            balanced = 0
        if bin1 == bin2:
            next
        else:
            DI[bin1] += balanced
            DI[bin2] -= balanced
    return DI

def calcDIBorderRobust(pixels,zeros,window=10,ncpu=8,calcDI_check=False):
    shape = pixels.bin2_id.max() + 1
    zeros = np.delete(zeros,np.where(zeros>=shape)[0]) # avoid index error
    di_sign_sum = np.zeros(shape)
    for w in range(window-2,window+3):
        pixels_ = pixels.query('bin2_id - bin1_id < ' + str(w))
        pixels_ = dd.from_pandas(pixels_,npartitions=2*ncpu)
        ini_dict = pixels_.map_partitions(calcDIDataFrame).compute()
        di = collections.Counter()
        for i in ini_dict:
            di.update(i)
        di_sign = np.array([np.sign(di.get(i) or 0) for i in range(shape)])
        di_sign_sum += di_sign
    di_merged_sign = np.sign(di_sign_sum)
    di_merged_sign = np.delete(di_merged_sign, zeros) # remove zeros

    di_ = di_merged_sign - np.array([0] + di_merged_sign[:-1].tolist())
    check_up = 2
    di_check_up = np.array([np.sum(di_merged_sign[i-check_up:i]) for i in range(di_merged_sign.shape[0])])
    di_check = np.logical_and(di_check_up == -check_up, di_ == 2)
    # border = np.where(di_ == 2)[0]
    border = np.where(di_check)[0]
    border = insert_nans(border,zeros) # correct border position
    border = np.concatenate(([0],border,[shape-1])) # add the start and end bin of each chromosome
    print('border number : '+str(len(border)))

    # below is the di_check_value calculation for final TAD merge
    if calcDI_check:
        check_up = 3
        check_down = 3
        di_check_up_value = np.array([np.sum(di_merged_sign[i-check_up:i]) for i in range(di_merged_sign.shape[0])])
        di_check_down_value = np.array([np.sum(di_merged_sign[i:i+check_down]) for i in range(di_merged_sign.shape[0])])
        di_check_value = np.nan_to_num(di_check_down_value) - np.nan_to_num(di_check_up_value)
        di_check_value = np.array([x if m else 0 for x,m in zip(di_check_value,di_check)]) # for best border alignment
        di_check_value = insert_nans_value(di_check_value,zeros)

        return (border,di_check_value)

    return border

def calcDIBorder(pixels,zeros,chrom,bw1,bw2,window=10,ncpu=8):
    # TODO: should use pixels without empty bins
    pixels = pixels.query('bin2_id - bin1_id < ' + str(window))
    shape = pixels.bin2_id.max() + 1
    zeros = np.delete(zeros,np.where(zeros>=shape))

    pixels = dd.from_pandas(pixels,npartitions=2*ncpu)

    ini_dict = pixels.map_partitions(calcDIDataFrame).compute()
    di = collections.Counter()
    for i in ini_dict:
        di.update(i)

    check_up = 0
    check_down = 0
    di_value = [float(di.get(i) or 0) for i in range(shape)]
    di_diff = np.array(di_value) - np.array([0] + di_value[:-1])
    di_sign = [np.sign(di.get(i) or 0) for i in range(shape)]
    di_check_up = np.array([np.sum(di_sign[i-check_up:i]) for i in range(shape)])
    di_check_down = np.array([np.sum(di_sign[i:i+check_down]) for i in range(shape)])
    di_ = np.array(di_sign) - np.array([0] + di_sign[:-1])
    # border = np.where(np.logical_and(di_ > 0, np.array(di) == 1))[0]
    # border = np.where(di_ == 2)[0]
    di_check = np.logical_and(di_check_up == -check_up, di_check_down == check_down)
    border = np.where(np.logical_and(di_ == 2, di_check))[0]
    # border = np.where(np.logical_and(di_ == 2, di_value > min_di))[0]
    # border = np.where(np.logical_and(di_ == 2, bs < bs_min))[0]
    print('border number:')
    print(len(border))

    bw1.addEntries(chrom,0,values = di_value,span=10000,step=10000)
    bw2.addEntries(chrom,0,values = di_diff ,span=10000,step=10000)

    return border

def calcBorderStrengthDf(df,window):
    segment = collections.Counter()
    for v in df.iterrows():
        try:
            bin1,bin2,count,balanced = v[1]
        except:
            bin1,bin2,balanced = v[1] # no balance data, use count as balance variable
        bin1 = int(bin1)
        bin2 = int(bin2)
        dis = bin2 - bin1
        if(dis <= window+1):
            left  = bin1 + 1
            right = bin1 + dis - 1
        else:
            left = bin1 + dis - window
            right = bin1 + window
        for i in range(left,right+1):
            segment[i] += balanced
    return segment


def calcBorder(pixels,zeros,window=10,ncpu=8):

    pixels = pixels.query('bin2_id - bin1_id < ' + str(window))
    shape = pixels.bin2_id.max() + 1
    zeros = np.delete(zeros,np.where(zeros>=shape))

    pixels = dd.from_pandas(pixels,npartitions=2*ncpu)

    horse = functools.partial(calcBorderStrengthDf,window=window)
    ini_dict = pixels.map_partitions(horse).compute()
    bs = functools.reduce(operator.add,ini_dict)

    bs = [bs.get(i) or 0 for i in range(shape)]
    upStream = np.array(bs[window:2*window] + bs[:(-window)])
    downStream = np.array(bs[window:] + bs[(-2*window):(-window)])
    try:
        bs = upStream + downStream - np.array(bs) # minus or divide?
    except:
        return np.array([])
    bs = np.delete(bs,zeros)

    # border = argrelmin(np.array(bs))[0]
    border = find_peaks(bs, distance = 2, width = 4,)[0]
    # border = find_peaks((-1) * np.array(bs))[0]
    border = insert_nans(border,zeros)
    return border

def getLatentTads(border,minDis=10,maxDis=100):
    tads = itertools.combinations(border,2)
    tads = list(filter(lambda x: x[1]-x[0] < maxDis and x[1]-x[0] > minDis, tads))
    print('find ' + str(len(tads)) + ' potential TADs')
    return tads

def latentDITadsLoop(c,window=10,ncpu=8,maxDis=100,calcDI_check = False,outDIcheck_file='di_check_value.10k'):
    tadLists = []
    if calcDI_check:
        di_checks = []
    for chrom in c.chromnames:
        chromsize = c.chromsizes[chrom]
        offset = c.offset(chrom)
        zeros = remove_nan(c.matrix(balance=False).fetch(chrom))
        try:
            pixels = c.matrix(balance=True, as_pixels=True).fetch(chrom)
        except:
            pixels = c.matrix(balance=False, as_pixels=True).fetch(chrom)
        pixels.bin1_id -= offset
        pixels.bin2_id -= offset
        if pixels.empty:
            continue
        # border = calcDIBorder(pixels,zeros,chrom,bw1,bw2,window,ncpu)
        if calcDI_check:
            border,di_check_value = calcDIBorderRobust(pixels,zeros,window,ncpu,calcDI_check)
            di_check_dataframe = pd.DataFrame({'chr':[chrom]*di_check_value.shape[0],
                                                'bin':np.arange(di_check_value.shape[0]),
                                                'score':di_check_value,
                                            },)
            di_checks.append(di_check_dataframe)
        else:
            border = calcDIBorderRobust(pixels,zeros,window,ncpu,calcDI_check)
        tads = getLatentTads(border,maxDis=maxDis)

        tadLists.append(pd.DataFrame({
            'chrom':chrom,
            'bin1':[x[0] for x in tads],
            'bin2':[x[1] for x in tads],
        }))
    if calcDI_check:
        di_check = pd.concat(di_checks)
        di_check.to_csv(outDIcheck_file,sep='\t',header=0,index=0)
    return pd.concat(tadLists)

# def latentTadsLoop(c,window=10,ncpu=8,maxDis=100):
#     tadLists = []
#     for chrom in c.chromnames:
#         offset = c.offset(chrom)
#         zeros = remove_nan(c.matrix(balance=False).fetch(chrom))
#         try:
#             pixels = c.matrix(balance=True, as_pixels=True).fetch(chrom)
#         except:
#             pixels = c.matrix(balance=False, as_pixels=True).fetch(chrom)
#         pixels.bin1_id -= offset
#         pixels.bin2_id -= offset
#         border = calcBorder(pixels,zeros,window,ncpu)
#         tads = getLatentTads(border,maxDis=maxDis)

#         tadLists.append(pd.DataFrame({
#             'chrom':chrom,
#             'bin1':[int(x[0]) for x in tads],
#             'bin2':[int(x[1]) for x in tads],
#         }))
#     return pd.concat(tadLists)

# function to remove full nan column and row
def remove_nan(sparse_mat):
    """  
    remove full zero/nan column and rows from symmetric sparse matrix
    return
    """
    # non_zero = sparse_mat.getnnz(0) > 0
    try: 
        non_zero = np.nansum(sparse_mat.toarray(),0) > 0
    except AttributeError:
        non_zero = np.nansum(sparse_mat,0) > 0
    # symmetric matrix, so the non-zeros of column and row are same
    # https://stackoverflow.com/questions/31188141/scipy-sparse-matrix-remove-the-rows-whose-all-elements-are-zero
    #M = sparse_mat[non_zero][:,non_zero]
    zero = np.where(np.logical_not(non_zero))[0]
    # return (M,zero)
    return (zero)

def insert_nans_value(value, zeros):
    index = insert_nans(np.arange(value.shape[0]),zeros)
    max_index = np.max(index)
    final_value = np.zeros(max_index+1)
    final_value[index] = value
    return final_value
    
def insert_nans(border, zeros):
    # correct border position
    try:
        # zero_count = 0
        for i in np.nditer(zeros):
            # border[border>=(i-zero_count)] += 1
            border[border>=i] += 1
            # zero_count += 1
    except:
        border = border
    return border



def callTads(mcool,res,prefix,calcDI_check=False,maxDis=2000000):
    # prefix = os.path.basename(mcool).split('.')[0]
    c = cooler.Cooler(mcool+'::/resolutions/'+str(res))
    maxDis = int(maxDis/res)
    if calcDI_check:
        tads = latentDITadsLoop(c,maxDis=maxDis, calcDI_check = calcDI_check, outDIcheck_file = prefix + '.di_check_value')
    else:
        tads = latentDITadsLoop(c,maxDis=maxDis)

    tads_ = tads.copy()
    tads_.bin1 = tads_.bin1.astype('int') * res
    tads_.bin2 = tads_.bin2.astype('int') * res
    bedpe = pd.concat([tads_,tads_],axis=1)
    bedpe.to_csv(prefix + '.latentTads.' + str(res//1000) + 'k.bedpe',sep='\t',header=False,index=False)

    featureDict = calcFeatures(tads,c,maxDis=maxDis,label=False)
    with open(prefix + '.featureDict.' + str(res//1000) + 'k.pickle', 'wb') as handle:
        pickle.dump(featureDict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# def trainModel():
#     with open('/store/wshen/7hic/7hic/dataLabel/labeledData.pkl','rb') as f:
#         data = pickle.load(f)
#     tests = data['features'][-1000:]
#     tests_label = np.array(data['labels'][-1000:])
#     trains = data['features'][:-1000]
#     trains_label = np.array(data['labels'][:-1000])
#     # with open('tests.pkl','rb') as f:
#     #     tests = pickle.load(f)
#     # label_dict = {'T':1,'F':0}

#     # trained_model = trainNeuralNetwork(trains,trains_label,tests,tests_label)
#     trained_model = trainNeuralNetworkWithoutTest(data['features'],np.array(data['labels']))

#     data = pd.read_pickle('/store/wshen/7hic/7hic/src/K562featureDict.80k.pickle')
#     features = torch.Tensor(data['features'])
#     res = trained_model(features)
#     predictions = res.argmax(1)
#     predictTads = data['tads'].copy()[(predictions == 1).tolist()]
#     predictTads.bin1 = predictTads.bin1 * 80000
#     predictTads.bin2 = predictTads.bin2 * 80000
#     bedpe = pd.concat([predictTads,predictTads],axis=1)
#     bedpe.to_csv('k562PredictedTads.80k.bedpe',sep='\t',header=False,index=False)
    # return trained_model
    # trainNeuralNetworkWithoutTest(trains['features'],np.array(trains['labels']))

    #rfc = RandomForestClassifier(n_estimators=20,random_state=0)
    #rfc.fit(trains['features'],trains['labels'])
    #score = rfc.score(tests['features'],tests['labels'])
    #print(score)
    ## save
    #joblib.dump(rfc, "rfc.20.joblib")

def mergeTADRes(tads):
    '''
        merge tads by overlap and select the higher resolution result to save
    '''

def predictTAD(modelfile,featurefile,resolution,amount,outfname='predictTads'):
    from keras.models import load_model

    model = load_model(modelfile)
    res = pd.read_pickle(featurefile)
    x = res['features']
    tads = res['tads']
    predicts = model.predict(x)
    # ind = np.argmax(predicts,axis=1)==1
    ind = np.argpartition(predicts[:,1],-amount)[-amount:]
    predTads = tads.iloc[ind]
    # predTadsProbs1 = predicts[np.argmax(predicts,axis=1)==1][:,0]
    predTadsProbs2 = predicts[ind][:,1]
    predTads.bin1 = predTads.bin1*resolution
    predTads.bin2 = predTads.bin2*resolution
    bedpe = pd.concat([predTads,predTads],axis=1)
    # bedpe['probs1'] = predTadsProbs1
    bedpe['probs2'] = predTadsProbs2
    bedpe.to_csv(outfname + '.' + str(resolution//1000) + 'k.bedpe',sep='\t',header=False,index=False)


def predictTAD_(modelfile,featurefile,resolution,outfname='predictTads'):
    from keras.models import load_model

    model = load_model(modelfile)
    res = pd.read_pickle(featurefile)
    x = res['features']
    tads = res['tads']
    predicts = model.predict(x)
    predTads = tads[np.argmax(predicts,axis=1)==1]
    # predTadsProbs1 = predicts[np.argmax(predicts,axis=1)==1][:,0]
    predTadsProbs2 = predicts[np.argmax(predicts,axis=1)==1][:,1]
    predTads.bin1 = predTads.bin1*resolution 
    predTads.bin2 = predTads.bin2*resolution 
    bedpe = pd.concat([predTads,predTads],axis=1)
    # bedpe['probs1'] = predTadsProbs1
    bedpe['probs2'] = predTadsProbs2
    bedpe.to_csv(outfname + '.' + str(resolution//1000) + 'k.bedpe',sep='\t',header=False,index=False)

def predictTADmultiRes(modelfile_list,featurefile_list,resolution_list = [5000,10000,20000,40000], outfname='predictTads'):
    if len(modelfile_list) == 1:
        modelfile = modelfile_list[0]
        for featurefile,resolution in zip(featurefile_list, resolution_list):
            predictTAD(modelfile,featurefile,resolution,int(20000/resolution*10000),outfname)
    else:
        for modelfile,featurefile,resolution in zip(modelfile_list,featurefile_list, resolution_list):
            predictTAD(modelfile,featurefile,resolution,int(20000/resolution*10000),outfname)

def predictTADmultiRes_(modelfile_list,featurefile_list,resolution_list = [5000,10000,20000,40000], outfname='predictTads'):
    if len(modelfile_list) == 1:
        modelfile = modelfile_list[0]
        for featurefile,resolution in zip(featurefile_list, resolution_list):
            predictTAD(modelfile,featurefile,resolution,outfname)
    else:
        for modelfile,featurefile,resolution in zip(modelfile_list,featurefile_list, resolution_list):
            predictTAD(modelfile,featurefile,resolution,outfname)


def align_boundary(domain_file_list,score_file,out_file,distance=6,res=10000):
    domains = []
    for domain_file in domain_file_list:
        domain_ = pd.read_csv(domain_file,delimiter='\t',names=['chr','start','end',], usecols = [0,1,2],
                                dtype={'chr':str,'start':int,'end':int})
        domains.append(domain_)
    domain = pd.concat(domains)
    domain['start'] = (domain['start']/res).astype(int)
    domain['end']   = (domain['end']/res).astype(int)
    chrlist = domain['chr'].unique()
    score_data = pd.read_csv(score_file,delimiter='\t',names=['chr','bin','score'],
                             dtype={'chr':str,'bin':int,'score':float})
    # final_domain_list = pd.DataFrame(columns=('chr','start','end'))
    final_domain_list = []
    for chrom in chrlist:
        print('now processing chromsome ' + chrom)
        sub_domain_list = domain[domain['chr']==chrom]
        # 5'
        sub_domain_list = merge_boundarys_horse(sub_domain_list,'start',score_data,distance,chrom)
        # 3'
        sub_domain_list = merge_boundarys_horse(sub_domain_list,'end',score_data,distance,chrom)
        
        final_domain_list.append(sub_domain_list)

    final_domain = pd.concat(final_domain_list)
    final_domain.drop_duplicates(inplace=True) # remove duplicates after boundary alignment
    final_domain = final_domain[final_domain['start']<final_domain['end']]
    final_domain['start'] = final_domain['start'] * res
    final_domain['end']   = final_domain['end'] * res
    final_domain.to_csv(out_file, sep = '\t', header = False, index =False) 




def merge_boundarys_horse(sub_domain_list,side,score_data,distance,chrom):
    sub_domain_list = sub_domain_list.sort_values(by=side)  # sort
    require_merge = np.diff(sub_domain_list[side]) <= distance
    value, count = map(list, zip(*((k, len(list(g))) for k, g in itertools.groupby(require_merge))))

    start = [x for x in itertools.accumulate([0]+count[:-1])]
    end   = [x for x in itertools.accumulate(count)]  
    start = [x for x in itertools.compress(start,value)]
    end   = [x+1 for x in itertools.compress(end,value)]

    for i,j in zip(start,end):
        bin_index_s = min(sub_domain_list[side][i:j])
        bin_index_e = max(sub_domain_list[side][i:j])
        #print('distance ',bin_index_e - bin_index_s)
        if bin_index_s == bin_index_e:
            #print('same boundaries, skip')
            next
        # find a bin with lowest score
        try:
            best_bin_ = score_data[(score_data['chr'] == chrom) &  (score_data['bin'] >= bin_index_s) & (score_data['bin'] <= bin_index_e)]['score'].idxmax()
        except ValueError:
            print('out of bin range, skip')
            print('start ',bin_index_s,'end ',bin_index_e)
            print('max bin of current chrom',max(score_data[score_data['chr'] == chrom]['bin']))
            next
        #best_bin_ = score_data[(score_data['chr'] == chrom) &  (score_data['bin'] >= bin_index_s) & (score_data['bin'] <= bin_index_e)]['score'].idxmin()
        best_bin  = score_data['bin'][best_bin_]
        sub_domain_list[side][i:j] = best_bin 

    return(sub_domain_list)


def calcMedianFilter(mat):
    return ndimage.median_filter(mat, size=5)

# def calcInsulation():
# def calcDISignal(pixels,zeros,window=10,ncpu=8):
#     shape = pixels.bin2_id.max() + 1
#     zeros = np.delete(zeros,np.where(zeros>=shape)[0]) # avoid index error
#     di_sign_sum = np.zeros(shape)
#     for w in range(window-2,window+3):
#         pixels_ = pixels.query('bin2_id - bin1_id < ' + str(window))
#         pixels_ = dd.from_pandas(pixels_,npartitions=2*ncpu)
#         ini_dict = pixels_.map_partitions(calcDIDataFrame).compute()
#         di = collections.Counter()
#         for i in ini_dict:
#             di.update(i)
#         di_sign = np.array([np.sign(di.get(i) or 0) for i in range(shape)])
#         di_sign_sum += di_sign
#     di_merged_sign = np.sign(di_sign_sum)
#     di_merged_sign = np.delete(di_merged_sign, zeros) # remove zeros

#     di_ = di_merged_sign - np.array([0] + di_merged_sign[:-1].tolist())
#     check_up = 2
#     di_check_up = np.array([np.sum(di_merged_sign[i-check_up:i]) for i in range(di_merged_sign.shape[0])])
#     di_check = np.logical_and(di_check_up == -check_up, di_ == 2)
#     # border = np.where(di_ == 2)[0]
#     border = np.where(di_check)[0]
#     border = insert_nans(border,zeros) # correct border position
#     border = np.append(border,[0,shape-1]) # add the start and end bin of each chromosome
#     print('border number:')
#     print(len(border))

#     # bw1.addEntries(chrom,0,values = di_value,span=10000,step=10000)
#     # bw2.addEntries(chrom,0,values = di_diff ,span=10000,step=10000)

#     return border

def predictTADbenchmark(modelfileList,featurefile,outLabel=''):
    '''
        model file list should be ordered
    '''
    from keras.models import load_model
    res = pd.read_pickle(featurefile)
    x = res['features']
    tads = res['tads']
    tadLabelList = []
    tadProbList = []
    for modelfile in modelfileList:
        model = load_model(modelfile)
        predicts = model.predict(x)
        tadLabels = np.argmax(predicts,axis=1)
        tadLabelList.append(tadLabels)
        tadProbs = predicts[:,1]/np.sum(predicts,axis=1)
        tadProbList.append(tadProbs)
    tadLabelMat = np.array(tadLabelList).T
    tadProbMat = np.array(tadProbList).T
    # print(tadMat)
    np.savez(outLabel+'_predictsMat.npz',tadMat = tadLabelMat, tadProbMat = tadProbMat)
def plot_predictTADbenchmark(matfileList,labelList,pdffilename):
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    import seaborn as sns
    # create a PdfPages object
    pdf = PdfPages(pdffilename)
    fig1 = plt.figure(figsize=(16,6))
    ax = fig1.add_subplot(111)
    # pdf format, 3 fig for each matfile
    tadMatList = [np.load(matfile)['tadMat'] for matfile in matfileList]
    tadProbMatList = [np.load(matfile)['tadProbMat'] for matfile in matfileList]
    width = 0.2
    pos = -1
    rects = []
    for tadMat,tag in zip(tadMatList,labelList):
        tadCount = np.sum(tadMat,axis=0)
        x = np.arange(tadCount.shape[0])
        # labels = ['round' + str(i+1) for i in x]
        labels = [str(i+1) for i in x]
        rect = ax.bar(x + pos*width, tadCount, width, label=tag)
        rects.append(rect)
        pos += 1

    ax.set_ylabel('TAD number')
    ax.set_xlabel('training round')
    # ax.set_title('Scores by group and gender')
    ax.set_xticks(x, labels)
    ax.set_ylim(2000)
    ax.legend()
    # for rect in rects:
    #     ax.bar_label(rect, padding=3)

    pdf.savefig(fig1)

    # fig2 = plt.figure()
    # ax1 = fig2.add_subplot(131)
    # ax2 = fig2.add_subplot(132)
    # ax3 = fig2.add_subplot(133)
    # axList = [ax1,ax2,ax3]
    # for ax,tadMat,tag in zip(axList,tadMatList,labelList):
    for tadMat,tag in zip(tadMatList,labelList):
        fig = plt.figure(figsize=(16,16))
        ax = fig.add_subplot(111)

        size = tadMat.shape[1]
        overlapMat = np.zeros((size,size))
        for i in range(size):
            for j in range(size):
                overlap = np.logical_and(tadMat[:,i] == 1, tadMat[:,j] == 1).sum()
                overlapRate = overlap/(tadMat[:,i] == 1).sum()
                # overlapMat[i,j] = overlap
                overlapMat[i,j] = round(overlapRate,2)
        im = ax.imshow(overlapMat)
        ax.set_xticks(np.arange(size), labels=range(1,size+1))
        ax.set_yticks(np.arange(size), labels=range(1,size+1))
        ax.set_title(tag)
        for i in range(size):
            for j in range(size):
                text = ax.text(j,i,overlapMat[i,j],ha="center",va="center",color="w",
                fontsize=6)
        fig.colorbar(im,ax=ax,shrink=0.7)
        pdf.savefig(fig)

    for tadMat,tadProbMat,tag in zip(tadMatList,tadProbMatList,labelList):
        # filter out latent TADs
        # tadMat = tadMat[np.sum(tadMat,axis=1)>0,:]
        tadProbMat = tadProbMat[np.sum(tadMat,axis=1)>0,:]
        tadProbMat = tadProbMat[np.sum(tadProbMat,axis=1)>0,:]
        # im = ax.imshow(tadMat, aspect='auto')
        cluster = sns.clustermap(tadProbMat,col_cluster=False,method='ward',cmap='vlag',center=0.5,
        metric='canberra',  yticklabels=False)
        # cluster.cax.set_visible(False)
        ax = cluster.fig.axes[2]
        # ax.set_xticks([x+0.5 for x in range(tadMat.shape[1])],[x+1 for x in range(tadMat.shape[1])])
        ax.set_xticks([x+0.5 for x in range(tadProbMat.shape[1])],[x+1 for x in range(tadProbMat.shape[1])])
        ax.set_title(tag)
        # ax.set_xlabel('training round')
        # cluster.fig.tight_layout()
        # cluster.savefig('heatmap'+tag+'.pdf')
        pdf.savefig(cluster.fig)
    # pdf.savefig(fig2)

    # remember to close the object to ensure writing multiple plots
    pdf.close()

# def calcEdgeStrengthList(tadsList,cList,labels):
#     for tad,c,label in zip(tadsList,cList,labels):
#         edgeS = calcEdgeStrength(tad,c)
#         outf  = label+'_edgeStrength.npy'
#         np.save(outf,edgeS)
        
#     edgeSList = []
#     for label in labels:
#         f  = label+'_edgeStrength.npy'
#         edgeS = np.load(f)
#         edgeSList.append(edgeS)
#     fig, ax = plt.subplots()
#     ax.boxplot(x=edgeSList,labels=labels)
#     fig.savefig('edgeStrengthBoxPlot.pdf')






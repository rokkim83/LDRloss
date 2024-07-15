import numpy as np
import cv2
cimport numpy as np
from libc.stdlib cimport malloc, free
np.import_array()

ctypedef np.uint8_t DTYPE_t
ctypedef np.float32_t DTYPE_f

def getH2d(np.ndarray[DTYPE_t, ndim=2] src):
    cdef long long row = src.shape[0]
    cdef long long col = src.shape[1]
        
    cdef np.ndarray[DTYPE_f, ndim=2] h2d = np.zeros([256, 256], dtype=np.float32)
    cdef np.ndarray[DTYPE_f, ndim=1] hvec = np.zeros([32640, ], dtype=np.float32)
    cdef np.ndarray[DTYPE_f, ndim=1] maskvec = np.zeros([32640, ], dtype=np.float32)
    cdef np.ndarray[DTYPE_f, ndim=1] layerMask = np.zeros([255, ], dtype=np.float32)
    cdef np.ndarray[DTYPE_f, ndim=1] hl


    #cdef int *h2d_array = <int*>malloc(256*256*sizeof(int))
    #cdef double *hvec = <double*>malloc(32640*sizeof(double))
    #cdef double *maskvec = <double*>malloc(32640*sizeof(double))
    #cdef double *layerMask = <double*>malloc(255*sizeof(double))
    #cdef double *hl = <double*>malloc(256*sizeof(double))
    
    cdef int i, j, x, y, vecindx, l, subindx
    cdef double sumhl, maxval
    cdef int ref, trg
    #try:
    for i in range(row):
        for j in range(col):
            ref = src[i, j]
            if i < row - 1:
                trg = src[i+1, j]
                h2d[max(trg, ref), min(trg, ref)] += 1
                #h2d_array[max(trg, ref)*col + min(trg, ref)] += 1
            if j < col - 1:
                trg = src[i, j+1]
                h2d[max(trg, ref), min(trg, ref)] += 1
                #h2d_array[max(trg, ref)*col + min(trg, ref)] += 1
                
    maxval = 0
    vecindx = 0
    
    for l in range(1, 256):
        subindx = 0
        sumhl = 0 
        hl = np.zeros((256-l, ), dtype=np.float32)

        for j in range(l, 256):
            i = j - l
            #hl[subindx] = np.log(h2d_array[j*col+i]+1)
            hl[subindx] = np.log(h2d[j, i]+1)
            h2d[j, 1] = hl[subindx]
            sumhl += hl[subindx]
            subindx += 1

        if sumhl == 0:
            vecindx += 256-l
            continue
        
        layerMask[l-1] = sumhl
        if maxval < sumhl:
            maxval = sumhl

        for j in range(256-l):
            maskvec[vecindx+j] = sumhl
            hvec[vecindx+j] = hl[j] / sumhl
        vecindx += 256 - l
    
    for i in range(255):
        layerMask[i] /= maxval
    for i in range(32640):
        maskvec[i] /= maxval
    
    #return [out_hvec for out_hvec in hvec[:32640]], [out_maskl for out_maskl in layerMask[:255]], [out_maskv for out_maskv in maskvec[:32640]]
    return h2d, hvec, layerMask, maskvec
    #finally:
    #    free(hvec)
    #    free(maskvec)
    #    free(layerMask)
    #    free(hl)
        

def pgetH2d(src):
		
    R = src.shape[0]
    C = src.shape[1]

    h2D_in = np.zeros((256, 256), dtype=np.float32)
    for r in range(0, R):
        for c in range(0, C):
            ref = src[r, c]
            if r < R - 1:
                trg = src[r+1, c]
                h2D_in[max(trg, ref), min(trg, ref)] = h2D_in[max(trg, ref), min(trg, ref)] + 1
            if c < C - 1:
                trg = src[r, c+1]
                h2D_in[max(trg, ref), min(trg, ref)] = h2D_in[max(trg, ref), min(trg, ref)] + 1

    hvec = np.zeros((32640,), dtype=np.float32)
    maskLayer = np.zeros((255, ), dtype=np.float32)
    maskvec = np.zeros((32640, ), dtype=np.float32)

    vecidx = 0
    for layer in range(1, 256):
        hl = np.zeros((256-layer, ), dtype=np.float32)

        tmp_idx = 0
        for j in range(layer, 256):
            i = j - layer
            hl[tmp_idx] = np.log(h2D_in[j, i]+1)
            tmp_idx += 1
        
        sumh = np.sum(hl) # reliability of dl is proportional to hl
        if sumh == 0:
            vecidx = vecidx + 256-layer
            continue

        maskLayer[layer-1] = sumh
        maskvec[vecidx:vecidx + 256-layer] = sumh
        hvec[vecidx:vecidx + 256-layer] = hl/sumh #  kappa = 1 / sum(hl) by eq. (3) and (11)
        vecidx = vecidx + 256-layer
        
    maskLayer = maskLayer / np.max(maskLayer) # by eq. (23)
    maskvec = maskvec / np.max(maskvec)

    return hvec, maskLayer, maskvec

#import time

#st = time.time()
#img = cv2.imread('001.png', cv2.IMREAD_GRAYSCALE)
#getH2d(img)
#print(f'cython time: {time.time()-st}')
#st = time.time()
#pgetH2d(img)
#print(f'python time: {time.time()-st}')
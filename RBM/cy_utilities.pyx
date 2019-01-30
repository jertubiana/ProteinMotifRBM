#!python
#cython: boundscheck=False, wraparound=False,nonecheck=False
# -*- coding: utf-8 -*-
"""
 Copyright 2018 - by Jerome Tubiana (jertubiana@@gmail.com)
     All rights reserved
     
     Permission is granted for anyone to copy, use, or modify this
     software for any uncommercial purposes, provided this copyright 
     notice is retained, and note is made of any changes that have 
     been made. This software is distributed without any warranty, 
     express or implied. In no event shall the author or contributors be 
     liable for any damage arising out of the use of this software.
     
     The publication of research using this software, modified or not, must include 
     appropriate citations to:
"""

import numpy as np
cimport numpy as np

FLOAT = np.float64
INT = np.int

ctypedef np.float64_t FLOAT_t
ctypedef np.int_t INT_t
cimport cython
from libc.math cimport exp



cdef compute_output_pointers(int M, int N, int n_c , int B, double *flat_output, INT_t *flat_config,double *flat_couplings):
    cdef int n,b,m
    cdef int i1 = 0
    cdef int i2 = 0
    cdef int i3 = 0
    for m in range(M):
        for n in range(N):
            i1 = m
            i2 = n
            for b in range(B):
                flat_output[i1] += flat_couplings[ i3+flat_config[i2] ]
                i1+=M
                i2+=N        
            i3 += n_c


cdef compute_output_pointers2(int M, int N, int n_c , int B, double *flat_output, INT_t *flat_config,double *flat_couplings):
    cdef int n,b,m
    cdef int i1 = 0
    cdef int i2 = 0
    cdef int i3 = 0
    for m in range(M):
        for n in range(N):
            i1 = m
            i2 = n
            for b in range(B):
                flat_output[i2] += flat_couplings[ i3+flat_config[i1] ]
                i1+=M
                i2+=N        
            i3 += n_c




cdef dot_Potts_pointers(int N, int n_c , int B, double *output, INT_t *flat_config,double *flat_fields):
    cdef int n,b
    cdef int i1 = 0
    cdef int i2 = 0
    for b in range(B):
        i1 = 0
        for n in range(N):
            output[b] += flat_fields[ i1 + flat_config[i2] ]
            i1 += n_c
            i2 += 1
            
cdef dot_Potts2_pointers(int N, int n_c , int B, double *output, INT_t *flat_config,double *flat_fields):
    cdef int n,b
    cdef int i1 = 0
    cdef int i2 = 0
    for b in range(B):
        for n in range(N):
            output[b] += flat_fields[i2+flat_config[i1]]
            i1+=1
            i2+=n_c            
            
            
cdef average_pointer(INT_t *out, INT_t *X, int B, int N, int n_c):
    cdef int b,n
    cdef int i1 = 0
    cdef int i2 = 0
    for b in range(B):
        i2 = 0
        for n in range(N):    
            out[i2 + X[i1] ] +=1
            i1 +=1
            i2 += n_c
            
cdef weighted_average_pointer(FLOAT_t *out, INT_t *X, FLOAT_t *weights, int B, int N, int n_c):
    cdef int b,n
    cdef int i1 = 0
    cdef int i2 = 0
    for b in range(B):
        i2 = 0
        for n in range(N):    
            out[i2 + X[i1] ] +=weights[b]
            i1 +=1
            i2 += n_c
            

cdef average_product_FxP_pointer(double *X1, INT_t *X2, double *out,int B, int M, int N, int n_c):
    cdef int b,n,c
    cdef int i1 = 0
    cdef int i2 = 0
    cdef int i3 = 0
    for m in range(M):
        for n in range(N):
            i2 = n
            i3 = m
            for b in range(B):
                out[i1+X2[i2]] += X1[i3]
                i2+=N
                i3+=M
            i1+=n_c    

cdef average_product_PxP_pointer(INT_t *X1, INT_t *X2, INT_t *out,int B, int M, int N, int n_c1,int n_c2):
    cdef int b,n,m
    cdef int i1 = 0
    cdef int i2 = 0
    cdef int i3 = 0
    for m in range(M):
        for n in range(N):
            i2 = m
            i3 = n
            for b in range(B):
                out[i1 + n_c2 * X1[i2] + X2[i3]] +=1
                i2 += M
                i3 += N
            i1 += n_c1 * n_c2            

cdef weighted_average_product_PxP_pointer(INT_t *X1, INT_t *X2, FLOAT_t *weights, FLOAT_t *out,int B, int M, int N, int n_c1,int n_c2):
    cdef int b,n,m
    cdef int i1 = 0
    cdef int i2 = 0
    cdef int i3 = 0
    for m in range(M):
        for n in range(N):
            i2 = m
            i3 = n
            for b in range(B):
                out[i1 + n_c2 * X1[i2] + X2[i3]] +=weights[b]
                i2 += M
                i3 += N
            i1 += n_c1 * n_c2    

            

cpdef compute_output_C(int N, int n_cd ,np.ndarray[INT_t,ndim=2,mode="c"] config, np.ndarray[FLOAT_t,ndim=3,mode="c"] couplings): # Direction = up (Second dimension is Potts)
    cdef int M = couplings.shape[0]
    cdef int B = config.shape[0]    
    cdef np.ndarray[FLOAT_t,ndim=2] output = np.zeros([B,M],dtype = FLOAT)
    compute_output_pointers(M,N,n_cd,B,&output[0,0],&config[0,0],&couplings[0,0,0])
    return output

cpdef compute_output_C2(int M, int n_cu ,np.ndarray[INT_t,ndim=2,mode="c"] config, np.ndarray[FLOAT_t,ndim=3,mode="c"] couplings): # Direction = down (First dimension is Potts)
    cdef int N = couplings.shape[1]
    cdef int B = config.shape[0] 
    cdef np.ndarray[FLOAT_t,ndim=2] output = np.zeros([B,N],dtype = FLOAT)
    compute_output_pointers2(M,N,n_cu,B,&output[0,0],&config[0,0],&couplings[0,0,0])
    return output


cpdef compute_output_Potts_C(int N, int n_cd, int n_cu ,np.ndarray[INT_t,ndim=2,mode="c"] config, np.ndarray[FLOAT_t,ndim=4,mode="c"] couplings): # Direction = up
    cdef int M = couplings.shape[0]
    cdef int B = config.shape[0]
    cdef int c
    cdef np.ndarray[FLOAT_t,ndim=3] output = np.zeros([B,M,n_cu],dtype = FLOAT)
    for c in range(n_cu):
        output[:,:,c] = compute_output_C(N,n_cd,config,couplings[:,:,c,:].astype(dtype=FLOAT,order="c"))
    return output

cpdef compute_output_Potts_C2(int M, int n_cu, int n_cd ,np.ndarray[INT_t,ndim=2,mode="c"] config, np.ndarray[FLOAT_t,ndim=4,mode="c"] couplings): # Direction = down
    cdef int N = couplings.shape[1]
    cdef int B = config.shape[0]
    cdef int c
    cdef np.ndarray[FLOAT_t,ndim=3] output = np.zeros([B,N,n_cd],dtype = FLOAT)
    for c in range(n_cd):
        output[:,:,c] = compute_output_C2(M,n_cu,config,couplings[:,:,:,c].astype(dtype=FLOAT,order="c"))
    return output


cpdef dot_Potts_C(int N, int n_c, np.ndarray[INT_t,ndim=2,mode="c"] config, np.ndarray[FLOAT_t,ndim=2,mode="c"] fields):
    cdef int B = config.shape[0]
    cdef np.ndarray[FLOAT_t,ndim=1] output = np.zeros(B,dtype=FLOAT)  
    dot_Potts_pointers(N,n_c,B,&output[0],&config[0,0],&fields[0,0])
    return output

cpdef dot_Potts2_C(int N, int n_c, np.ndarray[INT_t,ndim=2] config, np.ndarray[FLOAT_t,ndim=3] fields):
    cdef int B = config.shape[0]
    cdef np.ndarray[FLOAT_t,ndim=1] output = np.zeros(B,dtype=FLOAT)
    dot_Potts2_pointers(N,n_c,B,&output[0],&config[0,0],&fields[0,0,0])
    return output

cpdef average_C(np.ndarray[INT_t, ndim=2,mode="c"] X, int n_c):
    cdef int B = X.shape[0]
    cdef int N = X.shape[1]
    cdef np.ndarray[INT_t,ndim=2,mode="c"] out = np.zeros([N,n_c],order="c",dtype=INT)
    average_pointer(&out[0,0], &X[0,0],B,N,n_c)
    return np.asarray(out,dtype=FLOAT)/B

cpdef weighted_average_C(np.ndarray[INT_t, ndim=2,mode="c"] X, np.ndarray[FLOAT_t,ndim=1,mode="c"] weights, int n_c):
    cdef int B = X.shape[0]
    cdef int N = X.shape[1]
    cdef np.ndarray[FLOAT_t,ndim=2,mode="c"] out = np.zeros([N,n_c],order="c",dtype=FLOAT)
    weighted_average_pointer(&out[0,0], &X[0,0],&weights[0],B,N,n_c)
    return out/weights.sum()



cpdef average_product_FxP_C(np.ndarray[FLOAT_t, ndim=2,mode="c"] X1, np.ndarray[INT_t,ndim=2,mode="c"] X2,  int n_c):
    cdef int B = X1.shape[0]
    cdef int M = X1.shape[1]
    cdef int N = X2.shape[1]
    cdef np.ndarray[FLOAT_t,ndim=3,mode="c"] out = np.zeros([M,N,n_c],dtype=FLOAT,order="c")
    average_product_FxP_pointer(&X1[0,0],&X2[0,0],&out[0,0,0], B,M,N,n_c)
    return out/B

cpdef average_product_mPxP_C(np.ndarray[FLOAT_t, ndim=3,mode="c"] X1, np.ndarray[INT_t,ndim=2,mode="c"] X2,  int n_c1, int n_c2):
    cdef int B = X1.shape[0]
    cdef int M = X1.shape[1]
    cdef int N = X2.shape[1]
    cdef int i
    cdef np.ndarray[FLOAT_t,ndim=4,mode="c"] out = np.zeros([M,N,n_c1,n_c2],dtype=FLOAT,order="c")
    for c1 in range(n_c1):
        out[:,:,c1,:] = average_product_FxP_C(np.asarray(X1[:,:,c1],order="c"),X2,n_c2)
    return out

cpdef average_product_PxP_C(np.ndarray[INT_t, ndim=2,mode="c"] X1, np.ndarray[INT_t,ndim=2,mode="c"] X2,  int n_c1,int n_c2):
    cdef int B = X1.shape[0]
    cdef int M = X1.shape[1]
    cdef int N = X2.shape[1]
    cdef np.ndarray[INT_t,ndim=4,mode="c"] out = np.zeros([M,N,n_c1,n_c2],dtype=INT,order="c")
    average_product_PxP_pointer(&X1[0,0],&X2[0,0],&out[0,0,0,0], B,M,N,n_c1,n_c2)
    return np.asarray(out,dtype=FLOAT)/B
        

cpdef weighted_average_product_PxP_C(np.ndarray[INT_t, ndim=2,mode="c"] X1, np.ndarray[INT_t, ndim=2,mode="c"] X2,np.ndarray[FLOAT_t, ndim=1,mode="c"] weights,int n_c1,int n_c2):
    cdef int B = X1.shape[0]
    cdef int M = X1.shape[1]
    cdef int N = X2.shape[1]
    cdef np.ndarray[FLOAT_t,ndim=4,mode="c"] out = np.zeros([M,N,n_c1,n_c2],dtype=FLOAT,order="c")
    weighted_average_product_PxP_pointer(&X1[0,0],&X2[0,0],&weights[0],&out[0,0,0,0], B,M,N,n_c1,n_c2)
    return out/weights.sum()
    


cdef tower_sampling_pointer(int B, int N, int n_c, double *flat_cum_probabilities, double *rng):
    cdef int b,n
    cdef int low =0
    cdef int high = n_c
    cdef int middle = 0
    cdef int i1 = 0
    cdef int i2 = 0    
    for b in range(B):
        for n in range(N):
            low = 0
            high = n_c
            while low<high:
                middle = (low+high)/2
                if rng[i1]<flat_cum_probabilities[i2+middle]:
                    high = middle
                else:
                    low = middle +1
            rng[i1] = high
            i1+=1
            i2+=n_c


cpdef tower_sampling_C(int B, int N, int n_c, np.ndarray[FLOAT_t,ndim=3, mode="c"] cum_probabilities, np.ndarray[FLOAT_t,ndim=2, mode="c"] rng):
    tower_sampling_pointer(B,N,n_c,&cum_probabilities[0,0,0], &rng[0,0])
    return np.asarray(rng,dtype=INT)




def Bernoulli_Gibbs_free_C(int B, int N, FLOAT_t beta,np.ndarray[INT_t,ndim=2] x, np.ndarray[FLOAT_t,ndim =2] fields_eff, np.ndarray[FLOAT_t,ndim =1] fields0, np.ndarray[FLOAT_t,ndim=2] couplings, np.ndarray[INT_t,ndim=2] rng1,np.ndarray[FLOAT_t,ndim =2] rng2):
    cdef int b, n,n_
    cdef int pos = 0
    cdef int new = 0
    cdef int previous =0
    for b in range(B):
        for n_ in range(N):
            pos = rng1[b,n_]
            previous = x[b,pos]
            new = rng2[b,n_] < 1/(1+ exp(-beta*fields_eff[b,pos] -(1-beta)*fields0[pos] ))
            if new != previous:
                x[b,pos] = new
                for n in range(N):
                    fields_eff[b,n] += (new-previous)*couplings[n,pos]
    return x,fields_eff


def Spin_Gibbs_free_C(int B, int N, FLOAT_t beta,np.ndarray[INT_t,ndim=2] x, np.ndarray[FLOAT_t,ndim =2] fields_eff, np.ndarray[FLOAT_t,ndim =1] fields0, np.ndarray[FLOAT_t,ndim=2] couplings, np.ndarray[INT_t,ndim=2] rng1,np.ndarray[FLOAT_t,ndim =2] rng2):
    cdef int b, n,n_
    cdef int pos = 0
    cdef int new = 0
    cdef int previous =0
    for b in range(B):
        for n_ in range(N):
            pos = rng1[b,n_]
            previous = x[b,pos]
            new = 2* (rng2[b,n_] < 1/(1+ exp(-2*(beta*fields_eff[b,pos] + (1-beta) * fields0[pos]  )) ) ) -1
            if new != previous:
                x[b,pos] = new
                for n in range(N):
                    fields_eff[b,n] += (new-previous)*couplings[n,pos]
    return x,fields_eff


def Potts_Gibbs_free_C(int B, int N, int n_c,FLOAT_t beta,np.ndarray[INT_t,ndim=2] x, np.ndarray[FLOAT_t,ndim =3] fields_eff, np.ndarray[FLOAT_t,ndim=2] fields0,np.ndarray[FLOAT_t,ndim=4] couplings, np.ndarray[INT_t,ndim=2] rng1,np.ndarray[FLOAT_t,ndim =2] rng2):
    cdef int b, n,n_,c
    cdef int pos = 0
    cdef int new = 0
    cdef int previous =0
    cdef np.ndarray[FLOAT_t,ndim=1] cum_proba = np.zeros(n_c)
    cdef int low = 0
    cdef int high = n_c
    cdef int middle = 0
    for b in range(B):
        for n_ in range(N):
            pos = rng1[b,n_]
            previous = x[b,pos]
            
            for c in range(n_c):
                if c == 0:
                    cum_proba[c] = exp(  beta*fields_eff[b,pos,c] + (1-beta) * fields0[pos,c]) 
                else:
                    cum_proba[c] = cum_proba[c-1]+ exp(  beta*fields_eff[b,pos,c] + (1-beta) * fields0[pos,c]) 
            for c in range(n_c):
                cum_proba[c]/=cum_proba[n_c-1]
                
            low = 0
            high = n_c
            while low<high:
                middle = (low+high)/2
                if rng2[b,n_]<cum_proba[middle]:
                    high = middle
                else:
                    low = middle+1
            new = high
            if new != previous:
                x[b,pos] = new
                for n in range(N):
                    for c in range(n_c):
                        fields_eff[b,n,c] += couplings[n,pos,c,new] - couplings[n,pos,c,previous]
    return x,fields_eff


def Bernoulli_Gibbs_input_C(int B, int N, FLOAT_t beta,np.ndarray[INT_t,ndim=2] x, np.ndarray[FLOAT_t,ndim =2] fields_eff,  np.ndarray[FLOAT_t,ndim =2] inputs, np.ndarray[FLOAT_t,ndim=1] fields0, np.ndarray[FLOAT_t,ndim=2] couplings, np.ndarray[INT_t,ndim=2] rng1,np.ndarray[FLOAT_t,ndim =2] rng2):
    cdef int b, n,n_
    cdef int pos = 0
    cdef int new = 0
    cdef int previous =0
    for b in range(B):
        for n_ in range(N):
            pos = rng1[b,n_]
            previous = x[b,pos]
            new = rng2[b,n_] < 1/(1+ exp( - ( beta*fields_eff[b,pos] + beta*inputs[b,pos] + (1-beta) * fields0[pos])) )
            if new != previous:
                x[b,pos] = new
                for n in range(N):
                    fields_eff[b,n] += (new-previous)*couplings[n,pos]
    return x,fields_eff


def Spin_Gibbs_input_C(int B, int N, FLOAT_t beta,np.ndarray[INT_t,ndim=2] x, np.ndarray[FLOAT_t,ndim =2] fields_eff,  np.ndarray[FLOAT_t,ndim =2] inputs, np.ndarray[FLOAT_t,ndim=1] fields0, np.ndarray[FLOAT_t,ndim=2] couplings, np.ndarray[INT_t,ndim=2] rng1,np.ndarray[FLOAT_t,ndim =2] rng2):
    cdef int b, n,n_
    cdef int pos = 0
    cdef int new = 0
    cdef int previous =0
    for b in range(B):
        for n_ in range(N):
            pos = rng1[b,n_]
            previous = x[b,pos]
            new = 2* (rng2[b,n_] < 1/(1+ exp(-2*( beta* fields_eff[b,pos]+ beta*inputs[b,pos] +(1-beta)*fields0[pos] )) ) ) -1
            if new != previous:
                x[b,pos] = new
                for n in range(N):
                    fields_eff[b,n] += (new-previous)*couplings[n,pos]
    return x,fields_eff


def Potts_Gibbs_input_C(int B, int N, int n_c,FLOAT_t beta,np.ndarray[INT_t,ndim=2] x, np.ndarray[FLOAT_t,ndim =3] fields_eff,np.ndarray[FLOAT_t,ndim =3] inputs, np.ndarray[FLOAT_t,ndim=2] fields0,np.ndarray[FLOAT_t,ndim=4] couplings, np.ndarray[INT_t,ndim=2] rng1,np.ndarray[FLOAT_t,ndim =2] rng2):
    cdef int b, n,n_,c
    cdef int pos = 0
    cdef int new = 0
    cdef int previous =0
    cdef np.ndarray[FLOAT_t,ndim=1] cum_proba = np.zeros(n_c)
    cdef int low = 0
    cdef int high = n_c
    cdef int middle = 0
    for b in range(B):
        for n_ in range(N):
            pos = rng1[b,n_]
            previous = x[b,pos]
            for c in range(n_c):
                if c == 0:
                    cum_proba[c] = exp( beta*fields_eff[b,pos,c]+beta*inputs[b,pos,c] + (1-beta)*fields0[pos,c]) 
                else:
                    cum_proba[c] = cum_proba[c-1]+ exp( beta*fields_eff[b,pos,c]+beta*inputs[b,pos,c] + (1-beta)*fields0[pos,c]) 
            for c in range(n_c):
                cum_proba[c]/=cum_proba[n_c-1]
            low = 0
            high = n_c
            while low<high:
                middle = (low+high)/2
                if rng2[b,n_]<cum_proba[middle]:
                    high = middle
                else:
                    low = middle+1
            new = high
            if new != previous:
                x[b,pos] = new
                for n in range(N):
                    for c in range(n_c):
                        fields_eff[b,n,c] += couplings[n,pos,c,new] - couplings[n,pos,c,previous]
    return x,fields_eff




cdef substitute_pointer(int B,int N, int n_c, double *fields, double *out, INT_t *config):
    cdef int b,n
    cdef int i1=0
    cdef int i2 = 0
    for b in range(B):
        for n in range(N):
            out[i1] = fields[i2 + config[i1]  ]
            i1 +=1
            i2 += n_c
    

cpdef substitute_C(np.ndarray[FLOAT_t,ndim=3,mode="c"] fields, np.ndarray[INT_t,ndim=2,mode="c"] config):
    cdef int B = fields.shape[0]
    cdef int N = fields.shape[1]
    cdef int n_c = fields.shape[2]
    cdef np.ndarray[FLOAT_t,ndim=2,mode="c"] out = np.zeros([B,N],dtype=FLOAT)
    substitute_pointer(B,N,n_c, &fields[0,0,0],&out[0,0], &config[0,0] )
    return out


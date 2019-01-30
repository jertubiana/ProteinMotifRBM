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

import pandas as pd
import numpy as np
import utilities
import sequence_logo
import pickle
import matplotlib
import matplotlib.pyplot as plt
import os
import scipy
import rbm


aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V',  'W', 'Y','-']
aadict = {aa[k]:k for k in range(len(aa))}

aadict['X'] = len(aa)
aadict['B'] = len(aa)
aadict['Z'] = len(aa)
for k,key in enumerate(['a', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v',  'w', 'y']):
    aadict[key] = aadict[aa[k]]
aadict['x'] = len(aa)
aadict['b'] = len(aa)
aadict['z'] = -1
aadict['.'] = -1




def load_FASTA(filename,with_labels=False, remove_insertions = True,drop_duplicates=True):
    count = 0
    current_seq = ''
    all_seqs = []
    if with_labels:
        all_labels = []
    with open(filename,'r') as f:
        for line in f:
            if line[0] == '>':
                all_seqs.append(current_seq)
                current_seq = ''
                if with_labels:
                    all_labels.append(line[1:].replace('\n','').replace('\r',''))
            else:
                current_seq+=line.replace('\n','').replace('\r','')
                count+=1
        all_seqs.append(current_seq)
        all_seqs=np.array(map(lambda x: [aadict[y] for y in x],all_seqs[1:]),dtype=int,order="c")
    
    if remove_insertions:
        all_seqs = np.asarray(all_seqs[:, ((all_seqs == -1).max(0) == False) ],dtype='int',order='c')

    if drop_duplicates:
        all_seqs = pd.DataFrame(all_seqs).drop_duplicates()
        if with_labels:
            all_labels = np.array(all_labels)[all_seqs.index]
        all_seqs = np.array(all_seqs)
    
    if with_labels:
        return all_seqs, np.array(all_labels)
    else:
        return all_seqs

def write_FASTA(filename,all_data,all_labels=None):
    sequences = num2seq(all_data)
    if all_labels is None:
        all_labels = ['S%s'%k for k in range(len(sequences))]
    with open(filename,'wb') as fil:
        for seq, label in zip(sequences,all_labels):
            fil.write('>%s\n'%label)
            fil.write('%s\n'%seq)
    return 'done'


def seq2num(string):
    if type(string) == str:
        return np.array([aadict[x] for x in string])[np.newaxis,:]
    elif type(string) ==list:
        return np.array([[aadict[x] for x in string_] for string_ in string])


def num2seq(num):
    return [''.join([aa[x] for x in num_seq]) for num_seq in num]


def distance(MSA,verbose=False):
    B = MSA.shape[0]
    N = MSA.shape[1]
    distance = np.zeros([B,B])
    for b in range(B):
        if verbose:
            if b%1000 ==0:
                print b
        distance[b] =  ((MSA[b] != MSA).mean(1))
        distance[b,b] = 2.
    return distance

def count_neighbours(MSA,threshold = 0.1): # Compute reweighting
    B = MSA.shape[0]
    N = MSA.shape[1]
    num_neighbours = np.zeros(B)
    for b in range(B):
        if b%1000 ==0:
            print b
        num_neighbours[b] =  ((MSA[b] != MSA).mean(1) < threshold).sum()
    return num_neighbours


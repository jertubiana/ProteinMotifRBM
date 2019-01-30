
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

Probabilistic Graphical Model (PGM) Class.

A general class for any kind of PGM. Has a 'gen_data' method.


TO DO

Methods :
- gen_data
- markov_step

"""



import numpy as np
#import numbers
import utilities
import copy
from scipy.sparse import diags
import time
#import time,copy
#import random
#import itertools
#import matplotlib.pyplot as plt
#from scipy import weave
#import os


def couplings_gradients(W,X1_p,X1_n, X2_p,X2_n, n_c1, n_c2,mean1= False, mean2= False, l1 = 0, l1b = 0, l1c= 0, l2 = 0,l1_custom = None,l1b_custom=None,weights=None,weights_neg=None):
    update =utilities.average_product(X1_p, X2_p, c1 = n_c1,c2=n_c2, mean1 = mean1, mean2= mean2,weights=weights) - utilities.average_product(X1_n, X2_n, c1 = n_c1,c2=n_c2, mean1 = mean1, mean2= mean2,weights=weights_neg)
    if l2>0:
        update -= l2 * W
    if l1>0:
        update -= l1 * np.sign( W)
    if l1b>0: # NOT SUPPORTED FOR POTTS
        if n_c2 > 1: # Potts RBM.
            update -= l1b * np.sign(W) *  np.abs(W).mean(-1).mean(-1)[:,np.newaxis,np.newaxis]
        else:
            update -= l1b * np.sign( W) * (np.abs(W).sum(1))[:,np.newaxis]
    if l1c>0: # NOT SUPPORTED FOR POTTS
        update -= l1c * np.sign( W) * ((np.abs(W).sum(1))**2)[:,np.newaxis]
    if l1_custom is not None:
        update -= l1_custom * np.sign(W)
    if l1b_custom is not None:
        update -= l1b_custom[0] * (l1b_custom[1]* np.abs(W)).mean(-1).mean(-1)[:,np.newaxis,np.newaxis] *np.sign(W)

    if weights is not None:
        update *= weights.sum()/X1_p.shape[0]
    return update





def gauge_adjust_couplings(W,n_c1,n_c2,gauge='zerosum'):
    if gauge == 'zerosum':
        if (n_c1 >1) & (n_c2 >1):
            W =W
        elif (n_c1 ==1) & (n_c2 >1):
            W-= W.sum(-1)[:,:,np.newaxis]/n_c2
        elif (n_c1 >1) & (n_c2 ==1):
            W-= W.sum(-1)[:,:,np.newaxis]/n_c1
    elif gauge=='none':
        return W
    else:
        print 'adjust_couplings -> gauge not supported'
    return W



class PGM(object):
    def __init__(self, n_layers = 3, layers_name =['layer1','layer2','layer3'], layers_size = [100,20,30],layers_nature = ['Bernoulli','Bernoulli','Bernoulli'], layers_n_c = [None,None,None]):
        self.n_layers = n_layers
        self.layers_name = layers_name
        self.layers_size = layers_size
        self.layers_nature = layers_nature
        self.layers_n_c = layers_n_c


    def markov_step(self,config,beta =1):
        return config

    def markov_step_PT(self,config):
        for i,beta in zip(np.arange(self.N_PT), self.betas):
            config[i] = self.markov_step(config[i],beta =beta)
        return config

    def markov_step_and_energy(self, config,E, beta=1):
        return config,E


    def gen_data(self, Nchains = 10, Lchains = 100, Nthermalize = 0 ,Nstep = 1, N_PT =1, config_init = [], beta = 1,batches = None,reshape = True,record_replica = False, record_acceptance=None, update_betas = None,record_swaps = False):
        """
        Generate Monte Carlo samples from the RBM. Starting from random initial conditions, Gibbs updates are performed to sample from equilibrium.
        Inputs :
            Nchains (10): Number of Markov chains
            Lchains (100): Length of each chain
            Nthermalize (0): Number of Gibbs sampling steps to perform before the first sample of a chain.
            Nstep (1): Number of Gibbs sampling steps between each sample of a chain
            N_PT (1): Number of Monte Carlo Exchange replicas to use. This==useful if the mixing rate==slow. Watch self.acceptance_rates_g to check that it==useful (acceptance rates about 0==useless)
            batches (10): Number of batches. Must divide Nchains. higher==better for speed (more vectorized) but more RAM consuming.
            reshape (True): If True, the output==(Nchains x Lchains, n_visibles/ n_hiddens) (chains mixed). Else, the output==(Nchains, Lchains, n_visibles/ n_hiddens)
            config_init ([]). If not [], a Nchains X n_visibles numpy array that specifies initial conditions for the Markov chains.
            beta (1): The inverse temperature of the model.
        """
        if batches==None:
            batches = Nchains
        n_iter = Nchains/batches
        Ndata = Lchains * batches
        if record_replica:
            reshape = False

        if (N_PT>1):
            if record_acceptance==None:
                record_acceptance = True


            if update_betas ==None:
                update_betas = False

            if record_acceptance:
                self.mavar_gamma = 0.95

            if update_betas:
                record_acceptance = True
                self.update_betas_lr = 0.1
                self.update_betas_lr_decay = 1
        else:
            record_acceptance = False
            update_betas = False





        if (N_PT > 1) & record_replica:
            data = [ np.zeros([Nchains,N_PT,Lchains,self.layers_size[i]],dtype= getattr(self,self.layers_name[i]).type ) for i in range(self.n_layers) ]
        else:
            data = [ np.zeros([Nchains,Lchains,self.layers_size[i]],dtype= getattr(self,self.layers_name[i]).type ) for i in range(self.n_layers) ]

        if self.n_layers ==1:
            data = data[0]

        if config_init <> []:
            if type(config_init) == np.ndarray:
                config_init = [config_init.copy()] + [getattr(self,layer).random_init_config(batches) for layer in self.layers_name[1:]]


        for i in range(n_iter):
            if config_init == []:
                config = self._gen_data(Nthermalize, Ndata,Nstep, N_PT = N_PT, batches = batches, reshape = False,beta = beta,record_replica = record_replica, record_acceptance=record_acceptance,update_betas=update_betas,record_swaps=record_swaps)
            else:
                config = self._gen_data(Nthermalize, Ndata,Nstep, N_PT = N_PT, batches = batches, reshape = False,beta = beta,record_replica = record_replica, config_init = [config_init[l][batches*i:batches*(i+1)] for l in range(self.n_layers)],record_acceptance=record_acceptance,update_betas=update_betas,record_swaps=record_swaps)
            if (N_PT > 1) & record_replica:
                if self.n_layers==1:
                    data[batches*i:batches*(i+1),:,:,:] = copy.copy(np.swapaxes(config,0,2))
                else:
                    for l in range(self.n_layers):
                        data[l][batches*i:batches*(i+1),:,:,:] = copy.copy(np.swapaxes(config[l],0,2))
            else:
                if self.n_layers==1:
                    data[batches*i:batches*(i+1),:,:] = copy.copy(np.swapaxes(config,0,1))
                else:
                    for l in range(self.n_layers):
                        data[l][batches*i:batches*(i+1),:,:] = copy.copy(np.swapaxes(config[l],0,1))


        if reshape:
            if self.n_layers==1:
                return data.reshape([Nchains*Lchains,self.layers_size[0]])
            else:
                return [ data[layer].reshape([Nchains*Lchains,self.layers_size[layer]]) for layer in range(self.n_layers)  ]
        else:
            return data

    def _gen_data(self,Nthermalize,Ndata,Nstep, N_PT =1, batches = 1,reshape = True,config_init=[],beta = 1, record_replica = False,record_acceptance = True,update_betas = False,record_swaps = False):
        self.N_PT = N_PT
        if self.N_PT > 1:
            if update_betas | (not hasattr(self,'betas')):
                self.betas =  np.arange(N_PT)/float(N_PT-1) * beta
                self.betas = self.betas[::-1]
            if (len(self.betas) <> N_PT):
                self.betas =  np.arange(N_PT)/float(N_PT-1) * beta
                self.betas = self.betas[::-1]

            self.acceptance_rates = np.zeros(N_PT-1)
            self.mav_acceptance_rates = np.zeros(N_PT-1)
        self.count_swaps = 0
        self.record_swaps = record_swaps
        if self.record_swaps:
            self.particle_id = [np.arange(N_PT)[:,np.newaxis].repeat(batches,axis=1)]

        Ndata/= batches
        if N_PT >1:
            config = [ getattr(self,layer).random_init_config(batches,N_PT=N_PT) for layer in self.layers_name]
            if config_init <> []:
                for l in range(self.n_layers):
                    config[l][0] = config_init[l]
            energy = np.zeros([N_PT,batches])
        else:
            if config_init <> []:
                config = config_init
            else:
                config = [ getattr(self,layer).random_init_config(batches) for layer in self.layers_name]

        if self.n_layers==1:
            config = config[0] #no list



        for _ in range(Nthermalize):
            if N_PT >1:
                #config = self.markov_step_PT(config)
                #config,energy = self.exchange_step_PT(config,energy,record_acceptance=record_acceptance)
                config,energy = self.markov_step_PT2(config,energy)
                config,energy = self.exchange_step_PT(config,energy,record_acceptance=record_acceptance,compute_energy=False)
                if update_betas:
                    self.update_betas(beta=beta)
            else:
                config = self.markov_step(config, beta = beta)
        if self.n_layers==1:
            data = [ utilities.copy_config(config,N_PT=N_PT,record_replica=record_replica)]
        else:
            data = [ [utilities.copy_config(config[l],N_PT=N_PT,record_replica=record_replica)] for l in range(self.n_layers)]


        for _ in range(Ndata-1):
            for _ in range(Nstep):
                    if N_PT > 1:
                        #config = self.markov_step_PT(config)
                        #config,energy = self.exchange_step_PT(config,energy,record_acceptance=record_acceptance)
                        config,energy = self.markov_step_PT2(config,energy)
                        config,energy = self.exchange_step_PT(config,energy,record_acceptance=record_acceptance,compute_energy=False)
                        if update_betas:
                            self.update_betas(beta=beta)
                    else:
                        config = self.markov_step(config, beta = beta)
            if self.n_layers ==1:
                data.append( utilities.copy_config(config,N_PT=N_PT,record_replica=record_replica)  )
            else:
                for l in range(self.n_layers):
                    data[l].append( utilities.copy_config(config[l],N_PT=N_PT,record_replica=record_replica)  )


        if self.record_swaps:
            print 'cleaning particle trajectories'
            positions = np.array(self.particle_id)
            positions = np.array(tmpRBM.particle_id)
            invert = np.zeros([batches,Ndata,N_PT])
            for b in range(batches):
                for i in range(Ndata):
                    for k in range(N_PT):
                        invert[b,i,k] = np.nonzero( positions[i,:,b]==k)[0]
            self.particle_id = invert
            self.last_at_zero = np.zeros([batches,Ndata,N_PT])
            for b in range(batches):
                for i in range(Ndata):
                    for k in range(N_PT):
                        tmp = np.nonzero(self.particle_id[b,:i,k]==0)[0]
                        if len(tmp)>0:
                            self.last_at_zero[b,i,k] = i-1-tmp.max()
                        else:
                            lself.ast_at_zero[b,i,k] = 1000
            self.last_at_zero[:,0,0] = 0
        

            self.trip_duration = np.zeros([batches,Ndata])
            for b in range(batches):
                for i in range(Ndata):
                    self.trip_duration[b,i] = self.last_at_zero[b,i,np.nonzero(invert[b,i,:] == 9)[0]]




        if reshape:
            if self.n_layers==1:
                data= np.array(data).reshape([Ndata*batches,self.layers_size[0]])
            else:
                for l in range(self.n_layers):
                    data[l] = np.array(data[l]).reshape([Ndata*batches,self.layers_size[l]])
        else:
            if self.n_layers==1:
                data = np.array(data)
            else:
                for l in range(self.n_layers):
                    data[l] = np.array(data[l])
        return data



    def update_betas(self,beta=1):
        if self.acceptance_rates.mean()>0:
            self.stiffness = np.maximum(1 - (self.mav_acceptance_rates/self.mav_acceptance_rates.mean()),0) + 1e-4 * np.ones(self.N_PT-1)
            diag = self.stiffness[0:-1] + self.stiffness[1:]
            if self.N_PT>3:
                offdiag_g = -self.stiffness[1:-1]
                offdiag_d = -self.stiffness[1:-1]
                M = diags([offdiag_g,diag,offdiag_d],offsets = [-1,0,1],shape = [self.N_PT -2, self.N_PT-2]).toarray()
            else:
                M = diags([diag],offsets=[0],shape = [self.N_PT -2, self.N_PT-2]).toarray()
            B = np.zeros(self.N_PT-2)
            B[0] = self.stiffness[0] * beta
            self.betas[1:-1] = self.betas[1:-1] * (1- self.update_betas_lr) + self.update_betas_lr *  np.linalg.solve(M,B)
            self.update_betas_lr*= self.update_betas_lr_decay

    def AIS(self,M = 10, n_betas = 10000, batches = None, verbose = 0, beta_type = 'adaptive'):
        if beta_type == 'linear':
            betas = np.arange(n_betas)/float(n_betas-1)
        elif beta_type == 'root':
            betas = np.sqrt( np.arange(n_betas)/float(n_betas-1) )
        elif beta_type == 'adaptive':
            if hasattr(self,'betas'):
                if self.N_PT%2:
                    tmp =1
                else:
                    tmp=0
            else:
                tmp = 0
            if tmp:
                if verbose:
                    print 'Using previously computed betas: %s'%self.betas
                N_PT = len(self.betas)
            else:
                Nthermalize = 200
                Nchains = 20
                N_PT = 11
                self.adaptive_PT_lr = 0.05
                self.adaptive_PT_decay = True
                self.adaptive_PT_lr_decay = 10**(-1/float(Nthermalize))
                if verbose:
                    t = time.time()
                    print 'Learning betas...'
                self.gen_data(N_PT = N_PT, Nchains = Nchains, Lchains =1, Nthermalize = Nthermalize,update_betas = True)
                if verbose:
                    print 'Elapsed time: %s, Acceptance rates: %s'%(time.time() -t, self.mav_acceptance_rates)
            betas = []
            sparse_betas = self.betas[::-1]
            for i in range(N_PT - 1):
                betas+= list( sparse_betas[i] + (sparse_betas[i+1] - sparse_betas[i]) * np.arange(n_betas/(N_PT - 1))/float(n_betas/(N_PT - 1)-1) )
            betas = np.array(betas)
            # if verbose:
                # import matplotlib.pyplot as plt
                # plt.plot(betas); plt.title('Interpolating temperatures');plt.show()

            # Initialization.
        log_weights =  np.zeros(M)
        config = self.gen_data(Nchains=M,Lchains=1,Nthermalize=0,beta=0)
        if self.n_layers ==1:
            fields_eff = self.compute_fields_eff(config)
            config = (config,fields_eff)
        energy = np.zeros(M)

        log_Z_init = np.zeros(1)
        for N,layer in zip(self.layers_size, self.layers_name):
            log_Z_init += getattr(self,layer).logpartition(None, beta=0)

        if verbose:
            print 'Initial evaluation: log(Z) = %s'%log_Z_init

        for i in range(1,n_betas):
            if verbose:
                if (i%2000==0):
                    print 'Iteration %s, beta: %s'%(i,betas[i])
                    print 'Current evaluation: log(Z)= %s +- %s'%( (log_Z_init + log_weights).mean(), (log_Z_init + log_weights).std()/np.sqrt(M))

            config,energy = self.markov_step_and_energy(config,energy,beta=betas[i])
            log_weights += -(betas[i] - betas[i-1]) * energy
        self.log_Z_AIS = (log_Z_init + log_weights).mean()
        self.log_Z_AIS_std = (log_Z_init + log_weights).std()/np.sqrt(M)
        if verbose:
            print 'Final evaluation: log(Z)= %s +- %s'%(self.log_Z_AIS, self.log_Z_AIS_std)
        return self.log_Z_AIS, self.log_Z_AIS_std

    def likelihood(self,data,recompute_Z = False):
        if (not hasattr(self,'log_Z_AIS')) | recompute_Z:
            self.AIS()
        return -self.free_energy(data) - self.log_Z_AIS

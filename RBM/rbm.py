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
import layer
import pgm
import utilities
from utilities import check_random_state,gen_even_slices,logsumexp,log_logistic,average,average_product,saturate
#from scipy.special import erf,erfinv
#from scipy.sparse import diags
import time,copy
import itertools
import batch_norm_utils
#import random
#import itertools
#from scipy import weave
#import os


#%%

class RBM(pgm.PGM):
    def __init__(self,n_v = 100, n_h = 20,visible = 'Bernoulli', hidden='Bernoulli', n_cv =1, n_ch =1, random_state = None, gauge='zerosum',zero_field = False):
        self.n_v = n_v
        self.n_h = n_h
        self.n_visibles = n_v
        self.n_hiddens = n_h
        self.visible = visible
        self.hidden =hidden
        self.random_state = check_random_state(random_state)
        if self.visible == 'Potts':
            self.n_cv = n_cv
        else:
            self.n_cv = 1
        if self.hidden == 'Potts':
            self.n_ch = n_ch
        else:
            self.n_ch = 1

        super(RBM, self).__init__(n_layers = 2, layers_size = [self.n_v,self.n_h],layers_nature = [visible,hidden], layers_n_c = [self.n_cv,self.n_ch] , layers_name = ['vlayer','hlayer'] )


        self.gauge = gauge
        self.zero_field = zero_field
        self.vlayer = layer.Layer(N= self.n_v, nature = self.visible, position = 'visible', n_c = self.n_cv, random_state = self.random_state, zero_field = self.zero_field)
        self.hlayer = layer.Layer(N= self.n_h, nature = self.hidden, position = 'hidden', n_c = self.n_ch, random_state = self.random_state, zero_field = self.zero_field)
        self.init_weights(0.01)
        self.tmp_l2_fields = 0





    def init_weights(self,amplitude):
        if (self.n_ch >1) & (self.n_cv>1):
            self.weights = amplitude * self.random_state.randn(self.n_h, self.n_v,self.n_ch, self.n_cv)
            self.weights = pgm.gauge_adjust_couplings(self.weights,self.n_ch,self.n_cv,gauge=self.gauge)
        elif (self.n_ch >1) & (self.n_cv ==1):
            self.weights = amplitude * self.random_state.randn(self.n_h, self.n_v,self.n_ch)
            self.weights = pgm.gauge_adjust_couplings(self.weights,self.n_ch,self.n_cv,gauge=self.gauge)
        elif (self.n_ch ==1) & (self.n_cv>1):
            self.weights = amplitude * self.random_state.randn(self.n_h, self.n_v,self.n_cv)
            self.weights = pgm.gauge_adjust_couplings(self.weights,self.n_ch,self.n_cv,gauge=self.gauge)
        else:
            self.weights = amplitude * self.random_state.randn(self.n_h, self.n_v)


    def markov_step_PT(self,(v,h)):
        for i,beta in zip(np.arange(self.N_PT), self.betas):
            (v[i,:,:],h[i,:,:]) = self.markov_step((v[i,:,:],h[i,:,:]),beta =beta)
        return (v,h)


    def markov_step_PT2(self,(v,h),E):
        for i,beta in zip(np.arange(self.N_PT), self.betas):
            (v[i,:,:],h[i,:,:]),E[i,:]=  self.markov_step_and_energy((v[i,:,:],h[i,:,:]),E[:,i],beta =beta)
        return (v,h),E


    def exchange_step_PT(self,(v,h),E,record_acceptance=True,compute_energy=True):
        if compute_energy:
            for i in np.arange(self.N_PT):
                E[i,:] = self.energy( v[i,:,:],h[i,:,:],remove_init=True)

        if self.record_swaps:
            particle_id = self.particle_id[-1].copy()
        for i in np.arange(self.count_swaps%2,self.N_PT-1,2):
            proba = np.minimum( 1,  np.exp( (self.betas[i+1]-self.betas[i]) * (E[i+1,:]-E[i,:])   ) )
            swap = np.random.rand(proba.shape[0]) < proba
            if i>0:
                v[i:i+2,swap,:] = v[i+1:i-1:-1,swap ,:]
                h[i:i+2,swap,:] = h[i+1:i-1:-1,swap,:]
                E[i:i+2,swap] = E[i+1:i-1:-1,swap]
                if self.record_swaps:
                    particle_id[i:i+2,swap] = particle_id[i+1:i-1:-1,swap]

            else:
                v[i:i+2,swap,:] = v[i+1::-1,swap,:]
                h[i:i+2,swap,:] = h[i+1::-1,swap,:]
                E[i:i+2,swap] = E[i+1::-1,swap]
                if self.record_swaps:
                    particle_id[i:i+2,swap] = particle_id[i+1::-1,swap]


            if record_acceptance:
                self.acceptance_rates[i] = swap.mean()
                self.mav_acceptance_rates[i] = self.mavar_gamma * self.mav_acceptance_rates[i] +  self.acceptance_rates[i]*(1-self.mavar_gamma)

        if self.record_swaps:
            self.particle_id.append(particle_id)

        self.count_swaps +=1
        return (v,h),E


    def update_betas(self,beta=1):
        super(RBM,self).update_betas(beta=beta)










    def markov_step(self,(v,h),beta =1):
         h = self.hlayer.sample_from_inputs( self.vlayer.compute_output(v,self.weights, direction ='up') , beta = beta )
         v = self.vlayer.sample_from_inputs( self.hlayer.compute_output(h,self.weights,direction='down') , beta = beta )
         return (v,h)

    def markov_step_and_energy(self, (v,h),E, beta=1):
        h = self.hlayer.sample_from_inputs( self.vlayer.compute_output(v,self.weights,direction='up'), beta=beta )
        psi = self.hlayer.compute_output(h,self.weights,direction='down')
        v,E = self.vlayer.sample_and_energy_from_inputs(psi,beta=beta)
        E+= self.hlayer.energy(h,remove_init=True)
        return (v,h),E

    def input_hiddens(self,v):
        if v.ndim ==1: v = v[np.newaxis,:]
        return self.vlayer.compute_output(v,self.weights, direction = 'up')

    def mean_hiddens(self,v):
        if v.ndim ==1: v = v[np.newaxis,:]
        return self.hlayer.mean_from_inputs( self.vlayer.compute_output(v,self.weights, direction = 'up'))

    def mean_visibles(self,h):
        if h.ndim ==1: h = h[np.newaxis,:]
        return self.vlayer.mean_from_inputs( self.hlayer.compute_output(h,self.weights, direction = 'down'))

    def sample_hiddens(self,v):
        if v.ndim ==1: v = v[np.newaxis,:]
        return self.hlayer.sample_from_inputs( self.vlayer.compute_output(v,self.weights, direction = 'up'))

    def sample_visibles(self,h):
        if h.ndim ==1: h = h[np.newaxis,:]
        return self.vlayer.sample_from_inputs( self.hlayer.compute_output(h,self.weights, direction = 'down'))

    def energy(self,v,h,remove_init = False):
        if v.ndim ==1: v = v[np.newaxis,:]
        if h.ndim ==1: h = h[np.newaxis,:]
        return self.vlayer.energy(v,remove_init = remove_init) + self.hlayer.energy(h,remove_init = remove_init) - utilities.bilinear_form(self.weights,h,v,c1=self.n_ch,c2= self.n_cv)


    def free_energy(self,v):
        if v.ndim ==1: v = v[np.newaxis,:]
        return self.vlayer.energy(v) - self.hlayer.logpartition(self.vlayer.compute_output(v,self.weights,direction='up'))




    def free_energy_h(self,h):
        if h.ndim ==1: h = h[np.newaxis,:]
        return self.hlayer.energy(h) - self.vlayer.logpartition(self.hlayer.compute_output(h,self.weights,direction='down'))


    def compute_all_moments(self,from_hidden = True): # Compute all moments for RBMs with small number of hidden units.
        if self.hidden in ['ReLU','Gaussian' ,'ReLU+','dReLU']:
            from_hidden = False

        if from_hidden:
            configurations = utilities.make_all_discrete_configs(self.n_h,self.hidden,c=self.n_ch)
            weights = -self.free_energy_h(configurations)
            maxi = weights.max()
            weights -= maxi
            weights = np.exp(weights)
            Z = weights.sum()
            mean_hiddens = average(configurations,c = self.n_ch,weights = weights)
            mean_visibles = average(self.mean_visibles(configurations),weights = weights)
            covariance = average_product(configurations, self.mean_visibles(configurations),c1 = self.n_ch,c2=self.n_cv, mean1=False,mean2=True,weights=weights)
            Z = Z * np.exp(maxi)
            return Z,mean_visibles,mean_hiddens,covariance
        else:
            configurations = utilities.make_all_discrete_configs(self.n_v,self.visible,c=self.n_cv)
            weights = -self.free_energy(configurations)
            maxi = weights.max()
            weights -= maxi
            weights = np.exp(weights)
            Z = weights.sum()
            mean_visibles =  average(configurations,c = self.n_cv,weights = weights)
            mean_hiddens = average(self.mean_hiddens(configurations),weights = weights)
            covariance = average_product(self.mean_hiddens(configurations),configurations,c1 = self.n_ch,c2=self.n_cv, mean1=True,mean2=False,weights=weights)
            Z = Z * np.exp(maxi)
            return Z,mean_visibles,mean_hiddens,covariance





    def pseudo_likelihood(self,v):
        if self.visible not in ['Bernoulli','Spin','Potts','Bernoulli_coupled','Spin_coupled','Potts_coupled']:
            print 'PL not supported for continuous data'
        else:
            if self.visible == 'Bernoulli':
                ind = (np.arange(v.shape[0]),self.random_state.randint(0, self.n_v, v.shape[0]))
                v_ = v.copy()
                v_[ind] = 1-v[ind]
                fe = self.free_energy(v)
                fe_ = self.free_energy(v_)
                return log_logistic(fe_ - fe)
            elif self.visible =='Spin':
                ind = (np.arange(v.shape[0]),self.random_state.randint(0, self.n_v, v.shape[0]))
                v_ = v.copy()
                v_[ind] = - v[ind]
                fe = self.free_energy(v)
                fe_ = self.free_energy(v_)
                return log_logistic(fe_ - fe)
            elif self.visible =='Potts':
                config = v
                ind_x = np.arange(config.shape[0])
                ind_y = self.random_state.randint(0, self.n_v, config.shape[0])
                E_vlayer_ref = self.vlayer.energy(config) + self.vlayer.fields[ind_y,config[ind_x,ind_y]]
                output_ref = self.vlayer.compute_output(config,self.weights) - self.weights[:,ind_y,config[ind_x,ind_y]].T
                fe = np.zeros([config.shape[0], self.n_cv])
                for c in range(self.n_cv):
                    output = output_ref + self.weights[:,ind_y,c].T
                    E_vlayer = E_vlayer_ref - self.vlayer.fields[ind_y,c]
                    fe[:,c] = E_vlayer-self.hlayer.logpartition(output)
                return - fe[ind_x,config[ind_x,ind_y]] - logsumexp(- fe,1)




    def gen_data(self, Nchains = 10, Lchains = 100, Nthermalize = 0 ,Nstep = 1, N_PT =1, config_init = [], beta = 1,batches = None,reshape = True,record_replica = False, record_acceptance=None, update_betas = None,record_swaps=False):
        return super(RBM,self).gen_data(Nchains = Nchains,Lchains=Lchains,Nthermalize=Nthermalize,Nstep=Nstep, N_PT = N_PT, config_init=config_init, beta =beta, batches = batches,reshape=reshape,record_replica=record_replica,record_acceptance=record_acceptance,update_betas=update_betas,record_swaps=record_swaps)




    def fit(self,data, batch_size = 100, learning_rate = None, extra_params = None, init='independent', optimizer='SGD', batch_norm=None,CD = False,N_PT = 1, N_MC = 1, nchains = None, n_iter = 10,
            lr_decay = True,lr_final=None,decay_after = 0.5,l1 = 0, l1b = 0, l1c=0, l2 = 0,l2_fields =0,no_fields = False,weights = None,
            update_betas =None, record_acceptance = None, shuffle_data = True,epsilon=  1e-6, verbose = 1, record = [],record_interval = 100,data_test = None,weights_test=None,l1_custom=None,l1b_custom=None,M_AIS=10,n_betas_AIS=10000):

        self.batch_size = batch_size
        self.optimizer  = optimizer
        if self.hidden in ['Gaussian','ReLU','ReLU+']:
            if batch_norm is None:
                batch_norm = True
        else:
            if batch_norm is None:
                batch_norm = True
        self.batch_norm = batch_norm
        self.record_swaps = False

        self.n_iter = n_iter
        if self.n_iter == 1:
            lr_decay = False

        if learning_rate is None:
            if self.hidden in ['Bernoulli','Spin','Potts']:
                learning_rate = 0.1
            else:
                if self.batch_norm:
                    learning_rate = 0.1
                else:
                    learning_rate = 0.01

            if self.optimizer == 'ADAM':
                learning_rate *= 0.1

        self.learning_rate_init = copy.copy(learning_rate)
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        if self.lr_decay:
            self.decay_after = decay_after
            self.start_decay = self.n_iter*self.decay_after
            if lr_final is None:
                self.lr_final = 1e-2 * self.learning_rate
            else:
                self.lr_final = lr_final
            self.decay_gamma = (float(self.lr_final)/float(self.learning_rate))**(1/float(self.n_iter* (1-self.decay_after) ))

        self.no_fields = no_fields
        self.gradient = self.initialize_gradient_dictionary()
        self.do_grad_updates = self.initialize_do_grad_updates()

        if self.optimizer =='momentum':
            if extra_params is None:
                extra_params = 0.9
            self.momentum = extra_params
            self.previous_update = self.initialize_gradient_dictionary()



        elif self.optimizer == 'ADAM':
            if extra_params is None:
                extra_params = [0.9, 0.999, 1e-8]
            self.beta1 = extra_params[0]
            self.beta2 = extra_params[1]
            self.epsilon = extra_params[2]

            self.gradient_moment1 = self.initialize_gradient_dictionary()
            self.gradient_moment2 = self.initialize_gradient_dictionary()



        data = np.asarray(data,dtype=self.vlayer.type,order="c")
        if self.batch_norm:
            self.mu_data = utilities.average(data,c=self.n_cv,weights=weights)

#        if self.visible == 'Bernoulli':
#            data = np.asarray(data,dtype=float)

        n_samples = data.shape[0]
        n_batches = int(np.ceil(float(n_samples) / self.batch_size))
        batch_slices = list(gen_even_slices(n_batches * self.batch_size,
                                            n_batches, n_samples))


        if init <> 'previous':
            if self.hidden in ['Bernoulli','Spin','Potts']:
                norm_init = 0.01
            else:
                norm_init = np.sqrt(0.1/self.n_v)

            self.init_weights(norm_init)
            if init=='independent':
                self.vlayer.init_params_from_data(data,eps=epsilon,weights=weights)



        self.N_PT = N_PT
        self.N_MC = N_MC
        if N_MC == 0:
            if self.n_cv > 1:
                nchains = (self.n_cv)**self.n_v
            else:
                nchains = 2**self.n_v

        if nchains is None:
            self.nchains = self.batch_size
        else:
            self.nchains = nchains

        self.CD = CD
        self.l1= l1
        self.l1b = l1b
        self.l1c = l1c
        self.l1_custom = l1_custom
        self.l1b_custom = l1b_custom
        self.l2 = l2
        self.tmp_l2_fields = l2_fields



        if self.N_PT>1:
            if record_acceptance==None:
                record_acceptance = True
            self.record_acceptance = record_acceptance

            if self.N_PT >2:
                if update_betas ==None:
                    update_betas = True
            else:
                update_betas = False

            self._update_betas = update_betas

            if self.record_acceptance:
                self.mavar_gamma = 0.95
                self.acceptance_rates = np.zeros(N_PT-1)
                self.mav_acceptance_rates = np.zeros(N_PT-1)
            self.count_swaps = 0

            if self._update_betas:
                record_acceptance = True
                self.update_betas_lr = 0.1
                self.update_betas_lr_decay = 1

            if self._update_betas | (not hasattr(self,'betas')):
                self.betas =  np.arange(N_PT)/float(N_PT-1)
                self.betas = self.betas[::-1]
            if (len(self.betas) <> N_PT):
                self.betas =  np.arange(N_PT)/float(N_PT-1)
                self.betas = self.betas[::-1]





        if self.N_PT > 1:
            self.fantasy_v = self.vlayer.random_init_config(self.nchains*self.N_PT).reshape([self.N_PT,self.nchains,self.vlayer.N])
            self.fantasy_h = self.hlayer.random_init_config(self.nchains*self.N_PT).reshape([self.N_PT,self.nchains,self.hlayer.N])
            self.fantasy_E = np.zeros([self.N_PT,self.nchains])
        else:
            if self.N_MC == 0:
                self.fantasy_v = utilities.make_all_discrete_configs(self.n_v,self.visible,c=self.n_cv)
            else:
                self.fantasy_v = self.vlayer.random_init_config(self.nchains)
                self.fantasy_h = self.hlayer.random_init_config(self.nchains)



        if shuffle_data:
            permute = np.arange(data.shape[0])
            self.random_state.shuffle(permute)
            data = data[permute,:]            
            if weights is not None:
                weights = weights[permute]


        if weights is not None:
            weights = np.asarray(weights,dtype='float')
            weights/=weights.mean()

        self.count_updates = 0
        if verbose:
            if weights is not None:
                lik = (self.pseudo_likelihood(data) * weights).sum()/weights.sum()
            else:
                lik = self.pseudo_likelihood(data).mean()
            print 'Iteration number 0, pseudo-likelihood: %.2f'%lik


        result = {}
        if 'W' in record:
            result['W'] = []
        if 'FV' in record:
            result['FV'] = []
        if 'FH' in record:
            result['FH'] = []
        if 'TH' in record:
            result['TH'] = []
        if 'B' in record:
            result['B'] = []
        if 'beta' in record:
            result['beta'] = []
        if 'p' in record:
            result['p'] = []
        if 'PL' in record:
            result['PL'] = []
        if 'PL_test' in record:
            result['PL_test'] = []
        if 'L' in record:
            result['L'] = []
        if 'L_test' in record:
            result['L_test'] = []
        if 'AP' in record:
            result['AP'] =[]
        if 'AM' in record:
            result['AM'] = []
        if 'A' in record:
            result['A'] = []
        if 'ETA' in record:
            result['ETA'] = []
        if 'AP0' in record:
            result['AP0'] = []
        if 'AM0' in record:
            result['AM0'] = []
        if 'TAU' in record:
            result['TAU'] = []
            if self.N_PT>1:
                current = np.argmax(MOI.expectation(self.fantasy_v[0]),axis=-1)
            else:
                current = np.argmax(MOI.expectation(self.fantasy_v),axis=-1)
            joint_z = np.zeros([self.zlayer.n_c,self.zlayer.n_c])
                                    



        count = 0

        for epoch in xrange(1,n_iter+1):
            if verbose:
                begin = time.time()
            if self.lr_decay:
                if (epoch>self.start_decay):
                    self.learning_rate*= self.decay_gamma

            print 'Starting epoch %s'%(epoch)
            for batch_slice in batch_slices:
                if weights is None:
                    self.minibatch_fit(data[batch_slice],weights=None)
                else:
                    self.minibatch_fit(data[batch_slice],weights=weights[batch_slice])
                if np.isnan(self.weights).max():
                    print 'NAN in weights. Breaking'
                    return result

                if 'TAU' in record:
                    if self.N_PT>1:
                        current = np.argmax(MOI.expectation(self.fantasy_v[0]),axis=-1)
                    else:
                        current = np.argmax(MOI.expectation(self.fantasy_v),axis=-1)
                    joint_z = 0.1 * utilities.average_product(previous,current,c1=self.zlayer.n_c,c2=self.zlayer.n_c)[0,0] + 0.9 * joint_z
                    previous = current.copy()

                if (count%record_interval ==0):
                    if 'W' in record:
                        result['W'].append( self.weights.copy() )
                    if 'FV' in record:
                        result['FV'].append(self.vlayer.fields.copy())
                    if 'FH' in record:
                        result['FH'].append(self.hlayer.fields.copy())
                    if 'TH' in record:
                        if self.hidden == 'ReLU':
                            result['TH'].append( (self.hlayer.theta_plus + self.hlayer.theta_minus)/(2*self.hlayer.a) )
                        elif self.hidden == 'ReLU+':
                            result['TH'].append( (self.hlayer.theta_plus - self.hlayer.mu_psi)/self.hlayer.a)
                        else:
                            result['TH'].append(  (1-self.hlayer.eta**2)/2 * (self.hlayer.theta_plus + self.hlayer.theta_minus)/self.hlayer.a )
                    if 'PL' in record:
                        result['PL'].append( utilities.average(self.pseudo_likelihood(data),weights=weights) )
                    if 'PL_test' in record:
                        result['PL_test'].append( utilities.average(self.pseudo_likelihood(data_test),weights=weights_test) )
                    if 'L' in record:
                        N_PT = copy.copy(self.N_PT)
                        # Z,_,_,_ = self.compute_all_moments()
                        # self.log_Z_AIS = np.log(Z)
                        self.AIS(M=M_AIS,n_betas=n_betas_AIS,verbose=0)
                        self.N_PT = N_PT
                        result['L'].append(utilities.average(self.likelihood(data,recompute_Z=False),weights=weights))
                    if 'L_test' in record:
                        result['L_test'].append(utilities.average(self.likelihood(data_test,recompute_Z=False),weights=weights_test))
                    if 'beta' in record:
                        if self.n_cv >1:
                            result['beta'].append( (self.weights**2).sum(-1).sum(-1) )
                        else:
                            result['beta'].append( (self.weights**2).sum(-1) )
                    if 'p' in record:
                        if self.n_cv >1:
                            tmp = (self.weights**2).sum(-1)
                        else:
                            tmp = (self.weights**2)
                        a = 3
                        result['p'].append(  (tmp**a).sum(-1)**2/(tmp**(2*a)).sum(-1)/self.n_v  )
                    if 'AP' in record:
                        result['AP'].append(self.hlayer.a_plus.copy())
                    if 'AM' in record:
                        result['AM'].append(self.hlayer.a_minus.copy())
                    if 'A' in record:
                        result['A'].append(self.hlayer.a.copy())
                    if 'B' in record:
                        result['B'].append(self.hlayer.b.copy())
                    if 'ETA' in record:
                        result['ETA'].append(self.hlayer.eta.copy())
                    if 'AP0' in record:
                        result['AP0'].append(self.hlayer.a_plus0.copy())
                    if 'AM0' in record:
                        result['AM0'].append(self.hlayer.a_minus0.copy())
                    if 'TAU' in record:
                        Q = joint_z/joint_z.sum(0)[np.newaxis,:]
                        lam,v  = np.linalg.eig(Q)
                        lam = lam[np.argsort(np.abs(np.real(lam)))[::-1]]
                        tau = -1/np.log(np.abs(np.real(lam[1])))
                        result['TAU'].append(tau.copy())


                count +=1

            if verbose:
                end = time.time()
                if weights is not None:
                    lik = (self.pseudo_likelihood(data) * weights).sum()/weights.sum()
                else:
                    lik = self.pseudo_likelihood(data).mean()

                print("[%s] Iteration %d, pseudo-likelihood = %.2f,"
                      " time = %.2fs"
                      % (type(self).__name__, epoch,
                         lik, end - begin))

            if shuffle_data:
                if weights is not None:
                    permute = np.arange(data.shape[0])
                    self.random_state.shuffle(permute)
                    weights = weights[permute]
                    data = data[permute,:]
                else:
                    self.random_state.shuffle(data)

        return result



    def minibatch_fit(self,V_pos,weights = None):
        self.count_updates +=1
        if self.CD: # Contrastive divergence: initialize the Markov chain at the data point.
            self.fantasy_v = V_pos
        # Else: use previous value.
        for _ in range(self.N_MC):
            if self.N_PT>1:
                (self.fantasy_v,self.fantasy_h),self.fantasy_E = self.markov_step_PT2((self.fantasy_v,self.fantasy_h),self.fantasy_E)
                (self.fantasy_v,self.fantasy_h),self.fantasy_E = self.exchange_step_PT((self.fantasy_v,self.fantasy_h),self.fantasy_E,record_acceptance=self.record_acceptance,compute_energy=False)
                if self._update_betas:
                    self.update_betas()

            else:
                (self.fantasy_v,self.fantasy_h) = self.markov_step((self.fantasy_v,self.fantasy_h) )

        if self.N_PT>1:
            V_neg = self.fantasy_v[0,:,:]
        else:
            V_neg = self.fantasy_v

        if self.N_MC>0: # No Monte Carlo. Compute exhaustively the moments using all 2**N configurations.
            weights_neg = None
        else:
            F = self.free_energy(V_neg)
            F -= F.min()
            weights_neg = np.exp(-F)
            weights_neg /= weights_neg.sum()


        psi_pos = self.vlayer.compute_output(V_pos,self.weights)
        psi_neg = self.vlayer.compute_output(V_neg,self.weights)

        if self.batch_norm:
            if (self.n_cv >1) & (self.n_ch  == 1):
                mu_psi = np.tensordot(self.weights,self.mu_data,axes = [(1,2),(0,1)])
            elif (self.n_cv >1) & (self.n_ch  > 1):
                mu_psi = np.tensordot(self.weights,self.mu_data,axes = [(1,3),(0,1)])
            elif (self.n_cv  == 1) & (self.n_ch  > 1):
                mu_psi = np.tensordot(self.weights,self.mu_data,axes = [1,0])
            else:
                mu_psi = np.dot(self.weights,self.mu_data)
            delta_mu_psi = (mu_psi-self.hlayer.mu_psi)
            self.hlayer.mu_psi += delta_mu_psi
            if self.hidden in ['Bernoulli','Spin','Potts']:
                self.hlayer.fields -= delta_mu_psi
            elif self.hidden == 'Gaussian':
                self.hlayer.b += delta_mu_psi
            elif self.hidden == 'ReLU+':
                self.hlayer.b += delta_mu_psi
                self.hlayer.theta_plus = self.hlayer.b
            elif self.hidden in ['ReLU','dReLU']:
                self.hlayer.b += delta_mu_psi
                self.hlayer.theta_plus += delta_mu_psi
                self.hlayer.theta_minus -= delta_mu_psi

            if self.hidden in ['Gaussian','ReLU','dReLU','ReLU+']:
                if self.hidden == 'Gaussian':
                    var_e = utilities.average(psi_pos**2,weights=weights) - utilities.average(psi_pos,weights=weights)**2
                    mean_v = np.zeros(self.n_h)
                elif self.hidden in ['ReLU','dReLU','ReLU+']:
                    e = self.hlayer.mean_from_inputs(psi_pos) * self.hlayer.a[np.newaxis,:]
                    v = (self.hlayer.var_from_inputs(psi_pos) * self.hlayer.a[np.newaxis,:]-1)
                    var_e = utilities.average(e**2,weights=weights) - utilities.average(e,weights=weights)**2
                    mean_v = utilities.average(v,weights=weights)

                new_a = (1+mean_v+np.sqrt( (1+mean_v)**2+4*var_e))/2
                delta_a = self.learning_rate/self.learning_rate_init * (new_a  - self.hlayer.a )
                constraint = np.maximum(-self.hlayer.a/4,0.05-self.hlayer.a) # a cannot go below 0.05; a_new >= 0.75 * a.
                correction = 1/np.maximum(delta_a/constraint,1)
                correction[constraint==0] = 1.0 * (delta_a[constraint==0]>0) # Correction of np bug.
                delta_a *= correction
                self.hlayer.a += delta_a

                # delta_a = np.maximum(delta_a, -self.hlayer.a/4) # cannot reduce by more than a factor 3/4; in case of rare feature
                # self.hlayer.a += delta_a
                # self.hlayer.a = np.maximum(self.hlayer.a,0.05) # Minimum a: 0.05 (for very rare features).

                if self.hidden == 'dReLU':
                    self.hlayer.a_plus = self.hlayer.a/(1+self.hlayer.eta)
                    self.hlayer.a_minus = self.hlayer.a/(1-self.hlayer.eta)


        H = self.hlayer.mean_from_inputs(psi_pos)
        H_neg = self.hlayer.mean_from_inputs(psi_neg)

        self.gradient['vlayer'] = self.vlayer.internal_gradients(V_pos,V_neg,weights=weights,weights_neg=weights_neg,value='data')
        self.gradient['hlayer'] = self.hlayer.internal_gradients(psi_pos,psi_neg,weights=weights,weights_neg=weights_neg,value='input')
        self.gradient['W'] = pgm.couplings_gradients(self.weights,H,H_neg,V_pos,V_neg,self.n_ch, self.n_cv, mean1 = True, l1 = self.l1, l1b = self.l1b, l1c = self.l1c, l2 = self.l2,weights=weights,weights_neg=weights_neg,l1_custom=self.l1_custom,l1b_custom=self.l1b_custom)

        if self.batch_norm: # Modify gradients.
            if self.hidden in ['Bernoulli','Spin','Potts']:
                df_dw = -average(V_pos,c=self.n_cv,weights=weights)

                if (self.n_cv >1) & (self.n_ch == 1):
                    self.gradient['W'] += df_dw[np.newaxis,:,:] * self.gradient['hlayer']['fields'][:,np.newaxis,np.newaxis]
                elif (self.n_cv == 1) & (self.n_ch > 1):
                    self.gradient['W'] += df_dw[np.newaxis,:,np.newaxis] * self.gradient['hlayer']['fields'][:,:,np.newaxis]
                elif (self.n_cv > 1) & (self.n_ch > 1):
                    self.gradient['W'] += df_dw[np.newaxis,:,np.newaxis,:] * self.gradient['hlayer']['fields'][:,np.newaxis,:,np.newaxis]
                else:
                    self.gradient['W'] += df_dw[np.newaxis,:] * self.gradient['hlayer']['fields'][:,np.newaxis]



            elif self.hidden in ['Gaussian','ReLU','ReLU+','dReLU']:
                if self.hidden == 'Gaussian':
                    db_dw,da_db, da_dw = batch_norm_utils.get_cross_derivatives_Gaussian(V_pos,psi_pos, self.hlayer, self.n_cv,weights=weights)
                    self.gradient['hlayer']['b'] += self.gradient['hlayer']['a'] * da_db

                elif self.hidden == 'ReLU':
                    db_dw,da_db, da_dtheta, da_dw = batch_norm_utils.get_cross_derivatives_ReLU(V_pos,psi_pos, self.hlayer, self.n_cv,weights=weights)
                    self.gradient['da_db'] = da_db
                    self.gradient['da_dtheta'] = da_dtheta
                    self.gradient['hlayer']['b'] += self.gradient['hlayer']['a'] * da_db
                    self.gradient['hlayer']['theta'] += self.gradient['hlayer']['a'] * da_dtheta

                elif self.hidden == 'ReLU+':
                    db_dw,da_db, da_dw = batch_norm_utils.get_cross_derivatives_ReLU_plus(V_pos,psi_pos, self.hlayer, self.n_cv,weights=weights)
                    self.gradient['hlayer']['b'] += self.gradient['hlayer']['a'] * da_db
                    self.gradient['hlayer']['b'] = saturate(self.gradient['hlayer']['b'],1.0/self.learning_rate)

                elif self.hidden == 'dReLU':
                    db_dw,da_db, da_dtheta, da_deta, da_dw = batch_norm_utils.get_cross_derivatives_dReLU(V_pos,psi_pos, self.hlayer, self.n_cv,weights=weights)
                    da_db *=  self.learning_rate/self.learning_rate_init * correction
                    da_dtheta *=  self.learning_rate/self.learning_rate_init * correction
                    da_deta *=  self.learning_rate/self.learning_rate_init * correction
                    if self.n_cv >1:
                        da_dw *=  self.learning_rate/self.learning_rate_init * correction[:,np.newaxis,np.newaxis]
                    else:
                        da_dw *=  self.learning_rate/self.learning_rate_init * correction[:,np.newaxis]

                    self.gradient['hlayer']['b'] += self.gradient['hlayer']['a'] * da_db
                    self.gradient['hlayer']['theta'] += self.gradient['hlayer']['a'] * da_dtheta
                    self.gradient['hlayer']['eta'] += self.gradient['hlayer']['a'] * da_deta
                    self.gradient['hlayer']['theta'] = saturate(self.gradient['hlayer']['theta'],1.0)
                    self.gradient['hlayer']['b'] = saturate(self.gradient['hlayer']['b'],1.0)
                    self.gradient['hlayer']['eta'] = saturate(self.gradient['hlayer']['eta'],1.0)

                if (self.n_cv >1) & (self.n_ch == 1):
                    self.gradient['W'] += db_dw[np.newaxis,:,:] * self.gradient['hlayer']['b'][:,np.newaxis,np.newaxis] + da_dw * self.gradient['hlayer']['a'][:,np.newaxis,np.newaxis]
                elif (self.n_cv == 1) & (self.n_ch > 1):
                    self.gradient['W'] += db_dw[np.newaxis,:,np.newaxis] * self.gradient['hlayer']['b'][:,:,np.newaxis]
                elif (self.n_cv > 1) & (self.n_ch > 1):
                    self.gradient['W'] += db_dw[np.newaxis,:,np.newaxis,:] * self.gradient['hlayer']['b'][:,np.newaxis,:,np.newaxis]
                else:
                    self.gradient['W'] += db_dw[np.newaxis,:] * self.gradient['hlayer']['b'][:,np.newaxis] + da_dw * self.gradient['hlayer']['a'][:,np.newaxis]

        self.gradient['W'] = saturate(self.gradient['W'],1.0)
        if self.tmp_l2_fields>0:
            self.gradient['vlayer']['fields'] -= self.tmp_l2_fields *  self.vlayer.fields
        for layer_ in ['vlayer','hlayer']:
            for internal_param,gradient in self.gradient[layer_].items():
                if self.do_grad_updates[layer_][internal_param]:
                    current = getattr(getattr(self,layer_),internal_param)
                    if self.optimizer == 'SGD':
                        new = current + self.learning_rate * gradient
                    elif self.optimizer == 'momentum':
                        self.previous_update[layer_][internal_param] =  (1- self.momentum) * self.learning_rate * gradient + self.momentum * self.previous_update[layer_][internal_param]
                        new = current + self.previous_update[layer_][internal_param]
                    elif self.optimizer == 'ADAM':
                        self.gradient_moment1[layer_][internal_param] = (1- self.beta1) * gradient + self.beta1 * self.gradient_moment1[layer_][internal_param]
                        self.gradient_moment2[layer_][internal_param] = (1- self.beta2) * gradient**2 + self.beta2 * self.gradient_moment2[layer_][internal_param]

                        new = current + self.learning_rate * (self.gradient_moment1[layer_][internal_param]/(1-self.beta1**self.count_updates)) /(self.epsilon + np.sqrt( self.gradient_moment2[layer_][internal_param]/(1-self.beta2**self.count_updates ) ) )

                    setattr( getattr(self,layer_),internal_param, new )

        if self.do_grad_updates['W']:
            if self.optimizer == 'SGD':
                if (self.n_cv>1) | (self.n_ch >1):
                    self.weights += self.learning_rate * pgm.gauge_adjust_couplings(self.gradient['W'],self.n_ch,self.n_cv,gauge=self.gauge)
                else:
                    self.weights += self.learning_rate * self.gradient['W']

            elif self.optimizer == 'momentum':
                self.previous_update['W'] = (1-self.momentum) * self.learning_rate * self.gradient['W'] + self.momentum * self.previous_update['W']
                if (self.n_cv>1) | (self.n_ch >1):
                    update = pgm.gauge_adjust_couplings(self.previous_update['W'],self.n_ch,self.n_cv,gauge=self.gauge)
                else:
                    update = self.previous_update['W']
                self.weights += update

            elif self.optimizer == 'ADAM':
                self.gradient_moment1['W'] = (1- self.beta1) * self.gradient_moment1['W'] + self.beta1 * self.gradient['W']
                self.gradient_moment2['W'] = (1- self.beta2) * self.gradient_moment2['W'] + self.beta2 * self.gradient['W']**2
                update = self.learning_rate * (self.gradient_moment1['W']/(1-self.beta1**self.count_updates))/(self.epsilon + np.sqrt( self.gradient_moment2['W']/(1-self.beta2**self.count_updates  )))


                if (self.n_cv>1) | (self.n_ch >1):
                    update = pgm.gauge_adjust_couplings(update,self.n_ch,self.n_cv,gauge=self.gauge)
                self.weights += update

        if self.hidden =='ReLU+':
            self.hlayer.theta_plus = self.hlayer.b

        elif self.hidden == 'ReLU':
            self.hlayer.theta_plus = self.hlayer.theta + self.hlayer.b
            self.hlayer.theta_minus = self.hlayer.theta - self.hlayer.b

        elif self.hidden =='dReLU':
            self.hlayer.eta =  np.sign(self.hlayer.eta) * np.minimum(np.abs(self.hlayer.eta),0.95)
            self.hlayer.a_plus = self.hlayer.a/(1+self.hlayer.eta)
            self.hlayer.a_minus = self.hlayer.a/(1-self.hlayer.eta)
            self.hlayer.theta_plus = self.hlayer.b + self.hlayer.theta/(1+self.hlayer.eta)
            self.hlayer.theta_minus = -self.hlayer.b + self.hlayer.theta/(1-self.hlayer.eta)
            self.hlayer.a_plus0 = np.maximum(self.hlayer.a_plus0,0.05)
            self.hlayer.a_minus0 = np.maximum(self.hlayer.a_minus0,0.05)







    def initialize_gradient_dictionary(self):
        out = {}
        out['vlayer'] = self.vlayer.internal_gradients(np.zeros([1,self.n_v],dtype=self.vlayer.type), np.zeros([1,self.n_v],dtype=self.vlayer.type) )
        out['hlayer'] = self.hlayer.internal_gradients(np.zeros([1,self.n_h],dtype=self.hlayer.type), np.zeros([1,self.n_h],dtype=self.hlayer.type) )
        out['W'] = pgm.couplings_gradients(self.weights,np.zeros([1,self.n_h],dtype=self.hlayer.type),np.zeros([1,self.n_h],dtype=self.hlayer.type),np.zeros([1,self.n_v],dtype=self.vlayer.type),np.zeros([1,self.n_v],dtype=self.vlayer.type), self.n_ch, self.n_cv )
        return out

    def initialize_do_grad_updates(self):
        out = {'vlayer':{},'hlayer':{}}
        out['W'] = True
        if self.visible in ['Bernoulli','Spin','Potts']:
            out['vlayer']['fields'] = True
        elif self.visible in ['Bernoulli_coupled','Spin','Spin_coupled']:
            out['vlayer']['fields'] = True
            out['vlayer']['couplings'] = True
        elif self.visible == 'Gaussian':
            out['vlayer']['a'] = True
            out['vlayer']['b'] = True
        elif self.visible == 'ReLU+':
            out['vlayer']['a'] = True
            out['vlayer']['b'] = True
            out['vlayer']['theta_plus'] = False
        elif self.visible == 'ReLU':
            out['vlayer']['a'] = True
            out['vlayer']['b'] = False
            out['vlayer']['theta'] = False
            out['vlayer']['theta_plus'] = True
            out['vlayer']['theta_minus'] = True
        elif self.visible == 'dReLU':
            out['vlayer']['a'] = False
            out['vlayer']['b'] = False
            out['vlayer']['theta'] = False
            out['vlayer']['eta'] = False
            out['vlayer']['theta_plus'] = True
            out['vlayer']['theta_minus'] = True
            out['vlayer']['a_plus'] = True
            out['vlayer']['a_minus'] = True


        if self.hidden in ['Bernoulli','Spin','Potts']:
            out['hlayer']['fields'] = True
            out['hlayer']['fields0'] = True

        elif self.hidden in ['Bernoulli_coupled','Spin','Spin_coupled']:
            out['hlayer']['fields'] = True
            out['hlayer']['fields0'] = True
            out['hlayer']['couplings'] = True

        elif self.hidden == 'Gaussian':
            out['hlayer']['a'] = False
            out['hlayer']['b'] = False

        elif self.hidden == 'ReLU+':
            out['hlayer']['a'] = False
            out['hlayer']['b'] = True
            out['hlayer']['theta_plus'] = False
        elif self.hidden == 'ReLU':
            out['hlayer']['a'] = False
            out['hlayer']['b'] = True
            out['hlayer']['theta'] = True
            out['hlayer']['theta_plus'] = False
            out['hlayer']['theta_minus'] = False
        elif self.hidden =='dReLU':
            out['hlayer']['a'] = False
            out['hlayer']['b'] = True
            out['hlayer']['theta'] = True
            out['hlayer']['eta'] = True
            out['hlayer']['theta_plus'] = False
            out['hlayer']['theta_minus'] = False
            out['hlayer']['a_plus'] = False
            out['hlayer']['a_minus'] = False
            out['hlayer']['a0'] = False
            out['hlayer']['b0'] = False
            out['hlayer']['theta0'] = False
            out['hlayer']['eta0'] = False
            out['hlayer']['theta_plus0'] = True
            out['hlayer']['theta_minus0'] = True
            out['hlayer']['a_plus0'] = True
            out['hlayer']['a_minus0'] = True

        if self.no_fields:
            out['vlayer']['fields'] = False
            out['hlayer']['fields'] = False
        return out


#%%

    def single_mutation_deltaE(self,configuration,iterable = None):
        if self.visible <> 'Potts':
            print 'usefull only for Potts'
        else:
            if iterable is None:
                iterable = itertools.product(xrange(self.n_v),xrange(self.n_cv))
            input_ref = self.vlayer.compute_output(configuration,self.weights)
            result = []
            ev_ref = self.vlayer.energy(configuration)
            for (site,color) in iterable:
                input_mutated = input_ref - self.weights[:,site,configuration[site]] + self.weights[:,site,color]
                result.append( ev_ref + self.vlayer.fields[site,configuration[site]] - self.vlayer.fields[site,color] - self.hlayer.logpartition(input_mutated) )
            return np.array(result)

    def double_mutation_deltaE(self,configuration,iterable = None):
        if self.visible <> 'Potts':
            print 'usefull only for Potts'
        else:
            if iterable is None:
                iterable = itertools.product(xrange(self.n_v),xrange(self.n_v),xrange(self.n_cv),xrange(self.n_cv))
            input_ref = self.vlayer.compute_output(configuration,self.weights)
            result = []
            ev_ref = self.vlayer.energy(configuration)
            for (site1,site2,color1,color2) in iterable:
                input_mutated = input_ref - self.weights[:,site1,configuration[site1]] + self.weights[:,site1,color1] -self.weights[:,site2,configuration[site2]] + self.weights[:,site2,color2]
                result.append( ev_ref + self.vlayer.fields[site1,configuration[site1]] + self.vlayer.fields[site2,configuration[site2]] - self.vlayer.fields[site1,color1]- self.vlayer.fields[site2,color2] - self.hlayer.logpartition(input_mutated) )
            return np.array(result)


    def compute_conditional_mutual(self,configurations):
        conditional_mutual = np.zeros([self.n_v,self.n_v])
        mus = np.zeros([self.n_v,self.n_cv])
        for l in range(self.n_cv):
            mus[:,l] = (configurations ==l).mean(0)
        entropy = - (mus * np.log(mus)).sum(1)
        for i in range(self.n_v):
            for j in range(self.n_v):
                print i,j
                if j<i:
                    conditional_mutual[i,j] = conditional_mutual[j,i]
                elif j ==i:
                    conditional_mutual[i,j] = entropy[i] * configurations.shape[0]
                else:
                    for configuration in configurations:
                        iterable = itertools.product([i],[j], range(self.n_cv),range(self.n_cv) )
                        free_energies = self.double_mutation_deltaE(configuration,iterable=iterable)
                        free_energy = free_energies[configuration[i] + self.n_cv * configuration[j] ]
                        logp_ij = logsumexp( free_energy - free_energies)
                        logp_i = logsumexp( free_energy - free_energies[configuration[i] + self.n_cv * np.arange(self.n_cv)])
                        logp_j = logsumexp( free_energy - free_energies[np.arange(self.n_cv) + self.n_cv * configuration[j]])
                        conditional_mutual[i,j]+= logp_ij - logp_i - logp_j

        return conditional_mutual/configurations.shape[0]

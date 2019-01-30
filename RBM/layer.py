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

Layer class. A general class for all sorts of layers

Attributes :
- N (number of r.v. in the layer)
- nature ('Bernoulli' (0-1), Spin (-1,1),  'Potts','Gaussian','ReLU','ReLU+')
- internal_couplings (None) : if there internal couplings in the layer ('pairwise', 'negative Bernoulli'...)
- type-specific parameters : (fields, color-specific fields, thresholds...)
- random_state : the random seed.
- position ('visible','hidden'). Relevant for computing the gradients.
- PT_type (for parallel tempering sampling)

Methods :
<==== CRITICAL ===>
- sample_from_inputs(inputs, beta); where input is ndata X N [ X n_colors] and beta is the inverse temperature (=1 by default)
- mean_from_inputs(inputs, beta); where input is ndata X N [ X n_colors] and beta is the inverse temperature (=1 by default)
- mean2_from_inputs(inputs, beta); where input is ndata X N [ X n_colors] and beta is the inverse temperature (=1 by default)
- var_from_inputs(inputs, beta); where input is ndata X N [ X n_colors] and beta is the inverse temperature (=1 by default)
- compute_output(configuration, weights, direction); where configuration is ndata x N; weights are the inter-layer couplings, and direction is either 'top' or 'down' (basically, use weights or weights.T)

<=== OTHERS ===>
- transform(inputs); compute the representation of an input in the current layer. By definition : conditional means for B,I,G, and ReLU for ReLU,ReLU+.
- internal_gradients(input_pos,input_neg); given two mini-batches from positive and negative phases, compute the gradient of the layer parameters.
- random_init_config : generate typical starting points for the Markov Chains.

Examples :
- A Bernoulli-Bernoulli RBM is made by 2 'Bernoulli' layers
- A Hopfield model is made by 1 'Bernoulli' 1 'Gaussian' layer
- An Spin model is made by 1 'Bernoulli' layer with 'pairwise' internal couplings


"""
import sys
sys.path.append('rbm/PGM/')


import numpy as np
from scipy.special import erf,erfinv
from scipy.sparse import csr_matrix
# from scipy import weave
import cy_utilities
#from scipy.weave import converters

from utilities import check_random_state,logistic,softmax,invert_softmax,cumulative_probabilities,erf_times_gauss,log_erf_times_gauss,logsumexp,average,bilinear_form,average_product


#%% Layer class

class Layer():
    def __init__(self, N = 100, nature = 'Bernoulli', position = 'visible', n_c = 1, gauge = 'zerosum', random_state = None,zero_field = False):
        self.N = N
        self.nature = nature
        self.random_state = check_random_state(random_state)
        self.position = position
        self.tmp = 0
        self.mu_psi = np.zeros(N) # For batch_norm.

        if self.nature in  ['Bernoulli','Spin']:
            self.n_c = 1
            self.fields = np.zeros(N)
            self.fields0 = np.zeros(N) # useful for PT.

        elif self.nature == 'Potts':
            self.n_c = n_c
            self.gauge = gauge # for gradient
            self.fields = np.zeros([N, n_c])
            self.fields0 = np.zeros([N, n_c]) # useful for PT.
        elif self.nature == 'Gaussian':
            self.n_c = 1
            self.a = np.ones(N)
            self.a0 = np.ones(N)
            self.b = np.zeros(N)
            self.b0 = np.zeros(N)
        elif self.nature == 'ReLU':
            self.n_c = 1
            self.a = np.ones(N)
            self.b = np.zeros(N)
            self.theta = np.zeros(N)

            self.theta_plus = np.zeros(N)
            self.theta_minus = np.zeros(N)

            self.a0 = np.ones(N)
            self.theta_plus0 = np.zeros(N)
            self.theta_minus0 = np.zeros(N)

        elif self.nature == 'dReLU':
            self.n_c = 1
            self.a = np.ones(N)
            self.theta = np.zeros(N)
            self.b = np.zeros(N)
            self.eta = np.zeros(N)

            self.a_plus = np.ones(N)
            self.a_minus = np.ones(N)
            self.theta_plus = np.zeros(N)
            self.theta_minus = np.zeros(N)

            self.a_plus0 = np.ones(N)
            self.a_minus0 = np.ones(N)
            self.theta_plus0 = np.zeros(N)
            self.theta_minus0 = np.zeros(N)

        elif nature == 'ReLU+':
            self.n_c = 1
            self.a = np.ones(N)
            self.b = np.zeros(N)

            self.theta_plus = self.b
            self.a_plus = self.a

            self.a0 = np.ones(N)
            self.theta_plus0 = np.zeros(N)

        elif nature in ['Bernoulli_coupled','Spin_coupled']:
            self.n_c = 1
            self.fields = np.zeros(N)
            self.couplings = np.zeros([N,N])

            self.fields0 = np.zeros(N)
            self.couplings0 = np.zeros([N,N])
        elif nature == 'Potts_coupled':
            self.n_c = n_c
            self.gauge = gauge
            self.fields = np.zeros([N,n_c])
            self.fields0 = np.zeros([N,n_c])
            self.couplings = np.zeros([N,N,n_c,n_c])
            self.couplings0 = np.zeros([N,N,n_c,n_c])

        if self.nature in ['Gaussian','ReLU','dReLU','ReLU+']:
            self.type=np.float
        else:
            self.type = np.int


        self.zero_field = zero_field

    def mean_from_inputs(self,psi, beta = 1):
        if self.nature == 'Bernoulli':
            if beta == 1:
                return logistic(psi + self.fields[np.newaxis,:])
            else:
                return logistic(beta* psi + self.fields0[np.newaxis,:] + beta*(self.fields[np.newaxis,:] - self.fields0[np.newaxis,:]) )

        elif self.nature =='Spin':
            if beta ==1:
                return np.tanh(psi + self.fields[np.newaxis,:])
            else:
                return np.tanh(beta* psi + self.fields0[np.newaxis,:] + beta*(self.fields[np.newaxis,:] - self.fields0[np.newaxis,:]) )

        elif self.nature =='Potts':
            if beta ==1:
                return softmax(psi + self.fields[np.newaxis,:,:])
            else:
                return softmax(beta* psi + self.fields0[np.newaxis,:,:] + beta*(self.fields[np.newaxis,:,:] - self.fields0[np.newaxis,:,:]) )

        elif self.nature =='Gaussian':
            if beta ==1:
                return (psi - self.b[np.newaxis,:])/self.a[np.newaxis,:]
            else:
                return (beta*psi - (beta *self.b + (1-beta) * self.b0 )[np.newaxis,:])/( beta * self.a + (1-beta) * self.a0 )[np.newaxis,:]

        elif self.nature =='ReLU':
            if beta == 1:
                etg_plus = erf_times_gauss((-psi + self.theta_plus[np.newaxis,:])/np.sqrt(self.a[np.newaxis,:]))
                etg_minus = erf_times_gauss((psi + self.theta_minus[np.newaxis,:])/np.sqrt(self.a[np.newaxis,:]))
                p_plus = 1/(1+ etg_minus/etg_plus)
                p_minus = 1- p_plus
                mean_neg = (psi + self.theta_minus[np.newaxis,:])/self.a[np.newaxis,:] - 1/etg_minus/np.sqrt(self.a[np.newaxis,:])
                mean_pos = (psi - self.theta_plus[np.newaxis,:])/self.a[np.newaxis,:] + 1/etg_plus/np.sqrt(self.a[np.newaxis,:])

                return mean_pos * p_plus + mean_neg * p_minus
            else:
                etg_plus = erf_times_gauss((-beta*psi + (beta*self.theta_plus + (1-beta) * self.theta_plus0)[np.newaxis,:])/np.sqrt( beta*self.a + (1-beta)* self.a0)[np.newaxis,:])
                etg_minus = erf_times_gauss((beta*psi + (beta*self.theta_minus+(1-beta) * self.theta_minus0)[np.newaxis,:])/np.sqrt( beta*self.a + (1-beta)* self.a0)[np.newaxis,:])
                p_plus = 1/(1+ etg_minus/etg_plus)
                p_minus = 1- p_plus
                mean_neg = (beta*psi + (beta*self.theta_minus + (1-beta) * self.theta_minus0)[np.newaxis,:])/(beta*self.a+(1-beta)*self.a0)[np.newaxis,:] - 1/etg_minus/np.sqrt( beta*self.a + (1-beta)* self.a0)[np.newaxis,:]
                mean_pos = (beta*psi - (beta*self.theta_plus+(1-beta) * self.theta_plus0)[np.newaxis,:])/(beta*self.a+(1-beta)*self.a0)[np.newaxis,:] + 1/etg_plus/np.sqrt( beta*self.a + (1-beta)* self.a0)[np.newaxis,:]

                return mean_pos * p_plus + mean_neg * p_minus

        elif self.nature =='dReLU':
            if beta == 1:
                theta_plus = self.theta_plus[np.newaxis,:]
                theta_minus = self.theta_minus[np.newaxis,:]
                a_plus = self.a_plus[np.newaxis,:]
                a_minus = self.a_minus[np.newaxis,:]
            else:
                theta_plus = (beta * self.theta_plus + (1-beta) * self.theta_plus0)[np.newaxis,:]
                theta_minus = (beta * self.theta_minus + (1-beta) * self.theta_minus0)[np.newaxis,:]
                a_plus = (beta * self.a_plus + (1-beta) * self.a_plus0)[np.newaxis,:]
                a_minus = (beta * self.a_minus + (1-beta) * self.a_minus0)[np.newaxis,:]
                psi *= beta

            psi_plus = (-psi + theta_plus)/np.sqrt(a_plus)
            psi_minus = (psi + theta_minus)/np.sqrt(a_minus)

            etg_plus = erf_times_gauss(psi_plus)
            etg_minus = erf_times_gauss(psi_minus)

            p_plus = 1/(1+ (etg_minus/np.sqrt(a_minus))/(etg_plus/np.sqrt(a_plus)) )
            nans  = np.isnan(p_plus)
            p_plus[nans] = 1.0 * (np.abs(psi_plus[nans]) > np.abs(psi_minus[nans]) )
            p_minus = 1- p_plus

            mean_pos = (-psi_plus + 1/etg_plus) / np.sqrt(a_plus)
            mean_neg = (psi_minus - 1/etg_minus) / np.sqrt(a_minus)
            return mean_pos * p_plus + mean_neg * p_minus

        elif self.nature =='ReLU+':
            if beta ==1:
                etg_plus = erf_times_gauss((-psi + self.theta_plus[np.newaxis,:])/np.sqrt(self.a[np.newaxis,:]))
                return (psi - self.theta_plus[np.newaxis,:])/self.a[np.newaxis,:] + 1/etg_plus/np.sqrt(self.a[np.newaxis,:])
            else:
                etg_plus = erf_times_gauss((-beta*psi + (beta*self.theta_plus + (1-beta) * self.theta_plus0)[np.newaxis,:])/np.sqrt( beta*self.a + (1-beta)* self.a0)[np.newaxis,:])
                return (beta*psi - (beta*self.theta_plus+(1-beta) * self.theta_plus0)[np.newaxis,:])/(beta*self.a+(1-beta)*self.a0)[np.newaxis,:] + 1/etg_plus/np.sqrt( beta*self.a + (1-beta)* self.a0)[np.newaxis,:]
        elif self.nature in ['Bernoulli_coupled','Spin_coupled','Potts_coupled']:
            print 'mean_from_input not supported for %s'%self.nature


    def mean2_from_inputs(self,psi, beta = 1):
        if self.nature in ['Bernoulli','Potts']:
            return self.mean_from_inputs(psi,beta=beta)
        elif self.nature =='Spin':
            return np.ones(psi.shape)
        elif self.nature == 'Gaussian':
            if beta ==1:
                return self.mean_from_inputs(psi,beta=1)**2 + 1/self.a[np.newaxis,:]
            else:
                return self.mean_from_inputs(psi,beta=beta)**2 + 1/(beta*self.a+(1-beta)*self.a0)[np.newaxis,:]
        elif self.nature =='ReLU':
            if beta == 1:
                etg_plus = erf_times_gauss((-psi + self.theta_plus[np.newaxis,:])/np.sqrt(self.a[np.newaxis,:]))
                etg_minus = erf_times_gauss((psi + self.theta_minus[np.newaxis,:])/np.sqrt(self.a[np.newaxis,:]))
                p_plus = 1/(1+ etg_minus/etg_plus)
                p_minus = 1- p_plus
                mean2_pos = 1/self.a[np.newaxis,:] * (1 +     ((psi - self.theta_plus[np.newaxis,:])/np.sqrt(self.a[np.newaxis,:]))**2  -  ((-psi + self.theta_plus[np.newaxis,:])/np.sqrt(self.a[np.newaxis,:]))/etg_plus )
                mean2_neg = 1/self.a[np.newaxis,:] * (1 +     ((psi + self.theta_minus[np.newaxis,:])/np.sqrt(self.a[np.newaxis,:]))**2  -  ((psi + self.theta_minus[np.newaxis,:])/np.sqrt(self.a[np.newaxis,:]))/etg_minus )
            else:
                etg_plus = erf_times_gauss((-beta*psi + (beta*self.theta_plus + (1-beta) * self.theta_plus0)[np.newaxis,:])/np.sqrt( beta*self.a + (1-beta)* self.a0)[np.newaxis,:])
                etg_minus = erf_times_gauss((beta*psi + (beta*self.theta_minus+(1-beta) * self.theta_minus0)[np.newaxis,:])/np.sqrt( beta*self.a + (1-beta)* self.a0)[np.newaxis,:])
                p_plus = 1/(1+ etg_minus/etg_plus)
                p_minus = 1- p_plus
                mean2_pos = 1/(beta*self.a+(1-beta)*self.a0)[np.newaxis,:] * (1 +     ((beta*psi - (beta*self.theta_plus+(1-beta)*self.theta_plus0)[np.newaxis,:])/np.sqrt(beta*self.a+(1-beta)*self.a0)[np.newaxis,:])**2  -  ((-psi + (beta*self.theta_plus+(1-beta)*self.theta_plus0)[np.newaxis,:])/np.sqrt(beta*self.a+(1-beta)*self.a0)[np.newaxis,:])/etg_plus )
                mean2_neg = 1/(beta*self.a+(1-beta)*self.a0)[np.newaxis,:] * (1 +     ((beta*psi + (beta*self.theta_minus+(1-beta)*self.theta_minus0)[np.newaxis,:])/np.sqrt(beta*self.a+(1-beta)*self.a0)[np.newaxis,:])**2  -  ((psi + (beta*self.theta_minus+(1-beta)*self.theta_minus0)[np.newaxis,:])/np.sqrt(beta*self.a+(1-beta)*self.a0)[np.newaxis,:])/etg_minus )
            return (p_plus * mean2_pos + p_minus * mean2_neg)

        elif self.nature =='dReLU':
            if beta == 1:
                theta_plus = self.theta_plus[np.newaxis,:]
                theta_minus = self.theta_minus[np.newaxis,:]
                a_plus = self.a_plus[np.newaxis,:]
                a_minus = self.a_minus[np.newaxis,:]
            else:
                theta_plus = (beta * self.theta_plus + (1-beta) * self.theta_plus0)[np.newaxis,:]
                theta_minus = (beta * self.theta_minus + (1-beta) * self.theta_minus0)[np.newaxis,:]
                a_plus = (beta * self.a_plus + (1-beta) * self.a_plus0)[np.newaxis,:]
                a_minus = (beta * self.a_minus + (1-beta) * self.a_minus0)[np.newaxis,:]
                psi *= beta

            psi_plus = (-psi + theta_plus)/np.sqrt(a_plus)
            psi_minus = (psi + theta_minus)/np.sqrt(a_minus)
            etg_plus = erf_times_gauss(psi_plus)
            etg_minus = erf_times_gauss(psi_minus)
            p_plus = 1/(1+ (etg_minus/np.sqrt(a_minus))/(etg_plus/np.sqrt(a_plus)) )
            nans  = np.isnan(p_plus)
            p_plus[nans] = 1.0 * (np.abs(psi_plus[nans]) > np.abs(psi_minus[nans]) )
            p_minus = 1- p_plus

            mean2_pos = 1/a_plus * (1 +     psi_plus**2  -  psi_plus/etg_plus )
            mean2_neg = 1/a_minus * (1 +     psi_minus**2  -  psi_minus/etg_minus )
            return mean2_pos * p_plus + mean2_neg * p_minus

        elif self.nature =='ReLU+':
            if beta ==1:
                etg_plus = erf_times_gauss((-psi + self.theta_plus[np.newaxis,:])/np.sqrt(self.a[np.newaxis,:]))
                mean2_pos = 1/self.a[np.newaxis,:] * (1 +     ((psi - self.theta_plus[np.newaxis,:])/np.sqrt(self.a[np.newaxis,:]))**2  -  ((-psi + self.theta_plus[np.newaxis,:])/np.sqrt(self.a[np.newaxis,:]))/etg_plus )
            else:
                etg_plus = erf_times_gauss((-beta*psi + (beta*self.theta_plus + (1-beta) * self.theta_plus0)[np.newaxis,:])/np.sqrt( beta*self.a + (1-beta)* self.a0)[np.newaxis,:])
                mean2_pos = 1/(beta*self.a+(1-beta)*self.a0)[np.newaxis,:] * (1 +     ((beta*psi - (beta*self.theta_plus+(1-beta)*self.theta_plus0)[np.newaxis,:])/np.sqrt(beta*self.a+(1-beta)*self.a0)[np.newaxis,:])**2  -  ((-psi + (beta*self.theta_plus+(1-beta)*self.theta_plus0)[np.newaxis,:])/np.sqrt(beta*self.a+(1-beta)*self.a0)[np.newaxis,:])/etg_plus )
            return mean2_pos

        elif self.nature in ['Bernoulli_coupled','Spin_coupled','Potts_coupled']:
            print 'mean2_from_input not supported for %s'%self.nature









    def mean_pm_from_inputs(self,psi, beta = 1):
        if self.nature =='ReLU':
             if beta == 1:
                etg_plus = erf_times_gauss((-psi + self.theta_plus[np.newaxis,:])/np.sqrt(self.a[np.newaxis,:]))
                etg_minus = erf_times_gauss((psi + self.theta_minus[np.newaxis,:])/np.sqrt(self.a[np.newaxis,:]))
                p_plus = 1/(1+ etg_minus/etg_plus)
                p_minus = 1- p_plus
                mean_neg = (psi + self.theta_minus[np.newaxis,:])/self.a[np.newaxis,:] - 1/etg_minus/np.sqrt(self.a[np.newaxis,:])
                mean_pos = (psi - self.theta_plus[np.newaxis,:])/self.a[np.newaxis,:] + 1/etg_plus/np.sqrt(self.a[np.newaxis,:])
             else:
                etg_plus = erf_times_gauss((-beta*psi + (beta*self.theta_plus + (1-beta) * self.theta_plus0)[np.newaxis,:])/np.sqrt( beta*self.a + (1-beta)* self.a0)[np.newaxis,:])
                etg_minus = erf_times_gauss((beta*psi + (beta*self.theta_minus+(1-beta) * self.theta_minus0)[np.newaxis,:])/np.sqrt( beta*self.a + (1-beta)* self.a0)[np.newaxis,:])
                p_plus = 1/(1+ etg_minus/etg_plus)
                p_minus = 1- p_plus
                mean_neg = (beta*psi + (beta*self.theta_minus + (1-beta) * self.theta_minus0)[np.newaxis,:])/(beta*self.a+(1-beta)*self.a0)[np.newaxis,:] - 1/etg_minus/np.sqrt( beta*self.a + (1-beta)* self.a0)[np.newaxis,:]
                mean_pos = (beta*psi - (beta*self.theta_plus+(1-beta) * self.theta_plus0)[np.newaxis,:])/(beta*self.a+(1-beta)*self.a0)[np.newaxis,:] + 1/etg_plus/np.sqrt( beta*self.a + (1-beta)* self.a0)[np.newaxis,:]
             return (mean_pos * p_plus, mean_neg * p_minus)
        elif self.nature =='dReLU':
            if beta == 1:
                theta_plus = self.theta_plus[np.newaxis,:]
                theta_minus = self.theta_minus[np.newaxis,:]
                a_plus = self.a_plus[np.newaxis,:]
                a_minus = self.a_minus[np.newaxis,:]
            else:
                theta_plus = (beta * self.theta_plus + (1-beta) * self.theta_plus0)[np.newaxis,:]
                theta_minus = (beta * self.theta_minus + (1-beta) * self.theta_minus0)[np.newaxis,:]
                a_plus = (beta * self.a_plus + (1-beta) * self.a_plus0)[np.newaxis,:]
                a_minus = (beta * self.a_minus + (1-beta) * self.a_minus0)[np.newaxis,:]
                psi *= beta

            psi_plus = (-psi + theta_plus)/np.sqrt(a_plus)
            psi_minus = (psi + theta_minus)/np.sqrt(a_minus)

            etg_plus = erf_times_gauss(psi_plus)
            etg_minus = erf_times_gauss(psi_minus)

            p_plus = 1/(1+ (etg_minus/np.sqrt(a_minus))/(etg_plus/np.sqrt(a_plus)) )
            nans  = np.isnan(p_plus)
            p_plus[nans] = 1.0 * (np.abs(psi_plus[nans]) > np.abs(psi_minus[nans]) )
            p_minus = 1- p_plus

            mean_pos = (-psi_plus + 1/etg_plus) / np.sqrt(a_plus)
            mean_neg = (psi_minus - 1/etg_minus) / np.sqrt(a_minus)

            return (mean_pos * p_plus,mean_neg * p_minus)



        else:
             print 'not supported'

    def mean12_pm_from_inputs(self,psi, beta = 1):
        if self.nature =='dReLU':
            if beta == 1:
                theta_plus = self.theta_plus[np.newaxis,:]
                theta_minus = self.theta_minus[np.newaxis,:]
                a_plus = self.a_plus[np.newaxis,:]
                a_minus = self.a_minus[np.newaxis,:]
            else:
                theta_plus = (beta * self.theta_plus + (1-beta) * self.theta_plus0)[np.newaxis,:]
                theta_minus = (beta * self.theta_minus + (1-beta) * self.theta_minus0)[np.newaxis,:]
                a_plus = (beta * self.a_plus + (1-beta) * self.a_plus0)[np.newaxis,:]
                a_minus = (beta * self.a_minus + (1-beta) * self.a_minus0)[np.newaxis,:]
                psi *= beta

            psi_plus = (-psi + theta_plus)/np.sqrt(a_plus)
            psi_minus = (psi + theta_minus)/np.sqrt(a_minus)
            etg_plus = erf_times_gauss(psi_plus)
            etg_minus = erf_times_gauss(psi_minus)
            p_plus = 1/(1+ (etg_minus/np.sqrt(a_minus))/(etg_plus/np.sqrt(a_plus)) )

            nans  = np.isnan(p_plus)
            p_plus[nans] = 1.0 * (np.abs(psi_plus[nans]) > np.abs(psi_minus[nans]) )

            p_minus = 1- p_plus
            mean2_pos = 1/a_plus * (1 +     psi_plus**2  -  psi_plus/etg_plus )
            mean2_neg = 1/a_minus * (1 +     psi_minus**2  -  psi_minus/etg_minus )
            mean_pos = (-psi_plus + 1/etg_plus) / np.sqrt(a_plus)
            mean_neg = (psi_minus - 1/etg_minus) / np.sqrt(a_minus)
            return (p_plus* mean_pos, p_minus * mean_neg, p_plus * mean2_pos, p_minus * mean2_neg)






        else:
             print 'not supported'



    def var_from_inputs(self,psi,beta =1):
        if self.nature in ['Bernoulli','Potts']:
            mean = self.mean_from_inputs(psi,beta=beta)
            return mean * (1-mean)
        elif self.nature =='Spin':
            return 1- self.mean_from_inputs(psi,beta=beta)**2
        elif self.nature == 'Gaussian':
            if beta==1:
                return np.ones(psi.shape)/self.a[np.newaxis,:]
            else:
                return np.ones(psi.shape)/(beta*self.a +(1-beta)* self.a0)[np.newaxis,:]

        elif self.nature == 'ReLU':
            if beta == 1:
                etg_plus = erf_times_gauss((-psi + self.theta_plus[np.newaxis,:])/np.sqrt(self.a[np.newaxis,:]))
                etg_minus = erf_times_gauss((psi + self.theta_minus[np.newaxis,:])/np.sqrt(self.a[np.newaxis,:]))
                p_plus = 1/(1+ etg_minus/etg_plus)
                p_minus = 1- p_plus
                mean_neg = (psi + self.theta_minus[np.newaxis,:])/self.a[np.newaxis,:] - 1/etg_minus/np.sqrt(self.a[np.newaxis,:])
                mean_pos = (psi - self.theta_plus[np.newaxis,:])/self.a[np.newaxis,:] + 1/etg_plus/np.sqrt(self.a[np.newaxis,:])
                mean2_pos = 1/self.a[np.newaxis,:] * (1 +     ((psi - self.theta_plus[np.newaxis,:])/np.sqrt(self.a[np.newaxis,:]))**2  -  ((-psi + self.theta_plus[np.newaxis,:])/np.sqrt(self.a[np.newaxis,:]))/etg_plus )
                mean2_neg = 1/self.a[np.newaxis,:] * (1 +     ((psi + self.theta_minus[np.newaxis,:])/np.sqrt(self.a[np.newaxis,:]))**2  -  ((psi + self.theta_minus[np.newaxis,:])/np.sqrt(self.a[np.newaxis,:]))/etg_minus )
                return (p_plus * mean2_pos + p_minus * mean2_neg) - (p_plus * mean_pos + p_minus * mean_neg)**2
            else:
                etg_plus = erf_times_gauss((-beta*psi + (beta*self.theta_plus + (1-beta) * self.theta_plus0)[np.newaxis,:])/np.sqrt( beta*self.a + (1-beta)* self.a0)[np.newaxis,:])
                etg_minus = erf_times_gauss((beta*psi + (beta*self.theta_minus+(1-beta) * self.theta_minus0)[np.newaxis,:])/np.sqrt( beta*self.a + (1-beta)* self.a0)[np.newaxis,:])
                p_plus = 1/(1+ etg_minus/etg_plus)
                p_minus = 1- p_plus
                mean_neg = (beta*psi + (beta*self.theta_minus + (1-beta) * self.theta_minus0)[np.newaxis,:])/(beta*self.a+(1-beta)*self.a0)[np.newaxis,:] - 1/etg_minus/np.sqrt( beta*self.a + (1-beta)* self.a0)[np.newaxis,:]
                mean_pos = (beta*psi - (beta*self.theta_plus+(1-beta) * self.theta_plus0)[np.newaxis,:])/(beta*self.a+(1-beta)*self.a0)[np.newaxis,:] + 1/etg_plus/np.sqrt( beta*self.a + (1-beta)* self.a0)[np.newaxis,:]
                mean2_pos = 1/(beta*self.a+(1-beta)*self.a0)[np.newaxis,:] * (1 +     ((psi - (beta*self.theta_plus+(1-beta)*self.theta_plus0)[np.newaxis,:])/np.sqrt(beta*self.a+(1-beta)*self.a0)[np.newaxis,:])**2  -  ((-beta*psi + (beta*self.theta_plus+(1-beta)*self.theta_plus0)[np.newaxis,:])/np.sqrt(beta*self.a+(1-beta)*self.a0)[np.newaxis,:])/etg_plus )
                mean2_neg = 1/(beta*self.a+(1-beta)*self.a0)[np.newaxis,:] * (1 +     ((psi + (beta*self.theta_minus+(1-beta)*self.theta_minus0)[np.newaxis,:])/np.sqrt(beta*self.a+(1-beta)*self.a0)[np.newaxis,:])**2  -  ((beta*psi + (beta*self.theta_minus+(1-beta)*self.theta_minus0)[np.newaxis,:])/np.sqrt(beta*self.a+(1-beta)*self.a0)[np.newaxis,:])/etg_minus )

                return (p_plus * mean2_pos + p_minus * mean2_neg) - (p_plus * mean_pos + p_minus * mean_neg)**2
        elif self.nature =='ReLU+':
            if beta ==1:
                etg_plus = erf_times_gauss((-psi + self.theta_plus[np.newaxis,:])/np.sqrt(self.a[np.newaxis,:]))
                mean_pos = (psi - self.theta_plus[np.newaxis,:])/self.a[np.newaxis,:] + 1/etg_plus/np.sqrt(self.a[np.newaxis,:])
                mean2_pos = 1/self.a[np.newaxis,:] * (1 +     ((psi - self.theta_plus[np.newaxis,:])/np.sqrt(self.a[np.newaxis,:]))**2  -  ((-psi + self.theta_plus[np.newaxis,:])/np.sqrt(self.a[np.newaxis,:]))/etg_plus )
                return mean2_pos - mean_pos**2
            else:
                etg_plus = erf_times_gauss((-beta*psi + (beta*self.theta_plus + (1-beta) * self.theta_plus0)[np.newaxis,:])/np.sqrt( beta*self.a + (1-beta)* self.a0)[np.newaxis,:])
                mean_pos = (beta*psi - (beta*self.theta_plus+(1-beta) * self.theta_plus0)[np.newaxis,:])/(beta*self.a+(1-beta)*self.a0)[np.newaxis,:] + 1/etg_plus/np.sqrt( beta*self.a + (1-beta)* self.a0)[np.newaxis,:]
                mean2_pos = 1/(beta*self.a+(1-beta)*self.a0)[np.newaxis,:] * (1 +     ((beta*psi - (beta*self.theta_plus+(1-beta)*self.theta_plus0)[np.newaxis,:])/np.sqrt(beta*self.a+(1-beta)*self.a0)[np.newaxis,:])**2  -  ((-beta*psi + (beta*self.theta_plus+(1-beta)*self.theta_plus0)[np.newaxis,:])/np.sqrt(beta*self.a+(1-beta)*self.a0)[np.newaxis,:])/etg_plus )
                return mean2_pos - mean_pos**2
        elif self.nature =='dReLU':
            if beta == 1:
                theta_plus = self.theta_plus[np.newaxis,:]
                theta_minus = self.theta_minus[np.newaxis,:]
                a_plus = self.a_plus[np.newaxis,:]
                a_minus = self.a_minus[np.newaxis,:]
            else:
                theta_plus = (beta * self.theta_plus + (1-beta) * self.theta_plus0)[np.newaxis,:]
                theta_minus = (beta * self.theta_minus + (1-beta) * self.theta_minus0)[np.newaxis,:]
                a_plus = (beta * self.a_plus + (1-beta) * self.a_plus0)[np.newaxis,:]
                a_minus = (beta * self.a_minus + (1-beta) * self.a_minus0)[np.newaxis,:]
                psi *= beta

            psi_plus = (-psi + theta_plus)/np.sqrt(a_plus)
            psi_minus = (psi + theta_minus)/np.sqrt(a_minus)
            etg_plus = erf_times_gauss(psi_plus)
            etg_minus = erf_times_gauss(psi_minus)
            p_plus = 1/(1+ (etg_minus/np.sqrt(a_minus))/(etg_plus/np.sqrt(a_plus)) )

            nans  = np.isnan(p_plus)
            p_plus[nans] = 1.0 * (np.abs(psi_plus[nans]) > np.abs(psi_minus[nans]) )

            p_minus = 1- p_plus
            mean2_pos = 1/a_plus * (1 +     psi_plus**2  -  psi_plus/etg_plus )
            mean2_neg = 1/a_minus * (1 +     psi_minus**2  -  psi_minus/etg_minus )
            mean_pos = (-psi_plus + 1/etg_plus)/np.sqrt(a_plus)
            mean_neg = (psi_minus - 1/etg_minus)/np.sqrt(a_minus)
            return (p_plus * mean2_pos + p_minus * mean2_neg) - (p_plus * mean_pos + p_minus * mean_neg)**2



        elif self.nature in ['Bernoulli_coupled','Spin_coupled','Potts_coupled']:
            print 'var_from_input not supported for %s'%self.nature



    def sample_from_inputs(self,psi,beta=1,previous=(None,None)):
        if self.nature == 'Bernoulli':
            return (self.random_state.random_sample(size = psi.shape) < self.mean_from_inputs(psi,beta))
        elif self.nature == 'Spin':
            return np.sign( self.random_state.random_sample(size = psi.shape) - (1- self.mean_from_inputs(psi,beta))/2  )

        elif self.nature =='Potts':
            if beta ==1:
                cum_probas = psi + self.fields[np.newaxis,:,:]
            else:
                cum_probas = beta * psi + beta * self.fields[np.newaxis,:,:] + (1-beta)*  self.fields0[np.newaxis,:,:]
            cum_probas = cumulative_probabilities(cum_probas)
            # if np.isnan(cum_probas).max():
            #     print 'CUM_PROBAS NAAAAAAAAAN'
            #     cum_probas = psi + self.fields[np.newaxis,:,:]
            #     print cum_probas.min(),cum_probas.max()
            #     out = np.argmax(cum_probas,axis=2)
            #     out[0,0] = 294
            #     return out
            rng = self.random_state.rand(psi.shape[0],self.N)
            rng = cy_utilities.tower_sampling_C(psi.shape[0], self.N, self.n_c,cum_probas,rng)
            return rng

        elif self.nature =='Gaussian':
            if beta ==1:
                return self.mean_from_inputs(psi,beta=1) +  self.random_state.randn(psi.shape[0],psi.shape[1])/np.sqrt(self.a[np.newaxis,:])
            else:
                return self.mean_from_inputs(psi,beta=beta) +  self.random_state.randn(psi.shape[0],psi.shape[1])/np.sqrt(beta*self.a + (1-beta)* self.a0)[np.newaxis,:]
                return psi

        elif self.nature == 'ReLU':
            if beta ==1:
                etg_plus = erf_times_gauss((-psi + self.theta_plus[np.newaxis,:])/np.sqrt(self.a[np.newaxis,:]))
                etg_minus = erf_times_gauss((psi + self.theta_minus[np.newaxis,:])/np.sqrt(self.a[np.newaxis,:]))
                p_plus = 1/(1+ etg_minus/etg_plus)

                is_pos = self.random_state.random_sample(size=psi.shape) < p_plus
                rmax = 0 * p_plus
                rmin = 0 * p_plus
                rmin[is_pos] = erf( ((self.theta_plus[np.newaxis,:] - psi)/np.sqrt(self.a))[is_pos]/np.sqrt(2) )
                rmax[is_pos] = 1
                rmin[~is_pos] = -1
                rmax[~is_pos] = erf( -((psi + self.theta_minus[np.newaxis,:])/np.sqrt(self.a))[~is_pos]/np.sqrt(2) )

                h = np.zeros(psi.shape)
                tmp = (rmax - rmin > 1e-14)
                h = np.sqrt(2)/np.sqrt(self.a) * erfinv(rmin + (rmax - rmin) * self.random_state.random_sample(size = h.shape)   ) + psi/self.a[np.newaxis,:]
                h[is_pos] -= (self.theta_plus/self.a)[np.newaxis,:].repeat(psi.shape[0],axis=0)[is_pos]
                h[~is_pos] += (self.theta_minus/self.a)[np.newaxis,:].repeat(psi.shape[0],axis=0)[~is_pos]
                h[np.isinf(h) | np.isnan(h) | ~tmp] = 0
                return h
            else:
                etg_plus = erf_times_gauss((-beta*psi + (beta*self.theta_plus + (1-beta) * self.theta_plus0)[np.newaxis,:])/np.sqrt( beta*self.a + (1-beta)* self.a0)[np.newaxis,:])
                etg_minus = erf_times_gauss((beta*psi + (beta*self.theta_minus+(1-beta) * self.theta_minus0)[np.newaxis,:])/np.sqrt( beta*self.a + (1-beta)* self.a0)[np.newaxis,:])

                p_plus = 1/(1+ etg_minus/etg_plus)

                is_pos = self.random_state.random_sample(size=psi.shape) < p_plus
                rmax = 0 * p_plus
                rmin = 0 * p_plus
                rmin[is_pos] = erf( (( (beta*self.theta_plus+(1-beta)*self.theta_plus0)[np.newaxis,:] - beta*psi)/np.sqrt(beta*self.a+ (1-beta) *self.a0))[is_pos]/np.sqrt(2) )
                rmax[is_pos] = 1
                rmin[~is_pos] = -1
                rmax[~is_pos] = erf( -((beta*psi + (beta*self.theta_minus+(1-beta)*self.theta_minus0)[np.newaxis,:])/np.sqrt(beta*self.a+ (1-beta) *self.a0))[~is_pos]/np.sqrt(2) )

                h = np.zeros(psi.shape)
                tmp = (rmax - rmin > 1e-14)
                h = np.sqrt(2)/np.sqrt(beta*self.a+ (1-beta) *self.a0) * erfinv(rmin + (rmax - rmin) * self.random_state.random_sample(size = h.shape)   ) + beta*psi/(beta*self.a+ (1-beta) *self.a0)[np.newaxis,:]
                h[is_pos] -= ( (beta*self.theta_plus+(1-beta)*self.theta_plus0)/(beta*self.a+(1-beta)*self.a0))[np.newaxis,:].repeat(psi.shape[0],axis=0)[is_pos]
                h[~is_pos] += ((beta*self.theta_minus+(1-beta)*self.theta_minus0)/(beta*self.a+(1-beta)*self.a0))[np.newaxis,:].repeat(psi.shape[0],axis=0)[~is_pos]
                h[np.isinf(h) | np.isnan(h) | ~tmp] = 0
                return h
        elif self.nature == 'dReLU':
            if beta == 1:
                theta_plus = self.theta_plus[np.newaxis,:]
                theta_minus = self.theta_minus[np.newaxis,:]
                a_plus = self.a_plus[np.newaxis,:]
                a_minus = self.a_minus[np.newaxis,:]
            else:
                theta_plus = (beta * self.theta_plus + (1-beta) * self.theta_plus0)[np.newaxis,:]
                theta_minus = (beta * self.theta_minus + (1-beta) * self.theta_minus0)[np.newaxis,:]
                a_plus = (beta * self.a_plus + (1-beta) * self.a_plus0)[np.newaxis,:]
                a_minus = (beta * self.a_minus + (1-beta) * self.a_minus0)[np.newaxis,:]
                psi *= beta

            nans = np.isnan(psi)
            if nans.max():
                nan_unit = np.nonzero(nans.max(0))[0]
                print 'NAN IN INPUT (n_h = %s, nature =dReLU)'%(self.N,self.nature)
                print 'Hidden units', nan_unit
                print 'Theta:',self.theta[nan_unit]
                print 'eta:',self.eta[nan_unit]
                print 'a',self.a[nan_unit]
                print 'b',self.b[nan_unit]

            psi_plus = (-psi + theta_plus)/np.sqrt(a_plus)
            psi_minus = (psi + theta_minus)/np.sqrt(a_minus)

            etg_plus = erf_times_gauss(psi_plus)
            etg_minus = erf_times_gauss(psi_minus)

            p_plus = 1/(1+ (etg_minus/np.sqrt(a_minus))/(etg_plus/np.sqrt(a_plus)) )
            nans  = np.isnan(p_plus)
            p_plus[nans] = 1.0 * (np.abs(psi_plus[nans]) > np.abs(psi_minus[nans]) )
            p_minus = 1- p_plus

            is_pos = self.random_state.random_sample(size=psi.shape) < p_plus
            rmax = np.zeros(p_plus.shape)
            rmin = np.zeros(p_plus.shape)
            rmin[is_pos] = erf( psi_plus[is_pos]/np.sqrt(2) )
            rmax[is_pos] = 1
            rmin[~is_pos] = -1
            rmax[~is_pos] = erf( -psi_minus[~is_pos]/np.sqrt(2) )

            h = np.zeros(psi.shape)
            tmp = (rmax - rmin > 1e-14)
            h = np.sqrt(2) * erfinv(rmin + (rmax - rmin) * self.random_state.random_sample(size = h.shape)   )
            h[is_pos] -= psi_plus[is_pos]
            h[~is_pos] += psi_minus[~is_pos]
            h/= np.sqrt(is_pos * a_plus + (1-is_pos) * a_minus)
            h[np.isinf(h) | np.isnan(h) | ~tmp] = 0
            return h

        elif self.nature =='ReLU+':
            if beta == 1:
                etg_plus = erf_times_gauss((-psi + self.theta_plus[np.newaxis,:])/np.sqrt(self.a[np.newaxis,:]))
                rmin = erf( ((self.theta_plus[np.newaxis,:] - psi)/np.sqrt(self.a))/np.sqrt(2) )
                rmax = 1
                tmp = (rmax - rmin > 1e-14)
                h =  np.sqrt(2)/np.sqrt(self.a) * erfinv(rmin + (rmax - rmin) * self.random_state.random_sample(size = psi.shape)   ) + (psi- self.theta_plus[np.newaxis,:])/self.a[np.newaxis,:]
                h[np.isinf(h) | np.isnan(h)] = 0
                return h
            else:
                etg_plus = erf_times_gauss((-beta*psi + (beta*self.theta_plus + (1-beta) * self.theta_plus0)[np.newaxis,:])/np.sqrt( beta*self.a + (1-beta)* self.a0)[np.newaxis,:])
                rmin = erf( (( (beta*self.theta_plus+(1-beta)*self.theta_plus0)[np.newaxis,:] - beta*psi)/np.sqrt(beta*self.a+(1-beta)*self.a0))/np.sqrt(2) )
                rmax = 1
                tmp = (rmax - rmin > 1e-14)
                h =  np.sqrt(2)/np.sqrt(beta*self.a+(1-beta)*self.a0)* erfinv(rmin + (rmax - rmin) * self.random_state.random_sample(size = psi.shape)   ) + (beta*psi- (beta*self.theta_plus+(1-beta)*self.theta_plus0)[np.newaxis,:])/(beta*self.a+(1-beta)*self.a0)[np.newaxis,:]
                h[np.isinf(h) | np.isnan(h)] = 0
                return h
        elif self.nature == 'Bernoulli_coupled':#:in ['Bernoulli_coupled','Spin_coupled','Potts_coupled']:
            (x,fields_eff) = previous

            if x is None:
                x = np.random.randint(0,high=2,size=[psi.shape[0],self.N])
            if fields_eff is None:
                fields_eff = self.fields[np.newaxis] + self.compute_output(x,self.couplings)

            if psi is not None:
                x,fields_eff=cy_utilities.Bernoulli_Gibbs_input_C(x.shape[0], self.N,beta,x,fields_eff,psi, self.fields0, self.couplings, self.random_state.randint(0,high=self.N,size=[x.shape[0],self.N]), self.random_state.rand(x.shape[0],self.N) )
            else:
                x,fields_eff=cy_utilities.Bernoulli_Gibbs_free_C(x.shape[0], self.N,beta,x,fields_eff, self.fields0,self.couplings, self.random_state.randint(0,high=self.N,size=[x.shape[0],self.N]), self.random_state.rand(x.shape[0],self.N) )

            return (x,fields_eff)

        elif self.nature == 'Spin_coupled':
            (x,fields_eff) = previous
            if x is None:
                x = 2*np.random.randint(0,high=2,size=[psi.shape[0],self.N])-1

            if fields_eff is None:
                fields_eff = self.fields[np.newaxis] + self.compute_output(x,self.couplings)

            if psi is not None:
                x,fields_eff=cy_utilities.Spin_Gibbs_input_C(x.shape[0], self.N,beta,x,fields_eff,psi, self.fields0, self.couplings, self.random_state.randint(0,high=self.N,size=[x.shape[0],self.N]), self.random_state.rand(x.shape[0],self.N) )
            else:
                x,fields_eff=cy_utilities.Spin_Gibbs_free_C(x.shape[0], self.N,beta,x,fields_eff, self.fields0,self.couplings, self.random_state.randint(0,high=self.N,size=[x.shape[0],self.N]), self.random_state.rand(x.shape[0],self.N) )

            return (x,fields_eff)


        elif self.nature == 'Potts_coupled':
            (x,fields_eff) = previous
            if x is None:
                x = np.random.randint(0,high=self.n_c,size=[psi.shape[0],self.N])
            if fields_eff is None:
                fields_eff = self.fields[np.newaxis] + self.compute_output(x,self.couplings)

            if psi is not None:
                x,fields_eff=cy_utilities.Potts_Gibbs_input_C(x.shape[0], self.N,self.n_c,beta,x,fields_eff,psi, self.fields0, self.couplings, self.random_state.randint(0,high=self.N,size=[x.shape[0],self.N]), self.random_state.rand(x.shape[0],self.N) )
            else:
                x,fields_eff=cy_utilities.Potts_Gibbs_free_C(x.shape[0], self.N,self.n_c,beta,x,fields_eff, self.fields0,self.couplings, self.random_state.randint(0,high=self.N,size=[x.shape[0],self.N]), self.random_state.rand(x.shape[0],self.N) )
            return (x,fields_eff)






    def compute_output(self,config, couplings, direction='up'):

        if config.ndim == 1: config = config[np.newaxis, :] # ensure that the config data is a batch, at leas of just one vector
        n_data = config.shape[0]

        if direction == 'up':
            N_output_layer = couplings.shape[0]
            if self.nature in ['Potts','Potts_coupled']:
                if couplings.ndim == 4: # output layer is Potts
                    n_c_output_layer = couplings.shape[2]
                    if self.tmp:
                        output = np.zeros([n_data, N_output_layer,n_c_output_layer])
                        for color in range(self.n_c):
                            A = csr_matrix(config == color)
                            for color_out in range(n_c_output_layer):
                                output[:,:,color_out]+= A.dot(couplings[:,:,color_out,color].T)
                        return output
                    else:
                        return cy_utilities.compute_output_Potts_C(self.N, self.n_c,n_c_output_layer ,config,couplings)
                else:
                    if self.tmp:
                        output = np.zeros([n_data,N_output_layer])
                        for color in range(self.n_c):
                            A = csr_matrix(config == color)
                            output+= A.dot(couplings[:,:,color].T)
                        return output
                    else:
                        return cy_utilities.compute_output_C(self.N, self.n_c,config,couplings)
            else:
                return np.tensordot(config, couplings, axes = (1,1))
        elif direction == 'down':
            N_output_layer = couplings.shape[1]
            if self.nature in ['Potts','Potts_coupled']:
                if couplings.ndim ==4:
                    n_c_output_layer = couplings.shape[3]
                    if self.tmp:
                        output = np.zeros([n_data, N_output_layer,n_c_output_layer])
                        for color in range(self.n_c):
                            A = csr_matrix(config == color)
                            for color_out in range(n_c_output_layer):
                                output[:,:,color_out]+= A.dot(couplings[:,:,color,color_out])
                        return output
                    else:
                        return cy_utilities.compute_output_Potts_C2(self.N, self.n_c,n_c_output_layer ,config, couplings)
                else:
                    if self.tmp:
                        output = np.zeros([n_data,N_output_layer])
                        for color in range(self.n_c):
                            A = csr_matrix(config == color)
                            output+= A.dot(couplings[:,:,color])
                        return output
                    else:
                        return cy_utilities.compute_output_C2(self.N, self.n_c,config,couplings)

            else:
                return np.tensordot(config, couplings, axes = (1,0))


    def energy(self,config,remove_init = False):
        if config.ndim == 1: config = config[np.newaxis, :] # ensure that the config data is a batch, at leas of just one vector
        if self.nature in ['Bernoulli','Spin']:
            if remove_init:
                return -np.dot(config,self.fields - self.fields0)
            else:
                return -np.dot(config,self.fields)
        if self.nature == 'Potts':
            E = np.zeros(config.shape[0])
            for color in range(self.n_c):
                A = csr_matrix(config == color)
                if remove_init:
                    E -= A.dot(self.fields[:,color]- self.fields0[:,color])
                else:
                    E -= A.dot(self.fields[:,color])
            return E
        elif self.nature == 'Gaussian':
            if remove_init:
                return np.dot(config**2,self.a -self.a0)/2 + np.dot(config, self.b - self.b0)
            else:
                return np.dot(config**2,self.a)/2 + np.dot(config, self.b)
        elif self.nature == 'ReLU':
            if remove_init:
                return np.dot(config**2,self.a -self.a0)/2 + np.dot(np.maximum(config,0), self.theta_plus - self.theta_plus0) + np.dot(np.maximum(-config,0), self.theta_minus - self.theta_minus0)
            else:
                return np.dot(config**2,self.a)/2 + np.dot(np.maximum(config,0), self.theta_plus) + np.dot(np.maximum(-config,0), self.theta_minus)
        elif self.nature == 'dReLU':
            if remove_init:
                a_plus = self.a_plus - self.a_plus0
                a_minus = self.a_minus - self.a_minus0
                theta_plus = self.theta_plus - self.theta_plus0
                theta_minus = self.theta_minus - self.theta_minus0
            else:
                a_plus = self.a_plus
                a_minus = self.a_minus
                theta_plus = self.theta_plus
                theta_minus = self.theta_minus

            config_plus = np.maximum(config,0)
            config_minus = np.maximum(-config,0)
            return np.dot(config_plus**2, a_plus)/2 + np.dot(config_minus**2,a_minus)/2 + np.dot(config_plus,theta_plus) + np.dot(config_minus,theta_minus)


        elif self.nature == 'ReLU+':
            if remove_init:
                return np.dot(config**2,self.a -self.a0)/2 + np.dot(np.maximum(config,0), self.theta_plus - self.theta_plus0)
            else:
                return np.dot(config**2,self.a)/2 + np.dot(np.maximum(config,0), self.theta_plus)
        elif self.nature in ['Bernoulli_coupled','Spin_coupled']:
            if remove_init:
                return - np.dot(config,self.fields-self.fields0) - 0.5* (np.dot(config , self.couplings) * config).sum(1)
            else:
                return - np.dot(config,self.fields) - 0.5* (np.dot(config , self.couplings) * config).sum(1)
        elif self.nature == 'Potts_coupled':
            if remove_init:
                E = - cy_utilities.dot_Potts_C(self.N, self.n_c,config, self.fields-self.fields0)
            else:
                E = - cy_utilities.dot_Potts_C(self.N, self.n_c,config, self.fields)

                E -= 0.5 * bilinear_form(self.couplings, config,config,c1=self.n_c,c2=self.n_c)
            return E
        else:
            print 'nature not supported, energy'


    def logpartition(self,inputs,beta = 1):
        if inputs is None:
            if self.nature in ['Potts','Potts_coupled']:
                inputs = np.zeros([1,self.N,self.n_c])
            else:
                inputs = np.zeros([1,self.N])
        if self.nature == 'Bernoulli':
            if beta == 1:
                return np.log(1+ np.exp(self.fields[np.newaxis,:] + inputs )).sum(1)
            else:
                return np.log(1+ np.exp( (beta* self.fields + (1-beta) * self.fields0)[np.newaxis,:] + beta * inputs )).sum(1)
        elif self.nature == 'Spin':
            if beta == 1:
                tmp = self.fields[np.newaxis,:] + inputs
            else:
                tmp = (beta* self.fields + (1-beta) * self.fields0)[np.newaxis,:] + beta * inputs
            return np.logaddexp(tmp,-tmp).sum(1) # stable logcosh(inputs)
        elif self.nature == 'Potts':
            if beta == 1:
                return logsumexp(self.fields[np.newaxis,:,:]+ inputs,2).sum(1)
            else:
                return logsumexp( (beta * self.fields + (1-beta) * self.fields0)[np.newaxis,:]+ beta * inputs,2).sum(1)
        elif self.nature == 'Gaussian':
            if beta == 1:
                return (0.5 * (inputs - self.b[np.newaxis,:])**2/self.a).sum(1) + 0.5 * np.log(2*np.pi/self.a).sum()
            else:
                return (0.5 * (beta * inputs - (beta * self.b + (1-beta) * self.b0)[np.newaxis,:] )**2/(beta*self.a + (1-beta) * self.a0)[np.newaxis,:] ).sum(1)  + 0.5 * np.log(2*np.pi/(beta*self.a+(1-beta)*self.a0) ).sum()
        elif self.nature == 'ReLU':
            if beta ==1:
                return np.logaddexp( log_erf_times_gauss((-inputs + self.theta_plus[np.newaxis,:])/np.sqrt(self.a[np.newaxis,:]) ), log_erf_times_gauss ( (inputs + self.theta_minus[np.newaxis,:])/np.sqrt(self.a[np.newaxis,:]))).sum(1) + 0.5 * np.log(2*np.pi/self.a).sum()
            else:
                return np.logaddexp( log_erf_times_gauss((-inputs * beta + (beta*self.theta_plus+(1-beta) * self.theta_plus0)[np.newaxis,:])/np.sqrt( (beta*self.a+(1-beta) * self.a0)[np.newaxis,:]) ), log_erf_times_gauss ( ( beta *inputs + (beta*self.theta_minus+(1-beta) * self.theta_minus0)[np.newaxis,:])/np.sqrt( (beta*self.a+(1-beta) * self.a0)[np.newaxis,:]))).sum(1)   + 0.5 * np.log(2*np.pi/(beta*self.a+(1-beta)*self.a0) ).sum()
        elif self.nature ==  'dReLU':
            if beta == 1:
                theta_plus = self.theta_plus
                theta_minus = self.theta_minus
                a_plus = self.a_plus
                a_minus = self.a_minus
            else:
                theta_plus = beta * self.theta_plus + (1-beta) * self.theta_plus0
                theta_minus = beta * self.theta_minus + (1-beta) * self.theta_minus0
                a_plus = beta * self.a_plus + (1-beta) * self.a_plus0
                a_minus = beta * self.a_minus + (1-beta) * self.a_minus0
            return np.logaddexp( log_erf_times_gauss((-inputs + theta_plus[np.newaxis,:])/np.sqrt(a_plus[np.newaxis,:]) ) - 0.5* np.log(a_plus[np.newaxis,:]), log_erf_times_gauss ( (inputs + theta_minus[np.newaxis,:])/np.sqrt(a_minus[np.newaxis,:]))- 0.5* np.log(a_minus[np.newaxis,:])).sum(1) + 0.5 * np.log(2*np.pi) * self.N

        elif self.nature == 'ReLU+':
            if beta ==1:
                return log_erf_times_gauss((-inputs + self.theta_plus[np.newaxis,:])/np.sqrt(self.a[np.newaxis,:]) ).sum(1)  +0.5 * np.log(2*np.pi/self.a).sum()
            else:
                return log_erf_times_gauss((-inputs * beta + (beta *self.theta_plus+ (1-beta) * self.theta_plus0)[np.newaxis,:])/np.sqrt( (beta*self.a+(1-beta) * self.a0)[np.newaxis,:]) ).sum(1)   + 0.5 * np.log(2*np.pi/(beta*self.a+(1-beta)*self.a0) ).sum()
        elif self.nature == 'Bernoulli_coupled':
            if beta ==0:
                return np.log(1+ np.exp(self.fields0)[np.newaxis,:] + 0 * inputs ).sum(1)
            else:
                print 'mean field equations not implemented for Bernoulli_coupled'
        elif self.nature =='Spin_coupled':
            if beta == 0:
                tmp = self.fields0[np.newaxis,:] + 0*inputs
            else:
                print 'mean field equations not implemented for Spin_coupled'
            return np.logaddexp(tmp,-tmp).sum(1) # stable logcosh(inputs)
        elif self.nature == 'Potts_coupled':
            if beta == 0:
                return logsumexp( self.fields0[np.newaxis,:]+ 0 * inputs,2).sum(1)
            else:
                print 'mean field equations not implemented for Potts_coupled'



        else:
            print 'hidden type not supported'
            return






#%%

    def transform(self,psi):
        if self.nature == 'Bernoulli':
            return (psi+self.fields)>0
        elif self.nature == 'Spin':
            return np.sign(psi+self.fields)
        elif self.nature == 'Potts':
            return np.argmax(psi+self.fields,axis=2)
        if self.nature == 'Gaussian':
            return self.mean_from_inputs(psi)
        elif self.nature == 'ReLU':
            return 1/self.a[np.newaxis,:] * ( (psi+self.theta_minus[np.newaxis,:]) * (psi <= np.minimum(-self.theta_minus,(self.theta_plus-self.theta_minus)/2 )[np.newaxis,:] ) + (psi-self.theta_plus[np.newaxis,:]) * (psi>= np.maximum(self.theta_plus,(self.theta_plus-self.theta_minus)/2 )[np.newaxis,:]) )
        elif self.nature == 'dReLU':
            return ( (psi+self.theta_minus[np.newaxis,:]) * (psi <= np.minimum(-self.theta_minus,(self.theta_plus/np.sqrt(self.a_plus)-self.theta_minus/np.sqrt(self.a_minus) )/(1/np.sqrt(self.a_plus) + 1/np.sqrt(self.a_minus) ) )[np.newaxis,:] ))/self.a_minus[np.newaxis,:] + ((psi-self.theta_plus[np.newaxis,:]) * (psi>= np.maximum(self.theta_plus,(self.theta_plus/np.sqrt(self.a_plus)-self.theta_minus/np.sqrt(self.a_minus) )/(1/np.sqrt(self.a_plus) + 1/np.sqrt(self.a_minus) ) )[np.newaxis,:]))/self.a_plus[np.newaxis,:]
        elif self.nature == 'ReLU+':
            return np.maximum(psi - self.theta_plus[np.newaxis,:],0)/self.a[np.newaxis,:]
        else:
            print 'not supported'

    def random_init_config(self,n_samples,N_PT=1):
        if self.nature in ['Bernoulli','Spin']:
            if N_PT>1:
                return self.sample_from_inputs( np.zeros([N_PT*n_samples, self.N]) ,beta = 0).reshape([N_PT,n_samples,self.N])
            else:
                return self.sample_from_inputs( np.zeros([n_samples, self.N]) ,beta = 0)

        elif self.nature == 'Potts':
            if N_PT>1:
                return self.sample_from_inputs( np.zeros([n_samples * N_PT, self.N, self.n_c]) ,beta = 0).reshape([N_PT,n_samples,self.N])
            else:
                return self.sample_from_inputs( np.zeros([n_samples, self.N, self.n_c]) ,beta = 0)
        elif self.nature in ['Gaussian','ReLU','dReLU']:
            if N_PT>1:
                return self.random_state.randn(N_PT,n_samples, self.N)
            else:
                return self.random_state.randn(n_samples, self.N)
        elif self.nature == 'ReLU+':
            if N_PT>1:
                return np.maximum(self.random_state.randn(N_PT,n_samples, self.N),0)
            else:
                return np.maximum(self.random_state.randn(n_samples, self.N),0)
        elif self.nature in ['Bernoulli_coupled','Spin_coupled']:
            if N_PT>1:
                (x,fields_eff) = self.sample_from_inputs( np.zeros([N_PT*n_samples, self.N]) ,beta = 0)
                x = x.reshape([N_PT,n_samples,self.N])
                fields_eff = fields_eff.reshape([N_PT,n_samples,self.N])
            else:
                (x,fields_eff) = self.sample_from_inputs( np.zeros([N_PT*n_samples, self.N]) ,beta = 0)
            return (x,fields_eff)
        elif self.nature == 'Potts_coupled':
            if N_PT>1:
                (x,fields_eff) = self.sample_from_inputs( np.zeros([n_samples*N_PT, self.N, self.n_c]) ,beta = 0)
                x = x.reshape([N_PT,n_samples,self.N])
                fields_eff = fields_eff.reshape([N_PT,n_samples,self.N,self.n_c])
            else:
                (x,fields_eff) = self.sample_from_inputs( np.zeros([n_samples, self.N, self.n_c]) ,beta = 0)
            return (x,fields_eff)

    def internal_gradients(self,data_pos,data_neg, l1 = None, l2 = None,weights = None,weights_neg=None,value='data'):
        gradients = {}
        if self.nature in ['Bernoulli','Spin','Potts']:
            if value == 'data': # data_pos, data_neg are configurations
                mu_pos = average(data_pos,c=self.n_c,weights=weights)
                mu_neg = average(data_neg,c=self.n_c,weights=weights_neg)
            elif value == 'mean':
                mu_pos = average(data_pos,weights=weights)
                mu_neg = average(data_neg,weights=weights_neg)
            elif value == 'input':
                mu_pos = average( self.mean_from_inputs(data_pos),weights=weights)
                mu_neg = average( self.mean_from_inputs(data_neg),weights=weights_neg)

            gradients['fields'] =   mu_pos - mu_neg
            if weights is not None:
                 gradients['fields'] *= weights.sum()/data_pos.shape[0]
        elif self.nature == 'Gaussian':
            if value == 'data':
                mu2_pos = average(data_pos**2,weights=weights)
                mu2_neg = average(data_neg**2,weights=weights_neg)
                mu_pos = average(data_pos,weights=weights)
                mu_neg = average(data_neg,weights=weights_neg)
            elif value == 'mean':
                print 'gaussian mean not supported for internal gradient'
            elif value == 'input':
                mu2_pos = average(self.mean2_from_inputs(data_pos),weights=weights)
                mu2_neg = average(self.mean2_from_inputs(data_neg),weights=weights_neg)
                mu_pos = average(self.mean_from_inputs(data_pos),weights=weights)
                mu_neg = average(self.mean_from_inputs(data_neg),weights=weights_neg)

            # if (self.position =='visible'): # don't need to update a and b if it is a hidden layer.
            gradients['a'] = -0.5 * (mu2_pos - mu2_neg)
            gradients['b'] = -mu_pos+mu_neg

            if weights is not None:
                gradients['a'] *= weights.sum()/data_pos.shape[0]
                gradients['b'] *= weights.sum()/data_pos.shape[0]

        elif self.nature == 'ReLU':
            if value == 'data':
                mu2_pos = average(data_pos**2,weights=weights)
                mu2_neg = average(data_neg**2,weights=weights_neg)
                mu_p_plus = average(np.maximum(data_pos,0),weights=weights)
                mu_n_pos = average(np.minimum(data_pos,0),weights=weights)
                mu_p_minus = average(np.maximum(data_neg,0),weights=weights_neg)
                mu_n_neg = average(np.minimum(data_neg,0),weights=weights_neg)
            elif value == 'mean':
                print 'ReLU mean not supported for internal gradient'
            elif value == 'input':
                mu2_pos = average(self.mean2_from_inputs(data_pos),weights=weights)
                mu2_neg = average(self.mean2_from_inputs(data_neg),weights=weights_neg)
                mu_p_plus,mu_n_pos = self.mean_pm_from_inputs(data_pos)
                mu_p_plus = average(mu_p_plus,weights = weights)
                mu_n_pos = average(mu_n_pos,weights = weights)
                mu_p_minus,mu_n_neg = self.mean_pm_from_inputs(data_neg)
                mu_p_minus = average(mu_p_minus,weights=weights_neg)
                mu_n_neg = average(mu_n_neg,weights=weights_neg)
            # if self.position == 'visible':
            gradients['a'] = -0.5 * (mu2_pos - mu2_neg)
            if weights is not None:
                gradients['a'] *= weights.sum()/data_pos.shape[0]

            gradients['theta_plus'] = - mu_p_plus + mu_p_minus
            gradients['theta_minus'] = mu_n_pos - mu_n_neg

            if weights is not None:
                gradients['theta_plus'] *= weights.sum()/data_pos.shape[0]
                gradients['theta_minus'] *= weights.sum()/data_pos.shape[0]

            gradients['theta'] = gradients['theta_plus'] + gradients['theta_minus']
            gradients['b'] = gradients['theta_plus'] - gradients['theta_minus']


        elif self.nature == 'dReLU':
            if value == 'data':
                mu2_p_pos = average(np.maximum(data_pos,0)**2,weights=weights)
                mu2_n_pos = average(np.minimum(data_pos,0)**2,weights=weights)
                mu2_p_neg = average(np.maximum(data_neg,0)**2,weights=weights_neg)
                mu2_n_neg = average(np.minimum(data_neg,0)**2,weights=weights_neg)
                mu_p_pos = average(np.maximum(data_pos,0),weights=weights)
                mu_n_pos = average(np.minimum(data_pos,0),weights=weights)
                mu_p_neg = average(np.maximum(data_neg,0),weights=weights_neg)
                mu_n_neg = average(np.minimum(data_neg,0),weights=weights_neg)
            elif value == 'mean':
                print 'dReLU mean not supported for internal gradient'
            elif value == 'input':
                mu_p_pos,mu_n_pos,mu2_p_pos,mu2_n_pos = self.mean12_pm_from_inputs(data_pos)
                mu_p_pos = average(mu_p_pos,weights = weights)
                mu_n_pos = average(mu_n_pos,weights = weights)
                mu2_p_pos = average(mu2_p_pos,weights=weights)
                mu2_n_pos = average(mu2_n_pos,weights=weights)

                mu_p_neg,mu_n_neg,mu2_p_neg,mu2_n_neg = self.mean12_pm_from_inputs(data_neg)
                mu_p_neg = average(mu_p_neg,weights=weights_neg)
                mu_n_neg = average(mu_n_neg,weights=weights_neg)
                mu2_p_neg = average(mu2_p_neg,weights=weights_neg)
                mu2_n_neg = average(mu2_n_neg,weights=weights_neg)

            for moment in ['data_pos','data_neg','mu_p_pos','mu_n_pos','mu2_p_pos','mu2_n_pos']:
                exec('tmp = np.isnan(%s).max()'%moment)
                if tmp:
                    print 'NAN in %s'%moment
                    exec('index = np.isnan(%s)'%moment)
                    print np.nonzero(index)
                    print 'a', self.a[index]
                    print 'theta', self.theta[index]
                    print 'b', self.b[index]
                    print 'eta', self.eta[index]
                    print 'theta_plus',self.theta_plus[index]
                    print 'theta_minus',self.theta_minus[index]
                    print 'a_plus',self.a_plus[index]
                    print 'a_minus',self.a_minus[index]
                    break
                    # print 'data_pos',data_pos[:,index]



            gradients['a_plus'] = - 0.5 * (mu2_p_pos - mu2_p_neg)
            gradients['a_minus'] = -0.5 * (mu2_n_pos - mu2_n_neg)
            gradients['theta_plus'] = - mu_p_pos + mu_p_neg
            gradients['theta_minus'] = mu_n_pos - mu_n_neg


            if weights is not None:
               gradients['a_plus'] *= weights.sum()/data_pos.shape[0]
               gradients['a_minus'] *= weights.sum()/data_pos.shape[0]
               gradients['theta_plus'] *= weights.sum()/data_pos.shape[0]
               gradients['theta_minus'] *= weights.sum()/data_pos.shape[0]

            gradients['a'] = gradients['a_plus']/(1+self.eta) + gradients['a_minus']/(1-self.eta)
            gradients['b'] = gradients['theta_plus'] - gradients['theta_minus']
            gradients['theta'] = gradients['theta_plus']/(1+self.eta) + gradients['theta_minus']/(1-self.eta)
            gradients['eta'] = (- self.a/(1+self.eta)**2 * gradients['a_plus']
                                + self.a/(1-self.eta)**2 * gradients['a_minus']
                                - self.theta/(1+self.eta)**2 * gradients['theta_plus']
                                + self.theta/(1-self.eta)**2 * gradients['theta_minus'] )

            # gradients['eta'] = (- self.a_plus**2 * gradients['a_plus'] + self.a_minus**2 * gradients['a_minus'])/self.a

            for param in ['theta_plus','theta_minus','a_plus','a_minus']:
                if np.isnan(gradients[param]).max():
                    print 'NAN in gradient of %s'%param





        elif self.nature =='ReLU+':
            if value == 'data':
                mu2_pos = average(data_pos**2,weights=weights)
                mu2_neg = average(data_neg**2,weights=weights_neg)
                mu_pos = average(data_pos,weights=weights)
                mu_neg = average(data_neg,weights=weights_neg)
            elif value == 'mean':
                print 'gaussian mean not supported for internal gradient'
            elif value == 'input':
                mu2_pos = average(self.mean2_from_inputs(data_pos),weights=weights)
                mu2_neg = average(self.mean2_from_inputs(data_neg),weights=weights_neg)
                mu_pos = average(self.mean_from_inputs(data_pos),weights=weights)
                mu_neg = average(self.mean_from_inputs(data_neg),weights=weights_neg)

            gradients['a'] = -0.5 * (mu2_pos - mu2_neg)
            gradients['theta_plus'] = -mu_pos + mu_neg
            if weights is not None:
                gradients['theta_plus'] *= weights.sum()/data_pos.shape[0]
                gradients['a'] *= weights.sum()/data_pos.shape[0]

            gradients['b'] = gradients['theta_plus']

        elif self.nature in ['Bernoulli_coupled','Spin_coupled']:
            if value =='data':
                mu_pos = average(data_pos,weights=weights)
                mu_neg = average(data_neg,weights=weights_neg)
                comu_pos = average_product(data_pos, data_pos,weights=weights)
                comu_neg = average_product(data_neg, data_neg,weights=weights_neg)


            elif value == 'mean':
                mu_pos = data_pos[0]
                mu_neg = average(data_neg,weights=weights_neg)
                comu_pos = data_pos[1]
                comu_neg = average_product(data_neg, data_neg,weights=weights_neg)
            elif value == 'input':
                print 'not supported'
            if self.batch_norm:
                gradients['couplings'] = comu_pos - comu_neg - mu_pos[:,np.newaxis] * (mu_pos-mu_neg)[np.newaxis,:] - mu_pos[np.newaxis,:] * (mu_pos-mu_neg)[:,np.newaxis]
                gradients['fields'] = mu_pos-mu_neg - np.dot(gradient['couplings'],mu_pos)
            else:
                gradients['fields'] = mu_pos-mu_neg
                gradients['couplings'] = comu_pos - comu_neg
            if weights is not None:
                gradients['fields'] *= weights.sum()/data_pos.shape[0]
                gradients['couplings'] *= weights.sum()/data_pos.shape[0]

            if l2 is not None:
                gradients['couplings'] -= l2 * self.couplings
            if l1 is not None:
                gradients['couplings'] -= l1 * np.sign(self.couplings)

        elif self.nature == 'Potts_coupled':
            if value == 'data':
                mu_pos = average(data_pos,c=self.n_c,weights=weights)
                mu_neg = average(data_neg,c=self.n_c,weights=weights_neg)
                comu_pos = average_product(data_pos, data_pos,weights=weights,c1=self.n_c, c2= self.n_c)
                comu_neg = average_product(data_neg, data_neg,weights=weights_neg,c1=self.n_c, c2= self.n_c)
            elif value == 'mean':
                mu_pos = data_pos[0]
                mu_neg = average(data_neg,c=self.n_c,weights=weights_neg)
                comu_pos = data_pos[1]
                comu_neg = average_product(data_neg, data_neg,c1=self.n_c, c2= self.n_c,weights=weights_neg)
            elif value == 'input':
                print 'not supported'


            if self.batch_norm:
                gradients['couplings'] = comu_pos - comu_neg - mu_pos[:,np.newaxis,:,np.newaxis] * (mu_pos-mu_neg)[np.newaxis,:,np.newaxis,:] - mu_pos[np.newaxis,:,np.newaxis,:] * (mu_pos-mu_neg)[:,np.newaxis,:,np.newaxis]
                gradients['fields'] = mu_pos - mu_neg - np.tensordot(gradients['couplings'], mu_pos,axes=([1,3],[0,1]))
            else:
                gradients['fields'] =  mu_pos - mu_neg
                gradients['couplings'] = comu_pos - comu_neg
            if weights is not None:
                gradients['fields'] *= weights.sum()/data_pos.shape[0]
                gradients['couplings'] *= weights.sum()/data_pos.shape[0]


            if l2 is not None:
                gradients['couplings'] -= l2 * self.couplings
            if l1 is not None:
                gradients['couplings'] -= l1 * np.sign(self.couplings)




        if (self.position== 'hidden'):
            if self.nature in ['Bernoulli','Spin','Bernoulli_coupled','Spin_coupled']:
                mu_neg0 = self.mean_from_inputs( np.zeros([1,self.N]),beta=0)[0]
                gradients['fields0'] = mu_pos - mu_neg0
                if weights is not None:
                    gradients['fields0'] *= weights.sum()/data_pos.shape[0]
            elif self.nature in ['Potts','Potts_coupled']:
                mu_neg0 = self.mean_from_inputs( np.zeros([1,self.N,self.n_c]),beta=0)[0]
                gradients['fields0'] = mu_pos - mu_neg0
                if weights is not None:
                    gradients['fields0'] *= weights.sum()/data_pos.shape[0]

            # elif self.nature == 'Gaussian':
            #     if self.tmp == 1:
            #         gradients['a0'] = -self.b**2 + self.a *(1. - 2.* self.b *mu_pos - self.a * mu2_pos)
            #         gradients['b0'] = -self.b**3/self.a - 2. * self.b**2 * mu_pos -self.a * mu_pos - self.a* self.b * mu2_pos

            #     else:
            #         mu2_neg0 = self.mean2_from_inputs( np.zeros([1,self.N]),beta=0)[0]
            #         mu_neg0 = self.mean_from_inputs( np.zeros([1,self.N]),beta=0)[0]
            #         gradients['a0'] = -0.5 * ( mu2_pos - mu2_neg0)
            #         gradients['b0'] = -mu_pos + mu_neg0

            #     if weights is not None:
            #         gradients['a0'] *= weights.sum()/data_pos.shape[0]
            #         gradients['b0'] *= weights.sum()/data_pos.shape[0]

            elif self.nature == 'ReLU':
                (mu_p_minus0,mu_n_neg0) = self.mean_pm_from_inputs(np.zeros([1,self.N]), beta = 0)
                mu_p_minus0 = mu_p_minus0[0]
                mu_n_neg0 = mu_n_neg0[0]
                mu2_neg0 = - self.mean2_from_inputs( np.zeros([1,self.N]),beta=0)[0]
                gradients['a0'] = -0.5 * (mu2_pos - mu2_neg0)
                gradients['theta_plus0'] = (-mu_p_plus + mu_p_minus0)
                gradients['theta_minus0'] = (mu_n_pos - mu_n_neg0)
                if weights is not None:
                    gradients['a0'] *= weights.sum()/data_pos.shape[0]
                    gradients['theta_plus0'] *= weights.sum()/data_pos.shape[0]
                    gradients['theta_minus0'] *= weights.sum()/data_pos.shape[0]

            elif self.nature == 'dReLU':
                (mu_p_neg0,mu_n_neg0,mu2_p_neg0,mu2_n_neg0) = self.mean12_pm_from_inputs(np.zeros([1,self.N]), beta = 0)
                mu_p_neg0 = mu_p_neg0[0]
                mu_n_neg0 = mu_n_neg0[0]
                mu2_p_neg0 = mu2_p_neg0[0]
                mu2_n_neg0 = mu2_n_neg0[0]
                gradients['a_plus0'] = -0.5 * (mu2_p_pos - mu2_p_neg0)
                gradients['a_minus0'] = -0.5 * (mu2_n_pos - mu2_n_neg0)
                gradients['theta_plus0'] = (-mu_p_pos + mu_p_neg0)
                gradients['theta_minus0'] = (mu_n_pos - mu_n_neg0)
                if weights is not None:
                    gradients['a_plus0'] *= weights.sum()/data_pos.shape[0]
                    gradients['a_plus0'] *= weights.sum()/data_pos.shape[0]
                    gradients['theta_plus0'] *= weights.sum()/data_pos.shape[0]
                    gradients['theta_minus0'] *= weights.sum()/data_pos.shape[0]                   

            elif self.nature == 'ReLU+':
                mu2_neg0 = self.mean2_from_inputs( np.zeros([1,self.N]),beta=0)[0]
                mu_neg0 = self.mean_from_inputs( np.zeros([1,self.N]),beta=0)[0]

                gradients['a0'] = -0.5 * (mu2_pos  - mu2_neg0)
                gradients['theta_plus0'] = -mu_pos + mu_neg0
                if weights is not None:
                    gradients['a0'] *= weights.sum()/data_pos.shape[0]
                    gradients['theta_plus0'] *= weights.sum()/data_pos.shape[0]


        if self.zero_field:
            gradients['fields'] *= 0
        return gradients



    def init_params_from_data(self,X,eps=1e-6,mean=False,weights=None):
        if self.nature in ['Bernoulli','Bernoulli_coupled']:
            if mean:
                mu = X
            else:
                mu = average(X,weights=weights)
            self.fields = np.log((mu+ eps)/(1-mu + eps))
            self.fields0 = self.fields.copy()
        elif self.nature in ['Spin','Spin_coupled']:
            if mean:
                mu = X
            else:
                mu = average(X,weights=weights)

            self.fields= 0.5*np.log((1+mu + eps)/(1-mu + eps) )
            self.fields0 = self.fields.copy()
        elif self.nature == 'Gaussian':
            mu = average(X,weights=weights)
            var = average(X**2,weights=weights) - mu**2
            self.a = 1/(var+eps)
            self.b = - self.a * mu
            self.a0 = self.a.copy()
            self.b0 = self.b.copy()
        elif self.nature in ['Potts','Potts_coupled']:
            if mean:
                mu = X
            else:
                mu = average(X,weights=weights,c=self.n_c)
            self.fields = invert_softmax(mu,eps=eps, gauge = self.gauge)
            self.fields0 = self.fields.copy()
        elif self.nature == 'ReLU':
            mu = average(X,weights=weights)
            var = average(X**2,weights=weights) - mu**2
            l1 = np.abs(X).mean(axis=0)
        elif self.nature =='ReLU+':
            mu = average(X,weights=weights)
            var = average(X**2,weights=weights) - mu**2

        if self.nature in ['Bernoulli_coupled','Spin_coupled','Potts_coupled']:
            self.couplings *=0
            self.couplings0 *=0

        if self.zero_field:
            self.fields *=0
            self.fields0 *=0


    def sample_and_energy_from_inputs(self,psi,beta=1,previous=None):
        if self.nature == 'Potts':
            config=self.sample_from_inputs(psi,beta=beta)
            energy = -cy_utilities.dot_Potts2_C(self.N, self.n_c,config, self.fields[np.newaxis]-self.fields0[np.newaxis]+psi)
            return config,energy
        elif self.nature in ['Bernoulli_coupled','Spin_coupled','Potts_coupled']:
            (x,fields_eff) = self.sample_from_inputs(psi,beta=beta,previous=previous)
            if psi is None:
                    f = 0.5* (self.fields[np.newaxis] + fields_eff) - self.fields0[np.newaxis]
            else:
                    f = 0.5* (self.fields[np.newaxis] + fields_eff) + psi-self.fields0[np.newaxis]

            if self.nature =='Potts_coupled':
                energy = -cy_utilities.dot_Potts2_C(self.N, self.n_c,x, f)
            else:
                energy = -np.sum(x*f,1)
            return (x,fields_eff),energy

        else:
            config = self.sample_from_inputs(psi,beta=beta)
            energy = self.energy(config,remove_init=True) - (config*psi).sum(1)
        return config,energy

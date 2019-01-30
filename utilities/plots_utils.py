"""
 Copyright 2018 - by Jerome Tubiana (jertubiana@gmail.com)
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

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import sys,os
sys.path.append('../RBM/')

import numpy as np
import rbm
import Proteins_utils
import Proteins_RBM_utils
import sequence_logo

def get_ax(ax,i,nrows,ncols):
    if (ncols>1) & (nrows>1):
        col = i%ncols
        row = i/ncols
        ax_ = ax[row,col]
    elif (ncols>1) & (nrows==1):
        ax_ = ax[i]
    elif (ncols==1) & (nrows>1):
        ax_ = ax[i]
    else:
        ax_ = ax
    return ax_

def clean_ax(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)        
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)        
    ax.set_xticks([])
    ax.set_yticks([])        



def plot_input_mean(RBM,I, subset, I_range=None,mean=None,weights = None, ax = None, ncols = 1, xlabels = None, with_dots = None , figsize = (3,3), show= True):
    if type(subset) in [int,np.int64]:
        subset = [subset]
    if ax is not None:
        nfeatures = 1
        ncols = 1
        nrows = 1
        return_fig = False
    else:
        nfeatures = len(subset)
        nrows = int(np.ceil(nfeatures/float(ncols)))
        fig, ax = plt.subplots(nrows,ncols)
        fig.set_figheight( nrows * figsize[0] )
        fig.set_figwidth( ncols * figsize[1] )
        return_fig = True

    if I_range is None:
        I_min = I.min()
        I_max = I.max()
        I_range = (I_max-I_min) * np.arange(0,1+0.01,0.01) + I_min

    if mean is None:
        mean = RBM.hlayer.mean_from_inputs(np.repeat(I_range[:,np.newaxis],RBM.n_h,axis=1))

    if xlabels is None:
        xlabels = [r'Input $I_{%s}$'%(i+1) for i in range(nfeatures)]

    for i in range(nfeatures):
        ax_ = get_ax(ax,i,nrows,ncols)
        ax2_ = ax_.twinx()    
        ax2_.hist(I[:,subset[i]],normed=True,weights=weights,bins=100)
        ax_.plot(I_range,mean[:,subset[i]],c='black',linewidth=2)

        xmin = I[:,subset[i]].min()
        xmax = I[:,subset[i]].max()
        ymin = mean[:,subset[i]].min()
        ymax = mean[:,subset[i]].max()



        if with_dots is not None:
            for val in with_dots[i]:
                tmp= np.argmin(np.abs(mean[:,subset[i]]-val) )
                ax_.plot(I_range[tmp],mean[tmp,subset[i]],'.',c='red',markersize=20)
                xmin = min(I_range[tmp],xmin)
                xmax = max(I_range[tmp],xmax)
                ymin = min(mean[tmp,subset[i]], ymin  )
                ymax = max(mean[tmp,subset[i]], ymax  )


        ax_.set_xlim([xmin,xmax])
        step = int( (xmax-xmin )/4.0) +1
        xticks = np.arange(int(xmin), int(xmax)+1, step)
        ax_.set_xticks(xticks)
        ax_.set_xticklabels(xticks,fontsize=12)
        ax_.set_ylim([ymin,ymax])
        step = int( (ymax-ymin )/4.0)+1
        yticks = np.arange(int(ymin), int(ymax)+1, step)
        ax_.set_yticks(yticks)
        ax_.set_yticklabels(yticks,fontsize=12)
        ax2_.set_yticks([])
        for tl in ax_.get_yticklabels():
            tl.set_fontsize(14)
        ax_.set_zorder(ax2_.get_zorder()+1)
        ax_.patch.set_visible(False)

        ax_.set_xlabel(xlabels[i],fontsize=14)

    for i in range(nfeatures,ncols*nrows):
        ax_ = get_ax(ax,i,nrows,ncols)
        clean_ax(ax_)

    if return_fig:
        plt.tight_layout()
        if show:
            plt.show()
        return fig



def plot_input_classes(subset,I_background, I_test, labels, n_class=None, label_names=None, label_background = 'MSA',
                colors = None, xlabels = None , markers = None, beta_order=False,show=True,fontsize=10,nbins=10,ncols=1,ymax=None):
    if n_class is None:
        n_class = labels.max()+1
    if label_names is None:
        label_names = ['Class %s'%i for i in range(n_class)]
    if colors is None:
        colors = ['C%s'%(i%10) for i in range(n_class)]

    nfeatures = len(subset)
    nrows = int(np.ceil(nfeatures/float(ncols)))
    fig, ax = plt.subplots(nrows, ncols)
    fig.set_figheight(nrows * 3)
    fig.set_figwidth(ncols * 3)

    all_labels = [label_background] + label_names
    if xlabels is None:
        xlabels = [r'$I_{%s}$'%(i+1) for i in range(nfeatures)]
    
    for i in range(nfeatures):
        ax_ = get_ax(ax,i,nrows,ncols)
        row = i/ncols
        col = i%ncols

        all_hists = []
        ax_.hist(I_background[:,subset[i]],color='black',bins=100,normed=True, histtype='bar',alpha=0.25,label=label_background)
        F = patches.Rectangle((0,0),1,1,facecolor='black',alpha=0.25)
        all_hists.append(F)
        for k in range(n_class):
            subset_class = (labels == k)
            ax_.hist(I_test[subset_class,subset[i]],color=colors[k],normed=True, histtype='step',label=label_names[k],bins=nbins)
            F = patches.Rectangle((0,0),1,1,facecolor=colors[k])
            all_hists.append(F)
        ax_.set_xlabel(xlabels[i],fontsize=14)
        if ymax is not None:
            ax_.set_ylim([0,ymax[i]])
        if row==0:
            ax_.legend(all_hists[ (col*(n_class+1))/ncols : ((col+1)*(n_class+1))/ncols ] , all_labels[ (col*(n_class+1))/ncols : ((col+1)*(n_class+1))/ncols ] ,fontsize=fontsize,frameon=False,ncol=1)
        for line in all_hists:
            line.set_visible(False)
    for i in range(nfeatures,ncols*nrows):
        ax_ = get_ax(ax,i,nrows,ncols)
        clean_ax(ax_)        

    plt.tight_layout()
    if show:
        plt.show()    
    return fig


def plot_input_classes_scatter(subsets,I_background, I_test, labels, n_class=None, label_names=None,
                  label_background = 'MSA',figsize=8,colors = None, axis_labels = None ,
                  markers = None,show=True,markersize=3,fontsize=10,ncols=1,ncols_legend=1,
                 xlim = None,ylim=None):
    if n_class is None:
        n_class = labels.max()+1
    if label_names is None:
        label_names = ['Class %s'%i for i in range(n_class)]
    if colors is None:
        colors = ['C%s'%(i%10) for i in range(n_class)]
        
    if markers is None:
        markers = ['o' for i in range(n_class)]
        

    nfeatures = len(subsets)
    nrows = int(np.ceil(nfeatures/float(ncols)))
    fig, ax = plt.subplots(nrows, ncols)
    if type(figsize) == list:
        fig.set_figwidth(ncols * figsize[0])        
        fig.set_figheight(nrows * figsize[1])
    else:
        fig.set_figheight(nrows * figsize)
        fig.set_figwidth(ncols * figsize)

    all_labels = [label_background] + label_names

    if axis_labels is None:
        axis_labels = [ [r'$I_{%s}$'%(subset[0]+1), r'$I_{%s}$'%(subset[1]+1) ] for subset in subsets]

    if xlim is None:
        xlim = [None for l in range(nfeatures)]
    if ylim is None:
        ylim = [None for l in range(nfeatures)]        
        
    for l in range(nfeatures):
        ax_ = get_ax(ax,l,nrows,ncols)
        row = l/ncols
        col = l%ncols
        i = subsets[l][0]
        j = subsets[l][1]          

        all_scats = []
        S = ax_.scatter(I_background[:,i],I_background[:,j] ,c='black',s=0.5,label=label_background,alpha=0.25)
        all_scats.append(S)
        for k in range(n_class):
            subset_class = (labels == k)
            S=ax_.scatter(I_test[subset_class,i],I_test[subset_class,j],c=colors[k],s=markersize,label=label_names[k],marker=markers[k]) 
            all_scats.append(S)

        ax_.set_xlabel(axis_labels[l][0],fontsize=fontsize)
        ax_.set_ylabel(axis_labels[l][1],fontsize=fontsize)

        if row==0:
            ax_.legend(all_scats[ (col*(n_class+1))/ncols : ((col+1)*(n_class+1))/ncols ] , all_labels[ (col*(n_class+1))/ncols : ((col+1)*(n_class+1))/ncols ] ,fontsize=16,frameon=False,ncol=ncols_legend,markerscale=5.0,handletextpad=0.2)

        ax_.set_xlim(xlim[l])
        ax_.set_ylim(ylim[l])


    for i in range(nfeatures,ncols*nrows):
        ax_ = get_ax(ax,i,nrows,ncols)
        clean_ax(ax_)        
                
    plt.tight_layout()
    if show:
        plt.show()    
    return fig




def plot_top_activating_distance(RBM, I,data, subset, nseqs = 20,all_distances = None, distance_top_features = None,ax=None,xlabels=None,ncols=1,figsize=(3,3),show=True):
    if type(subset) in [int,np.int64]:
        subset = [subset]
    if all_distances is None:
        sub = np.argsort(np.random.randn(len(data)))[:1000]
        all_distances = Proteins_utils.distance(data[sub]).flatten()

    if distance_top_features is None:
        I_ = I.copy()
        if RBM.hidden == 'dReLU':
            eta = RBM.hlayer.eta
            I_ *= np.sign(eta)[np.newaxis,:]
        q = np.percentile(I_,100* (1-float(nseqs)/data.shape[0]),axis=0)
        distance_top_features = []
        for k in range(RBM.n_h):
            subgroup = data[I_[:,k]>q[k]]
            distance_top_features.append( Proteins_utils.distance(subgroup).flatten() )

    if ax is not None:
        nfeatures = 1
        ncols = 1
        nrows = 1
        return_fig = False
    else:
        nfeatures = len(subset)
        nrows = int(np.ceil(nfeatures/float(ncols)))
        fig, ax = plt.subplots(nrows,ncols)
        fig.set_figheight( nrows * figsize[0] )
        fig.set_figwidth( ncols * figsize[1] )
        return_fig = True

    if xlabels is None:
        xlabels = ['Distance %s'%(i+1) for i in range(nfeatures)]

    for i in range(nfeatures):
        ax_ = get_ax(ax,i,nrows,ncols)
        ax_.hist(all_distances,bins=RBM.n_v,normed=True,range=(0,1),color = 'gray',alpha = 0.5)
        ax_.hist(distance_top_features[subset[i]],bins=RBM.n_v,normed=True,range=(0,1),alpha=0.5)
        ax_.set_xticks([0,0.5,1])
        ax_.set_xticklabels([0,0.5,1],fontsize=12)
        ax_.set_yticks([])
        ax_.set_xlabel(xlabels[i],fontsize=14)
    for i in range(nfeatures,ncols*nrows):
        ax_ = get_ax(ax,i,nrows,ncols)
        clean_ax(ax_)

    if return_fig:
        plt.tight_layout()
        if show:
            plt.show()
        return fig


def make_all_weights(RBM,data,nweights = 10, weights=None,name = 'all_weights.pdf',figsize=None,sort='beta',dpi=200,ticks_every=5):
    mini_name = name[:-4]
    n_h = RBM.n_h
    I = RBM.vlayer.compute_output(data,RBM.weights)
    I_min = I.min()
    I_max = I.max()
    I_range = (I_max-I_min) * np.arange(0,1+0.01,0.01) + I_min
    mean = RBM.hlayer.mean_from_inputs(np.repeat(I_range[:,np.newaxis],n_h,axis=1))

    betas = Proteins_RBM_utils.get_beta(RBM.weights,include_gaps=True)    
    if sort == 'beta':
        order = np.argsort(betas)[::-1]
    elif sort == 'p':
        p = Proteins_utils.get_sparsity(RBM.weights,include_gaps=True)
        order = np.argsort(p)
    else:
        order = np.arange(n_h)

    titles = [r'$||W||_2 = %.2f$'%betas[order[i]] for i in range(n_h)]
    ylabels = ['Weights %s'%(i+1) for i in range(n_h)]


    sub = np.argsort(np.random.randn(len(data)))[:1000]
    all_distances = Proteins_utils.distance(data[sub]).flatten()

    I_ = I.copy()
    if RBM.hidden == 'dReLU':
        eta = RBM.hlayer.eta
        I_ *= np.sign(eta)[np.newaxis,:]

    q = np.percentile(I_,100* (1-20.0/data.shape[0]),axis=0)

    distance_top_features = []
    for k in range(n_h):
        subgroup = data[I_[:,k]>q[k]]
        distance_top_features.append( Proteins_utils.distance(subgroup).flatten() )

    if figsize is None:
        figsize = (  max(int(0.3 * RBM.n_v), 2)  ,  3)        


    n_pages = int(np.ceil(n_h/float(nweights)))
    for k in range(n_pages):
        fig = plt.figure(figsize = (figsize[0]+figsize[1],figsize[1]*nweights))
        gs = gridspec.GridSpec(2*nweights, 2,width_ratios = [figsize[0],figsize[1]])

        for i in range(k*nweights,(k+1)*nweights):
            ii = i%nweights
            ax1 = fig.add_subplot(gs[2*ii:2*ii+2, 0])
            ax2 = fig.add_subplot(gs[2*ii, 1])
            ax3 = fig.add_subplot(gs[2*ii+1, 1])
            sequence_logo.Sequence_logo(RBM.weights[order[i]],ax=ax1,
                ylabel = ylabels[i], title=titles[i]
                ,ticks_every=ticks_every,ticks_labels_size=14,title_size=20,show=False)
            plot_input_mean(RBM,I, order[i], I_range=I_range,mean=mean,weights = weights, ax = ax2,xlabels=[''])
            plot_top_activating_distance(RBM, I, None,order[i], nseqs = 20,all_distances = all_distances, distance_top_features = distance_top_features,ax=ax3,xlabels=[''])
        plt.tight_layout()
        fig.savefig(mini_name+'tmp_#%s.png'%k,dpi=dpi)
        fig.clear()

    command = 'pdfjoin ' + mini_name+'tmp_#*.png -o %s'%name
    os.system(command)
    command = 'rm '+mini_name+'tmp_#*.png'
    os.system(command)
    print 'Make all weights: Done.'
    return 'done'
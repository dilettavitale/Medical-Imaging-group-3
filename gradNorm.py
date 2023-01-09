
import argparse
import torch
from torch import nn
import numpy as np
#from dataset import RegressionDataset
#from model import RegressionModel, RegressionTrain
import matplotlib.pyplot as plt

from torch.utils import data
from torch.autograd import Variable

import torch.nn.functional as F

class gradNorm:
    def __init__(self,model,optimizer):
        self.model = model
        self.optimizer = optimizer
      
    def forward(self,loss):
        '''# use CUDA if available
        if torch.cuda.is_available():
            self.model.cuda()
        # run n_iter iterations of training'''
        task_losses = []
        loss_ratios = []
        weights = []
        grad_norm_losses = []
        

        #out = UNET.model(X)  #one forward pass in train.py is computed every epoch
        #loss = MultiTaskLoss.forward(out,mask,lab,intens) #in train.py is computed every epoch
        
        weighted_task_loss = torch.mul(self.model.weights,loss)
        
        # initialize the initial loss L(0) if i=0
        initial_task_loss = loss.data.cpu().numpy()
        print(initial_task_loss)

        # get the total loss
        total_loss = torch.sum(weighted_task_loss)
        # clear the gradients
        self.optimizer.zero_grad(set_to_none=True)
        # do the backward pass to compute the gradients for the whole set of weights
        # This is equivalent to compute each \nabla_W L_i(t)
        total_loss.backward(retain_graph=True)

        # set the gradients of w_i(t) to zero because these gradients have to be updated using the GradNorm loss
        #print('Before turning to 0: {}'.format(model.weights.grad))
        self.model.weights.grad.data = self.model.weights.grad.data * 0.0 
        #print('Turning to 0: {}'.format(model.weights.grad))
        W = self.model.get_last_shared_layer()
        #get the gradient norms for each of the tasks 
        #G^{(i)}_w(t)
        norms=[]
        for i in range (len(loss)):
            #get the gradient of this task loss with respect to the shared parameters
            gygw = torch.autograd.grad(loss[i],W.parameters(),retain_graph=True)
            #compute the norm
            norms.append(torch.norm(torch.mul(self.model.weights[i],gygw[0])))
        norms = torch.stack(norms)
        #compute the inverse training rate r_i(t)
        loss_ratio = loss.data.cpu().numpy() / initial_task_loss
        inverse_train_rate = loss_ratio/np.mean(loss_ratio)

        #compute the mean norm \tilde{G}_w(t)
        mean_norm=np.mean(norms.data.cpu().numpy())
        
        #compute the GradNorm loss
        #this term has to remain constant
        constant_term = torch.tensor(mean_norm * (inverse_train_rate ** 0.06),requires_grad = False)
        #this is the GradNorm loss itself
        constant_term = constant_term.cuda()
        Gradloss = nn.L1Loss(reduction='sum')
        grad_norm_loss = 0
        for loss_index in range (0, len(loss)):
            grad_norm_loss = torch.add(grad_norm_loss,Gradloss(norms [loss_index], constant_term[loss_index]))
        # compute the gradient for the weights
        self.model.weights.grad = torch.autograd.grad(grad_norm_loss, self.model.weights)[0]
        self.optimizer.step()
        normalize_coeff=3/torch.sum(self.model.weights.data,dim=0)
        self.model.weights.data =  self.model.weights.data * normalize_coeff
        
    
            # record
        if torch.cuda.is_available():
            task_losses.append(loss.data.cpu().numpy())
            loss_ratios.append(np.sum(task_losses[-1] / task_losses[0]))
            weights.append(self.model.weights.data.cpu().numpy())
            grad_norm_losses.append(grad_norm_loss.data.cpu().numpy())
        else:
            task_losses.append(loss.data.numpy())
            loss_ratios.append(np.sum(task_losses[-1] / task_losses[0]))
            weights.append(self.model.weights.data.numpy())
            grad_norm_losses.append(grad_norm_loss.data.numpy())

        if i % 100 == 0:  
            if torch.cuda.is_available():
                print('{}/{}: loss_ratio={}, weights={}, task_loss={}, grad_norm_loss={}'.format(
                                    i, args.n_iter, loss_ratio[-1], self.model.weights.data.cpu().numpy(), loss.data.cpu().numpy(), grad_norm_loss.data.cpu().numpy()))
            else:
                print('{}/{}: loss_ratio={}, weights={}, task_loss={}, grad_norm_loss={}'.format(
                    i, args.n_iter, loss_ratio[-1], self.model.weights.data.numpy(), loss.data.numpy(), grad_norm_loss.data.numpy()))

        task_losses = np.array(task_losses)
        weights = np.array(weights)
        return task_losses
      
        

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np

import torch
import torch.utils.data as data


from torch.autograd import Variable

from utils import *


class MODEL_LSTM(nn.Module):
    '''
        !!note : with tying weights, emb_size must be equal to hidden_size. Setting hidden_size to emb_size
    '''
    def __init__(self, input_size, hidden_size, vocab_size,weight_tyining = True,variational_dropout = True , pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1,dropprob=0.2):
        super(MODEL_LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, input_size,padding_idx=pad_index)
        #self.emblin = nn.Linear(hidden_size, input_size) #used to allow weight tying with hidden_size != emb_size
        self.lstm = nn.LSTM(input_size, hidden_size,n_layers)
        self.pad_token = pad_index
        self.output = nn.Linear(hidden_size, vocab_size)
        self.embedding_dropout = nn.Dropout(emb_dropout)
        self.variational_dropout = variational_dropout
        self.out_dropout = out_dropout

        if weight_tyining:
            # Weight tying: condivisione dei pesi tra il layer di output e l'embedding
            self.output.weight = self.embedding.weight  # Tying the weights

        if self.variational_dropout:
            self.var_dropout_layer= VariationalDropout(dropprob) #da vedere se funzia anche con (out_dropout)
        else:
            self.var_dropout_layer = nn.Dropout(out_dropout)
            
        #before last layer add out_droput
        # self.output_dropout = nn.Dropout(self.var_out_dropout)
        # self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        emb = self.embedding_dropout(emb)

        if self.variational_dropout:
            emb = self.var_dropout_layer(emb)
        lstm_out, _ = self.lstm(emb)

        if self.variational_dropout:
            lstm_out = self.var_dropout_layer(lstm_out)

        # output = self.output_dropout(lstm_out) #droppout before last linear layer
        output = self.output(lstm_out).permute(0,2,1)
        return output

    def get_word_embedding(self, token):
        '''
            This method returns the embedding vector associated with a specific word. 
            It is used to obtain the vector representation of a word in the context 
            of the embedding space of the model. 
        '''
        return self.embedding(token).squeeze(0).detach().cpu().numpy()

    def get_most_similar(self, vector, top_k=10):
        '''
            This method returns the most similar words to a specified embedding vector. 
            It is used to find the most similar words in the vocabulary to a given embedding vector
        '''
        embs = self.embedding.weight.detach().cpu().numpy()
        #Our function that we used before
        scores = []
        for i, x in enumerate(embs):
            if i != self.pad_token:
                scores.append(cosine_similarity(x, vector))
        # Take ids of the most similar tokens
        scores = np.asarray(scores)
        indexes = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[indexes]
        return (indexes, top_scores)


class VariationalDropout(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.dropout = dropout

    def forward(self, input_tensor ):
        '''
            defines the behavior of the class when forwarding data through the model
            input_tensor = represents the input tensor on which to apply the variable dropout
        '''
        if not self.training or not self.dropout:
            return input_tensor
        m = input_tensor.data.new(1, input_tensor.size(1), input_tensor.size(2)).bernoulli_(1 - self.dropout)
        mask = Variable(m, requires_grad=False) / (1 - self.dropout)
        mask = mask.expand_as(input_tensor)
        return mask * input_tensor

class NTASGD(optim.Optimizer):
    '''
        Class that implement a custom Non-monotonic Adaptive Stochastic Gradient Descent.
        Adjust the learning rate during training so that it can more effectively 
        handle cases where the loss function does not monotonically decrease over iterations.
        
        Take example of this function by the following repo:
            https://github.com/ahmetumutdurmus/awd-lstm/blob/master/ntasgd.py
    '''
    def __init__(self, params, lr=1, n=5):
        defaults = dict(lr=lr, n=n, t0=10e7, t=0, logs=[])
        super(NTASGD, self).__init__(params, defaults)

    def check(self, v):
        group = self.param_groups[0]
        #Training
        if group['t'] > group['n'] and v > min(group['logs'][:-group['n']]):
            group['t0'] = self.state[next(iter(group['params']))]['step']
            print("Non-monotonic condition is triggered!")
            return True
        group['logs'].append(v)
        group['t'] += 1

    def set_lr(self, lr):
        for group in self.param_groups:
            group['lr'] = lr

    def step(self):
        group = self.param_groups[0]
        for p in group['params']:
            grad = p.grad.data
            state = self.state[p]
            # State initialization
            if len(state) == 0:
                state['step'] = 0
                state['mu'] = 1
                state['ax'] = torch.zeros_like(p.data)
            state['step'] += 1

            p.data.add_(other=grad, alpha=-group['lr'])
            # averaging
            if state['mu'] != 1:
                state['ax'].add_(p.data.sub(state['ax']).mul(state['mu']))
            else:
                state['ax'].copy_(p.data)
            # update mu
            state['mu'] = 1 / max(1, state['step'] - group['t0'])


def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 14:19:58 2021

@author: black
"""

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
import os



#Adding checkpoint functionality inside the class definition, BUT
#ONLY FOR meaningful names: this could be done outside the llop just as easily 
class DeepQNetwork(nn.Module):
  def __init__(self, input_dims, n_actions, name="None", chkpt_dir="None", lr=0.001):
    super(DeepQNetwork, self).__init__()

    self.input_dims = input_dims
    
    self.checkpoint_dir = chkpt_dir
    self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
  
    #I have no idea if I have the correct shape input_dims[1] = NOT CORRECT
    #input_dims[0] is the correct one (which means I have an error with the env setup)
    #input_dims[0] correspongs to the number of channels I 've got, in our case 
    #4 frames with one channel for grayscale image
    self.conv1 = nn.Conv2d(input_dims[0], 32, kernel_size=8, stride=4)

    self.conv2 = nn.Conv2d(32, 64,  kernel_size=4, stride=2)

    self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

    self.fc1 = nn.Linear(self.print_cnn_dim(), 512)

    self.fc2 = nn.Linear(512, n_actions)

    
    self.loss = nn.MSELoss()
    self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
    self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
    self.to(self.device)

  def forward(self, state):
    #NOTE: USE BETTER TERMINOLOGY THAN "x" and "out" like "state" and "action"
    out = F.relu(self.conv1(state))
    out = F.relu(self.conv2(out))
    out = F.relu(self.conv3(out)) #shape: (Batch_Size, no_filters, H,W )
    out = out.view(-1, self.print_cnn_dim() )  #np.reshape equivalent
    #out = out.view(out.size()[0], -1 )  #equivalent to above
    out = F.relu(self.fc1(out))
    actions = self.fc2(out)
    return actions

  def print_cnn_dim(self):
    '''
    Usage: Pass in the matrix of zeros with a batch size of one and dimensions 
    of input shape. What dimension comes out use out.view(dim, -1) to input to fc
    '''
    x = T.zeros((1, *self.input_dims))
    #print(x.shape)
    out = self.conv1(x)
    out = self.conv2(out)
    out = self.conv3(out)
    #print(f"FC-dimensions: {out.shape}")
    return T.prod(T.tensor(out.shape[1:])).item()

  def save_checkpoint(self):
    print("... saving checkpount ...")
    T.save(self.state_dict(), self.checkpoint_file)
  
  def load_checkpoint(self):
    print("... loading checkpoint ...")
    self.load_state_dict(T.load(self.checkpoint_file))
    

#Adding checkpoint functionality inside the class definition, BUT
#ONLY FOR meaningful names: this could be done outside the llop just as easily 
class DuelingDeepQNetwork(nn.Module):
  def __init__(self, input_dims, n_actions, name="None", chkpt_dir="None", lr=0.001):
    super(DuelingDeepQNetwork, self).__init__()

    self.input_dims = input_dims
    
    self.checkpoint_dir = chkpt_dir
    self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
  

    self.conv1 = nn.Conv2d(input_dims[0], 32, kernel_size=8, stride=4)

    self.conv2 = nn.Conv2d(32, 64,  kernel_size=4, stride=2)

    self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

    self.fc1 = nn.Linear(self.print_cnn_dim(), 512)  # => this remains as-is
    
    self.value_output = nn.Linear(512, 1)

    self.advantage_output = nn.Linear(512, n_actions)

    
    self.loss = nn.MSELoss()
    self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
    self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
    self.to(self.device)

  def forward(self, state):
    #NOTE: USE BETTER TERMINOLOGY THAN "x" and "out" like "state" and "action"
    out = F.relu(self.conv1(state))
    out = F.relu(self.conv2(out))
    out = F.relu(self.conv3(out)) #shape: (Batch_Size, no_filters, H,W )
    out = out.view(-1, self.print_cnn_dim() )  #np.reshape equivalent
    out = F.relu(self.fc1(out))

    state_value = self.value_output(out)
    advantage_values = self.advantage_output(out)

    #different from model code which returns both values, sticking to mine since
    #iny my view it is simpler and requires no further changes in agent code
    q_values = state_value + ( advantage_values -  advantage_values.mean(axis=1).unsqueeze(1)) 
    return q_values
    #return state_value, advantage_values



  def print_cnn_dim(self):
    '''
    Usage: Pass in the matrix of zeros with a batch size of one and dimensions 
    of input shape. What dimension comes out use out.view(dim, -1) to input to fc
    '''
    x = T.zeros((1, *self.input_dims))
    #print(x.shape)
    out = self.conv1(x)
    out = self.conv2(out)
    out = self.conv3(out)
    #print(f"FC-dimensions: {out.shape}")
    return T.prod(T.tensor(out.shape[1:])).item()

  def save_checkpoint(self):
    print("... saving checkpount ...")
    T.save(self.state_dict(), self.checkpoint_file)
  
  def load_checkpoint(self):
    print("... loading checkpoint ...")
    self.load_state_dict(T.load(self.checkpoint_file))


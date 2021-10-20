# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 14:19:52 2021

@author: black
"""

import numpy as np
#Solution code (Compatibility reasons)

#THIS CODE along with the rest of the enviro can be used for ANY atari
#game since it is totally generic
class ReplayBuffer():
  def __init__(self, max_size, input_shape, n_actions):
    self.mem_size = max_size 

    #position of last stored memory
    self.mem_cntr = 0

    #input_shape is for handling states of variable length 
    #DTYPE is for pytorch particulars plus 64 bit too much memory
    self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
    
    self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)

    self.action_memory = np.zeros(self.mem_size, dtype=np.int64)

    self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)

    self.terminal_memory = np.zeros(self.mem_size, dtype=np.uint8)

  def store_transition(self,state ,action, reward, state_, done):
    
    #cleverer way to handle it, what they did in GDRL
    index = self.mem_cntr % self.mem_size
    self.state_memory[index] = state
    self.action_memory[index] = action
    self.reward_memory[index] = reward
    self.new_state_memory[index] = state_
    self.terminal_memory[index] = done
    self.mem_cntr += 1

  def sample_buffer(self, batch_size):
    max_mem = min(self.mem_cntr, self.mem_size)

    batch = np.random.choice(max_mem, batch_size, replace=False)

    states = self.state_memory[batch]
    actions = self.action_memory[batch]
    rewards = self.reward_memory[batch]
    next_states = self.new_state_memory[batch]
    dones = self.terminal_memory[batch]

    return states, actions, rewards, next_states, dones


#MY CODE - should work equally well, try it out later
class MemoryBuffer():
  def __init__(self, env, N):
    self.N = N
    self.buffer_shape = tuple([self.N] + list(env.reset().shape ))
    self.memory_ind = 0
    self.actions_buffer = np.zeros( self.N  , dtype=np.uint8) 
    self.rewards_buffer = np.zeros( self.N )    #better to use states as np.float32. ONLY USE UINT8 for PLOTTING PURPOSES
    self.states_buffer = np.zeros( self.buffer_shape  , dtype=np.float32) 
    self.next_states_buffer = np.zeros( self.buffer_shape  , dtype=np.float32) 
    self.done_buffer = np.zeros( self.N  , dtype=np.uint8) 

  def print_self(self):
    print(f"buffer shape: {self.buffer_shape}")
    print(f"actions shape (1-D): {self.actions_buffer.shape}")
    print(f"states shape (3-D): {self.next_states_buffer.shape}")


  def store_experience(self, ind, state, action, reward, new_state, done):
    #cucling indexes will be taken care outside of the function
    #beter yet handle it inside here, with modulo:

    ind =  self.memory_ind % self.N
    self.actions_buffer[ind] = action
    self.rewards_buffer[ind] = reward
    self.states_buffer[ind] = state
    self.next_states_buffer[ind] = new_state
    self.done_buffer[ind] = done
    self.memory_ind += 1

    #NOW IT DOESN'T NEED THE outer loop modification

  def retrieve_experiences(self, batch_size=16):
    #suggestion: use np.vstack() on each element to return np.arrays per elem entry
    random_indices = np.random.choice(self.N, size=batch_size, replace=False)
    action = self.actions_buffer[random_indices]
    reward = self.rewards_buffer[random_indices]
    state = self.states_buffer[random_indices]
    new_state = self.next_states_buffer[random_indices]
    done = self.done_buffer[random_indices]
    return [state, action, reward, new_state, done]

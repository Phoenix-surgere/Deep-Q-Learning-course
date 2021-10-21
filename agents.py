# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 14:44:49 2021

@author: black
"""
import numpy as np
import torch as T
from nn_networks import DeepQNetwork, DuelingDeepQNetwork
from replay_memories import ReplayBuffer



class Agent():
  def __init__(self, gamma, epsilon, lr, n_actions, input_dims, 
               mem_size, batch_size, eps_min=0.01, eps_dec = 5e-7, 
               #what is replace for? A: Every "replace" steps update target network
               replace=1000, algo=None, env_name=None, chkpt_dir="tmp/dqn"):
    
    self.gamma = gamma
    self.epsilon = epsilon
    self.lr = lr
    self.n_actions = n_actions
    self.input_dims = input_dims
    self.batch_size = batch_size
    self.eps_min = eps_min
    self.eps_dec = eps_dec
    self.replace_target_cnt = replace
    self.algo = algo
    self.env_name = env_name
    self.chkpt_dir = chkpt_dir
    self.action_space = [i for i in range(self.n_actions)]  #for e-greeedy selection
    
    #usefl to count number of times online network ran so we know when to update
    #target network with eval network weights
    self.learn_step_counter = 0

    self.memory = ReplayBuffer(mem_size, input_dims, n_actions)


  def store_transition(self, state, action, reward, next_state, done):
    self.memory.store_transition(state, action, reward, next_state, done)

  
  def sample_memory(self):
    state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)
    states = T.tensor(state).to(self.q_online.device)
    actions = T.tensor(action).to(self.q_online.device)
    rewards = T.tensor(reward).to(self.q_online.device)
    next_states = T.tensor(next_state).to(self.q_online.device)
    dones = T.tensor(done).to(self.q_online.device)
    return states, actions, rewards, next_states, dones


   #That I did not do because it is very simple and will rework it later 
 #on to utilize more schemas for experimenting
  def decrement_epsilon(self):
    self.epsilon = self.epsilon - self.eps_dec \
                if self.epsilon > self.eps_min else self.eps_min

  
  #Saving and loading models -> That I didn't do because I couldn't be botherd
  def save_models(self):
    self.q_online.save_checkpoint()
    self.q_target.save_checkpoint()

  def load_models(self):
    self.q_online.save_checkpoint()
    self._q_target.save_checkpoint()
  

  def replace_target_network(self):
    if self.replace_target_cnt is not None and self.learn_step_counter % self.replace_target_cnt == 0:
      self.q_target.load_state_dict(self.q_online.state_dict())
  
  #Since I made the dueling Q-Net return Q-values directly I didn't actually have to 
  #modify my choose_action at any point so it is considered a constant 
  def choose_action(self, observation):
    #Need [observation] because torch expect (batch_size, input_dims) 
    #and without them it will only be input_dims
    if np.random.random() > self.epsilon:
        state = T.tensor( [observation] , dtype=T.float).to(self.q_online.device)
        actions = self.q_online.forward(state)
        action = T.argmax(actions).item() #gym onlci accepts integers and numpy arrays
    else:
        action = np.random.choice(self.action_space)
    return action
    

  def learn(self):
    raise NotImplementedError


class DQNAgent_NEW(Agent):
  #args: keyword arguments without default values
  #kwargs: keyword argument WITH default values (eg path="temp")
  #so I can now pass an arbitrary number of both of these using this syntax
  def __init__(self, *args, **kwargs):
    #Calls base constructor and pass in all of our arguments 
    super(DQNAgent_NEW, self).__init__(*args, **kwargs)
    
    self.q_online = DeepQNetwork(lr=self.lr, n_actions=self.n_actions, 
                               input_dims = self.input_dims,
                               name = self.env_name+"_"+ self.algo+'_q_online',
                              chkpt_dir = self.chkpt_dir)

    
    self.q_target = DeepQNetwork(lr=self.lr, n_actions=self.n_actions, 
                               input_dims = self.input_dims,
                               name = self.env_name+"_"+ self.algo+ '_q_target',
                              chkpt_dir = self.chkpt_dir)

  def learn(self):
    #bail out if we don't have more than batch_size items in our memory buffer
    if self.memory.mem_cntr < self.batch_size:
      return
    
    self.q_online.optimizer.zero_grad()

    self.replace_target_network()

    states, actions, rewards, next_states, dones = self.sample_memory()

    #way to deal with the wrong demension
    indices = np.arange(self.batch_size)
    q_pred = self.q_online.forward(states)[indices, actions] #dims -> batch_size x n_actions

    # max returns (max_value, index_of_max) hence we only want the first
    q_next = self.q_target.forward(next_states).max(dim=1)[0]
    
    #if state is terminal target = reward otherwise it's td-target as usual
    #could do a conditional but here's a smarter way with done as a mask:
    #NOTE that this is the only detail I didn't get from my own implementation
    q_next[dones] = 0.0
    #if dones.any() == 1: print(q_next, dones)

    q_target = rewards + self.gamma * q_next

    loss = self.q_online.loss(q_target, q_pred).to(self.q_online.device)
    loss.backward()
    self.q_online.optimizer.step()
    self.learn_step_counter += 1

    self.decrement_epsilon()

    
class DDQNAgent_NEW(Agent):
  def __init__(self, *args, **kwargs):
    #Calls base constructor and pass in all of our arguments 
    super(DDQNAgent_NEW, self).__init__(*args, **kwargs)
    
    self.q_online = DeepQNetwork(lr=self.lr, n_actions=self.n_actions, 
                               input_dims = self.input_dims,
                               name = self.env_name+"_"+ self.algo+'_q_online',
                              chkpt_dir = self.chkpt_dir)

    
    self.q_target = DeepQNetwork(lr=self.lr, n_actions=self.n_actions, 
                               input_dims = self.input_dims,
                               name = self.env_name+"_"+ self.algo+ '_q_target',
                              chkpt_dir = self.chkpt_dir)


  def learn(self):
    if self.memory.mem_cntr < self.batch_size:
      return

    self.q_online.optimizer.zero_grad()

    self.replace_target_network()

    states, actions, rewards, states_, dones = self.sample_memory()

    indices = np.arange(self.batch_size)

    q_pred = self.q_online.forward(states)[indices, actions]
    q_next = self.q_target.forward(states_)
    q_eval = self.q_online.forward(states_)

    max_actions = T.argmax(q_eval, dim=1)
    q_next[dones] = 0.0

    q_target = rewards + self.gamma*q_next[indices, max_actions]
    loss = self.q_online.loss(q_target, q_pred).to(self.q_online.device)
    loss.backward()

    self.q_online.optimizer.step()
    self.learn_step_counter += 1

    self.decrement_epsilon()
  
  def learn_BUGGED(self):
    #bail out if we don't have more than batch_size items in our memory buffer
    if self.memory.mem_cntr < self.batch_size:
      return
    
    self.q_online.optimizer.zero_grad()

    self.replace_target_network()

    states, actions, rewards, next_states, dones = self.sample_memory()

    #way to deal with the wrong demension
    indices = np.arange(self.batch_size)
    q_pred = self.q_online.forward(states)[indices, actions] #dims -> batch_size x n_actions

    #q_next = self.q_target.forward(next_states)
    q_eval = self.q_online.forward(next_states)

    #print(f"q_eval (from which we select argmax: {q_eval}")

    acts = q_eval.max(dim=1)[1]
    
    #print(f"acts: {acts}, shape: {acts.shape}")

    q_next = self.q_target.forward(next_states).gather(1, acts.unsqueeze(0)) 
    #print(f"q_next: {q_next}, shape: {q_next.shape}")
    #print(f"dones: {dones.shape}, dones: {dones}")
   
    #print(f"q_next init: {q_next}")
    q_next[dones.type(T.int64) ] = 0.0
    #q_next.gather(1, dones.unsqueeze(0).type(T.int64) )
    #print(f"q_next {q_next}")

    q_target = rewards + self.gamma * q_next

    loss = self.q_online.loss(q_target, q_pred).to(self.q_online.device)
    loss.backward()
    self.q_online.optimizer.step()
    self.learn_step_counter += 1

    self.decrement_epsilon()
    
  
class DuelingQNAgent_NEW(Agent):
  def __init__(self, *args, **kwargs):
    #Calls base constructor and pass in all of our arguments 
    super(DuelingQNAgent_NEW, self).__init__(*args, **kwargs)

    self.q_online = DuelingDeepQNetwork(lr=self.lr, n_actions=self.n_actions, 
                                 input_dims = self.input_dims,
                                 name = self.env_name+"_"+ self.algo+'_q_online',
                                chkpt_dir = self.chkpt_dir)


    self.q_target = DuelingDeepQNetwork(lr=self.lr, n_actions=self.n_actions, 
                                 input_dims = self.input_dims,
                                 name = self.env_name+"_"+ self.algo+ '_q_target',
                                chkpt_dir = self.chkpt_dir)
    
  def learn(self):
    if self.memory.mem_cntr < self.batch_size:
      return

    self.q_online.optimizer.zero_grad()

    self.replace_target_network()

    states, actions, rewards, states_, dones = self.sample_memory()

    indices = np.arange(self.batch_size)

    q_pred = self.q_online.forward(states)[indices, actions]
    q_next = self.q_target.forward(states_)
    q_eval = self.q_online.forward(states_)

    max_actions = T.argmax(q_eval, dim=1)
    q_next[dones] = 0.0

    q_target = rewards + self.gamma*q_next[indices, max_actions]
    loss = self.q_online.loss(q_target, q_pred).to(self.q_online.device)
    loss.backward()

    self.q_online.optimizer.step()
    self.learn_step_counter += 1

    self.decrement_epsilon()  
  
class DuelingDDQNAgent_NEW(Agent):
  def __init__(self, *args, **kwargs):
    #Calls base constructor and pass in all of our arguments 
    super(DuelingDDQNAgent_NEW, self).__init__(*args, **kwargs)

    self.q_online = DuelingDeepQNetwork(lr=self.lr, n_actions=self.n_actions, 
                                 input_dims = self.input_dims,
                                 name = self.env_name+"_"+ self.algo+'_q_online',
                                chkpt_dir = self.chkpt_dir)


    self.q_target = DuelingDeepQNetwork(lr=self.lr, n_actions=self.n_actions, 
                                 input_dims = self.input_dims,
                                 name = self.env_name+"_"+ self.algo+ '_q_target',
                                chkpt_dir = self.chkpt_dir)
 

  def learn(self):
    if self.memory.mem_cntr < self.batch_size:
        return

    self.q_online.optimizer.zero_grad()

    self.replace_target_network()

    states, actions, rewards, states_, dones = self.sample_memory()

    indices = np.arange(self.batch_size)

    q_pred = self.q_online.forward(states)[indices, actions]
    q_next = self.q_target.forward(states_)
    q_eval = self.q_online.forward(states_)

    max_actions = T.argmax(q_eval, dim=1)
    q_next[dones] = 0.0

    q_target = rewards + self.gamma*q_next[indices, max_actions]
    loss = self.q_online.loss(q_target, q_pred).to(self.q_online.device)
    loss.backward()

    self.q_online.optimizer.step()
    self.learn_step_counter += 1

    self.decrement_epsilon()  

  
  def learn_BUGGED(self):
    #bail out if we don't have more than batch_size items in our memory buffer
    if self.memory.mem_cntr < self.batch_size:
      return
    
    self.q_online.optimizer.zero_grad()

    self.replace_target_network()

    states, actions, rewards, next_states, dones = self.sample_memory()

    #way to deal with the wrong demension
    indices = np.arange(self.batch_size)
    q_pred = self.q_online.forward(states)[indices, actions] #dims -> batch_size x n_actions

    #q_next = self.q_target.forward(next_states)
    q_eval = self.q_online.forward(next_states)

    #print(f"q_eval (from which we select argmax: {q_eval}")

    acts = q_eval.max(dim=1)[1]
    
    #print(f"acts: {acts}, shape: {acts.shape}")

    q_next = self.q_target.forward(next_states).gather(1, acts.unsqueeze(0)) 
    #print(f"q_next: {q_next}, shape: {q_next.shape}")
    #print(f"dones: {dones.shape}, dones: {dones}")
   
    #print(f"q_next init: {q_next}")
    q_next[dones.type(T.int64) ] = 0.0
    #q_next.gather(1, dones.unsqueeze(0).type(T.int64) )
    #print(f"q_next {q_next}")

    q_target = rewards + self.gamma * q_next

    loss = self.q_online.loss(q_target, q_pred).to(self.q_online.device)
    loss.backward()
    self.q_online.optimizer.step()
    self.learn_step_counter += 1

    self.decrement_epsilon()

    
    
 #OLD NON-FACTORED CODE------------------------------------
#MODEL SOLUTION (mainly for compatibility and comparison purposes)
#NOTE: I do have my own skeleton, could replace the model solution with it
#if I ensure it works
class DQNAgent():
  def __init__(self, gamma, epsilon, lr, n_actions, input_dims, 
               mem_size, batch_size, eps_min=0.01, eps_dec = 5e-7, 
               #what is replace for? A: Every "replace" steps update target network
               replace=1000, algo=None, env_name=None, chkpt_dir="tmp/dqn"):
    
    self.gamma = gamma
    self.epsilon = epsilon
    self.lr = lr
    self.n_actions = n_actions
    self.input_dims = input_dims
    self.batch_size = batch_size
    self.eps_min = eps_min
    self.eps_dec = eps_dec
    self.replace_target_cnt = replace
    self.algo = algo
    self.env_name = env_name
    self.chkpt_dir = chkpt_dir
    self.action_space = [i for i in range(self.n_actions)]  #for e-greeedy selection
    
    #usefl to count number of times online network ran so we know when to update
    #target network with eval network weights
    self.learn_step_counter = 0

    self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

    self.q_online = DeepQNetwork(lr=self.lr, n_actions=self.n_actions, 
                               input_dims = self.input_dims,
                               name = self.env_name+"_"+ self.algo+'_q_online',
                              chkpt_dir = self.chkpt_dir)

    
    self.q_target = DeepQNetwork(lr=self.lr, n_actions=self.n_actions, 
                               input_dims = self.input_dims,
                               name = self.env_name+"_"+ self.algo+ '_q_target',
                              chkpt_dir = self.chkpt_dir)

  def choose_action(self, observation):
    #Need [observation] because torch expect (batch_size, input_dims) 
    #and without them it will only be input_dims
    if np.random.random() > self.epsilon:
        state = T.tensor( [observation] , dtype=T.float).to(self.q_online.device)
        actions = self.q_online.forward(state)
        action = T.argmax(actions).item() #gym onlci accepts integers and numpy arrays
    else:
        action = np.random.choice(self.action_space)
    return action
    

  def store_transition(self, state, action, reward, next_state, done):
    self.memory.store_transition(state, action, reward, next_state, done)

  
  def sample_memory(self):
    state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)
    states = T.tensor(state).to(self.q_online.device)
    actions = T.tensor(action).to(self.q_online.device)
    rewards = T.tensor(reward).to(self.q_online.device)
    next_states = T.tensor(next_state).to(self.q_online.device)
    dones = T.tensor(done).to(self.q_online.device)
    return states, actions, rewards, next_states, dones

  
  def replace_target_network(self):
    if self.replace_target_cnt is not None and self.learn_step_counter % self.replace_target_cnt == 0:
      self.q_target.load_state_dict(self.q_online.state_dict())
 
 #That I did not do because it is very simple and will rework it later 
 #on to utilize more schemas for experimenting
  def decrement_epsilon(self):
    self.epsilon = self.epsilon - self.eps_dec \
                if self.epsilon > self.eps_min else self.eps_min

  
  #Saving and loading models -> That I didn't do because I couldn't be botherd
  def save_models(self):
    self.q_online.save_checkpoint()
    self.q_target.save_checkpoint()

  def load_models(self):
    self.q_online.save_checkpoint()
    self._q_target.save_checkpoint()
  
  def learn(self):
    #bail out if we don't have more than batch_size items in our memory buffer
    if self.memory.mem_cntr < self.batch_size:
      return
    
    self.q_online.optimizer.zero_grad()

    self.replace_target_network()

    states, actions, rewards, next_states, dones = self.sample_memory()

    #way to deal with the wrong demension
    indices = np.arange(self.batch_size)
    q_pred = self.q_online.forward(states)[indices, actions] #dims -> batch_size x n_actions

    # max returns (max_value, index_of_max) hence we only want the first
    q_next = self.q_target.forward(next_states).max(dim=1)[0]
    
    #if state is terminal target = reward otherwise it's td-target as usual
    #could do a conditional but here's a smarter way with done as a mask:
    #NOTE that this is the only detail I didn't get from my own implementation
    q_next[dones] = 0.0
    #if dones.any() == 1: print(q_next, dones)

    q_target = rewards + self.gamma * q_next

    loss = self.q_online.loss(q_target, q_pred).to(self.q_online.device)
    loss.backward()
    self.q_online.optimizer.step()
    self.learn_step_counter += 1

    self.decrement_epsilon()


class DDQNAgent():
  def __init__(self, gamma, epsilon, lr, n_actions, input_dims, 
               mem_size, batch_size, eps_min=0.01, eps_dec = 5e-7, 
               #what is replace for? A: Every "replace" steps update target network
               replace=1000, algo=None, env_name=None, chkpt_dir="tmp/dqn"):
    
    self.gamma = gamma
    self.epsilon = epsilon
    self.lr = lr
    self.n_actions = n_actions
    self.input_dims = input_dims
    self.batch_size = batch_size
    self.eps_min = eps_min
    self.eps_dec = eps_dec
    self.replace_target_cnt = replace
    self.algo = algo
    self.env_name = env_name
    self.chkpt_dir = chkpt_dir
    self.action_space = [i for i in range(self.n_actions)]  #for e-greeedy selection
    
    #usefl to count number of times online network ran so we know when to update
    #target network with eval network weights
    self.learn_step_counter = 0

    self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

    self.q_online = DeepQNetwork(lr=self.lr, n_actions=self.n_actions, 
                               input_dims = self.input_dims,
                               name = self.env_name+"_"+ self.algo+'_q_online',
                              chkpt_dir = self.chkpt_dir)

    
    self.q_target = DeepQNetwork(lr=self.lr, n_actions=self.n_actions, 
                               input_dims = self.input_dims,
                               name = self.env_name+"_"+ self.algo+ '_q_target',
                              chkpt_dir = self.chkpt_dir)

  def choose_action(self, observation):
    #Need [observation] because torch expect (batch_size, input_dims) 
    #and without them it will only be input_dims
    if np.random.random() > self.epsilon:
        state = T.tensor( [observation] , dtype=T.float).to(self.q_online.device)
        actions = self.q_online.forward(state)
        action = T.argmax(actions).item() #gym onlci accepts integers and numpy arrays
    else:
        action = np.random.choice(self.action_space)
    return action
    

  def store_transition(self, state, action, reward, next_state, done):
    self.memory.store_transition(state, action, reward, next_state, done)

  
  def sample_memory(self):
    state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)
    states = T.tensor(state).to(self.q_online.device)
    actions = T.tensor(action).to(self.q_online.device)
    rewards = T.tensor(reward).to(self.q_online.device)
    next_states = T.tensor(next_state).to(self.q_online.device)
    dones = T.tensor(done).to(self.q_online.device)
    return states, actions, rewards, next_states, dones

  
  def replace_target_network(self):
    if self.replace_target_cnt is not None and self.learn_step_counter % self.replace_target_cnt == 0:
      self.q_target.load_state_dict(self.q_online.state_dict())
 
 #That I did not do because it is very simple and will rework it later 
 #on to utilize more schemas for experimenting
  def decrement_epsilon(self):
    self.epsilon = self.epsilon - self.eps_dec \
                if self.epsilon > self.eps_min else self.eps_min

  
  #Saving and loading models -> That I didn't do because I couldn't be botherd
  def save_models(self):
    self.q_online.save_checkpoint()
    self.q_target.save_checkpoint()

  def load_models(self):
    self.q_online.save_checkpoint()
    self._q_target.save_checkpoint()
  
  def learn(self):
    if self.memory.mem_cntr < self.batch_size:
      return

    self.q_online.optimizer.zero_grad()

    self.replace_target_network()

    states, actions, rewards, states_, dones = self.sample_memory()

    indices = np.arange(self.batch_size)

    q_pred = self.q_online.forward(states)[indices, actions]
    q_next = self.q_target.forward(states_)
    q_eval = self.q_online.forward(states_)

    max_actions = T.argmax(q_eval, dim=1)
    q_next[dones] = 0.0

    q_target = rewards + self.gamma*q_next[indices, max_actions]
    loss = self.q_online.loss(q_target, q_pred).to(self.q_online.device)
    loss.backward()

    self.q_online.optimizer.step()
    self.learn_step_counter += 1

    self.decrement_epsilon()
  
  def learn_BUGGED(self):
    #bail out if we don't have more than batch_size items in our memory buffer
    if self.memory.mem_cntr < self.batch_size:
      return
    
    self.q_online.optimizer.zero_grad()

    self.replace_target_network()

    states, actions, rewards, next_states, dones = self.sample_memory()

    #way to deal with the wrong demension
    indices = np.arange(self.batch_size)
    q_pred = self.q_online.forward(states)[indices, actions] #dims -> batch_size x n_actions

    #q_next = self.q_target.forward(next_states)
    q_eval = self.q_online.forward(next_states)

    #print(f"q_eval (from which we select argmax: {q_eval}")

    acts = q_eval.max(dim=1)[1]
    
    #print(f"acts: {acts}, shape: {acts.shape}")

    q_next = self.q_target.forward(next_states).gather(1, acts.unsqueeze(0)) 
    #print(f"q_next: {q_next}, shape: {q_next.shape}")
    #print(f"dones: {dones.shape}, dones: {dones}")
   
    #print(f"q_next init: {q_next}")
    q_next[dones.type(T.int64) ] = 0.0
    #q_next.gather(1, dones.unsqueeze(0).type(T.int64) )
    #print(f"q_next {q_next}")

    q_target = rewards + self.gamma * q_next

    loss = self.q_online.loss(q_target, q_pred).to(self.q_online.device)
    loss.backward()
    self.q_online.optimizer.step()
    self.learn_step_counter += 1

    self.decrement_epsilon()
    
  
    
    
class DuelingQNAgent():
  def __init__(self, gamma, epsilon, lr, n_actions, input_dims, 
               mem_size, batch_size, eps_min=0.01, eps_dec = 5e-7, 
               #what is replace for? A: Every "replace" steps update target network
               replace=1000, algo=None, env_name=None, chkpt_dir="tmp/dqn"):
    
    self.gamma = gamma
    self.epsilon = epsilon
    self.lr = lr
    self.n_actions = n_actions
    self.input_dims = input_dims
    self.batch_size = batch_size
    self.eps_min = eps_min
    self.eps_dec = eps_dec
    self.replace_target_cnt = replace
    self.algo = algo
    self.env_name = env_name
    self.chkpt_dir = chkpt_dir
    self.action_space = [i for i in range(self.n_actions)]  #for e-greeedy selection
    
    #usefl to count number of times online network ran so we know when to update
    #target network with eval network weights
    self.learn_step_counter = 0

    self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

    self.q_online = DuelingDeepQNetwork(lr=self.lr, n_actions=self.n_actions, 
                               input_dims = self.input_dims,
                               name = self.env_name+"_"+ self.algo+'_q_online',
                              chkpt_dir = self.chkpt_dir)

    
    self.q_target = DuelingDeepQNetwork(lr=self.lr, n_actions=self.n_actions, 
                               input_dims = self.input_dims,
                               name = self.env_name+"_"+ self.algo+ '_q_target',
                              chkpt_dir = self.chkpt_dir)

  def choose_action(self, observation):
    #Need [observation] because torch expect (batch_size, input_dims) 
    #and without them it will only be input_dims
    if np.random.random() > self.epsilon:
        state = T.tensor( [observation] , dtype=T.float).to(self.q_online.device)
        actions = self.q_online.forward(state)
        action = T.argmax(actions).item() #gym onlci accepts integers and numpy arrays
    else:
        action = np.random.choice(self.action_space)
    return action
    

  def store_transition(self, state, action, reward, next_state, done):
    self.memory.store_transition(state, action, reward, next_state, done)

  
  def sample_memory(self):
    state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)
    states = T.tensor(state).to(self.q_online.device)
    actions = T.tensor(action).to(self.q_online.device)
    rewards = T.tensor(reward).to(self.q_online.device)
    next_states = T.tensor(next_state).to(self.q_online.device)
    dones = T.tensor(done).to(self.q_online.device)
    return states, actions, rewards, next_states, dones

  
  def replace_target_network(self):
    if self.replace_target_cnt is not None and self.learn_step_counter % self.replace_target_cnt == 0:
      self.q_target.load_state_dict(self.q_online.state_dict())

 
 #That I did not do because it is very simple and will rework it later 
 #on to utilize more schemas for experimenting
  def decrement_epsilon(self):
    self.epsilon = self.epsilon - self.eps_dec \
                if self.epsilon > self.eps_min else self.eps_min

  
  #Saving and loading models -> That I didn't do because I couldn't be botherd
  def save_models(self):
    self.q_online.save_checkpoint()
    self.q_target.save_checkpoint()

  def load_models(self):
    self.q_online.save_checkpoint()
    self._q_target.save_checkpoint()
  
  def learn(self):
    #bail out if we don't have more than batch_size items in our memory buffer
    if self.memory.mem_cntr < self.batch_size:
      return
    
    self.q_online.optimizer.zero_grad()

    self.replace_target_network()

    states, actions, rewards, next_states, dones = self.sample_memory()

    #way to deal with the wrong demension
    indices = np.arange(self.batch_size)
    q_pred = self.q_online.forward(states)[indices, actions] #dims -> batch_size x n_actions

    # max returns (max_value, index_of_max) hence we only want the first
    q_next = self.q_target.forward(next_states).max(dim=1)[0]
    
    #if state is terminal target = reward otherwise it's td-target as usual
    #could do a conditional but here's a smarter way with done as a mask:
    #NOTE that this is the only detail I didn't get from my own implementation
    q_next[dones] = 0.0
    #if dones.any() == 1: print(q_next, dones)

    q_target = rewards + self.gamma * q_next

    loss = self.q_online.loss(q_target, q_pred).to(self.q_online.device)
    loss.backward()
    self.q_online.optimizer.step()
    self.learn_step_counter += 1

    self.decrement_epsilon()
    

class DuelingDDQNAgent():
  def __init__(self, gamma, epsilon, lr, n_actions, input_dims, 
               mem_size, batch_size, eps_min=0.01, eps_dec = 5e-7, 
               #what is replace for? A: Every "replace" steps update target network
               replace=1000, algo=None, env_name=None, chkpt_dir="tmp/dqn"):
    
    self.gamma = gamma
    self.epsilon = epsilon
    self.lr = lr
    self.n_actions = n_actions
    self.input_dims = input_dims
    self.batch_size = batch_size
    self.eps_min = eps_min
    self.eps_dec = eps_dec
    self.replace_target_cnt = replace
    self.algo = algo
    self.env_name = env_name
    self.chkpt_dir = chkpt_dir
    self.action_space = [i for i in range(self.n_actions)]  #for e-greeedy selection
    
    #usefl to count number of times online network ran so we know when to update
    #target network with eval network weights
    self.learn_step_counter = 0

    self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

    self.q_online = DuelingDeepQNetwork(lr=self.lr, n_actions=self.n_actions, 
                               input_dims = self.input_dims,
                               name = self.env_name+"_"+ self.algo+'_q_online',
                              chkpt_dir = self.chkpt_dir)

    
    self.q_target = DuelingDeepQNetwork(lr=self.lr, n_actions=self.n_actions, 
                               input_dims = self.input_dims,
                               name = self.env_name+"_"+ self.algo+ '_q_target',
                              chkpt_dir = self.chkpt_dir)

  def choose_action(self, observation):
    #Need [observation] because torch expect (batch_size, input_dims) 
    #and without them it will only be input_dims
    if np.random.random() > self.epsilon:
        state = T.tensor( [observation] , dtype=T.float).to(self.q_online.device)
        actions = self.q_online.forward(state)
        action = T.argmax(actions).item() #gym onlci accepts integers and numpy arrays
    else:
        action = np.random.choice(self.action_space)
    return action
    

  def store_transition(self, state, action, reward, next_state, done):
    self.memory.store_transition(state, action, reward, next_state, done)

  
  def sample_memory(self):
    state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)
    states = T.tensor(state).to(self.q_online.device)
    actions = T.tensor(action).to(self.q_online.device)
    rewards = T.tensor(reward).to(self.q_online.device)
    next_states = T.tensor(next_state).to(self.q_online.device)
    dones = T.tensor(done).to(self.q_online.device)
    return states, actions, rewards, next_states, dones

  
  def replace_target_network(self):
    if self.replace_target_cnt is not None and self.learn_step_counter % self.replace_target_cnt == 0:
      self.q_target.load_state_dict(self.q_online.state_dict())
 
 #That I did not do because it is very simple and will rework it later 
 #on to utilize more schemas for experimenting
  def decrement_epsilon(self):
    self.epsilon = self.epsilon - self.eps_dec \
                if self.epsilon > self.eps_min else self.eps_min

  
  #Saving and loading models -> That I didn't do because I couldn't be botherd
  def save_models(self):
    self.q_online.save_checkpoint()
    self.q_target.save_checkpoint()

  def load_models(self):
    self.q_online.save_checkpoint()
    self._q_target.save_checkpoint()
    
  def learn(self):
    if self.memory.mem_cntr < self.batch_size:
      return

    self.q_online.optimizer.zero_grad()

    self.replace_target_network()

    states, actions, rewards, states_, dones = self.sample_memory()

    indices = np.arange(self.batch_size)

    q_pred = self.q_online.forward(states)[indices, actions]
    q_next = self.q_target.forward(states_)
    q_eval = self.q_online.forward(states_)

    max_actions = T.argmax(q_eval, dim=1)
    q_next[dones] = 0.0

    q_target = rewards + self.gamma*q_next[indices, max_actions]
    loss = self.q_online.loss(q_target, q_pred).to(self.q_online.device)
    loss.backward()

    self.q_online.optimizer.step()
    self.learn_step_counter += 1

    self.decrement_epsilon()  
  
  
  def learn_BUGGED(self):
    #bail out if we don't have more than batch_size items in our memory buffer
    if self.memory.mem_cntr < self.batch_size:
      return
    
    self.q_online.optimizer.zero_grad()

    self.replace_target_network()

    states, actions, rewards, next_states, dones = self.sample_memory()

    #way to deal with the wrong demension
    indices = np.arange(self.batch_size)
    q_pred = self.q_online.forward(states)[indices, actions] #dims -> batch_size x n_actions

    #q_next = self.q_target.forward(next_states)
    q_eval = self.q_online.forward(next_states)

    #print(f"q_eval (from which we select argmax: {q_eval}")

    acts = q_eval.max(dim=1)[1]
    
    #print(f"acts: {acts}, shape: {acts.shape}")

    q_next = self.q_target.forward(next_states).gather(1, acts.unsqueeze(0)) 
    #print(f"q_next: {q_next}, shape: {q_next.shape}")
    #print(f"dones: {dones.shape}, dones: {dones}")
   
    #print(f"q_next init: {q_next}")
    q_next[dones.type(T.int64) ] = 0.0
    #q_next.gather(1, dones.unsqueeze(0).type(T.int64) )
    #print(f"q_next {q_next}")

    q_target = rewards + self.gamma * q_next

    loss = self.q_online.loss(q_target, q_pred).to(self.q_online.device)
    loss.backward()
    self.q_online.optimizer.step()
    self.learn_step_counter += 1

    self.decrement_epsilon()









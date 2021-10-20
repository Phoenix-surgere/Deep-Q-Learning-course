# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 13:46:48 2021

@author: black
"""
from collections import deque
import cv2
import numpy as np
import matplotlib.pyplot as plt
import gym

class RepeatActionAndMaxFrame(gym.Wrapper):
  def __init__(self, env, n_frames=4, clip_reward=False, no_ops=0, fire_first=False):
    super(RepeatActionAndMaxFrame, self).__init__(env)
    
    #Is that unnecessary?
    #self.env = env
    
    self.n_frames = n_frames

    #That's apparently wrong
    self.frame_buffer = np.zeros( tuple( [2] + list(env.observation_space.shape)) , dtype=np.uint8) 
    #self.shape = env.observation_space.low.shape
    #self.frame_buffer = np.zeros_like((2, self.shape))

    #print(self.frame_buffer.shape)

    #optinoal 
    self.clip_reward = clip_reward
    self.no_ops = no_ops
    self.fire_first = fire_first


  #not "def action", since there is a cretain terminology 'round here
  def step(self, action):
    total_reward, done  = 0.0, False
    
    #don't quite understand how many times to loop for
    #ANSWER: n_frames! 
    for i in range(self.n_frames):
      obs, reward, done, info = self.env.step(action)

      if self.clip_reward:
        reward = np.clip(np.array(reward), -1, 1  )[0]

      total_reward += reward
      #addition: didn't quite understand where that came from
      idx = i % 2

      #not obs[0] which is our image? had to replace it with just obs
      self.frame_buffer[idx] = obs
      if done:
        break
    
    #ANSWER: My implementation was correct! It's the same thing these are doing
    return self.frame_buffer.max(axis=0), total_reward, done, info


    #Model solution
    #max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])
    #return max_frame, total_reward, done, info

  def reset(self):
    obs_init = self.env.reset()
    no_ops = np.random.randint(self.no_ops) + 1 if self.no_ops > 0 else 0
    for _ in range(no_ops):
        _, _, done, _ = self.env.step(0)
        if done:
            self.env.reset()
    if self.fire_first:
      assert self.env.unwrapped.get_action_meanings()[1] == 'FIRE'
      obs, _, _, _ = self.env.step(1)

    self.frame_buffer[:] = 0
    self.frame_buffer[0] = obs_init
    return obs_init
 


class PreprocessFrame(gym.ObservationWrapper):

  def __init__(self, env, shape):
    super(PreprocessFrame, self).__init__(env)
    
    #THis apparently is not useful
    #self.env = env
    
    #NOT WRONG but below version parameterized
    #self.new_shape = (3, 210, 60)
    
    #more modular I GUESS if shape is the shape of the image we pass, (3,210, 60)
    self.new_shape  = (shape[2], shape[0], shape[1])
    
    #changed dtype from uint to float
    self.observation_space =  gym.spaces.Box(low= 0.0, high=1.0, shape= self.new_shape, dtype=np.float32)

  def observation(self, obs):
    #obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
    
    #Apprently there are two conversionf of RGB to gray - will follow the one
    #here in the solution though the web only showed the first one
    new_frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    

    #That's entirely wrong? => NO APPARENTLY it's just the last step, I skipped the
    #strange interplotation one with the "resized_Screen"
    #obs = obs.reshape( self.new_shape )
    
    #The interplotation bit was new nad I hadn't really understood I should do that
    resized_screen = cv2.resize(new_frame, self.new_shape[1: ],  interpolation=cv2.INTER_AREA)
    
    new_obs = np.array(resized_screen, dtype=np.uint8).reshape(self.new_shape)

    new_obs = new_obs / 255.0 
    return new_obs

class StackFrames(gym.ObservationWrapper):

  def __init__(self, env, stack_size=4):
    super(StackFrames, self).__init__(env)
    #self.env = env
    
    #NEED TO CHANGE observation_Space: Instead of a single RGB frame shape (210, 160, 3)
    #we need to make it shape (4, 210, 160) e.g 4 frames per obs; hence we use the Observation wrapper 

    #THIS I didn't get at all
    self.observation_space = gym.spaces.Box(
                                        env.observation_space.low.repeat(stack_size, axis=0),
                                        env.observation_space.high.repeat(stack_size, axis=0),
                                        dtype= np.float32  )
    
    #didn't add the maxlen param
    self.frame_stack = deque(maxlen=stack_size)

  def reset(self):

    #intiial code wasn't wrong but since there's a dedicated clear method...
    self.frame_stack.clear()

    obs = self.env.reset()

    for i in range(self.frame_stack.maxlen):
      self.frame_stack.append(obs)


   #  reshape stack to observation space low shape 
   #EDIT: removed first assigning this to self.frame_stack because it BUGS afterwards
    return np.array(self.frame_stack).reshape(self.observation_space.shape)


  def observation(self, obs):
    self.frame_stack.append(obs)

    #same thing here as with reset
    return np.array(self.frame_stack).reshape(self.observation_space.shape) 



#the params after stack_size are optional during testing
def make_env(env_name, new_shape= (84, 84, 1), stack_size=4, clip_rewards=False,
             no_ops=0, fire_first=False):
  env = gym.make(env_name)
  #print(f"Original Obs shape: {env.reset().shape}")
  env = RepeatActionAndMaxFrame(env, stack_size, clip_rewards, no_ops, fire_first )
  #print(f"Repeat and Maxed frame: {env.reset().shape}")
  env = PreprocessFrame(env, new_shape)
  #print(f"Preprocessed frame: {env.reset().shape}")
  env = StackFrames(env, stack_size)
  #print(f"Stacked Frames: {env.reset().shape}")

  return env


def plot_learning_curve(x, scores, epsilons, filename):
  fig  = plt.figure()
  ax = fig.add_subplot(111, label="1")
  ax2 = fig.add_subplot(111, label="2", frame_on = False)

  ax.plot(x, epsilons, color = "C0")
  ax.set_xlabel("Training Steps", color = "C0")
  ax.set_ylabel("Epsilon", color = "C0")
  ax.tick_params(axis='x', colors = "C0")
  ax.tick_params(axis = "y", colors="C0")

  N = len(scores)
  running_avg = np.empty(N)
  for t in range(N):
    running_avg[t] = np.mean( scores[max(0, t-100):(t+1)])

  ax2.scatter(x, running_avg, color="C1")
  ax2.axes.get_xaxis().set_visible(False)
  ax2.yaxis.tick_right()
  ax2.set_ylabel("Score", color="C1")
  ax2.yaxis.set_label_position('right')
  ax2.tick_params(axis="y", colors= "C1")

  plt.savefig(filename)